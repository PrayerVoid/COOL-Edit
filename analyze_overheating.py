"""
更新模式分析 —— 验证「更新集中在少数神经元上、导致参数变形与特异性下降」这一假设。

离线分析：只读取实验目录中的 updates/iter_*.pt、original_params.pt 与 results.json，
不需要加载模型。CPU 分块流式处理，每块加载即释放，峰值内存可控；支持断点续跑与实时进度文件。

逐层计算（阶段 A~E）：
  A+B. 每次迭代更新量的集中度（Gini、归一化熵、top-1/2/5/10/20/50% 质量占比）
       与各神经元的累积更新量
  C.   更新方向的一致性：相邻迭代（反复编辑时目标方向每次翻转）与同向迭代（隔一次）
       之间更新向量的余弦相似度；反向迭代取相邻序列中后一次迭代编号为奇数的那一半
  D.   参数变形：编辑后权重与原始权重的逐神经元余弦相似度，
       以及累积更新量与变形程度的对应关系
  E.   集中度 / 变形与 Specificity 的相关性
之后（阶段 F）做跨实验对比并绘制总览 dashboard。

用法:
  python analyze_overheating.py \
      --exp_dirs results/repeated_alphaedit results/repeated_random \
                 results/repeated_nmke results/repeated_cool \
      --labels alphaedit random nmke cool \
      --output_dir ./results/analysis_report

输出:
  - analysis_params.json、progress.json、checkpoint 文件
  - scalars/<label>/layer_<l>_<analysis>.json    逐层标量结果
  - tensors/<label>/layer_<l>_<analysis>.npz     逐层张量结果
  - cross_comparison.json                        跨实验对比汇总
  - *.png                                        集中度曲线、Lorenz 曲线、方向一致性、
                                                 变形、Specificity 相关性、跨实验对比、dashboard
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress, pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ===========================================================================
#  常量
# ===========================================================================

EDIT_LAYERS = [4, 5, 6, 7, 8]
CHUNK_SIZE = 20
PARALLEL = 2

# ===========================================================================
#  实时进度
# ===========================================================================

_PROGRESS_FILE = None  # 由 main 设置


def _log(msg: str, *, end: str = "\n") -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", end=end, flush=True)


def _write_progress(data: Dict) -> None:
    global _PROGRESS_FILE
    if _PROGRESS_FILE is None:
        return
    tmp = _PROGRESS_FILE.parent / "progress.json.tmp"
    try:
        data["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _PROGRESS_FILE)
    except Exception:
        pass


# ===========================================================================
#  工具函数
# ===========================================================================

def gini_np(x: np.ndarray) -> np.ndarray:
    B, N = x.shape
    total = x.sum(axis=1)
    mask = total > 1e-12
    out = np.zeros(B, dtype=np.float64)
    if not mask.any():
        return out
    xm = np.sort(x[mask], axis=1)
    idx = np.arange(1, N + 1, dtype=np.float64)
    g = 2.0 * (idx * xm).sum(axis=1) / (N * xm.sum(axis=1)) - (N + 1.0) / N
    out[mask] = g
    return out


def col_norms(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr.astype(np.float64), axis=1).astype(np.float32)


def pair_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    dot = (af * bf).sum(axis=1)
    denom = (np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1)) + 1e-12
    return (dot / denom).astype(np.float32)


# ===========================================================================
#  数据加载器
# ===========================================================================

class ExperimentLoader:
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        orig_path = exp_dir / "original_params.pt"
        if not orig_path.exists():
            raise FileNotFoundError(f"original_params.pt 不存在: {orig_path}")
        raw = torch.load(orig_path, map_location="cpu", weights_only=True)
        self.original_params: Dict[int, np.ndarray] = {}
        for k, v in raw.items():
            self.original_params[int(k.split("_")[1])] = v.float().numpy()

        results_path = exp_dir / "results.json"
        self.results: List[Dict] = []
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)

        updates_dir = exp_dir / "updates"
        if not updates_dir.exists():
            raise FileNotFoundError(f"updates/ 目录不存在: {updates_dir}")
        self.pt_files = sorted(
            updates_dir.glob("iter_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not self.pt_files:
            raise FileNotFoundError(f"updates/ 为空: {updates_dir}")
        self.num_iters = len(self.pt_files)

    def load_one_layer(self, layer: int, idx: int) -> np.ndarray:
        data = torch.load(self.pt_files[idx], map_location="cpu", weights_only=True)
        arr = data[str(layer)].float().numpy()
        del data
        return arr

    def load_chunk(self, layer: int, start: int, end: int) -> np.ndarray:
        parts = []
        for i in range(start, end):
            parts.append(self.load_one_layer(layer, i))
        arr = np.stack(parts)
        del parts
        gc.collect()
        return arr


# ===========================================================================
#  A+B. 集中度 + 累积更新（合并单遍 I/O）
# ===========================================================================

def _phase_a_b(loader, layer, chunk_size, label, progress_data):
    T = loader.num_iters
    n_neurons = loader.original_params[layer].shape[1]
    cumulative = np.zeros(n_neurons, dtype=np.float64)
    total_chunks = (T + chunk_size - 1) // chunk_size

    first_end = min(chunk_size, T)
    chunk = loader.load_chunk(layer, 0, first_end)
    norms0 = col_norms(chunk)
    del chunk; gc.collect()

    x = norms0.astype(np.float64) + 1e-12
    sorted_x = np.sort(x, axis=1)[:, ::-1]
    total = sorted_x.sum(axis=1, keepdims=True)
    g0 = gini_np(x)
    p = x / total
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    norm_ent = ent / np.log(n_neurons)

    all_gini = np.empty(T, dtype=np.float32)
    all_ent = np.empty(T, dtype=np.float32)
    all_norm_ent = np.empty(T, dtype=np.float32)
    top_pcts = [1, 2, 5, 10, 20, 50]
    top_arrays = {kp: np.empty(T, dtype=np.float32) for kp in top_pcts}

    all_gini[:first_end] = g0.astype(np.float32)
    all_ent[:first_end] = ent.astype(np.float32)
    all_norm_ent[:first_end] = norm_ent.astype(np.float32)
    for kp in top_pcts:
        k = max(1, int(n_neurons * kp / 100))
        top_arrays[kp][:first_end] = (sorted_x[:, :k].sum(axis=1) / total.squeeze(1)).astype(np.float32)

    cumulative += norms0.sum(axis=0).astype(np.float64)

    snapshot_indices = set(max(1, int(T * p)) - 1 for p in [0.1, 0.25, 0.5, 0.75, 1.0])
    snapshots = {}
    all_norms_for_freq = [norms0[t].copy() for t in range(first_end)]
    for t_global in range(first_end):
        if t_global in snapshot_indices:
            snapshots[t_global] = {"gini": float(all_gini[t_global])}

    del x, sorted_x, total, g0, p, ent, norm_ent, norms0; gc.collect()

    _log(f"      [A+B] 1/{total_chunks} ({first_end}/{T}) done")

    for chunk_idx, start in enumerate(range(chunk_size, T, chunk_size), start=2):
        end = min(start + chunk_size, T)
        B = end - start

        chunk = loader.load_chunk(layer, start, end)
        norms = col_norms(chunk)
        del chunk; gc.collect()

        x = norms.astype(np.float64) + 1e-12
        sorted_x = np.sort(x, axis=1)[:, ::-1]
        total = sorted_x.sum(axis=1, keepdims=True)
        all_gini[start:end] = gini_np(x).astype(np.float32)
        p = x / total
        all_ent[start:end] = (-(p * np.log(p + 1e-12)).sum(axis=1)).astype(np.float32)
        all_norm_ent[start:end] = (all_ent[start:end] / np.log(n_neurons)).astype(np.float32)

        for kp in top_pcts:
            k = max(1, int(n_neurons * kp / 100))
            top_arrays[kp][start:end] = (sorted_x[:, :k].sum(axis=1) / total.squeeze(1)).astype(np.float32)

        cumulative += norms.sum(axis=0).astype(np.float64)

        for t_local in range(B):
            t_global = start + t_local
            if t_global in snapshot_indices:
                snapshots[t_global] = {"gini": float(all_gini[t_global])}
            all_norms_for_freq.append(norms[t_local].copy())

        del x, sorted_x, total, p, norms; gc.collect()

        _log(f"      [A+B] {chunk_idx}/{total_chunks} ({end}/{T}) done")
        _write_progress({**progress_data, "phase": "A+B",
                         "chunk": f"{chunk_idx}/{total_chunks}", "iter": f"{end}/{T}"})

    all_norms_arr = np.stack(all_norms_for_freq)
    threshold = float(np.median(all_norms_arr))
    update_freq = (all_norms_arr > threshold).sum(axis=0).astype(np.float32)
    del all_norms_arr, all_norms_for_freq; gc.collect()

    concentration = {"gini": all_gini, "entropy": all_ent, "normalized_entropy": all_norm_ent}
    for kp in top_pcts:
        concentration[f"top_{kp}pct_ratio"] = top_arrays[kp]

    cumulative_result = {
        "cumulative_magnitude": cumulative.astype(np.float32),
        "update_frequency": update_freq,
        "cumulative_snapshots": snapshots,
        "n_neurons": n_neurons,
    }
    return concentration, cumulative_result


# ===========================================================================
#  C. 拔河效应
# ===========================================================================

def _phase_c(loader, layer, chunk_size, label, progress_data):
    T = loader.num_iters
    adj_mean, adj_med = [], []
    same_mean, same_med = [], []
    carry = []
    total_chunks = (T + chunk_size - 1) // chunk_size
    chunk_idx = 0

    for start in range(0, T, chunk_size):
        chunk_idx += 1
        end = min(start + chunk_size, T)
        actual_start = max(0, start - len(carry)) if carry else start

        chunk = loader.load_chunk(layer, actual_start, end)

        if carry:
            extended = np.concatenate([np.stack(carry), chunk], axis=0)
        else:
            extended = chunk
        EB = extended.shape[0]

        if EB >= 2:
            cs = pair_cosine(extended[1:], extended[:-1])
            adj_mean.extend(cs.mean(axis=1).tolist())
            adj_med.extend(np.median(cs, axis=1).tolist())
            del cs

        if EB >= 3:
            cs = pair_cosine(extended[2:], extended[:-2])
            same_mean.extend(cs.mean(axis=1).tolist())
            same_med.extend(np.median(cs, axis=1).tolist())
            del cs

        for arr in carry: del arr
        carry = [extended[-2].copy()] if EB >= 2 else []
        del extended, chunk; gc.collect()

        _log(f"      [C] {chunk_idx}/{total_chunks} ({end}/{T}) done")
        _write_progress({**progress_data, "phase": "C",
                         "chunk": f"{chunk_idx}/{total_chunks}", "iter": f"{end}/{T}"})

    for arr in carry: del arr
    gc.collect()

    opp_mean = [adj_mean[t - 1] for t in range(1, T) if t % 2 == 1]
    opp_med = [adj_med[t - 1] for t in range(1, T) if t % 2 == 1]

    return {
        "adjacent_cos_mean": adj_mean, "adjacent_cos_median": adj_med,
        "same_dir_cos_mean": same_mean, "same_dir_cos_median": same_med,
        "opposite_dir_cos_mean": opp_mean, "opposite_dir_cos_median": opp_med,
    }


# ===========================================================================
#  D. 参数变形
# ===========================================================================

def _phase_d(loader, layer, chunk_size, label, progress_data):
    T = loader.num_iters
    w0_f64 = loader.original_params[layer].astype(np.float64, copy=False)
    n_neurons = w0_f64.shape[1]
    cumulative = np.zeros_like(w0_f64)
    total_chunks = (T + chunk_size - 1) // chunk_size

    snapshot_indices = set(max(1, int(T * p)) - 1 for p in [0.1, 0.25, 0.5, 0.75, 1.0])
    cos_at_snapshot: Dict[int, np.ndarray] = {}
    deformation_norms: List[float] = []

    for chunk_idx, start in enumerate(range(0, T, chunk_size), start=1):
        end = min(start + chunk_size, T)
        B = end - start

        chunk = loader.load_chunk(layer, start, end)
        chunk_f64 = chunk.astype(np.float64)
        del chunk

        for t_local in range(B):
            cumulative += chunk_f64[t_local]
            deformation_norms.append(float(np.linalg.norm(cumulative, axis=0).mean()))
            t_global = start + t_local
            if t_global in snapshot_indices:
                w_curr = w0_f64 + cumulative
                dot = (w0_f64 * w_curr).sum(axis=0)
                cs = dot / (np.linalg.norm(w0_f64, axis=0) * np.linalg.norm(w_curr, axis=0) + 1e-12)
                cos_at_snapshot[t_global] = cs.astype(np.float32)
                del w_curr

        del chunk_f64; gc.collect()

        _log(f"      [D] {chunk_idx}/{total_chunks} ({end}/{T}) done")
        _write_progress({**progress_data, "phase": "D",
                         "chunk": f"{chunk_idx}/{total_chunks}", "iter": f"{end}/{T}"})

    w_final = w0_f64 + cumulative
    dot = (w0_f64 * w_final).sum(axis=0)
    n0 = np.linalg.norm(w0_f64, axis=0)
    nc = np.linalg.norm(w_final, axis=0)
    final_cos = (dot / (n0 * nc + 1e-12)).astype(np.float32)
    del w_final

    worst_k = min(20, n_neurons)
    worst_idx = np.argsort(final_cos)[:worst_k]
    cum_norms = np.linalg.norm(cumulative, axis=0).astype(np.float32)
    top_updated_idx = np.argsort(-cum_norms)[:worst_k]
    overlap = len(set(worst_idx.tolist()) & set(top_updated_idx.tolist()))

    return {
        "final_cos_sim": final_cos,
        "cos_at_snapshots": cos_at_snapshot,
        "deformation_norm_mean": deformation_norms,
        "worst_deformed_neurons": worst_idx.tolist(),
        "worst_cos_values": final_cos[worst_idx].tolist(),
        "top_updated_neurons": top_updated_idx.tolist(),
        "overlap_top_worst": overlap,
        "cumulative_per_neuron_norm": cum_norms,
    }


# ===========================================================================
#  E. 特异性关联
# ===========================================================================

def analyze_specificity_correlation(
    concentration: Dict[int, Dict[str, np.ndarray]],
    deformation: Dict[int, Dict[str, Any]],
    results: List[Dict],
    edit_layers: List[int],
) -> Dict[str, Any]:
    specificity = np.array(
        [r.get("neighborhood_success", float("nan")) for r in results], dtype=np.float64)
    valid = ~np.isnan(specificity)
    if valid.sum() < 5:
        return {"error": "Not enough valid specificity points"}
    spec_valid = specificity[valid]
    n = len(spec_valid)

    layer_correlations = {}
    for layer in edit_layers:
        conc = concentration.get(layer, {})
        corr = {}
        for metric, values in conc.items():
            if len(values) != len(specificity):
                continue
            vals = values[valid].astype(np.float64)
            if np.std(vals) < 1e-12:
                continue
            r_p, p_p = pearsonr(vals, spec_valid)
            r_s, p_s = spearmanr(vals, spec_valid)
            corr[metric] = {"pearson_r": round(float(r_p), 4), "pearson_p": round(float(p_p), 6),
                            "spearman_r": round(float(r_s), 4), "spearman_p": round(float(p_s), 6)}
        lag_analysis = {}
        for metric in ["gini", "normalized_entropy", "top_10pct_ratio"]:
            if metric not in conc:
                continue
            vals = conc[metric][valid].astype(np.float64)
            if np.std(vals) < 1e-12:
                continue
            lags = {}
            for lag in [1, 2, 3, 5, 10]:
                if lag >= n - 2:
                    break
                r, p = pearsonr(vals[:n - lag], spec_valid[lag:])
                lags[f"lag_{lag}"] = {"pearson_r": round(float(r), 4),
                                       "pearson_p": round(float(p), 6)}
            lag_analysis[metric] = lags
        layer_correlations[layer] = {"correlations": corr, "lag_analysis": lag_analysis}

    output: Dict[str, Any] = {"layer_correlations": layer_correlations}
    for layer in edit_layers:
        bl = deformation.get(layer, {})
        dn = bl.get("deformation_norm_mean", [])
        if len(dn) == len(specificity):
            dn_val = np.array(dn, dtype=np.float64)[valid]
            if np.std(dn_val) > 1e-12:
                r_p, p_p = pearsonr(dn_val, spec_valid)
                output[f"deformation_vs_specificity_layer_{layer}"] = {
                    "pearson_r": round(float(r_p), 4), "pearson_p": round(float(p_p), 6)}
    return output


# ===========================================================================
#  F. 跨实验对比
# ===========================================================================

def cross_experiment_comparison(exp_results: Dict[str, Dict]) -> Dict[str, Any]:
    comparison = {}
    for name, data in exp_results.items():
        rl = data.get("results", [])
        if not rl: continue
        sv = np.array([r.get("neighborhood_success", float("nan")) for r in rl])
        sv = sv[~np.isnan(sv)]
        conc = data.get("concentration", {})
        gini_trends = {}
        for layer, metrics in conc.items():
            gv = metrics.get("gini")
            if gv is not None and len(gv) >= 2:
                slope, _, _, _, _ = linregress(range(len(gv)), gv)
                gini_trends[f"layer_{layer}"] = round(float(slope), 8)
        comparison[name] = {
            "specificity_first": round(float(sv[0]), 4) if len(sv) else None,
            "specificity_last": round(float(sv[-1]), 4) if len(sv) else None,
            "specificity_decay": round(float(sv[-1] - sv[0]), 4) if len(sv) >= 2 else None,
            "gini_trends": gini_trends, "num_iterations": len(rl)}
    return comparison


# ===========================================================================
#  绘图
# ===========================================================================

def plot_concentration(concentration, edit_layers, exp_label, output_dir):
    d = output_dir / "concentration" / exp_label; d.mkdir(parents=True, exist_ok=True)
    for metric in ["gini", "normalized_entropy", "top_5pct_ratio", "top_10pct_ratio"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for layer in edit_layers:
            vals = concentration.get(layer, {}).get(metric)
            if vals is not None and len(vals):
                ax.plot(vals, marker=".", markersize=2, linewidth=1, alpha=0.8, label=f"Layer {layer}")
        ax.set_xlabel("Iteration"); ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{exp_label} - {metric.replace('_', ' ').title()}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"{metric}.png", dpi=150); plt.close(fig)


def plot_cumulative(cumulative, edit_layers, exp_label, output_dir):
    d = output_dir / "cumulative" / exp_label; d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for layer in edit_layers:
        cm = cumulative.get(layer, {}).get("cumulative_magnitude")
        if cm is not None:
            sm = np.sort(cm)[::-1]; cs = np.cumsum(sm) / (sm.sum() + 1e-12)
            ax.plot(np.linspace(0, 1, len(cs)), cs, linewidth=1.5, label=f"Layer {layer}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Uniform")
    ax.set_xlabel("Cumulative Fraction of Neurons"); ax.set_ylabel("Cumulative Fraction of Update Magnitude")
    ax.set_title(f"{exp_label} - Lorenz Curve"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(d / "lorenz_curve.png", dpi=150); plt.close(fig)
    for layer in edit_layers:
        cm = cumulative.get(layer, {}).get("cumulative_magnitude")
        if cm is None: continue
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(cm, bins=100, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(x=np.median(cm), color="red", linestyle="--", linewidth=1.2,
                   label=f"Median = {np.median(cm):.4f}")
        ax.set_xlabel("Cumulative Update Magnitude"); ax.set_ylabel("Neuron Count")
        ax.set_title(f"{exp_label} - Layer {layer} Cumulative Update Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"cumulative_hist_layer_{layer}.png", dpi=150); plt.close(fig)


def plot_tug_of_war(tow, edit_layers, exp_label, output_dir):
    d = output_dir / "tug_of_war" / exp_label; d.mkdir(parents=True, exist_ok=True)
    for layer in edit_layers:
        ld = tow.get(layer, {})
        if not ld: continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for key, color, label in [("adjacent_cos_mean", "b", "Adjacent (alternating)"),
                                   ("same_dir_cos_mean", "g", "Same-direction (skip 1)"),
                                   ("opposite_dir_cos_mean", "r", "Opposite-direction")]:
            if ld.get(key):
                ax.plot(ld[key], color + ".-", markersize=2, linewidth=0.8, alpha=0.7, label=label)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Iteration Pair Index"); ax.set_ylabel("Mean Column Cosine Similarity")
        ax.set_title(f"{exp_label} - Layer {layer}: Update Direction Consistency")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"direction_consistency_layer_{layer}.png", dpi=150); plt.close(fig)


def plot_deformation(deformation, edit_layers, exp_label, output_dir):
    d = output_dir / "deformation" / exp_label; d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer in edit_layers:
        norms = deformation.get(layer, {}).get("deformation_norm_mean", [])
        if norms: ax.plot(norms, linewidth=1.2, label=f"Layer {layer}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Per-Neuron Deformation Norm")
    ax.set_title(f"{exp_label} - Mean Deformation Norm"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(d / "deformation_norm.png", dpi=150); plt.close(fig)
    for layer in edit_layers:
        bl = deformation.get(layer, {})
        cs_arr = bl.get("final_cos_sim")
        if cs_arr is None: continue
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(cs_arr, bins=80, alpha=0.7, color="coral", edgecolor="white", linewidth=0.3)
        ax.axvline(x=np.median(cs_arr), color="blue", linestyle="--", linewidth=1.2,
                   label=f"Median = {np.median(cs_arr):.4f}")
        ax.set_xlabel("Cosine Similarity to Original (final)"); ax.set_ylabel("Neuron Count")
        ax.set_title(f"{exp_label} - Layer {layer}: Per-Neuron Cosine Similarity")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"final_cos_sim_layer_{layer}.png", dpi=150); plt.close(fig)
        cum_norms = bl.get("cumulative_per_neuron_norm")
        if cum_norms is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(cum_norms, cs_arr, s=1, alpha=0.5, c="steelblue", edgecolors="none")
            ax.set_xlabel("Cumulative Update Magnitude")
            ax.set_ylabel("Cosine Similarity to Original (final)")
            ax.set_title(f"{exp_label} - Layer {layer}: Magnitude vs Deformation")
            if len(cum_norms) > 1:
                z = np.polyfit(cum_norms, cs_arr, 1); xl = np.linspace(cum_norms.min(), cum_norms.max(), 100)
                ax.plot(xl, np.polyval(z, xl), "r-", linewidth=1.5, label=f"y={z[0]:.2e}x+{z[1]:.3f}")
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(d / f"magnitude_vs_deformation_layer_{layer}.png", dpi=150); plt.close(fig)


def plot_specificity_vs_concentration(concentration, results, edit_layers, exp_label, output_dir):
    d = output_dir / "correlation" / exp_label; d.mkdir(parents=True, exist_ok=True)
    spec = np.array([r.get("neighborhood_success", float("nan")) for r in results])
    for layer in edit_layers:
        conc = concentration.get(layer, {})
        if not conc: continue
        for metric in ["gini", "normalized_entropy", "top_10pct_ratio"]:
            mv = conc.get(metric)
            if mv is None or len(mv) == 0: continue
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.set_xlabel("Iteration"); ax1.set_ylabel("Specificity", color="tab:red")
            ax1.plot(spec, "r.-", markersize=3, linewidth=1, alpha=0.7, label="Specificity")
            ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.set_ylim(-0.05, 1.05)
            ax2 = ax1.twinx()
            ax2.set_ylabel(metric.replace("_", " ").title(), color="tab:blue")
            ax2.plot(mv, "b.-", markersize=3, linewidth=1, alpha=0.7,
                     label=metric.replace("_", " ").title())
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            ax1.set_title(f"{exp_label} - Layer {layer}: Spec vs {metric.replace('_', ' ').title()}")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(d / f"specificity_vs_{metric}_layer_{layer}.png", dpi=150); plt.close(fig)


def plot_cross_experiment(exp_results, comparison, output_dir):
    d = output_dir / "cross_experiment"; d.mkdir(parents=True, exist_ok=True)
    names = list(exp_results.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in names:
        rl = exp_results[name].get("results", [])
        if not rl: continue
        ax.plot([r.get("neighborhood_success", float("nan")) for r in rl], linewidth=1.5, alpha=0.8, label=name)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Specificity"); ax.set_title("Specificity Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(d / "specificity_comparison.png", dpi=150); plt.close(fig)
    first_layer = next((k for n in names for k in exp_results[n].get("concentration", {}).keys()), None)
    if first_layer is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in names:
            gv = exp_results[name].get("concentration", {}).get(first_layer, {}).get("gini")
            if gv is not None and len(gv): ax.plot(gv, linewidth=1.5, alpha=0.8, label=name)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Gini Coefficient")
        ax.set_title(f"Gini Comparison (Layer {first_layer})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(d / "gini_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    spec_decays = [comparison.get(n, {}).get("specificity_decay", 0) for n in names]
    ax.barh(names, spec_decays, color=["green" if sd >= 0 else "red" for sd in spec_decays], alpha=0.7)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Specificity Change (Last - First)"); ax.set_title("Specificity Decay by Experiment")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(d / "specificity_decay_bars.png", dpi=150); plt.close(fig)


def plot_dashboard(exp_results, edit_layers, output_dir):
    d = output_dir / "dashboard"; d.mkdir(parents=True, exist_ok=True)
    names = list(exp_results.keys()); n_exps = len(names)
    if n_exps == 0: return
    fig = plt.figure(figsize=(16, 4 * n_exps + 2))
    gs = GridSpec(n_exps + 1, 4, figure=fig, hspace=0.35, wspace=0.3)
    for ax, title in [(fig.add_subplot(gs[0, 0]), "Specificity"), (fig.add_subplot(gs[0, 1]), "Gini"),
                       (fig.add_subplot(gs[0, 2]), "Mean Deformation"), (fig.add_subplot(gs[0, 3]), "Spec vs Gini")]:
        ax.set_title(title)
    ax_spec, ax_gini, ax_deform, ax_corr = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for name in names:
        rl = exp_results[name].get("results", [])
        if rl: ax_spec.plot([r.get("neighborhood_success", float("nan")) for r in rl],
                             linewidth=1.2, alpha=0.8, label=name)
    ax_spec.set_ylim(-0.05, 1.05); ax_spec.legend(fontsize=6); ax_spec.grid(True, alpha=0.3)
    first_layer = next((k for n in names for k in exp_results[n].get("concentration", {}).keys()), None)
    if first_layer is not None:
        for name in names:
            gv = exp_results[name].get("concentration", {}).get(first_layer, {}).get("gini")
            if gv is not None and len(gv): ax_gini.plot(gv, linewidth=1.2, alpha=0.8, label=name)
            dn = exp_results[name].get("deformation", {}).get(first_layer, {}).get("deformation_norm_mean", [])
            if dn: ax_deform.plot(dn, linewidth=1.2, alpha=0.8, label=name)
    ax_gini.grid(True, alpha=0.3); ax_deform.legend(fontsize=6); ax_deform.grid(True, alpha=0.3)
    all_s, all_g, all_c = [], [], []
    for i, name in enumerate(names):
        conc = exp_results[name].get("concentration", {})
        if first_layer and first_layer in conc and "gini" in conc[first_layer]:
            gv = conc[first_layer]["gini"]
            sv = [r.get("neighborhood_success", float("nan")) for r in exp_results[name].get("results", [])]
            m = min(len(gv), len(sv)); all_g.extend(gv[:m]); all_s.extend(sv[:m])
            all_c.extend([f"C{i}"] * m)
    if all_s:
        ax_corr.scatter(all_g, all_s, c=all_c, s=3, alpha=0.5)
        if len(all_g) > 1:
            r_val, _ = pearsonr(all_g, all_s); ax_corr.set_title(f"Spec vs Gini (r={r_val:.3f})")
        ax_corr.set_xlabel("Gini"); ax_corr.set_ylabel("Specificity"); ax_corr.grid(True, alpha=0.3)
    for exp_idx, name in enumerate(names):
        row = exp_idx + 1
        rl = exp_results[name].get("results", [])
        if not rl: continue
        ax_cum = fig.add_subplot(gs[row, 0])
        for layer in edit_layers:
            cm = exp_results[name].get("cumulative", {}).get(layer, {}).get("cumulative_magnitude")
            if cm is not None:
                sm = np.sort(cm)[::-1]; cs = np.cumsum(sm) / (sm.sum() + 1e-12)
                ax_cum.plot(np.linspace(0, 1, len(cs)), cs, linewidth=1, label=f"L{layer}")
        ax_cum.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.4)
        ax_cum.set_title(f"{name}: Lorenz"); ax_cum.legend(fontsize=5)
        ax_tow = fig.add_subplot(gs[row, 1])
        for layer in edit_layers:
            adj = exp_results[name].get("tug_of_war", {}).get(layer, {}).get("adjacent_cos_mean", [])
            if adj: ax_tow.plot(adj, linewidth=0.8, alpha=0.7, label=f"L{layer}")
        ax_tow.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_tow.set_title(f"{name}: Adjacent Cos"); ax_tow.legend(fontsize=5); ax_tow.grid(True, alpha=0.3)
        ax_def = fig.add_subplot(gs[row, 2])
        for layer in edit_layers:
            cs_arr = exp_results[name].get("deformation", {}).get(layer, {}).get("final_cos_sim")
            if cs_arr is not None: ax_def.hist(cs_arr, bins=50, alpha=0.5, label=f"L{layer}", density=True)
        ax_def.set_title(f"{name}: Final Cos Dist"); ax_def.legend(fontsize=5)
        ax_ts = fig.add_subplot(gs[row, 3]); ax_ts2 = ax_ts.twinx()
        ax_ts.plot([r.get("neighborhood_success", float("nan")) for r in rl], "r-", linewidth=1, alpha=0.8)
        if first_layer:
            gv = exp_results[name].get("concentration", {}).get(first_layer, {}).get("gini")
            if gv is not None and len(gv): ax_ts2.plot(gv, "b-", linewidth=1, alpha=0.8)
        ax_ts.set_title(f"{name}: Spec vs Gini")
    fig.suptitle("Repeated Editing Analysis Dashboard", fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(d / "dashboard.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ===========================================================================
#  保存
# ===========================================================================

def save_layer_scalar(exp_name, layer, analysis_name, data, output_dir):
    sdir = output_dir / "scalars" / exp_name; sdir.mkdir(parents=True, exist_ok=True)
    ser = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            ser[k] = v.tolist() if v.ndim <= 1 else f"<array {list(v.shape)}>"
        elif isinstance(v, dict):
            ser[k] = {str(sk): (sv.tolist() if isinstance(sv, np.ndarray) else sv) for sk, sv in v.items()}
        else: ser[k] = v
    with open(sdir / f"layer_{layer}_{analysis_name}.json", "w", encoding="utf-8") as f:
        json.dump(ser, f, indent=2, ensure_ascii=False)


def save_layer_tensors(exp_name, layer, analysis_name, data, output_dir):
    tdir = output_dir / "tensors" / exp_name; tdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tdir / f"layer_{layer}_{analysis_name}.npz", **data)


# ===========================================================================
#  单实验处理
# ===========================================================================

def process_experiment(exp_dir, label, output_dir, chunk_size, global_start_time,
                         exp_idx, n_exps, steps_per_layer) -> Dict[str, Any]:
    """处理单个实验，返回该实验的全部逐层分析结果。"""
    output_dir = Path(output_dir)
    loader = ExperimentLoader(Path(exp_dir))
    results_list = loader.results
    edit_layers = EDIT_LAYERS

    concentration_all = {}
    cumulative_all = {}
    tug_of_war_all = {}
    deformation_all = {}

    completed = _load_ckpt(output_dir)
    done_exp_offset = sum(1 for k, v in completed.items() if v and k.startswith(label))

    for layer_idx, layer in enumerate(edit_layers):
        n_chunks = (loader.num_iters + chunk_size - 1) // chunk_size
        _log(f"  [{label}] Layer {layer} ({layer_idx+1}/{len(edit_layers)}, {loader.num_iters} iters, "
             f"{n_chunks} chunks)")

        # 计算全局进度基数
        base_done = exp_idx * len(edit_layers) * steps_per_layer + layer_idx * steps_per_layer
        base_total = n_exps * len(edit_layers) * steps_per_layer

        progress_data = {
            "label": label, "layer": layer,
            "layer_idx": f"{layer_idx+1}/{len(edit_layers)}",
            "exp_idx": f"{exp_idx+1}/{n_exps}",
            "done": base_done, "total": base_total,
        }

        # --- A+B ---
        ckpt_key = f"{label}_layer{layer}_ab"
        if not completed.get(ckpt_key):
            _log(f"  [{label}]   [A+B] Concentration + Cumulative...")
            t0 = time.time()
            concentration_all[layer], cumulative_all[layer] = _phase_a_b(
                loader, layer, chunk_size, label, {**progress_data, "done": base_done + 1, "total": base_total})
            save_layer_scalar(label, layer, "concentration", concentration_all[layer], output_dir)
            cum_scalar = {"n_neurons": cumulative_all[layer]["n_neurons"],
                          "snapshot_gini": {str(si): sd["gini"]
                                            for si, sd in cumulative_all[layer]["cumulative_snapshots"].items()}}
            save_layer_scalar(label, layer, "cumulative", cum_scalar, output_dir)
            save_layer_tensors(label, layer, "cumulative",
                               {"cumulative_magnitude": cumulative_all[layer]["cumulative_magnitude"],
                                "update_frequency": cumulative_all[layer]["update_frequency"]}, output_dir)
            completed[ckpt_key] = True; _save_ckpt(output_dir, completed)
            _log(f"  [{label}]   [A+B] done ({time.time() - t0:.0f}s)")
        else:
            _log(f"  [{label}]   [A+B] cached")

        # --- C ---
        ckpt_key = f"{label}_layer{layer}_tug_of_war"
        if not completed.get(ckpt_key):
            _log(f"  [{label}]   [C] Tug-of-war...")
            t0 = time.time()
            tug_of_war_all[layer] = _phase_c(loader, layer, chunk_size, label,
                                             {**progress_data, "done": base_done + 2, "total": base_total})
            save_layer_scalar(label, layer, "tug_of_war", tug_of_war_all[layer], output_dir)
            completed[ckpt_key] = True; _save_ckpt(output_dir, completed)
            _log(f"  [{label}]   [C] done ({time.time() - t0:.0f}s)")
        else:
            _log(f"  [{label}]   [C] cached")

        # --- D ---
        ckpt_key = f"{label}_layer{layer}_deformation"
        if not completed.get(ckpt_key):
            _log(f"  [{label}]   [D] Deformation...")
            t0 = time.time()
            deformation_all[layer] = _phase_d(loader, layer, chunk_size, label,
                                              {**progress_data, "done": base_done + 3, "total": base_total})
            deform_scalar = {"deformation_norm_mean": deformation_all[layer]["deformation_norm_mean"],
                             "overlap_top_worst": deformation_all[layer]["overlap_top_worst"],
                             "worst_deformed_neurons": deformation_all[layer]["worst_deformed_neurons"][:20],
                             "worst_cos_values": deformation_all[layer]["worst_cos_values"],
                             "top_updated_neurons": deformation_all[layer]["top_updated_neurons"][:20]}
            save_layer_scalar(label, layer, "deformation", deform_scalar, output_dir)
            save_layer_tensors(label, layer, "deformation",
                               {"final_cos_sim": deformation_all[layer]["final_cos_sim"],
                                "cumulative_per_neuron_norm": deformation_all[layer]["cumulative_per_neuron_norm"]},
                               output_dir)
            completed[ckpt_key] = True; _save_ckpt(output_dir, completed)
            _log(f"  [{label}]   [D] done ({time.time() - t0:.0f}s)")
        else:
            _log(f"  [{label}]   [D] cached")

        gc.collect()

    # --- E ---
    ckpt_key = f"{label}_correlation"
    correlation = {}
    if not completed.get(ckpt_key):
        _log(f"  [{label}] [E] Specificity correlations...")
        t0 = time.time()
        correlation = analyze_specificity_correlation(concentration_all, deformation_all, results_list, edit_layers)
        save_layer_scalar(label, 0, "correlation", correlation, output_dir)
        completed[ckpt_key] = True; _save_ckpt(output_dir, completed)
        _log(f"  [{label}] [E] done ({time.time() - t0:.0f}s)")
    else:
        _log(f"  [{label}] [E] cached")

    # final progress update
    eta = (time.time() - global_start_time) / (exp_idx + 1) * (n_exps - exp_idx - 1)
    _write_progress({"label": label, "status": "completed",
                      "exp_idx": f"{exp_idx+1}/{n_exps}",
                      "elapsed": round(time.time() - global_start_time),
                      "eta_seconds": round(eta)})

    return {"label": label, "exp_dir": str(exp_dir), "edit_layers": edit_layers, "results": results_list,
            "num_iters": loader.num_iters, "concentration": concentration_all, "cumulative": cumulative_all,
            "tug_of_war": tug_of_war_all, "deformation": deformation_all, "correlation": correlation}


def _save_ckpt(output_dir, completed):
    ckpt = output_dir / "analysis_checkpoint.json"
    tmp = output_dir / "analysis_checkpoint.json.tmp"
    with open(tmp, "w") as f: json.dump({"completed_analyses": completed}, f)
    os.replace(tmp, ckpt)


def _load_ckpt(output_dir):
    p = output_dir / "analysis_checkpoint.json"
    if p.exists():
        try:
            with open(p) as f: return json.load(f).get("completed_analyses", {})
        except Exception: pass
    return {}


# ===========================================================================
#  主程序
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="更新模式分析（离线，仅需 CPU）")
    parser.add_argument("--exp_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--parallel", type=int, default=PARALLEL)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    parallel_n = args.parallel
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global _PROGRESS_FILE
    _PROGRESS_FILE = output_dir / "progress.json"

    if args.labels is None:
        labels = [Path(d).name for d in args.exp_dirs]
    else:
        labels = args.labels
    assert len(labels) == len(args.exp_dirs), "labels 和 exp_dirs 数量不一致"

    with open(output_dir / "analysis_params.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    per_iter_fp32 = 4096 * 14336 * 4 / 1e9
    per_iter_fp64 = 4096 * 14336 * 8 / 1e9
    peak = chunk_size * (per_iter_fp32 + per_iter_fp64) + 1
    _log(f"{'='*50}")
    _log(f"更新模式分析")
    _log(f"  实验数: {len(labels)}  编辑层: {EDIT_LAYERS}")
    _log(f"  分块: {chunk_size} iters/块  并行: {parallel_n}")
    _log(f"  内存预估: ~{peak:.1f}GB/进程 × {parallel_n} = ~{peak*parallel_n:.1f}GB")
    _log(f"  进度文件: {_PROGRESS_FILE}")
    _log(f"{'='*50}")

    steps_per_layer = 4  # A+B、C、D 三步 + 1 步预留
    total_steps = len(labels) * len(EDIT_LAYERS) * steps_per_layer

    _write_progress({"status": "starting", "overall_pct": 0, "overall_done": 0,
                      "overall_total": total_steps, "experiments": labels})

    completed = _load_ckpt(output_dir) if args.resume else {}
    if completed:
        _log(f"断点续跑: {sum(1 for v in completed.values() if v)} 个分析已缓存")

    all_exp_results: Dict[str, Dict[str, Any]] = {}
    global_start = time.time()

    if parallel_n > 1 and len(labels) > 1:
        actual = min(parallel_n, len(labels))
        _log(f"启动 {actual} 个子进程...")

        remaining = list(zip(args.exp_dirs, labels))
        while remaining:
            batch = remaining[:actual]; remaining = remaining[actual:]
            with Pool(processes=len(batch)) as pool:
                asyncs = []
                for exp_dir, label in batch:
                    exp_idx = labels.index(label)
                    r = pool.apply_async(process_experiment,
                                         (exp_dir, label, str(output_dir), chunk_size,
                                          global_start, exp_idx, len(labels), steps_per_layer))
                    asyncs.append((label, r))
                for label, r in asyncs:
                    all_exp_results[label] = r.get()
                    _log(f"  ✅ {label} complete ({len(all_exp_results)}/{len(labels)})")
    else:
        for exp_idx, (exp_dir, label) in enumerate(zip(args.exp_dirs, labels)):
            _log(f"{'='*50}")
            _log(f"Experiment [{exp_idx+1}/{len(labels)}]: {label}")
            _log(f"  {exp_dir}")
            _log(f"{'='*50}")
            all_exp_results[label] = process_experiment(
                exp_dir, label, output_dir, chunk_size, global_start, exp_idx, len(labels), steps_per_layer)

    # --- F ---
    _log("[F] Cross-experiment comparison...")
    comparison = cross_experiment_comparison(all_exp_results)
    with open(output_dir / "cross_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # --- 绘图 ---
    if not args.skip_plots:
        _log("Plotting...")
        for name, data in all_exp_results.items():
            plot_concentration(data.get("concentration", {}), EDIT_LAYERS, name, output_dir)
            plot_cumulative(data.get("cumulative", {}), EDIT_LAYERS, name, output_dir)
            plot_tug_of_war(data.get("tug_of_war", {}), EDIT_LAYERS, name, output_dir)
            plot_deformation(data.get("deformation", {}), EDIT_LAYERS, name, output_dir)
            plot_specificity_vs_concentration(data.get("concentration", {}), data.get("results", []),
                                              EDIT_LAYERS, name, output_dir)
        plot_cross_experiment(all_exp_results, comparison, output_dir)
        plot_dashboard(all_exp_results, EDIT_LAYERS, output_dir)
        _log("Plots saved.")

    _write_progress({"status": "completed", "overall_pct": 100, "overall_done": total_steps,
                      "overall_total": total_steps, "elapsed_seconds": round(time.time() - global_start)})

    _log(f"\n分析完成! 总用时 {time.time() - global_start:.0f}s")
    _log(f"输出: {output_dir}")
    for name in all_exp_results:
        c = comparison.get(name, {})
        if c:
            _log(f"  {name}: specificity {c.get('specificity_first')} -> "
                 f"{c.get('specificity_last')} ({c.get('specificity_decay', 0):+.4f})")
            if c.get("gini_trends"):
                _log(f"         avg gini trend: {np.mean(list(c['gini_trends'].values())):+.2e}/iter")


if __name__ == "__main__":
    main()
