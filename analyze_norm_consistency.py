"""
神经元更新范数一致性与变形分析。

单进程、单遍扫描磁盘上所有 iter_*.pt，每次加载一个文件就提取全部层的
delta，分别计算列 L2 范数。最少 I/O、最少 CPU、最少内存。

用法:
  # 顺序编辑全部分析（LLaMA 8B 默认层 [4,5,6,7,8]）
  python analyze_norm_consistency.py ^
      --exp_dirs results/sequential_alphaedit ^
      --labels sequential ^
      --output_dir ./results/analysis_report/norm_consistency

  # GPT-2 XL 分析（指定层）
  python analyze_norm_consistency.py ^
      --exp_dirs results/gpt2_repeated ^
      --labels gpt2 ^
      --layers 13 14 15 16 17 ^
      --output_dir ./results/analysis_report/norm_consistency

  # 反复编辑仅一致性（直方图和变形已有结果）
  python analyze_norm_consistency.py ^
      --exp_dirs results/repeated_alphaedit ^
      --labels repeated_ae ^
      --output_dir ./results/analysis_report/norm_consistency ^
      --skip_deformation

  # 多个实验串行（默认），每实验单线程跑满磁盘带宽
  python analyze_norm_consistency.py ^
      --exp_dirs results/seq results/rep ^
      --labels seq rep ^
      --output_dir ./results/analysis_report/norm_consistency

输出:
  - norms/<label>_layer<l>_norms.npy        每次迭代的各神经元更新范数 [T, neurons]
  - norms/<label>_layer<l>_cumulative.npy   累积更新量 [neurons]（供 analyze_neuron_restore.py 使用）
  - consistency/consistency_stats_<label>.json
  - consistency_decay_*.png / cumulative_hist_*.png / final_cos_sim_*.png / mag_vs_cos_*.png
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore")

# ===========================================================================
#  常量
# ===========================================================================

N_COLORS = 10
COLORS = plt.cm.Set1(np.linspace(0, 1, N_COLORS))


# ===========================================================================
#  数据加载器
# ===========================================================================


class ExperimentLoader:
    """加载实验目录中的更新矩阵和原始权重。"""

    def __init__(self, exp_dir: str, edit_layers: List[int]):
        exp_dir = Path(exp_dir)
        self.exp_dir = exp_dir
        self.edit_layers = edit_layers

        orig_path = exp_dir / "original_params.pt"
        if not orig_path.exists():
            raise FileNotFoundError(f"original_params.pt 不存在: {orig_path}")
        raw_orig = torch.load(orig_path, map_location="cpu", weights_only=True)
        self.original_params: Dict[int, np.ndarray] = {}
        for k, v in raw_orig.items():
            self.original_params[int(k.split("_")[1])] = v.float().numpy()
        del raw_orig

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

        # 推断维度
        first = torch.load(self.pt_files[0], map_location="cpu", weights_only=True)
        first_layer_key = str(self.edit_layers[0])
        if first_layer_key not in first:
            first_layer_key = str(list(self.original_params.keys())[0])
        example = first[first_layer_key]
        self.hidden_dim = example.shape[0]
        self.intermediate_dim = example.shape[1]
        del first, example
        gc.collect()

        print(f"  Loaded {self.num_iters} files ({self.hidden_dim}, {self.intermediate_dim})")


# ===========================================================================
#  单实验处理函数（核心逻辑）
# ===========================================================================


def process_experiment(
    exp_dir: str,
    label: str,
    edit_layers: List[int],
    output_dir: Path,
    skip_deformation: bool,
) -> Dict[str, Any]:
    """处理单个实验。

    I/O 策略 —— 单遍扫描：
      对 t in range(T):
        加载 iter_{t:03d}.pt（瞬态 1.17 GB）
        遍历每层，提取 delta、计算列范数
        释放文件
    总 I/O: T × ~1.17 GB = 234 GB (@200 iters) → ~78s (@3 GB/s NVMe)
    稳态内存: norms 矩阵 5×11MB + cum_delta 5×470MB(deform, float64) ≈ 2.4 GB
    """
    print(f"\n{'=' * 50}")
    print(f"Processing: {label}")
    print(f"  Dir: {exp_dir}")
    print(f"{'=' * 50}")

    t_start = time.time()
    loader = ExperimentLoader(exp_dir, edit_layers)
    T = loader.num_iters
    layers_present = [l for l in edit_layers if l in loader.original_params]
    n_layers = len(layers_present)
    n_neurons = loader.intermediate_dim
    hidden_dim = loader.hidden_dim

    # 预分配
    norms = np.zeros((n_layers, T, n_neurons), dtype=np.float32)
    cum_norms = np.zeros((n_layers, n_neurons), dtype=np.float64)

    cum_weight_delta: Optional[np.ndarray] = None
    if not skip_deformation:
        cum_weight_delta = np.zeros((n_layers, hidden_dim, n_neurons), dtype=np.float64)

    # --- 单遍扫描 ---
    report_interval = max(1, T // 20)
    for t in range(T):
        data = torch.load(loader.pt_files[t], map_location="cpu", weights_only=True)

        for li, layer in enumerate(layers_present):
            delta = data[str(layer)].float().numpy()  # [hidden, neurons]
            col_n = np.linalg.norm(delta, axis=0)      # [neurons]

            norms[li, t] = col_n.astype(np.float32)
            cum_norms[li] += col_n.astype(np.float64)

            if cum_weight_delta is not None:
                cum_weight_delta[li] += delta.astype(np.float64)

        del data
        gc.collect()

        if (t + 1) % report_interval == 0:
            pct = 100 * (t + 1) // T
            mem_mb = cum_weight_delta.nbytes / 1e6 if cum_weight_delta is not None else 0
            print(f"  [{label}] {t + 1}/{T} ({pct}%), "
                  f"norms mem: {norms.nbytes / 1e6:.0f} MB"
                  + (f", cum_delta mem: {mem_mb:.0f} MB" if mem_mb else ""))

    # --- consistency（Spearman 秩相关）---
    print(f"  Computing Spearman rank correlations...")
    t0 = time.time()
    consistency: Dict[int, np.ndarray] = {}
    for li, layer in enumerate(layers_present):
        ref = norms[li, 0]
        rhos = np.full(T, 1.0, dtype=np.float64)
        for t in range(1, T):
            r, _ = spearmanr(ref, norms[li, t])
            rhos[t] = r if not np.isnan(r) else 0.0
        consistency[layer] = rhos
    print(f"  Spearman done ({time.time() - t0:.1f}s)")

    # 保存中间数据 norms
    norms_dir = output_dir / "norms"
    norms_dir.mkdir(parents=True, exist_ok=True)
    for li, layer in enumerate(layers_present):
        np.save(norms_dir / f"{label}_layer{layer}_norms.npy", norms[li].astype(np.float32))
        np.save(norms_dir / f"{label}_layer{layer}_cumulative.npy", cum_norms[li].astype(np.float64))

    # --- 绘图：一致性衰减 ---
    _plot_consistency_decay(consistency, label, output_dir)

    # --- 保存 stats ---
    stats = {}
    for layer, rhos in consistency.items():
        stats[f"layer_{layer}"] = {
            "rho_mean": float(rhos.mean()),
            "rho_first": float(rhos[0]),
            "rho_last": float(rhos[-1]),
            "rho_std": float(rhos.std()),
        }
    stats_path = output_dir / "consistency" / f"consistency_stats_{label}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 直方图 + 变形（非跳过模式）---
    deformation: Dict[int, Dict[str, Any]] = {}
    if not skip_deformation and cum_weight_delta is not None:
        for li, layer in enumerate(layers_present):
            W_orig = loader.original_params[layer].astype(np.float64)
            W_final = W_orig + cum_weight_delta[li]

            cum = cum_norms[li].copy()
            _phase_histogram(cum, layer, label, output_dir)

            deform = _phase_deformation(W_final, W_orig, cum, layer, label, output_dir)
            deformation[layer] = deform
            stats[f"layer_{layer}"].update(deform)

            del W_final, cum
            gc.collect()

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\n  [{label}] Done in {elapsed:.1f}s")

    return {
        "label": label,
        "consistency": consistency,
        "cumulative_norms": {l: cum_norms[li] for li, l in enumerate(layers_present)},
        "deformation": deformation,
    }


# ===========================================================================
#  绘图
# ===========================================================================


def _plot_consistency_decay(
    consistency: Dict[int, np.ndarray],
    label: str,
    output_dir: Path,
):
    """单实验的一致性衰减曲线（跨层叠加）。"""
    d = output_dir / "consistency"
    d.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, layer in enumerate(sorted(consistency.keys())):
        rhos = consistency[layer]
        c = COLORS[idx % N_COLORS]
        ax.plot(rhos, color=c, linewidth=1.2, label=f"Layer {layer}")

    ax.set_xlabel("Step t (reference = step 0)")
    ax.set_ylabel("Spearman rho(N[0], N[t])")
    ax.set_title(f"{label}: Consistency Decay")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / f"consistency_decay_{label}.png", dpi=150)
    plt.close(fig)


def _plot_consistency_decay_comparison(
    all_consistency: Dict[str, Dict[int, np.ndarray]],
    output_dir: Path,
):
    """跨实验的一致性衰减对比。"""
    d = output_dir / "consistency"
    d.mkdir(parents=True, exist_ok=True)

    all_layers = sorted({l for c in all_consistency.values() for l in c})

    # 按层分图
    fig, axes = plt.subplots(
        len(all_layers), 1, figsize=(10, 3 * len(all_layers)), squeeze=False
    )
    for row, layer in enumerate(all_layers):
        ax = axes[row, 0]
        for idx, (lab, consistency) in enumerate(all_consistency.items()):
            if layer not in consistency:
                continue
            rhos = consistency[layer]
            c = COLORS[idx % N_COLORS]
            ax.plot(rhos, color=c, linewidth=1.2, alpha=0.8, label=lab)
        ax.set_title(f"Layer {layer}")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == len(all_layers) - 1:
            ax.set_xlabel("Step t (reference = step 0)")
        ax.set_ylabel("Spearman rho")
    fig.tight_layout()
    fig.savefig(d / "consistency_decay_comparison_all.png", dpi=150)
    plt.close(fig)

    # 跨层平均后跨实验对比
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (lab, consistency) in enumerate(all_consistency.items()):
        rhos_mean = np.mean([consistency[l] for l in consistency], axis=0)
        c = COLORS[idx % N_COLORS]
        ax.plot(rhos_mean, color=c, linewidth=1.5, label=lab)
    ax.set_xlabel("Step t (reference = step 0)")
    ax.set_ylabel("Spearman rho (mean across layers)")
    ax.set_title("Consistency Decay: Cross-Experiment Comparison")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "consistency_decay_comparison_all_avg.png", dpi=150)
    plt.close(fig)


def _phase_histogram(
    cumulative: np.ndarray,
    layer: int,
    label: str,
    output_dir: Path,
):
    """累积更新量分布直方图。"""
    d = output_dir / "histograms"
    d.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(cumulative, bins=100, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.3)
    median_val = float(np.median(cumulative))
    ax.axvline(x=median_val, color="red", linestyle="--", linewidth=1.2,
               label=f"Median = {median_val:.4f}")
    ax.set_xlabel("Cumulative Update Magnitude (sum of per-step norms)")
    ax.set_ylabel("Neuron Count")
    ax.set_title(f"{label} - Layer {layer} Cumulative Update Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / f"cumulative_hist_{label}_layer{layer}.png", dpi=150)
    plt.close(fig)


def _phase_histogram_comparison(
    all_cumulative: Dict[str, Dict[int, np.ndarray]],
    output_dir: Path,
):
    """跨实验对比直方图，按层叠加。"""
    d = output_dir / "histograms"
    d.mkdir(parents=True, exist_ok=True)

    all_layers = sorted({l for cm in all_cumulative.values() for l in cm})

    for layer in all_layers:
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, (lab, cm) in enumerate(all_cumulative.items()):
            if layer not in cm:
                continue
            c = COLORS[idx % N_COLORS]
            ax.hist(cm[layer], bins=100, alpha=0.5, color=c,
                    edgecolor="white", linewidth=0.2, label=lab)
        ax.set_xlabel("Cumulative Update Magnitude")
        ax.set_ylabel("Neuron Count")
        ax.set_title(f"Layer {layer}: Cumulative Update Distribution Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(d / f"cumulative_hist_comparison_layer{layer}.png", dpi=150)
        plt.close(fig)


def _phase_deformation(
    W_final: np.ndarray,
    W_orig: np.ndarray,
    cumulative_norms: np.ndarray,
    layer: int,
    label: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """余弦相似度直方图 + 累积范数 vs 余弦散点图。"""
    d = output_dir / "deformation"
    d.mkdir(parents=True, exist_ok=True)

    # 余弦相似度
    n0 = np.linalg.norm(W_orig, axis=0)
    nf = np.linalg.norm(W_final, axis=0)
    dot = (W_orig * W_final).sum(axis=0)
    cos_sim = (dot / (n0 * nf + 1e-12)).astype(np.float32)

    stats: Dict[str, Any] = {
        "cos_sim_median": float(np.median(cos_sim)),
        "cos_sim_mean": float(np.mean(cos_sim)),
        "cos_sim_std": float(np.std(cos_sim)),
        "cos_sim_below_0.9": float((cos_sim < 0.9).mean()),
        "cos_sim_below_0.5": float((cos_sim < 0.5).mean()),
    }

    # 直方图
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(cos_sim, bins=80, alpha=0.7, color="coral",
            edgecolor="white", linewidth=0.3)
    median_cos = float(np.median(cos_sim))
    ax.axvline(x=median_cos, color="blue", linestyle="--", linewidth=1.2,
               label=f"Median = {median_cos:.4f}")
    ax.set_xlabel("Cosine Similarity to Original (final)")
    ax.set_ylabel("Neuron Count")
    ax.set_title(f"{label} - Layer {layer}: Per-Neuron Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / f"final_cos_sim_{label}_layer{layer}.png", dpi=150)
    plt.close(fig)

    # 散点图
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cumulative_norms, cos_sim, s=1, alpha=0.5,
               c="steelblue", edgecolors="none")
    ax.set_xlabel("Cumulative Update Magnitude")
    ax.set_ylabel("Cosine Similarity to Original (final)")
    ax.set_title(f"{label} - Layer {layer}: Magnitude vs Deformation")
    if len(cumulative_norms) > 1:
        z = np.polyfit(cumulative_norms, cos_sim, 1)
        xl = np.linspace(cumulative_norms.min(), cumulative_norms.max(), 100)
        ax.plot(xl, np.polyval(z, xl), "r-", linewidth=1.5,
                label=f"y={z[0]:.2e}x+{z[1]:.3f}")
        ax.legend(fontsize=8)
        stats["linear_fit_slope"] = float(z[0])
        stats["linear_fit_intercept"] = float(z[1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / f"mag_vs_cos_{label}_layer{layer}.png", dpi=150)
    plt.close(fig)

    return stats


# ===========================================================================
#  主程序
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="神经元更新范数一致性与变形分析"
    )
    parser.add_argument(
        "--exp_dirs", type=str, nargs="+", required=True,
        help="实验目录列表（需含 original_params.pt 和 updates/）",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="实验标签（数量须与 --exp_dirs 一致）",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[4, 5, 6, 7, 8],
        help="编辑层号列表，默认 [4,5,6,7,8]（LLaMA 8B）；GPT-2 XL 使用 [13,14,15,16,17]",
    )
    parser.add_argument(
        "--skip_deformation", action="store_true",
        help="跳过直方图和变形分析（反复编辑已有现成结果时使用）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edit_layers = args.layers

    labels: List[str]
    if args.labels is None:
        labels = [Path(d).name for d in args.exp_dirs]
    else:
        labels = args.labels
    assert len(labels) == len(args.exp_dirs), \
        f"labels ({len(labels)}) 和 exp_dirs ({len(args.exp_dirs)}) 数量不一致"

    with open(output_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Neuron Norm Consistency Analysis")
    print(f"  Experiments: {len(labels)} (skipping deformation: {args.skip_deformation})")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")

    # 串行处理每个实验（每实验内单线程单遍扫描）
    all_consistency: Dict[str, Dict[int, np.ndarray]] = {}
    all_cumulative: Dict[str, Dict[int, np.ndarray]] = {}

    for exp_idx, (exp_dir, label) in enumerate(zip(args.exp_dirs, labels)):
        result = process_experiment(exp_dir, label, edit_layers, output_dir, args.skip_deformation)
        all_consistency[label] = result["consistency"]
        all_cumulative[label] = result.get("cumulative_norms", {})

    # --- 跨实验对比图 ---
    if len(labels) > 1:
        _plot_consistency_decay_comparison(all_consistency, output_dir)
        if not args.skip_deformation:
            _phase_histogram_comparison(all_cumulative, output_dir)

    # 汇总
    summary = {}
    for label in labels:
        cons = all_consistency.get(label, {})
        entry: Dict[str, Any] = {}
        for layer, rhos in cons.items():
            entry[f"layer_{layer}_rho_last"] = float(rhos[-1]) if len(rhos) else None
        if cons:
            first_layer = list(cons.keys())[0]
            entry["num_iters"] = len(cons[first_layer])
        summary[label] = entry

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete!")
    for label in labels:
        n = summary[label].get("num_iters", 0)
        print(f"  {label}: {n} iterations")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
