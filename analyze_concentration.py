"""
更新集中度分析：对数据集前 N 条知识依次用 AlphaEdit 编辑（每条仅编辑一次），
每次编辑后计算更新矩阵的熵、基尼系数、幅值等集中度指标。

用法:
  python analyze_concentration.py ^
      --hparams hparams/AlphaEdit/llama3.1-8b.yaml ^
      --dataset data/counterfact.json ^
      --num_edits 100 --output_dir ./results/concentration

输出:
  - original_params.pt    编辑前原始权重
  - updates/iter_*.pt     每次编辑的更新量 delta
  - metrics/layer_*_metrics.json  每层的集中度指标序列
  - results.json          编辑评估结果
  - resume.pt             断点检查点（支持 --resume 续跑）

断点续跑:
  python analyze_concentration.py ^
      --hparams ... --dataset ... ^
      --num_edits 200 --output_dir ./results/concentration ^
      --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import hmean

# ── EasyEdit 路径 ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from easyeditor import BaseEditor
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
from easyeditor.models.alphaedit.AlphaEdit_main import (
    get_context_templates,
    upd_matrix_match_shape,
)
from easyeditor.models.alphaedit.compute_ks import compute_ks
from easyeditor.models.alphaedit.compute_z import (
    compute_z,
    get_module_input_output_at_words,
)
from easyeditor.util import nethook

# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════


def gini_np(x: np.ndarray) -> np.ndarray:
    """计算基尼系数。x: [B, N]，返回 [B]。"""
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


def compute_concentration_metrics(
    delta: torch.Tensor,
) -> Dict[str, float]:
    """从单层更新矩阵计算集中度指标。

    delta shape: [hidden_dim, intermediate_dim]
    对列（intermediate_dim 维度，即神经元维度）取 L2 范数，
    然后计算 Gini、Entropy、Top-k 比率。

    Returns:
        {
            "gini": float,
            "entropy": float,
            "normalized_entropy": float,
            "magnitude": float,           # 整矩阵 L2 范数
            "mean_neuron_magnitude": float, # 列范数均值
            "top_1pct_ratio": float,
            "top_2pct_ratio": float,
            "top_5pct_ratio": float,
            "top_10pct_ratio": float,
            "top_20pct_ratio": float,
            "top_50pct_ratio": float,
        }
    """
    per_neuron = torch.linalg.norm(delta, dim=0)  # [intermediate_dim]
    magnitude = torch.linalg.norm(delta).item()
    mean_neuron_mag = per_neuron.mean().item()
    n_neurons = per_neuron.numel()

    x = per_neuron.float().numpy() + 1e-12
    sorted_x = np.sort(x)[::-1]
    total = sorted_x.sum()
    p = x / total

    gini = float(gini_np(x[None, :])[0])
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    norm_entropy = entropy / np.log(n_neurons) if n_neurons > 1 else 0.0

    top_k = {}
    for pct in [1, 2, 5, 10, 20, 50]:
        k = max(1, int(n_neurons * pct / 100))
        top_k[f"top_{pct}pct_ratio"] = float(sorted_x[:k].sum() / total)

    return {
        "gini": gini,
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "magnitude": magnitude,
        "mean_neuron_magnitude": mean_neuron_mag,
        **top_k,
    }


# ═══════════════════════════════════════════════════════════════════
#  P 矩阵加载
# ═══════════════════════════════════════════════════════════════════


def ensure_projection_cache(model, tok, hparams: AlphaEditHyperParams):
    """从 P_loc 加载 P 矩阵，同时初始化 cache_c。"""
    import os
    W_out = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
    )
    mname = hparams.model_name.lower()
    if "llama" in mname or "qwen" in mname or "gpt-j-6b" in mname:
        dim = W_out.shape[1]
    elif "gpt2" in mname:
        dim = W_out.shape[0]
    else:
        dim = W_out.shape[1]
    del W_out

    P = torch.zeros((len(hparams.layers), dim, dim), device="cpu")
    cache_c = torch.zeros((len(hparams.layers), dim, dim), device="cpu")

    p_loc = hparams.P_loc
    if not os.path.exists(p_loc):
        raise FileNotFoundError(
            f"P 矩阵文件不存在: {p_loc}。请先运行 AlphaEdit 计算并保存 P 矩阵。"
        )

    print(f"Loading P from {p_loc}")
    P_loaded = torch.load(p_loc, map_location="cpu")
    for i in range(len(hparams.layers)):
        P[i, :, :] = P_loaded[i, :, :]

    return P, cache_c


# ═══════════════════════════════════════════════════════════════════
#  单次编辑函数（AlphaEdit，无掩码）
# ═══════════════════════════════════════════════════════════════════


def do_single_edit(
    model,
    tok,
    request: Dict,
    hparams: AlphaEditHyperParams,
    P: torch.Tensor,
    cache_c: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    对模型执行一次 AlphaEdit 编辑，返回各层的更新量矩阵 {layer: delta}。

    delta 的 shape 与 down_proj.weight 相同（LLaMA: [hidden_dim, intermediate_dim]）。
    """
    device = f"cuda:{hparams.device}"
    edit_layers = hparams.layers
    context_templates = get_context_templates(model, tok)

    req = deepcopy(request)
    if req["target_new"]["str"][0] != " ":
        req["target_new"]["str"] = " " + req["target_new"]["str"]
    req_target = req["target_new"]["str"]

    z_layer = edit_layers[-1]
    req_for_z = deepcopy(req)
    req_for_z["target_new"] = req_target
    z = compute_z(model, tok, req_for_z, hparams, z_layer, context_templates)

    upd_matrices: Dict[int, torch.Tensor] = {}

    for i, layer in enumerate(edit_layers):
        k = compute_ks(model, tok, [req_for_z], hparams, layer, context_templates).T

        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[req["prompt"]],
            words=[req["subject"]],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        targets = z[:, None] - cur_zs
        repeat_factor = k.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(edit_layers) - i)

        k_gpu = k.to(device)
        pg = P[i, :, :].to(device)
        cg = cache_c[i, :, :].to(device)

        A = pg @ (k_gpu @ k_gpu.T + cg) + hparams.L2 * torch.eye(
            k_gpu.shape[0], dtype=torch.float, device=device
        )
        B = pg @ k_gpu @ resid.T.to(device)
        del pg, cg, k_gpu
        torch.cuda.empty_cache()

        upd = torch.linalg.solve(A, B)
        del A, B
        torch.cuda.empty_cache()

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        weight = nethook.get_parameter(model, weight_name)
        upd_full = upd_matrix_match_shape(upd, weight.shape)

        with torch.no_grad():
            weight[...] = weight + upd_full.float()

        upd_matrices[layer] = upd_full.detach().cpu()

        for x in [k, cur_zs, targets, resid, upd, upd_full]:
            try:
                x.cpu()
            except Exception:
                pass
            del x
        torch.cuda.empty_cache()

    for i, layer in enumerate(edit_layers):
        k = compute_ks(model, tok, [req_for_z], hparams, layer, context_templates).T
        cache_c[i, :, :] += k.cpu() @ k.cpu().T

    return upd_matrices


# ═══════════════════════════════════════════════════════════════════
#  评估函数
# ═══════════════════════════════════════════════════════════════════


def test_batch_prediction(
    model, tok, prefixes: List[str], which_correct: List[int],
    target_new: str, target_true: str,
):
    """与 BLUE test_batch_prediction 一致的 token-level log-prob 评估。"""
    device = next(model.parameters()).device

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok = tok(f" {target_new}", add_special_tokens=False)["input_ids"]
    b_tok = tok(f" {target_true}", add_special_tokens=False)["input_ids"]
    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    corrects: List[bool] = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        correct_condition = (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        )
        if correct_condition:
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            corrects.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], corrects


def evaluate_direction(
    model, tok, record: Dict[str, Any],
    target_new: str, target_true: str,
) -> Dict[str, float]:
    """
    评估编辑效果。

    由于是单向编辑（target_new 方向），
    rewrite / paraphrase 期望 target_new 被预测，
    neighborhood 期望 target_true 被预测（保持原知识不变）。
    """
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]

    rewrite_prompts = [rewrite["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    prob_prompts = [rewrite_prompts, paraphrase_prompts, neighborhood_prompts]
    which_correct = [
        [0] * len(rewrite_prompts),       # rewrite: target_new 正确
        [0] * len(paraphrase_prompts),    # paraphrase: target_new 正确
        [1] * len(neighborhood_prompts),  # neighborhood: target_true 正确
    ]

    probs, corrects = test_batch_prediction(
        model, tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new, target_true,
    )

    cutoffs = [0] + np.cumsum([len(p) for p in prob_prompts]).tolist()
    ret_probs = [probs[cutoffs[i - 1]:cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [corrects[cutoffs[i - 1]:cutoffs[i]] for i in range(1, len(cutoffs))]

    eval_result = {
        "rewrite_prompts_probs": ret_probs[0],
        "paraphrase_prompts_probs": ret_probs[1] if len(ret_probs) > 1 else [],
        "neighborhood_prompts_probs": ret_probs[2] if len(ret_probs) > 2 else [],
        "rewrite_prompts_correct": ret_corrects[0],
        "paraphrase_prompts_correct": ret_corrects[1] if len(ret_corrects) > 1 else [],
        "neighborhood_prompts_correct": ret_corrects[2] if len(ret_corrects) > 2 else [],
    }

    metrics: Dict[str, float] = {}

    # rewrite（期望 target_new 的 nlp 更低，即概率更高 → 成功）
    if eval_result.get("rewrite_prompts_probs"):
        metrics["rewrite_success"] = float(np.mean([
            1.0 if x["target_new"] < x["target_true"] else 0.0
            for x in eval_result["rewrite_prompts_probs"]
        ]))
        metrics["rewrite_acc"] = float(np.mean(
            eval_result.get("rewrite_prompts_correct", [])
        )) if eval_result.get("rewrite_prompts_correct") else float("nan")

    # paraphrase
    if eval_result.get("paraphrase_prompts_probs"):
        metrics["paraphrase_success"] = float(np.mean([
            1.0 if x["target_new"] < x["target_true"] else 0.0
            for x in eval_result["paraphrase_prompts_probs"]
        ]))
        metrics["paraphrase_acc"] = float(np.mean(
            eval_result.get("paraphrase_prompts_correct", [])
        )) if eval_result.get("paraphrase_prompts_correct") else float("nan")

    # neighborhood（期望 target_true 的 nlp 更低，即概率更高 → 保持原知识）
    if eval_result.get("neighborhood_prompts_probs"):
        metrics["neighborhood_success"] = float(np.mean([
            1.0 if x["target_true"] < x["target_new"] else 0.0
            for x in eval_result["neighborhood_prompts_probs"]
        ]))
        metrics["neighborhood_acc"] = float(np.mean(
            eval_result.get("neighborhood_prompts_correct", [])
        )) if eval_result.get("neighborhood_prompts_correct") else float("nan")

    keys = ["rewrite_success", "paraphrase_success", "neighborhood_success"]
    vals = [metrics[k] for k in keys if k in metrics and not np.isnan(metrics[k])]
    if len(vals) == 3:
        metrics["score"] = float(hmean(vals))

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="更新集中度分析：对数据集前 N 条知识依次用 AlphaEdit 编辑"
    )
    parser.add_argument(
        "--hparams", type=str, required=True,
        help="AlphaEdit 超参文件路径",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="数据集路径（如 data/counterfact.json）",
    )
    parser.add_argument(
        "--num_edits", type=int, default=10,
        help="要依次编辑的数据条目数",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/concentration",
        help="输出目录",
    )
    parser.add_argument(
        "--save_every", type=int, default=1,
        help="每 N 次编辑保存一次更新量到磁盘（默认每次保存）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从已有输出目录的断点恢复",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updates_dir = output_dir / "updates"
    updates_dir.mkdir(exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # 保存参数
    with open(output_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── 加载模型 ──
    print(f"Loading hparams from {args.hparams}")
    hparams = AlphaEditHyperParams.from_hparams(args.hparams)
    editor = BaseEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    edit_layers = hparams.layers
    print(f"Model loaded, edit layers: {edit_layers}")

    # ── 加载数据 ──
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        records = json.load(f)
    records = records[:args.num_edits]
    print(f"Loaded {len(records)} records for editing")

    # ── P 矩阵 ──
    print("Loading projection matrix P...")
    P, cache_c = ensure_projection_cache(model, tok, hparams)

    # ── 保存原始权重 ──
    original_params_path = output_dir / "original_params.pt"
    if not original_params_path.exists():
        original_params = {}
        for layer in edit_layers:
            name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            original_params[f"layer_{layer}"] = (
                nethook.get_parameter(model, name).detach().cpu()
            )
        torch.save(original_params, original_params_path)
        print("Original params saved.")
    else:
        print(f"Original params already exist, skipping.")

    # ── 断点续跑 / 重建 ──
    resume_path = output_dir / "resume.pt"
    completed_edits: set[int] = set()
    results: List[Dict[str, Any]] = []
    metrics_by_layer: Dict[str, Dict[str, list]] = {
        f"layer_{l}": {
            "gini": [], "entropy": [], "normalized_entropy": [],
            "magnitude": [], "mean_neuron_magnitude": [],
            "top_1pct_ratio": [], "top_2pct_ratio": [],
            "top_5pct_ratio": [], "top_10pct_ratio": [],
            "top_20pct_ratio": [], "top_50pct_ratio": [],
        }
        for l in edit_layers
    }
    start_edit_idx = 0

    def _reconstruct_from_artifacts() -> int:
        """从 original_params.pt + updates/iter_*.pt 重建模型状态。

        cache_c 无法重建，从零开始。metrics 从已保存的 JSON 恢复。
        返回下一个编辑的索引。
        """
        print("从已有的更新量重建模型状态...")

        orig_params = torch.load(
            original_params_path, map_location="cpu", weights_only=True
        )
        pt_files = sorted(
            updates_dir.glob("iter_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not pt_files:
            print("错误: updates/ 目录为空")
            return 0

        for layer_name, orig_w in orig_params.items():
            layer = int(layer_name.split("_")[1])
            w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            with torch.no_grad():
                nethook.get_parameter(model, w_name)[...] = (
                    orig_w.to(next(model.parameters()).device, orig_w.dtype)
                )

        for pf in pt_files:
            it = int(pf.stem.split("_")[1])
            deltas = torch.load(pf, map_location="cpu", weights_only=True)
            for k, delta in deltas.items():
                layer = int(k)
                w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                with torch.no_grad():
                    w = nethook.get_parameter(model, w_name)
                    w[...] = w + delta.to(w.device, w.dtype)

        # 从已保存的 metrics JSON 恢复
        for layer in edit_layers:
            mpath = metrics_dir / f"layer_{layer}_metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    saved = json.load(f)
                for key in metrics_by_layer[f"layer_{layer}"]:
                    if key in saved:
                        metrics_by_layer[f"layer_{layer}"][key] = saved[key]

        max_iter = max(int(p.stem.split("_")[1]) for p in pt_files)
        print(f"重建完成：{len(pt_files)} 次编辑（0~{max_iter})")
        return max_iter + 1

    if args.resume:
        if resume_path.exists():
            resume_data = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
            for layer in edit_layers:
                w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                saved_w = resume_data["edit_layer_weights"][f"layer_{layer}"]
                with torch.no_grad():
                    nethook.get_parameter(model, w_name)[...] = saved_w.to(
                        next(model.parameters()).device, saved_w.dtype
                    )
            cache_c = resume_data["cache_c"].to(cache_c.device)
            results = resume_data.get("results", [])

            # 恢复 metrics_by_layer
            saved_metrics = resume_data.get("metrics_by_layer", {})
            for layer_key in metrics_by_layer:
                if layer_key in saved_metrics:
                    for metric_key in metrics_by_layer[layer_key]:
                        if metric_key in saved_metrics[layer_key]:
                            metrics_by_layer[layer_key][metric_key] = list(
                                saved_metrics[layer_key][metric_key]
                            )

            completed_edits = set(resume_data.get("completed_edits", []))
            start_edit_idx = max(completed_edits) + 1 if completed_edits else 0
            print(
                f"从 resume.pt 恢复：{len(completed_edits)} 次编辑，"
                f"从第 {start_edit_idx} 条继续"
            )
        else:
            start_edit_idx = _reconstruct_from_artifacts()
            if start_edit_idx > 0:
                completed_edits = set(range(start_edit_idx))
                print(f"从 artifact 重建：{start_edit_idx} 次编辑继续")
            else:
                print("artifact 重建失败，从头开始")

    # ═══════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════

    for edit_idx in range(args.num_edits):
        if edit_idx < start_edit_idx:
            continue

        record = records[edit_idx]
        req = record["requested_rewrite"]

        # 构建 request
        request = {
            "prompt": req["prompt"],
            "subject": req["subject"],
            "target_new": {"str": req["target_new"]["str"]},
            "target_true": {"str": req["target_true"]["str"]},
        }
        prompt_str = req["prompt"].format(req["subject"])

        print(f"\n{'=' * 60}")
        print(f"Edit {edit_idx + 1:3d}/{args.num_edits}: {prompt_str}")
        print(f"  target: {req['target_new']['str']}")
        print(f"{'=' * 60}")

        iter_start = time.time()

        # 执行编辑
        upd_matrices = do_single_edit(
            model, tok, request, hparams, P, cache_c,
        )

        # 保存更新量
        if edit_idx % args.save_every == 0:
            torch.save(
                {str(k): v.cpu() for k, v in upd_matrices.items()},
                updates_dir / f"iter_{edit_idx:03d}.pt",
            )

        # 计算每层的集中度指标
        layer_metrics = {}
        for layer in edit_layers:
            delta = upd_matrices[layer]
            metrics = compute_concentration_metrics(delta)
            layer_metrics[layer] = metrics

            # 追加到累积记录
            layer_key = f"layer_{layer}"
            for k, v in metrics.items():
                metrics_by_layer[layer_key][k].append(v)

        # 评估编辑效果
        target_new = req["target_new"]["str"]
        target_true = req["target_true"]["str"]
        eval_metrics = evaluate_direction(
            model, tok, record, target_new, target_true,
        )

        # 记录结果
        result_entry = {
            "edit_idx": edit_idx,
            "prompt": prompt_str,
            "subject": req["subject"],
            "target_new": target_new,
            "time_seconds": round(time.time() - iter_start, 2),
        }
        for layer in edit_layers:
            for k in ["magnitude", "gini", "normalized_entropy"]:
                result_entry[f"layer_{layer}_{k}"] = layer_metrics[layer][k]
        result_entry.update(eval_metrics)
        results.append(result_entry)

        # ── 保存 resume.pt ──
        edit_layer_weights = {}
        for layer in edit_layers:
            name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            edit_layer_weights[f"layer_{layer}"] = (
                nethook.get_parameter(model, name).detach().cpu()
            )
        completed_edits.add(edit_idx)
        torch.save(
            {
                "edit_layer_weights": edit_layer_weights,
                "cache_c": cache_c.detach().cpu(),
                "results": results,
                "completed_edits": sorted(completed_edits),
                "metrics_by_layer": metrics_by_layer,
            },
            resume_path,
        )

        # 每步 flush metrics JSON
        for layer in edit_layers:
            mpath = metrics_dir / f"layer_{layer}_metrics.json"
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(metrics_by_layer[f"layer_{layer}"], f, indent=2)

        # 打印摘要
        m4 = layer_metrics[edit_layers[0]]
        rew = eval_metrics.get("rewrite_success", float("nan"))
        para = eval_metrics.get("paraphrase_success", float("nan"))
        neig = eval_metrics.get("neighborhood_success", float("nan"))
        sc = eval_metrics.get("score", float("nan"))
        print(f"  Layer {edit_layers[0]}: |upd|={m4['magnitude']:.4f}, "
              f"gini={m4['gini']:.4f}, norm_entropy={m4['normalized_entropy']:.4f}")
        print(f"  Eval: rewrite={rew:.4f}, paraphrase={para:.4f}, "
              f"neighborhood={neig:.4f}, score={sc:.4f}")
        print(f"  Time: {result_entry['time_seconds']:.1f}s")

    # ── 保存最终结果 ──
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"完成！{args.num_edits} 条数据编辑完毕。")
    print(f"结果保存到: {output_dir.resolve()}")
    print(f"  updates/              更新量文件 iter_*.pt")
    for layer in edit_layers:
        print(f"  metrics/layer_{layer}_metrics.json  集中度指标")
    print(f"  resume.pt            断点检查点")
    print(f"  results.json         编辑记录")


if __name__ == "__main__":
    main()
