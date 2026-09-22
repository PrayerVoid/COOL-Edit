"""
神经元回退实验。

对重复编辑后的模型，按累积更新量排序，逐比例将神经元权重恢复为原始值，
观察 specificity + target_true 负对数概率变化。支持两种选择模式：
  - global:    所有层神经元统一排序，取全局 top-p%
  - per_layer: 每层独立排序，各取该层 top-p%（总比例相同）
每个比例都同时给出随机回退（--random_trials 次取均值）作为对照。

用法:
  python analyze_neuron_restore.py ^
      --hparams hparams/AlphaEdit/llama3.1-8b.yaml ^
      --exp_dirs ./results/repeated_alphaedit ./results/repeated_cool ^
      --labels alphaedit cool ^
      --output_dir ./results/analysis_report/neuron_restore ^
      --percentages "0,0.1,0.5,1,2,5,10,20,50,100" ^
      --random_trials 5
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from easyeditor import BaseEditor  # noqa: E402
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams  # noqa: E402

# ===========================================================================
#  实验参数
# ===========================================================================

SUBJECT = "Danielle Darrieux"
PROMPT_TEMPLATE = "The mother tongue of {} is"
TARGET_TRUE = "French"
CURRENT_TARGET = "Spanish"

PARAPHRASE_PROMPTS = [
    "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native",
    "An album was recorded for Capitol Nashville but never released. Danielle Darrieux spoke the language",
]

NEIGHBORHOOD_PROMPTS = [
    "The mother tongue of Léon Blum is",
    "The native language of Montesquieu is",
    "François Bayrou, a native",
    "The native language of Raymond Barre is",
    "Michel Rocard is a native speaker of",
    "Jacques Chaban-Delmas is a native speaker of",
    "The native language of François Bayrou is",
    "Maurice Genevoix, speaker of",
    "The mother tongue of François Bayrou is",
    "Melchior de Vogüé, speaker of",
]

EDIT_LAYERS = [4, 5, 6, 7, 8]


# ===========================================================================
#  评估
# ===========================================================================


def test_batch_prediction_3way(model, tok, prefixes, candidate_a, candidate_b, target_true):
    device = next(model.parameters()).device
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    candidates = [candidate_a, candidate_b, target_true]
    cand_toks = [tok(f" {c}", add_special_tokens=False)["input_ids"] for c in candidates]
    cand_lens = [len(t) for t in cand_toks]

    prompt_tok = tok(
        [f"{prefix} {cand}" for prefix in prefixes for cand in candidates],
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = []
    for i in range(logits.size(0)):
        prefix_idx = i // 3
        cand_idx = i % 3
        cur_len = cand_lens[cand_idx]
        nlp = 0.0
        for j in range(cur_len):
            cur_tok = cand_toks[cand_idx][j]
            nlp += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[prefix_idx] + j - 1, :], dim=0
            )[cur_tok].item()
        nlp /= cur_len
        if cand_idx == 0:
            results.append({"candidate_a": nlp, "candidate_b": 0.0, "target_true": 0.0})
        elif cand_idx == 1:
            results[-1]["candidate_b"] = nlp
        else:
            results[-1]["target_true"] = nlp
    return results


def evaluate(model, tok, record, current_target, target_true) -> Tuple[float, float]:
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]
    candidates = ["English", "Spanish"]
    other = [c for c in candidates if c != current_target][0]

    neighborhood_prompts = record.get("neighborhood_prompts", [])
    rewrite_prompts = [rewrite["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])

    all_prefixes = list(chain(rewrite_prompts, paraphrase_prompts, neighborhood_prompts))
    probs = test_batch_prediction_3way(model, tok, all_prefixes, current_target, other, target_true)

    n_nei = len(neighborhood_prompts)
    if n_nei == 0:
        return float("nan"), float("nan")

    nei_start = len(rewrite_prompts) + len(paraphrase_prompts)
    nei = probs[nei_start:nei_start + n_nei]

    spec = float(np.mean([
        1.0 if (x["target_true"] < x["candidate_a"] and x["target_true"] < x["candidate_b"]) else 0.0
        for x in nei
    ]))
    target_nlp = float(np.mean([x["target_true"] for x in nei]))
    return spec, target_nlp


def build_record() -> Dict[str, Any]:
    return {
        "requested_rewrite": {
            "prompt": PROMPT_TEMPLATE,
            "subject": SUBJECT,
            "target_new": {"str": CURRENT_TARGET},
            "target_true": {"str": TARGET_TRUE},
        },
        "paraphrase_prompts": PARAPHRASE_PROMPTS,
        "neighborhood_prompts": NEIGHBORHOOD_PROMPTS,
    }


# ===========================================================================
#  工具
# ===========================================================================


def apply_restore(info, targets, source_key: str):
    for l, idx in targets.items():
        if idx.numel() == 0:
            continue
        with torch.no_grad():
            info[l]["param"][:, idx] = info[l][source_key][:, idx]


# ===========================================================================
#  单实验处理
# ===========================================================================


def process_experiment(
    model, tok, hparams, exp_dir: Path, label: str,
    percentages: List[float], random_trials: int, norms_dir: Path, output_dir: Path,
):
    """处理单个实验，保存结果到 output_dir/{label}_*。

    返回该实验的 {label: results_dict}，供后续汇总用。
    """
    edit_layers = hparams.layers
    print(f"\n{'#' * 60}")
    print(f"Experiment: {label}")
    print(f"{'#' * 60}")

    # ── 权重 ──
    raw = torch.load(exp_dir / "original_params.pt", map_location="cpu", weights_only=True)
    W_orig = {int(k.split("_")[1]): v.float() for k, v in raw.items()}
    del raw

    resume_path = exp_dir / "resume.pt"
    if not resume_path.exists():
        cand = sorted(exp_dir.glob("resume_iter_*.pt"))
        resume_path = cand[-1] if cand else resume_path
    assert resume_path.exists(), f"resume.pt not found in {exp_dir}"
    rd = torch.load(resume_path, map_location="cpu", weights_only=True)
    W_edited = {int(k.split("_")[1]): v.float() for k, v in rd["edit_layer_weights"].items()}
    print(f"Loaded {len(W_edited)} layers from {resume_path}")

    # ── 写入模型 ──
    for layer in edit_layers:
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        p = model.get_parameter(weight_name)
        with torch.no_grad():
            p[...] = W_edited[layer].to(p.device, p.dtype)

    # ── 预缓存 ──
    info = {}
    n_neurons_per_layer = {}
    device_map = {}
    for layer in edit_layers:
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        p = model.get_parameter(weight_name)
        d = p.device
        n_neurons_per_layer[layer] = p.shape[1]
        device_map[layer] = d
        info[layer] = {
            "param": p,
            "W_edited": W_edited[layer].to(d, p.dtype),
            "W_orig": W_orig[layer].to(d, p.dtype),
        }

    # ── 排序数据 ──
    sorted_indices_per_layer: Dict[int, np.ndarray] = {}
    global_all: List[Tuple[float, int, int]] = []
    for layer in edit_layers:
        cp = norms_dir / f"{label}_layer{layer}_cumulative.npy"
        if not cp.exists():
            cp = norms_dir / f"repeated_layer{layer}_cumulative.npy"
        if cp.exists():
            cum = np.load(cp)
        else:
            cum = torch.linalg.norm(W_orig[layer] - W_edited[layer], dim=0).numpy()
        sorted_indices_per_layer[layer] = np.argsort(cum)[::-1].copy()
        for ni in range(cum.shape[0]):
            global_all.append((float(cum[ni]), layer, ni))
        print(f"  Layer {layer}: {cum.shape[0]} neurons")
    global_all.sort(key=lambda x: -x[0])
    n_total = len(global_all)

    record = build_record()

    # ── 工具函数 ──
    def targets_global(n):
        by_layer: Dict[int, List[int]] = {l: [] for l in edit_layers}
        for _, layer, ni in global_all[:n]:
            by_layer[layer].append(ni)
        return {l: torch.tensor(sorted(set(v)), device=device_map[l])
                for l, v in by_layer.items() if v}

    def targets_global_from_indices(indices):
        by_layer: Dict[int, List[int]] = {l: [] for l in edit_layers}
        for gi in indices:
            _, layer, ni = global_all[gi]
            by_layer[layer].append(ni)
        return {l: torch.tensor(sorted(set(v)), device=device_map[l])
                for l, v in by_layer.items() if v}

    # ═══ global mode ═══
    print(f"\n  --- global mode ---")
    g_spec_topk, g_spec_rnd = [], []
    g_nlp_topk, g_nlp_rnd = [], []

    for pct in percentages:
        n_r = max(1, int(n_total * pct / 100))

        tgt = targets_global(n_r)
        apply_restore(info, tgt, "W_orig")
        spec, nlp = evaluate(model, tok, record, CURRENT_TARGET, TARGET_TRUE)
        apply_restore(info, tgt, "W_edited")
        g_spec_topk.append(spec)
        g_nlp_topk.append(nlp)

        rnd_specs, rnd_nlps = [], []
        for trial in range(random_trials):
            rng = np.random.RandomState(trial * 1000 + int(pct * 100))
            gi = rng.choice(n_total, size=n_r, replace=False)
            rtgt = targets_global_from_indices(gi)
            apply_restore(info, rtgt, "W_orig")
            s, n = evaluate(model, tok, record, CURRENT_TARGET, TARGET_TRUE)
            rnd_specs.append(s)
            rnd_nlps.append(n)
            apply_restore(info, rtgt, "W_edited")
        g_spec_rnd.append(float(np.mean(rnd_specs)))
        g_nlp_rnd.append(float(np.mean(rnd_nlps)))

        print(f"    p={pct:6.1f}%: spec top-k={spec:.4f} rnd={np.mean(rnd_specs):.4f}  "
              f"nlp top-k={nlp:.4f} rnd={np.mean(rnd_nlps):.4f}")

    g_out = {"percentages": percentages,
             "topk_specificity": g_spec_topk, "random_specificity": g_spec_rnd,
             "topk_target_nlp": g_nlp_topk, "random_target_nlp": g_nlp_rnd}

    # ═══ per-layer mode ═══
    print(f"\n  --- per-layer mode ---")
    p_spec_topk, p_spec_rnd = [], []
    p_nlp_topk, p_nlp_rnd = [], []

    for pct in percentages:
        n_pl = max(1, int(n_neurons_per_layer[EDIT_LAYERS[0]] * pct / 100))

        tgt = {
            l: torch.tensor(sorted_indices_per_layer[l][:n_pl].copy(),
                            device=device_map[l])
            for l in edit_layers
        }
        apply_restore(info, tgt, "W_orig")
        spec, nlp = evaluate(model, tok, record, CURRENT_TARGET, TARGET_TRUE)
        apply_restore(info, tgt, "W_edited")
        p_spec_topk.append(spec)
        p_nlp_topk.append(nlp)

        rnd_specs, rnd_nlps = [], []
        for trial in range(random_trials):
            rng = np.random.RandomState(trial * 1000 + int(pct * 100) + 5000)
            rtgt = {
                l: torch.tensor(
                    rng.choice(n_neurons_per_layer[l], size=n_pl, replace=False),
                    device=device_map[l],
                )
                for l in edit_layers
            }
            apply_restore(info, rtgt, "W_orig")
            s, n = evaluate(model, tok, record, CURRENT_TARGET, TARGET_TRUE)
            rnd_specs.append(s)
            rnd_nlps.append(n)
            apply_restore(info, rtgt, "W_edited")
        p_spec_rnd.append(float(np.mean(rnd_specs)))
        p_nlp_rnd.append(float(np.mean(rnd_nlps)))

        print(f"    p={pct:6.1f}%: spec top-k={spec:.4f} rnd={np.mean(rnd_specs):.4f}  "
              f"nlp top-k={nlp:.4f} rnd={np.mean(rnd_nlps):.4f}")

    p_out = {"percentages": percentages,
             "topk_specificity": p_spec_topk, "random_specificity": p_spec_rnd,
             "topk_target_nlp": p_nlp_topk, "random_target_nlp": p_nlp_rnd}

    # ── 保存 ──
    for suffix, data in [("global", g_out), ("per_layer", p_out)]:
        with open(output_dir / f"{label}_results_{suffix}.json", "w") as f:
            json.dump(data, f, indent=2)

    _plot_dual(output_dir, f"{label}_global", percentages, g_spec_topk, g_spec_rnd, g_nlp_topk, g_nlp_rnd)
    _plot_dual(output_dir, f"{label}_per_layer", percentages, p_spec_topk, p_spec_rnd, p_nlp_topk, p_nlp_rnd)

    return {label: {"global": g_out, "per_layer": p_out}}


# ===========================================================================
#  主程序
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--exp_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--percentages", type=str,
                        default="0,0.1,0.5,1,2,5,10,20,50,100")
    parser.add_argument("--random_trials", type=int, default=5)
    parser.add_argument("--norms_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = args.labels or [Path(d).name for d in args.exp_dirs]
    assert len(labels) == len(args.exp_dirs), \
        f"labels ({len(labels)}) 和 exp_dirs ({len(args.exp_dirs)}) 数量不一致"
    percentages = [float(p) for p in args.percentages.split(",")]

    with open(output_dir / "params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 模型（所有实验共用） ──
    print(f"Loading model from {args.hparams}...")
    hparams = AlphaEditHyperParams.from_hparams(args.hparams)
    editor = BaseEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    print(f"Layers: {hparams.layers}")

    # ── 默认 norms_dir ──
    default_norms_dir = None
    if args.norms_dir:
        default_norms_dir = Path(args.norms_dir)

    # ── 串行处理每个实验 ──
    all_results: Dict[str, Any] = {}
    for exp_dir_str, label in zip(args.exp_dirs, labels):
        exp_dir = Path(exp_dir_str)

        norms_dir = default_norms_dir or (
            exp_dir.parent / "analysis_report" / "norm_consistency" / "norms"
        )

        result = process_experiment(
            model, tok, hparams, exp_dir, label,
            percentages, args.random_trials, norms_dir, output_dir,
        )
        all_results.update(result)

    # ── 跨实验对比图（至少两个实验时） ──
    if len(labels) >= 2:
        for mode in ["global", "per_layer"]:
            _plot_comparison_specificity(output_dir, all_results, percentages, mode)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Done! Results saved to {output_dir}")
    print(f"{'=' * 50}")


# ===========================================================================
#  绘图
# ===========================================================================

def _plot_dual(output_dir, prefix, percentages,
               spec_topk, spec_rnd, nlp_topk, nlp_rnd):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentages, spec_topk, "b.-", lw=1.5, ms=6, label="Top-k restore")
    ax.plot(percentages, spec_rnd, "r.--", lw=1.5, ms=6, label="Random restore")
    ax.set_xlabel("Percentage of neurons restored")
    ax.set_ylabel("Specificity (neighborhood_success)")
    ax.set_title(f"{prefix}: Specificity vs Neuron Restore Ratio")
    ax.set_xscale("log")
    ax.set_xticks(percentages)
    ax.set_xticklabels([f"{p}" for p in percentages], fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=spec_topk[0], color="gray", ls=":", alpha=0.5,
               label=f"Baseline ({spec_topk[0]:.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_specificity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentages, nlp_topk, "b.-", lw=1.5, ms=6, label="Top-k restore")
    ax.plot(percentages, nlp_rnd, "r.--", lw=1.5, ms=6, label="Random restore")
    ax.set_xlabel("Percentage of neurons restored")
    ax.set_ylabel("Target_true mean neg log-prob (lower = better)")
    ax.set_title(f"{prefix}: Target_true NLL vs Neuron Restore Ratio")
    ax.set_xscale("log")
    ax.set_xticks(percentages)
    ax.set_xticklabels([f"{p}" for p in percentages], fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_target_nlp.png", dpi=150)
    plt.close(fig)


def _plot_comparison_specificity(output_dir, all_results, percentages, mode):
    """多实验间 top-k 特异性对比。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, (label, results) in enumerate(all_results.items()):
        data = results[mode]["topk_specificity"]
        c = plt.cm.Set1(idx % 10)
        ax.plot(percentages, data, ".-", color=c, lw=1.5, ms=6, label=label)
    ax.set_xlabel("Percentage of neurons restored")
    ax.set_ylabel("Specificity (neighborhood_success)")
    ax.set_title(f"Cross-Experiment Comparison ({mode})")
    ax.set_xscale("log")
    ax.set_xticks(percentages)
    ax.set_xticklabels([f"{p}" for p in percentages], fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_{mode}_specificity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
