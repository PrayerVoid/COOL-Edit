"""
反复编辑 / 顺序编辑实验脚本。

支持的编辑模式：
  - alphaedit: 标准 AlphaEdit，不屏蔽神经元
  - random: 每次随机选取 neuron_ratio 比例的神经元编辑
  - heat: 基于神经元热度（heat）的概率采样，即 COOL-Edit
  - nmke: 基于神经元重要性评分的掩码（共振 + 爆发）

支持的编辑方式：
  - repeated_3way: 对同一条知识反复交替编辑，候选目标 English ←→ Spanish，
    固定 target_true=French
  - sequential: 对数据集前 N 条知识依次各编辑一次

支持模型：LLaMA 系列（包括 Qwen、Mistral）和 GPT-2 系列。

用法示例：
  # 反复编辑 + alphaedit
  python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml ^
      --mode alphaedit --edit_type repeated_3way --num_iterations 50 --output_dir ./results/alphaedit

  # 反复编辑 + heat
  python run_repeated_edit.py --hparams ... --mode heat --neuron_ratio 0.5 --heat_decay 0.9 ^
      --edit_type repeated_3way --num_iterations 50 --output_dir ./results/cool

  # 反复编辑 + nmke
  python run_repeated_edit.py --hparams ... --mode nmke --resonance_ratio 0.3 --burst_ratio 0.3 ^
      --edit_type repeated_3way --num_iterations 50 --output_dir ./results/nmke

  # 顺序编辑
  python run_repeated_edit.py --hparams ... --mode alphaedit --edit_type sequential ^
      --dataset data/counterfact.json --num_iterations 100 --output_dir ./results/sequential

  # GPT-2 模型
  python run_repeated_edit.py --hparams hparams/AlphaEdit/gpt2-xl.yaml ^
      --mode nmke --edit_type repeated_3way --num_iterations 50 --output_dir ./results/gpt2_nmke

  # 续跑
  python run_repeated_edit.py --hparams ... --mode heat --edit_type repeated_3way ^
      --num_iterations 200 --output_dir ./results/cool --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import hmean  # noqa: E402

# ── EasyEdit 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from easyeditor import BaseEditor  # noqa: E402
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams  # noqa: E402
from easyeditor.models.alphaedit.AlphaEdit_main import (  # noqa: E402
    get_context_templates,
    upd_matrix_match_shape,
)
from easyeditor.models.alphaedit.compute_ks import compute_ks  # noqa: E402
from easyeditor.models.alphaedit.compute_z import (  # noqa: E402
    compute_z,
    get_module_input_output_at_words,
)
from easyeditor.util import nethook  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  模型类型检测
# ═══════════════════════════════════════════════════════════════════


def get_model_type(hparams: AlphaEditHyperParams) -> str:
    """根据模型名称判断模型类型：'llama' 或 'gpt'。"""
    name = hparams.model_name.lower()
    if any(k in name for k in ["llama", "qwen", "mistral"]):
        return "llama"
    if "gpt" in name or "gpt-j" in name:
        return "gpt"
    # 默认 LLaMA（大多数现代模型）
    return "llama"


# ═══════════════════════════════════════════════════════════════════
#  随机神经元选择
# ═══════════════════════════════════════════════════════════════════


def generate_random_mask(
    total_neurons: int,
    neuron_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """随机选取 neuron_ratio 比例的神经元，返回 0/1 掩码。"""
    num_selected = max(1, int(total_neurons * neuron_ratio))
    indices = torch.randperm(total_neurons, device=device)[:num_selected]
    mask = torch.zeros(total_neurons, device=device)
    mask[indices] = 1.0
    return mask


# ═══════════════════════════════════════════════════════════════════
#  热度神经元选择
# ═══════════════════════════════════════════════════════════════════


def compute_selection_probs(
    heat: torch.Tensor,
    neuron_ratio: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    基于热度计算每个神经元被选中的独立伯努利概率。
    heat[i] 越高 → 概率越低。
    缩放至 sum(p) = neuron_ratio * N，保证期望选中数匹配目标比例。
    """
    N = heat.numel()
    if N == 0:
        return heat.new_zeros(N)

    logits = -heat / temperature
    logits = logits - logits.max()
    softmax_probs = torch.softmax(logits, dim=0)

    target_sum = neuron_ratio * N
    raw_sum = softmax_probs.sum()
    if raw_sum < 1e-12:
        return torch.full_like(heat, neuron_ratio)

    scale = target_sum / raw_sum
    probs = softmax_probs * scale
    probs = probs.clamp(0.0, 1.0)
    return probs


def sample_neurons(probs: torch.Tensor) -> torch.Tensor:
    """独立伯努利采样，返回 0/1 掩码（1=选中参与编辑）。"""
    return torch.bernoulli(probs)


def update_heat(
    heat: torch.Tensor,
    selection_mask: torch.Tensor,
    upd_norms: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """
    更新热度向量。
    heat_new = heat_old * decay + selection_mask * upd_norms
    upd_norms[i] = |upd[i, :]|（该神经元更新向量的 L2 模长）。
    """
    if upd_norms.ndim > 1:
        upd_norms = upd_norms.norm(dim=1)
    heat = heat * decay
    heat = heat + selection_mask * upd_norms
    return heat


# ═══════════════════════════════════════════════════════════════════
#  NMKE 重要性评分（hook 版，支持 LLaMA 和 GPT-2）
# ═══════════════════════════════════════════════════════════════════


def get_importance_scores_via_hooks(
    model, tok, input_prompts: List[str], layer: int,
    hparams: AlphaEditHyperParams,
) -> torch.Tensor:
    """
    通过 hook 捕获中间激活，计算每个神经元的重要性评分。
    支持 LLaMA（gate_proj + up_proj + down_proj + RMSNorm）和
    GPT-2（c_fc + c_proj + GELU + LayerNorm）。

    返回 [n_prompts, n_neurons] 的评分矩阵。
    """
    device = f"cuda:{hparams.device}"
    model_type = get_model_type(hparams)
    input_tok = tok(input_prompts, return_tensors="pt", padding=True).to(device)
    captured: Dict[str, torch.Tensor] = {}

    def make_hook(key):
        def hook(module, input_, output):
            if isinstance(output, tuple):
                output = output[0]
            captured[key] = output.detach()
        return hook

    def make_input_hook(key):
        def hook(module, input_, output):
            inp = input_[0] if isinstance(input_, tuple) else input_
            captured[key] = inp.detach()
        return hook

    handles = []

    if model_type == "llama":
        mlp = model.model.layers[layer].mlp
        handles.append(mlp.gate_proj.register_forward_hook(make_hook("gate")))
        handles.append(mlp.up_proj.register_forward_hook(make_hook("up")))
        handles.append(
            model.model.layers[layer].post_attention_layernorm.register_forward_hook(
                make_input_hook("residual")
            )
        )
        handles.append(
            model.model.norm.register_forward_hook(make_input_hook("final_hidden"))
        )
    else:  # gpt
        mlp = model.transformer.h[layer].mlp
        handles.append(mlp.c_fc.register_forward_hook(make_hook("cfc")))
        handles.append(
            model.transformer.h[layer].ln_2.register_forward_hook(
                make_input_hook("residual")
            )
        )
        handles.append(
            model.transformer.ln_f.register_forward_hook(make_input_hook("final_hidden"))
        )

    with torch.no_grad():
        logits = model(**input_tok).logits
    for h in handles:
        h.remove()

    predicted_top1 = [
        torch.argmax(logits[i, -1, :]).item() for i in range(logits.shape[0])
    ]

    residual = captured["residual"]
    final_hidden = captured["final_hidden"]

    if model_type == "llama":
        act_fn = mlp.act_fn
        gate_out = captured["gate"]
        up_out = captured["up"]
        coeffs = act_fn(gate_out) * up_out
        fc2_vectors = mlp.down_proj.weight.data  # [hidden, intermediate]
        final_var = final_hidden.pow(2).mean(-1, keepdim=True)  # RMS
    else:
        act_fn = mlp.act
        cfc_out = captured["cfc"]
        coeffs = act_fn(cfc_out)
        fc2_vectors = mlp.c_proj.weight.data.T  # transpose to [hidden, intermediate]
        final_var = torch.var(final_hidden, dim=-1, unbiased=False, keepdim=True).sqrt() + 1e-5  # std

    fc2_vectors = fc2_vectors.float().to(device)
    AMPLIFY_FACTOR = 30
    scores_all = []

    for b in range(input_tok["input_ids"].shape[0]):
        c = coeffs[b, -1, :].float()
        r = residual[b, -1, :].float()
        fv = final_var[b, -1, :].float()
        ffn_subvalues = (c * fc2_vectors).T  # [intermediate, hidden]
        pred_idx = predicted_top1[b]

        if model_type == "llama":
            # RMSNorm: x / sqrt(mean(x^2)) * weight
            origin_prob = torch.softmax(
                model.lm_head(
                    (r.unsqueeze(0) * torch.rsqrt(fv.unsqueeze(0) + 1e-6)
                     ).to(model.model.norm.weight.device) * model.model.norm.weight
                ), dim=-1
            )[0, pred_idx]
            cur_plus = ffn_subvalues * AMPLIFY_FACTOR + r.unsqueeze(0)
            cur_probs = torch.softmax(
                model.lm_head(
                    (cur_plus * torch.rsqrt(fv.unsqueeze(0) + 1e-6)
                     ).to(model.model.norm.weight.device) * model.model.norm.weight
                ), dim=-1
            )[:, pred_idx]
        else:
            # LayerNorm: (x - mean) / std * weight
            E = r.mean(-1, keepdim=True)
            r_norm = (r.unsqueeze(0) - E.unsqueeze(0)) / fv.unsqueeze(0)
            r_norm = r_norm.to(model.transformer.ln_f.weight.device) * model.transformer.ln_f.weight.data

            origin_prob = torch.softmax(
                model.lm_head(r_norm), dim=-1
            )[0, pred_idx]

            cur_plus = ffn_subvalues * AMPLIFY_FACTOR + r.unsqueeze(0)
            E_cur = cur_plus.mean(-1, keepdim=True)
            cur_norm = (cur_plus - E_cur) / fv.unsqueeze(0)
            cur_norm = cur_norm.to(model.transformer.ln_f.weight.device) * model.transformer.ln_f.weight.data
            cur_probs = torch.softmax(
                model.lm_head(cur_norm), dim=-1
            )[:, pred_idx]

        origin_prob = origin_prob.clamp(min=1e-8, max=1.0)
        cur_probs = cur_probs.clamp(min=1e-8, max=1.0)
        s = torch.log(torch.clamp(cur_probs, min=1e-8)) - torch.log(
            torch.clamp(origin_prob, min=1e-8)
        )
        scores_all.append(s)

    scores_all = torch.stack(scores_all).to(device)
    scores_all = torch.nan_to_num(scores_all, nan=0.0, posinf=0.0, neginf=0.0)

    del input_tok, logits
    for v in captured.values():
        del v
    torch.cuda.empty_cache()
    return scores_all


# ═══════════════════════════════════════════════════════════════════
#  NMKE 混合共振掩码
# ═══════════════════════════════════════════════════════════════════


def compute_hybrid_resonant_mask(
    score_matrix: torch.Tensor,
    resonance_ratio: float = 0.25,
    burst_ratio: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算混合共振掩码，返回 (final_mask, resonance_mask, burst_mask)。"""
    normalized = (score_matrix - score_matrix.mean(dim=1, keepdim=True)) / (
        score_matrix.std(dim=1, keepdim=True) + 1e-6
    )
    resonance_counts = (normalized > 0.0).float().sum(dim=0)
    resonance_cut = torch.quantile(resonance_counts, 1 - resonance_ratio)
    resonance_mask = (resonance_counts >= resonance_cut).float()
    burst_score = score_matrix.max(dim=0).values
    burst_cut = torch.quantile(burst_score, 1 - burst_ratio)
    burst_mask = (burst_score >= burst_cut).float()
    final_mask = torch.clamp(resonance_mask + burst_mask, max=1.0)
    return final_mask, resonance_mask, burst_mask


def entropy_adaptive_mask_ratio(
    score_matrix: torch.Tensor,
    resonance_bounds: Tuple[float, float] = (0.3, 0.4),
    burst_bounds: Tuple[float, float] = (0.3, 0.4),
    gamma_r: float = 3.0,
    gamma_b: float = 2.0,
    alpha: float = 30.0,
) -> Tuple[float, float]:
    """基于熵动态调整 resonance/burst 比例。"""
    D = score_matrix.shape[1]
    logD = np.log(D)
    shifted = score_matrix - score_matrix.max(dim=1, keepdim=True)[0]
    scaled = shifted * alpha
    softmax_scores = torch.softmax(scaled, dim=1)
    entropy_prompt = -(softmax_scores * (softmax_scores + 1e-8).log()).sum(dim=1)
    entropy_r = entropy_prompt.mean() / logD
    resonance_ratio = resonance_bounds[0] + (
        resonance_bounds[1] - resonance_bounds[0]
    ) * entropy_r.clamp(0, 1).pow(gamma_r)
    max_acts = score_matrix.max(dim=0).values
    max_acts = torch.clamp(max_acts, min=0.0)
    if max_acts.sum() < 1e-8:
        return float(resonance_ratio), burst_bounds[0]
    burst_probs = max_acts / (max_acts.sum() + 1e-8)
    entropy_burst = -(burst_probs * (burst_probs + 1e-8).log()).sum()
    entropy_b = entropy_burst / logD
    burst_ratio = burst_bounds[0] + (
        burst_bounds[1] - burst_bounds[0]
    ) * entropy_b.clamp(0, 1).pow(gamma_b)
    return float(resonance_ratio), float(burst_ratio)


# ═══════════════════════════════════════════════════════════════════
#  P 矩阵加载
# ═══════════════════════════════════════════════════════════════════


def ensure_projection_cache(model, tok, hparams: AlphaEditHyperParams):
    """加载 P 矩阵（不存在则报错），同时初始化 cache_c。"""
    W_out = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
    )
    model_type = get_model_type(hparams)
    if model_type == "llama":
        dim = W_out.shape[1]
    elif model_type == "gpt":
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
#  集中度指标计算
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
    """从单层更新矩阵计算集中度指标。"""
    per_neuron = torch.linalg.norm(delta, dim=0)
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
#  单次编辑
# ═══════════════════════════════════════════════════════════════════


def do_single_edit(
    model, tok, request: Dict, hparams: AlphaEditHyperParams,
    P: torch.Tensor, cache_c: torch.Tensor,
    mode: str = "alphaedit",
    neuron_ratio: float = 0.5,
    heat_decay: float = 0.9,
    heat_temperature: float = 1.0,
    resonance_ratio: float = 0.3,
    burst_ratio: float = 0.3,
    use_adaptive_ratio: bool = False,
    heat_vectors: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[Dict[int, torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    """
    执行单次编辑。

    mode 参数决定掩码策略：
      - alphaedit: 无掩码
      - random: 随机选择神经元
      - heat: 热度驱动的概率采样
      - nmke: 神经元重要性评分掩码 (resonance + burst)

    Returns:
        upd_matrices: {layer: update_tensor}
        heat_vectors: {layer: heat_tensor} — heat 模式下返回更新后的热度，否则 None
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

    # ── NMKE 模式：预先计算所有层的掩码 ──
    nmke_masks: Dict[int, torch.Tensor] = {}
    if mode == "nmke":
        input_prompts = [
            context.format(req["prompt"].replace("{}", req["subject"]))
            for context_type in context_templates
            for context in context_type
        ]
        for layer in edit_layers:
            score_matrix = get_importance_scores_via_hooks(
                model, tok, input_prompts, layer, hparams
            )
            if use_adaptive_ratio:
                rr, br = entropy_adaptive_mask_ratio(score_matrix)
                print(f"  Layer {layer} adaptive: res={rr:.4f}, bur={br:.4f}")
            else:
                rr, br = resonance_ratio, burst_ratio
            final_mask, _, _ = compute_hybrid_resonant_mask(
                score_matrix,
                resonance_ratio=max(rr, 0.01),
                burst_ratio=max(br, 0.01),
            )
            kept = int(final_mask.sum().item())
            total = final_mask.numel()
            print(f"  Layer {layer} NMKE mask: {kept}/{total} ({kept / total:.2%})")
            nmke_masks[layer] = final_mask
            del score_matrix, final_mask
            torch.cuda.empty_cache()

    upd_matrices: Dict[int, torch.Tensor] = {}
    new_heat_vectors: Optional[Dict[int, torch.Tensor]] = (
        {} if mode == "heat" else None
    )

    for i, layer in enumerate(edit_layers):
        print(f"\n--- Layer {layer} ({i + 1}/{len(edit_layers)}) ---")

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

        num_neurons = k.shape[0]

        # ── 随机/热度掩码：计算采样掩码 ──
        selection_mask: Optional[torch.Tensor] = None
        if mode == "random" and neuron_ratio > 0.0 and neuron_ratio < 1.0:
            selection_mask = generate_random_mask(num_neurons, neuron_ratio, device=f"cuda:{hparams.device}")
            kept = int(selection_mask.sum().item())
            print(f"  Layer {layer} random mask: {kept}/{num_neurons} ({kept / num_neurons:.2%})")

        elif mode == "heat" and neuron_ratio > 0.0 and neuron_ratio < 1.0:
            heat = heat_vectors.get(layer) if heat_vectors else None
            if heat is None:
                heat = torch.zeros(num_neurons, device=f"cuda:{hparams.device}")
            else:
                heat = heat.to(f"cuda:{hparams.device}")

            probs = compute_selection_probs(heat, neuron_ratio, heat_temperature)
            selection_mask = sample_neurons(probs)
            kept = int(selection_mask.sum().item())
            print(f"  Layer {layer} heat selection: {kept}/{num_neurons} "
                  f"({kept / num_neurons:.2%}), "
                  f"heat mean={heat.mean().item():.4f}, max={heat.max().item():.4f}")

        k_gpu = k.to(device)
        pg = P[i, :, :].to(device)
        cg = cache_c[i, :, :].to(device)

        # 构建 A 矩阵
        A = pg @ (k_gpu @ k_gpu.T + cg) + hparams.L2 * torch.eye(
            k_gpu.shape[0], dtype=torch.float, device=device
        )
        B = pg @ k_gpu @ resid.T.to(device)
        del pg, cg, k_gpu
        torch.cuda.empty_cache()

        upd = torch.linalg.solve(A, B)
        del A, B
        torch.cuda.empty_cache()

        # ── zero 掩码：直接置零 ──
        if selection_mask is not None:
            upd = selection_mask[:, None] * upd

        # ── NMKE 掩码 ──
        if mode == "nmke" and layer in nmke_masks:
            mask = nmke_masks[layer].to(upd.device)
            upd = mask[:, None] * upd

        # ── 更新热度 ──
        if mode == "heat" and selection_mask is not None:
            heat = heat.to(device)
            heat = update_heat(heat, selection_mask, upd, heat_decay)
            new_heat_vectors[layer] = heat.detach().cpu()
        elif mode == "heat" and heat_vectors is not None and layer in heat_vectors:
            new_heat_vectors[layer] = heat_vectors[layer].cpu()

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        weight = nethook.get_parameter(model, weight_name)
        upd_full = upd_matrix_match_shape(upd, weight.shape)
        print(f"  orig norm: {torch.linalg.norm(weight):.4f}, "
              f"upd norm: {torch.linalg.norm(upd_full):.4f}")

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

    print("Single edit completed.")
    return upd_matrices, new_heat_vectors


# ═══════════════════════════════════════════════════════════════════
#  三路评估
# ═══════════════════════════════════════════════════════════════════


def test_batch_prediction_3way(
    model, tok, prefixes: List[str],
    candidate_a: str, candidate_b: str, target_true: str,
):
    """
    计算三个候选项的 token-level 平均 neg log-prob。
    返回 [{"candidate_a": float, "candidate_b": float, "target_true": float}, ...]
    """
    device = next(model.parameters()).device

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    candidates = [candidate_a, candidate_b, target_true]
    cand_toks = [
        tok(f" {c}", add_special_tokens=False)["input_ids"]
        for c in candidates
    ]
    cand_lens = [len(t) for t in cand_toks]

    prompt_tok = tok(
        [
            f"{prefix} {cand}"
            for prefix in prefixes
            for cand in candidates
        ],
        padding=True,
        return_tensors="pt",
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


def evaluate_3way(
    model, tok, record: Dict[str, Any],
    current_target: str,
    target_true: str,
):
    """
    三路评估。
    - rewrite/paraphrase: current_target 须比 target_true 和另一候选都低（nlp 越低越好）
    - neighborhood: target_true 须比两个候选都低
    """
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]
    candidates = ["English", "Spanish"]
    other = [c for c in candidates if c != current_target][0]

    rewrite_prompts = [rewrite["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    prob_prompts = [rewrite_prompts, paraphrase_prompts, neighborhood_prompts]

    all_prefixes = list(chain(*prob_prompts))
    probs = test_batch_prediction_3way(
        model, tok, all_prefixes, current_target, other, target_true,
    )

    cutoffs = [0] + np.cumsum([len(p) for p in prob_prompts]).tolist()
    ret_probs = [probs[cutoffs[i - 1]:cutoffs[i]] for i in range(1, len(cutoffs))]

    metrics: Dict[str, float] = {}

    if ret_probs[0]:
        metrics["rewrite_success"] = float(np.mean([
            1.0 if (
                x["candidate_a"] < x["candidate_b"]
                and x["candidate_a"] < x["target_true"]
            ) else 0.0
            for x in ret_probs[0]
        ]))

    if len(ret_probs) > 1 and ret_probs[1]:
        metrics["paraphrase_success"] = float(np.mean([
            1.0 if (
                x["candidate_a"] < x["candidate_b"]
                and x["candidate_a"] < x["target_true"]
            ) else 0.0
            for x in ret_probs[1]
        ]))

    if len(ret_probs) > 2 and ret_probs[2]:
        metrics["neighborhood_success"] = float(np.mean([
            1.0 if (
                x["target_true"] < x["candidate_a"]
                and x["target_true"] < x["candidate_b"]
            ) else 0.0
            for x in ret_probs[2]
        ]))

    keys = ["rewrite_success", "paraphrase_success", "neighborhood_success"]
    vals = [metrics[k] for k in keys if k in metrics and not np.isnan(metrics[k])]
    if len(vals) == 3:
        metrics["score"] = float(hmean(vals))

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  顺序编辑评估（与 BLUE 一致）
# ═══════════════════════════════════════════════════════════════════


def test_batch_prediction(
    model, tok, prefixes: List[str], which_correct: List[int],
    target_new: str, target_true: str,
):
    """token-level log-prob 评估（顺序编辑用）。"""
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


def evaluate_sequential(
    model, tok, record: Dict[str, Any],
    target_new: str, target_true: str,
) -> Dict[str, float]:
    """
    评估顺序编辑效果。
    rewrite / paraphrase 期望 target_new 被预测，
    neighborhood 期望 target_true 被预测。
    """
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]

    rewrite_prompts = [rewrite["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    prob_prompts = [rewrite_prompts, paraphrase_prompts, neighborhood_prompts]
    which_correct = [
        [0] * len(rewrite_prompts),
        [0] * len(paraphrase_prompts),
        [1] * len(neighborhood_prompts),
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

    metrics: Dict[str, float] = {}

    if ret_probs[0]:
        metrics["rewrite_success"] = float(np.mean([
            1.0 if x["target_new"] < x["target_true"] else 0.0
            for x in ret_probs[0]
        ]))
        metrics["rewrite_acc"] = float(np.mean(
            ret_corrects[0]
        )) if ret_corrects[0] else float("nan")

    if len(ret_probs) > 1 and ret_probs[1]:
        metrics["paraphrase_success"] = float(np.mean([
            1.0 if x["target_new"] < x["target_true"] else 0.0
            for x in ret_probs[1]
        ]))
        metrics["paraphrase_acc"] = float(np.mean(
            ret_corrects[1]
        )) if ret_corrects[1] else float("nan")

    if len(ret_probs) > 2 and ret_probs[2]:
        metrics["neighborhood_success"] = float(np.mean([
            1.0 if x["target_true"] < x["target_new"] else 0.0
            for x in ret_probs[2]
        ]))
        metrics["neighborhood_acc"] = float(np.mean(
            ret_corrects[2]
        )) if ret_corrects[2] else float("nan")

    keys = ["rewrite_success", "paraphrase_success", "neighborhood_success"]
    vals = [metrics[k] for k in keys if k in metrics and not np.isnan(metrics[k])]
    if len(vals) == 3:
        metrics["score"] = float(hmean(vals))

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════


def plot_results(results: List[Dict[str, Any]], output_dir: Path):
    """绘制指标随迭代次数的变化曲线。"""
    iters = [r["iteration"] if "iteration" in r else r.get("edit_idx", 0) for r in results]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    single_plots = [
        ("rewrite_success", "Rewrite Success", "rewrite_success.png"),
        ("paraphrase_success", "Generalization", "generalization.png"),
        ("neighborhood_success", "Specificity", "specificity.png"),
        ("score", "Score (HMean)", "score.png"),
    ]
    for key, ylabel, fname in single_plots:
        vals = [r.get(key, float("nan")) for r in results]
        valid = [(i, v) for i, v in zip(iters, vals) if not np.isnan(v)]
        if not valid:
            continue
        xs, ys = zip(*valid)
        plt.figure(figsize=(10, 5))
        plt.plot(xs, ys, marker=".", markersize=5, linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Iteration")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close()

    plt.figure(figsize=(10, 5))
    for key, label, style in [
        ("rewrite_success", "Rewrite", "b.-"),
        ("paraphrase_success", "Generalization", "c.-"),
        ("neighborhood_success", "Specificity", "m.-"),
        ("score", "Score (HMean)", "r.-"),
    ]:
        vals = [r.get(key, float("nan")) for r in results]
        valid = [(i, v) for i, v in zip(iters, vals) if not np.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            plt.plot(xs, ys, style, label=label, markersize=4, linewidth=1.2)
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.title("All Metrics vs Iteration")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / "all_metrics.png", dpi=150)
    plt.close()


def plot_heat_evolution(
    heat_history: List[Dict[int, torch.Tensor]],
    edit_layers: List[int],
    output_dir: Path,
):
    """绘制各层神经元热度的演化过程。"""
    if not heat_history:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    save_interval = max(1, len(heat_history) // 50)

    for layer in edit_layers:
        layer_heats = []
        for step, hv in enumerate(heat_history):
            if step % save_interval == 0 and layer in hv:
                h = hv[layer]
                layer_heats.append((step, h.clone()))

        if not layer_heats:
            continue

        steps = [s for s, _ in layer_heats]
        means = [h.mean().item() for _, h in layer_heats]
        maxs = [h.max().item() for _, h in layer_heats]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, means, "b-", label="Mean heat", linewidth=1.5)
        plt.plot(steps, maxs, "r-", label="Max heat", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Heat")
        plt.title(f"Layer {layer} — Neuron Heat Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"heat_layer_{layer}.png", dpi=150)
        plt.close()


# ═══════════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="知识编辑实验（支持 alphaedit / random / heat / nmke，"
                    "支持 repeated_3way / sequential 两种编辑方式）"
    )
    # 基础参数
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="从已有输出目录的断点恢复")

    # 模式选择
    parser.add_argument(
        "--mode", type=str, default="alphaedit",
        choices=["alphaedit", "random", "heat", "nmke"],
        help="编辑模式",
    )
    parser.add_argument(
        "--edit_type", type=str, default="repeated_3way",
        choices=["repeated_3way", "sequential"],
        help="编辑方式",
    )

    # 随机/热度参数
    parser.add_argument(
        "--neuron_ratio", type=float, default=0.5,
        help="random/heat 模式下编辑的神经元比例（0~1，取 0 或 1 时不掩码）",
    )
    parser.add_argument(
        "--heat_decay", type=float, default=0.9,
        help="heat 模式下热度衰减系数（0~1）",
    )
    parser.add_argument(
        "--heat_temperature", type=float, default=1.0,
        help="heat 模式下 softmax 温度参数",
    )

    # NMKE 参数
    parser.add_argument(
        "--resonance_ratio", type=float, default=0.3,
        help="nmke 模式共振比例",
    )
    parser.add_argument(
        "--burst_ratio", type=float, default=0.3,
        help="nmke 模式爆发比例",
    )
    parser.add_argument(
        "--adaptive_ratio", action="store_true",
        help="nmke 模式使用自适应比例",
    )

    # 数据集参数（顺序编辑用）
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="数据集路径（edit_type=sequential 时必需）",
    )

    # repeated_3way 的事实与三路评估参数
    parser.add_argument("--subject", type=str, default="Danielle Darrieux")
    parser.add_argument("--prompt_template", type=str, default="The mother tongue of {} is")
    parser.add_argument("--target_true", type=str, default="French")
    parser.add_argument(
        "--candidate_list", type=str, nargs="+", default=["English", "Spanish"],
    )

    args = parser.parse_args()

    # ── 校验参数 ──
    if args.edit_type == "sequential" and not args.dataset:
        parser.error("edit_type=sequential 时必须指定 --dataset")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updates_dir = output_dir / "updates"
    updates_dir.mkdir(exist_ok=True)
    if args.edit_type == "sequential":
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

    with open(output_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── 加载模型 ──
    print(f"Loading hparams from {args.hparams}")
    hparams = AlphaEditHyperParams.from_hparams(args.hparams)
    editor = BaseEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    edit_layers = hparams.layers
    model_type = get_model_type(hparams)
    print(f"Model type: {model_type}, edit layers: {edit_layers}")

    # ── 加载数据集（顺序编辑） ──
    records = None
    if args.edit_type == "sequential":
        print(f"Loading dataset: {args.dataset}")
        with open(args.dataset, "r", encoding="utf-8") as f:
            records = json.load(f)
        records = records[:args.num_iterations]
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
        print("Original params already exist, skipping.")

    # ═══════════════════════════════════════════════════════════════
    #  断点续跑 / 重建
    # ═══════════════════════════════════════════════════════════════

    resume_path = output_dir / "resume.pt"
    completed_iterations: set[int] = set()
    results: List[Dict[str, Any]] = []
    heat_history: List[Dict[int, torch.Tensor]] = []
    heat_vectors: Dict[int, torch.Tensor] = {}
    start_iteration = 0

    # 顺序编辑特有的 metrics_by_layer
    metrics_by_layer: Optional[Dict[str, Dict[str, list]]] = None
    if args.edit_type == "sequential":
        metrics_by_layer = {
            f"layer_{l}": {
                "gini": [], "entropy": [], "normalized_entropy": [],
                "magnitude": [], "mean_neuron_magnitude": [],
                "top_1pct_ratio": [], "top_2pct_ratio": [],
                "top_5pct_ratio": [], "top_10pct_ratio": [],
                "top_20pct_ratio": [], "top_50pct_ratio": [],
            }
            for l in edit_layers
        }

    def _reconstruct_from_artifacts():
        """从 original_params.pt + updates/iter_*.pt 重建模型状态。"""
        nonlocal heat_vectors, heat_history, metrics_by_layer

        print("从已有 artifact 重建模型状态...")
        if not original_params_path.exists():
            return 0
        if not updates_dir.exists():
            return 0
        orig_params = torch.load(
            original_params_path, map_location="cpu", weights_only=True
        )
        pt_files = sorted(
            updates_dir.glob("iter_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not pt_files:
            return 0
        for layer_name, orig_w in orig_params.items():
            layer = int(layer_name.split("_")[1])
            w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            with torch.no_grad():
                nethook.get_parameter(model, w_name)[...] = orig_w.to(
                    next(model.parameters()).device, orig_w.dtype
                )
        found_iters = []
        for pf in pt_files:
            it = int(pf.stem.split("_")[1])
            found_iters.append(it)
            deltas = torch.load(pf, map_location="cpu", weights_only=True)
            for k, delta in deltas.items():
                layer = int(k)
                w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                with torch.no_grad():
                    w = nethook.get_parameter(model, w_name)
                    w[...] = w + delta.to(w.device, w.dtype)
        print("注意: cache_c 将从零开始累积")

        loaded_results: List[Dict] = []
        rpath = output_dir / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                loaded_results = json.load(f)
            loaded_results = [r for r in loaded_results if r["iteration"] <= max(found_iters)]
            loaded_results.sort(key=lambda r: r["iteration"])

        # 恢复 metrics_by_layer（顺序编辑）
        if metrics_by_layer is not None:
            for layer in edit_layers:
                mpath = metrics_dir / f"layer_{layer}_metrics.json"
                if mpath.exists():
                    with open(mpath) as f:
                        saved = json.load(f)
                    for key in metrics_by_layer[f"layer_{layer}"]:
                        if key in saved:
                            metrics_by_layer[f"layer_{layer}"][key] = saved[key]

        max_iter = max(found_iters)
        print(f"重建完成：{len(found_iters)} 次迭代（0~{max_iter}）")
        return max_iter + 1, loaded_results

    if args.resume:
        if resume_path.exists():
            data = torch.load(resume_path, map_location="cpu", weights_only=True)
            for layer in edit_layers:
                w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                saved_w = data["edit_layer_weights"][f"layer_{layer}"]
                with torch.no_grad():
                    nethook.get_parameter(model, w_name)[...] = saved_w.to(
                        next(model.parameters()).device, saved_w.dtype
                    )
            cache_c = data["cache_c"].to(cache_c.device)
            results = data["results"]
            completed_iterations = set(data.get("completed_iterations", []))
            start_iteration = max(completed_iterations) + 1 if completed_iterations else 0
            heat_vectors = data.get("heat_vectors", {})
            heat_history = data.get("heat_history", [])

            # 恢复 metrics_by_layer
            saved_metrics = data.get("metrics_by_layer", {})
            if metrics_by_layer is not None:
                for layer_key in metrics_by_layer:
                    if layer_key in saved_metrics:
                        for metric_key in metrics_by_layer[layer_key]:
                            if metric_key in saved_metrics[layer_key]:
                                metrics_by_layer[layer_key][metric_key] = list(
                                    saved_metrics[layer_key][metric_key]
                                )

            print(f"从 resume.pt 恢复：{len(completed_iterations)} 次迭代，从 {start_iteration} 继续")
            if heat_vectors:
                print(f"  热度向量已恢复（{len(heat_vectors)} 层）")
        else:
            ret = _reconstruct_from_artifacts()
            if isinstance(ret, tuple) and len(ret) == 2:
                start_iteration, results = ret
            elif isinstance(ret, int):
                start_iteration = ret
                results = []
            if start_iteration > 0:
                completed_iterations = set(range(start_iteration))
                print(f"从 artifact 重建：{start_iteration} 次迭代，从 {start_iteration} 继续")

    # ═══════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════

    for iteration in range(args.num_iterations):
        if iteration < start_iteration:
            continue

        # ── 确定编辑方向和数据 ──
        if args.edit_type == "repeated_3way":
            current = args.candidate_list[iteration % 2]
            direction_label = "English" if iteration % 2 == 0 else "Spanish"

            request = {
                "prompt": args.prompt_template,
                "subject": args.subject,
                "target_new": {"str": current},
                "target_true": {"str": args.target_true},
            }
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration:3d}/{args.num_iterations} [{direction_label}]")
            print(f"{'=' * 60}")

        else:  # sequential
            record = records[iteration]
            req = record["requested_rewrite"]
            current = req["target_new"]["str"]
            target_true = req["target_true"]["str"]

            request = {
                "prompt": req["prompt"],
                "subject": req["subject"],
                "target_new": {"str": current},
                "target_true": {"str": target_true},
            }

            prompt_str = req["prompt"].format(req["subject"])
            print(f"\n{'=' * 60}")
            print(f"Edit {iteration + 1:3d}/{args.num_iterations}: {prompt_str}")
            print(f"  target: {current}")
            print(f"{'=' * 60}")

        iter_start = time.time()

        # ── 执行编辑 ──
        upd_matrices, new_heat_vectors = do_single_edit(
            model, tok, request, hparams, P, cache_c,
            mode=args.mode,
            neuron_ratio=args.neuron_ratio if args.mode in ("random", "heat") else 0.0,
            heat_decay=args.heat_decay,
            heat_temperature=args.heat_temperature,
            resonance_ratio=args.resonance_ratio,
            burst_ratio=args.burst_ratio,
            use_adaptive_ratio=args.adaptive_ratio,
            heat_vectors=heat_vectors if heat_vectors else None,
        )

        # 更新热度状态
        if args.mode == "heat" and new_heat_vectors is not None:
            heat_vectors = new_heat_vectors
            if args.edit_type == "repeated_3way":
                heat_snapshot = {k: v.clone() for k, v in heat_vectors.items()}
                heat_history.append(heat_snapshot)

        # ── 保存更新量 ──
        if iteration % args.save_every == 0:
            torch.save(
                {str(k): v.cpu() for k, v in upd_matrices.items()},
                updates_dir / f"iter_{iteration:03d}.pt",
            )

        layer_norms = {}
        for layer in edit_layers:
            layer_norms[f"upd_norm_layer_{layer}"] = float(
                torch.linalg.norm(upd_matrices[layer]).item()
            )

        # ── 评估 ──
        if args.edit_type == "repeated_3way":
            eval_metrics = evaluate_3way(model, tok, {
                "requested_rewrite": {"prompt": args.prompt_template, "subject": args.subject,
                                      "target_new": {"str": current},
                                      "target_true": {"str": args.target_true}},
                "paraphrase_prompts": [
                    "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native",
                    "An album was recorded for Capitol Nashville but never released. Danielle Darrieux spoke the language"
                ],
                "neighborhood_prompts": [
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
                ],
            }, current_target=current, target_true=args.target_true)

        else:  # sequential
            eval_metrics = evaluate_sequential(
                model, tok, record, current, target_true,
            )

            # 计算集中度指标（顺序编辑特有）
            layer_metrics = {}
            for layer in edit_layers:
                delta = upd_matrices[layer]
                metrics = compute_concentration_metrics(delta)
                layer_metrics[layer] = metrics

                layer_key = f"layer_{layer}"
                for k, v in metrics.items():
                    metrics_by_layer[layer_key][k].append(v)

            # 保存 metrics JSON
            for layer in edit_layers:
                mpath = metrics_dir / f"layer_{layer}_metrics.json"
                with open(mpath, "w", encoding="utf-8") as f:
                    json.dump(metrics_by_layer[f"layer_{layer}"], f, indent=2)

        iter_time = time.time() - iter_start

        # ── 记录结果 ──
        result_entry: Dict[str, Any] = {
            "iteration": iteration,
            "time_seconds": round(iter_time, 2),
        }
        if args.edit_type == "repeated_3way":
            result_entry["direction"] = direction_label
        else:
            result_entry["edit_idx"] = iteration
            result_entry["prompt"] = prompt_str
            result_entry["subject"] = req["subject"]
            result_entry["target_new"] = current

        result_entry.update(eval_metrics)
        result_entry.update(layer_norms)

        # 顺序编辑额外记录集中度信息
        if args.edit_type == "sequential":
            for layer in edit_layers:
                for k in ["magnitude", "gini", "normalized_entropy"]:
                    result_entry[f"layer_{layer}_{k}"] = layer_metrics[layer][k]

        results.append(result_entry)

        # ── 断点保存 ──
        edit_layer_weights = {}
        for layer in edit_layers:
            name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            edit_layer_weights[f"layer_{layer}"] = (
                nethook.get_parameter(model, name).detach().cpu()
            )
        completed_iterations.add(iteration)

        checkpoint: Dict[str, Any] = {
            "edit_layer_weights": edit_layer_weights,
            "cache_c": cache_c.detach().cpu(),
            "results": results,
            "completed_iterations": sorted(completed_iterations),
        }
        if args.mode == "heat" and heat_vectors:
            checkpoint["heat_vectors"] = {k: v.cpu() for k, v in heat_vectors.items()}
            checkpoint["heat_history"] = heat_history
        if metrics_by_layer is not None:
            checkpoint["metrics_by_layer"] = metrics_by_layer
        torch.save(checkpoint, resume_path)

        if (iteration + 1) % 10 == 0:
            backup_path = output_dir / f"resume_iter_{iteration:03d}.pt"
            torch.save(checkpoint, backup_path)

        # ── 打印摘要 ──
        print(f"  Rewrite success:    {eval_metrics.get('rewrite_success', 0):.4f}")
        print(f"  Generalization:     {eval_metrics.get('paraphrase_success', 0):.4f}")
        print(f"  Specificity:        {eval_metrics.get('neighborhood_success', 0):.4f}")
        print(f"  Score:              {eval_metrics.get('score', 0):.4f}")
        norm_str = ", ".join(f"L{k}={v:.4f}" for k, v in layer_norms.items())
        print(f"  Layer norms: {norm_str}")
        print(f"  Time: {iter_time:.1f}s")

    # ── 保存结果 & 绘图 ──
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved.")

    plot_results(results, output_dir)

    if args.mode == "heat" and heat_history:
        plot_heat_evolution(heat_history, edit_layers, output_dir)
        heat_dir = output_dir / "heat_snapshots"
        heat_dir.mkdir(exist_ok=True)
        torch.save(heat_history, heat_dir / "heat_history.pt")
        print(f"Heat history saved ({len(heat_history)} snapshots).")

    first_r, last_r = results[0], results[-1]
    print(f"\nSummary: {args.num_iterations} iterations (mode={args.mode}, edit_type={args.edit_type})")
    print(f"  First → rewrite={first_r.get('rewrite_success', 0):.4f}, "
          f"score={first_r.get('score', 0):.4f}")
    print(f"  Last  → rewrite={last_r.get('rewrite_success', 0):.4f}, "
          f"score={last_r.get('score', 0):.4f}")


if __name__ == "__main__":
    main()
