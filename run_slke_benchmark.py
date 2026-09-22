"""
sLKE 基准实验（2000 步串行终身知识编辑）。

对 counterfact_nX.json 格式的数据（每条知识有多个 target_new），
按照索引文件指定的顺序依次编辑，在指定 checkpoint 时评估所有已编辑知识。

评估标准（N 路比较）：
  - rewrite/paraphrase: 当前 target 的 nlp < 每一条其他 target_new 和 target_true
  - neighborhood: target_true 的 nlp < 每一条 target_new

例子：
python run_slke_benchmark.py \
    --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --dataset_dir data/sLKE \
    --index_file data/sLKE/index_2000steps.json \
    --output_dir ./results/slke_alphaedit \
    --mode alphaedit \
    --checkpoint_list 10 100 500 1000 1500 2000

"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easyeditor import BaseEditor
from easyeditor.models.alphaedit import AlphaEditHyperParams
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

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════


def ensure_projection_cache(
    model, tok, hparams: AlphaEditHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载 P 矩阵，初始化 cache_c。"""
    P = torch.load(hparams.P_loc, map_location="cpu", weights_only=True)
    n_layers = len(hparams.layers)
    hidden_dim = P.shape[-1]
    if P.ndim == 2:
        P = P.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
    if P.shape[0] != n_layers:
        P = P[:n_layers]
    cache_c = torch.zeros(n_layers, hidden_dim, hidden_dim)
    print(f"P matrix loaded: {P.shape}, cache_c initialized: {cache_c.shape}")
    return P, cache_c


def chunks(lst, n):
    """将列表分割成大小为 n 的块。"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def hmean(vals: List[float]) -> float:
    """调和平均数（忽略 nan）。"""
    valid = [v for v in vals if not math.isnan(v) and v > 0]
    if len(valid) < 3:
        return float("nan")
    return len(valid) / sum(1.0 / v for v in valid)


# ═══════════════════════════════════════════════════════════════════
#  掩码策略
# ═══════════════════════════════════════════════════════════════════


def generate_random_mask(
    num_neurons: int, ratio: float, device: str = "cpu",
) -> torch.Tensor:
    """随机选择 ratio 比例的神经元，返回 0/1 掩码。"""
    k = max(1, int(num_neurons * ratio))
    mask = torch.zeros(num_neurons, device=device)
    idxs = torch.randperm(num_neurons, device=device)[:k]
    mask[idxs] = 1.0
    return mask


def compute_selection_probs(
    heat: torch.Tensor, ratio: float, temperature: float = 1.0,
) -> torch.Tensor:
    """基于热度计算每个神经元的伯努利概率。"""
    heat_norm = (heat - heat.mean()) / (heat.std() + 1e-8)
    scores = -heat_norm / temperature
    probs = torch.sigmoid(scores)
    target_frac = probs.mean().item()
    if target_frac > 0:
        scale = ratio / target_frac
        probs = torch.clamp(probs * scale, 0.0, 1.0)
    return probs


def sample_neurons(probs: torch.Tensor) -> torch.Tensor:
    """独立伯努利采样，返回 0/1 掩码。"""
    return (torch.rand_like(probs) < probs).float()


def update_heat(
    heat: torch.Tensor,
    selection_mask: torch.Tensor,
    upd: torch.Tensor,
    decay: float = 0.9,
) -> torch.Tensor:
    """更新热度向量。"""
    heat = heat * decay
    update_norm = torch.linalg.norm(upd, dim=1)
    heat = heat + (1 - decay) * selection_mask * update_norm
    return heat


def get_model_type(hparams) -> str:
    """检测模型类型。"""
    name = hparams.model_name.lower()
    if "llama" in name or "qwen" in name:
        return "llama"
    return "gpt"


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


def compute_hybrid_resonant_mask(
    score_matrix: torch.Tensor,
    resonance_ratio: float = 0.25,
    burst_ratio: float = 0.15,
    use_resonance: bool = True,
    use_burst: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算混合共振/爆发掩码。

    - Resonance: 统计每个神经元 Z-score > 0 的 prompt 数，取 top ratio
    - Burst: 取 top burst_ratio 个按最大激活值排序的神经元
    """
    device = score_matrix.device
    n_prompts, n_neurons = score_matrix.shape
    resonance_mask = torch.zeros(n_neurons, device=device)
    burst_mask = torch.zeros(n_neurons, device=device)

    if use_resonance and resonance_ratio > 0:
        mean_s = score_matrix.mean(dim=0, keepdim=True)
        std_s = score_matrix.std(dim=0, keepdim=True) + 1e-8
        z_scores = (score_matrix - mean_s) / std_s
        pos_count = (z_scores > 0).sum(dim=0).float()
        n_resonance = max(1, int(n_neurons * resonance_ratio))
        _, resonance_idxs = torch.topk(pos_count, n_resonance)
        resonance_mask[resonance_idxs] = 1.0

    if use_burst and burst_ratio > 0:
        max_vals = score_matrix.max(dim=0).values
        n_burst = max(1, int(n_neurons * burst_ratio))
        _, burst_idxs = torch.topk(max_vals, n_burst)
        burst_mask[burst_idxs] = 1.0

    final_mask = torch.clamp(resonance_mask + burst_mask, 0.0, 1.0)
    return final_mask, resonance_mask, burst_mask


def entropy_adaptive_mask_ratio(
    score_matrix: torch.Tensor,
    resonance_bounds: Tuple[float, float] = (0.3, 0.4),
    burst_bounds: Tuple[float, float] = (0.3, 0.4),
    gamma_r: float = 3.0,
    gamma_b: float = 2.0,
    alpha: float = 30.0,
) -> Tuple[float, float]:
    """基于熵动态调整掩码比例。"""
    mean_scores = score_matrix.mean(dim=0)
    p = mean_scores / (mean_scores.sum() + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum().item()
    max_entropy = math.log(len(p))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

    if math.isnan(norm_entropy) or math.isinf(norm_entropy):
        norm_entropy = 0.5

    res_ratio = resonance_bounds[0] + (resonance_bounds[1] - resonance_bounds[0]) * (
        1.0 / (1.0 + math.exp(-alpha * (norm_entropy - 0.5)))
    )
    bur_ratio = burst_bounds[0] + (burst_bounds[1] - burst_bounds[0]) * (
        1.0 / (1.0 + math.exp(alpha * (norm_entropy - 0.5)))
    )
    return res_ratio, bur_ratio


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
        heat_vectors: heat 模式下返回更新后的热度，否则 None
    """
    device = f"cuda:{hparams.device}"
    edit_layers = hparams.layers
    context_templates = get_context_templates(model, tok)

    req = deepcopy(request)
    if not req["target_new"]["str"].startswith(" "):
        req["target_new"]["str"] = " " + req["target_new"]["str"]
    req_target = req["target_new"]["str"]

    z_layer = edit_layers[-1]
    req_for_z = deepcopy(req)
    req_for_z["target_new"] = req_target
    z = compute_z(model, tok, req_for_z, hparams, z_layer, context_templates)

    # -- NMKE 模式：预计算所有层的掩码 --
    nmke_masks: Dict[int, torch.Tensor] = {}
    if mode == "nmke":
        input_prompts = [
            context.format(req["prompt"].replace("{}", req["subject"]))
            for context_type in context_templates
            for context in context_type
        ]
        for layer in edit_layers:
            score_matrix = get_importance_scores_via_hooks(
                model, tok, input_prompts, layer, hparams,
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

        # -- 随机/热度掩码 --
        selection_mask: Optional[torch.Tensor] = None
        if mode == "random" and 0.0 < neuron_ratio < 1.0:
            selection_mask = generate_random_mask(
                num_neurons, neuron_ratio, device=device,
            )
            kept = int(selection_mask.sum().item())
            print(f"  Layer {layer} random mask: {kept}/{num_neurons} "
                  f"({kept / num_neurons:.2%})")

        elif mode == "heat" and 0.0 < neuron_ratio < 1.0:
            heat = heat_vectors.get(layer) if heat_vectors else None
            if heat is None:
                heat = torch.zeros(num_neurons, device=device)
            else:
                heat = heat.to(device)

            probs = compute_selection_probs(heat, neuron_ratio, heat_temperature)
            selection_mask = sample_neurons(probs)
            kept = int(selection_mask.sum().item())
            print(f"  Layer {layer} heat selection: {kept}/{num_neurons} "
                  f"({kept / num_neurons:.2%}), "
                  f"heat mean={heat.mean().item():.4f}, max={heat.max().item():.4f}")

        k_gpu = k.to(device)
        pg = P[i, :, :].to(device)
        cg = cache_c[i, :, :].to(device)

        A = pg @ (k_gpu @ k_gpu.T + cg) + hparams.L2 * torch.eye(
            k_gpu.shape[0], dtype=torch.float, device=device,
        )
        B = pg @ k_gpu @ resid.T.to(device)
        del pg, cg, k_gpu
        torch.cuda.empty_cache()

        upd = torch.linalg.solve(A, B)
        del A, B
        torch.cuda.empty_cache()

        # 应用掩码
        if selection_mask is not None:
            upd = selection_mask[:, None] * upd
        if mode == "nmke" and layer in nmke_masks:
            mask = nmke_masks[layer].to(upd.device)
            upd = mask[:, None] * upd

        # 更新热度
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

        del k, cur_zs, targets, resid, upd, upd_full
        torch.cuda.empty_cache()

    # 累积协方差
    for i, layer in enumerate(edit_layers):
        k = compute_ks(model, tok, [req_for_z], hparams, layer, context_templates).T
        cache_c[i, :, :] += k.cpu() @ k.cpu().T

    print("Single edit completed.")
    return upd_matrices, new_heat_vectors


# ═══════════════════════════════════════════════════════════════════
#  N 路评测
# ═══════════════════════════════════════════════════════════════════


def test_batch_prediction_nway(
    model, tok,
    prefixes: List[str],
    current_target: str,
    all_new_targets: List[str],
    target_true: str,
):
    """
    N 路 token-level neg log-prob 计算。

    对每条前缀，计算当前 target、其他每个 target_new、target_true 的 nlp，
    返回每个前缀的详细比较结果。
    """
    device = next(model.parameters()).device

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]

    # 构建所有候选
    candidates = [current_target] + [
        t for t in all_new_targets if t != current_target
    ] + [target_true]

    cand_toks = [
        tok(f" {c}", add_special_tokens=False)["input_ids"]
        for c in candidates
    ]
    cand_lens = [len(t) for t in cand_toks]

    # 构造大 batch: prefixes × candidates
    prompt_tok = tok(
        [f"{prefix} {cand}" for prefix in prefixes for cand in candidates],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    n_candidates = len(candidates)
    results = []
    for i in range(logits.size(0)):
        prefix_idx = i // n_candidates
        cand_idx = i % n_candidates
        cur_len = cand_lens[cand_idx]
        nlp = 0.0
        for j in range(cur_len):
            cur_tok = cand_toks[cand_idx][j]
            nlp += -F.log_softmax(
                logits[i, prefix_lens[prefix_idx] + j - 1, :], dim=0,
            )[cur_tok].item()
        nlp /= cur_len

        if cand_idx == 0:
            results.append({
                "nlp_current": nlp,
                "nlp_others": [],
                "nlp_true": 0.0,
            })
        elif cand_idx < n_candidates - 1:
            results[-1]["nlp_others"].append(nlp)
        else:
            results[-1]["nlp_true"] = nlp

    return results


def evaluate_multitarget(
    model, tok, record: Dict[str, Any],
    edit_state: Dict[int, int],
    case_id: int,
) -> Dict[str, float]:
    """
    多目标评估。

    从 edit_state 获取该知识的当前编辑进度，在 rewrite/paraphrase/neighborhood
    三类 prompt 上分别做 N 路比较。
    """
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]
    target_idx = edit_state[case_id] - 1  # 已应用的目标的索引
    all_new = [t["str"] for t in rewrite["target_new"]]
    current_target = all_new[target_idx]
    target_true = rewrite["target_true"]["str"]

    rewrite_prompts = [rewrite["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    prob_prompts = [rewrite_prompts, paraphrase_prompts, neighborhood_prompts]

    all_prefixes = list(chain(*prob_prompts))
    probs = test_batch_prediction_nway(
        model, tok, all_prefixes,
        current_target, all_new, target_true,
    )

    cutoffs = [0] + np.cumsum([len(p) for p in prob_prompts]).tolist()
    ret_probs = [
        probs[cutoffs[i - 1]:cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    metrics: Dict[str, float] = {}

    for prompt_type, prefix_results in zip(
        ["rewrite", "paraphrase", "neighborhood"], ret_probs,
    ):
        success_key = f"{prompt_type}_success"
        if not prefix_results:
            continue

        if prompt_type in ("rewrite", "paraphrase"):
            # 当前 target 必须优于所有 other targets + target_true
            success = float(np.mean([
                1.0 if (
                    x["nlp_current"] < min(x["nlp_others"] + [x["nlp_true"]])
                ) else 0.0
                for x in prefix_results
            ]))
        else:
            # neighborhood: target_true 优于所有 target_new
            success = float(np.mean([
                1.0 if (
                    x["nlp_true"] < min([x["nlp_current"]] + x["nlp_others"])
                ) else 0.0
                for x in prefix_results
            ]))
        metrics[success_key] = success

    keys = ["rewrite_success", "paraphrase_success", "neighborhood_success"]
    vals = [metrics.get(k, float("nan")) for k in keys]
    valid_vals = [v for v in vals if not math.isnan(v) and v > 0]
    if len(valid_vals) == 3:
        metrics["score"] = hmean(valid_vals)
    else:
        metrics["score"] = float("nan")

    return metrics


def chain(*iterables):
    """类似 itertools.chain 的简单实现。"""
    for it in iterables:
        for item in it:
            yield item


# ═══════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_results(results: List[Dict[str, Any]], output_dir: Path):
    """绘制指标随 checkpoint 的变化曲线。"""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    iters = [r["checkpoint"] for r in results]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    single_plots = [
        ("rewrite_success", "Rewrite Success", "rewrite_success.png"),
        ("paraphrase_success", "Generalization", "generalization.png"),
        ("neighborhood_success", "Specificity", "specificity.png"),
        ("score", "Score (HMean)", "score.png"),
    ]
    for key, ylabel, fname in single_plots:
        vals = [r.get("avg_" + key, float("nan")) for r in results]
        valid = [(i, v) for i, v in zip(iters, vals) if not math.isnan(v)]
        if not valid:
            continue
        xs, ys = zip(*valid)
        plt.figure(figsize=(10, 5))
        plt.plot(xs, ys, marker=".", markersize=5, linewidth=1.5)
        plt.xlabel("Edit Count")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Edit Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close()

    # 合并图
    plt.figure(figsize=(10, 5))
    for key, label, style in [
        ("rewrite_success", "Rewrite", "b.-"),
        ("paraphrase_success", "Generalization", "c.-"),
        ("neighborhood_success", "Specificity", "m.-"),
        ("score", "Score (HMean)", "r.-"),
    ]:
        vals = [r.get("avg_" + key, float("nan")) for r in results]
        valid = [(i, v) for i, v in zip(iters, vals) if not math.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            plt.plot(xs, ys, style, label=label, markersize=4, linewidth=1.2)
    plt.xlabel("Edit Count")
    plt.ylabel("Metric")
    plt.title("All Metrics vs Edit Count")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / "all_metrics.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="sLKE 基准实验（2000 步串行终身知识编辑，N 路比较评估）",
    )
    # 基础参数
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="数据文件夹路径，自动加载其中所有 counterfact_n*.json")
    parser.add_argument("--index_file", type=str, required=True,
                        help="索引文件路径（指定编辑顺序）")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="从已有输出目录的 resume.pt 恢复")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定 checkpoint 文件恢复（优先于 --resume）")

    # 模式选择
    parser.add_argument(
        "--mode", type=str, default="alphaedit",
        choices=["alphaedit", "random", "heat", "nmke"],
        help="编辑模式",
    )

    # 随机/热度参数
    parser.add_argument("--neuron_ratio", type=float, default=0.5,
                        help="random/heat 模式下编辑的神经元比例 (0~1)")
    parser.add_argument("--heat_decay", type=float, default=0.9,
                        help="heat 模式下热度衰减系数")
    parser.add_argument("--heat_temperature", type=float, default=1.0,
                        help="heat 模式下 softmax 温度参数")

    # NMKE 参数
    parser.add_argument("--resonance_ratio", type=float, default=0.3,
                        help="nmke 模式共振比例")
    parser.add_argument("--burst_ratio", type=float, default=0.3,
                        help="nmke 模式爆发比例")
    parser.add_argument("--adaptive_ratio", action="store_true",
                        help="nmke 模式使用自适应比例")

    # Checkpoint 参数
    parser.add_argument(
        "--checkpoint_list", type=int, nargs="+", default=None,
        help="指定编辑次数时评测，如 --checkpoint_list 1 5 10 20 50 100",
    )

    args = parser.parse_args()

    if not args.checkpoint_list:
        parser.error("必须指定 --checkpoint_list")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Edit layers: {edit_layers}")

    # ── 加载数据集（自动识别文件夹中所有 counterfact_n*.json） ──
    data_dir = Path(args.dataset_dir)
    if not data_dir.is_dir():
        parser.error(f"dataset_dir 不是有效目录: {args.dataset_dir}")
    data_paths = sorted(data_dir.glob("counterfact_n*.json"))
    if not data_paths:
        parser.error(f"在 {data_dir} 中未找到 counterfact_n*.json 文件")
    print(f"Loading datasets from: {data_dir}")
    records_map = {}
    for ds_path in data_paths:
        with open(ds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            records_map[r["case_id"]] = r
        print(f"  {ds_path.name}: {len(data)} records")
    print(f"Total: {len(records_map)} unique records")

    # ── 加载索引 ──
    print(f"Loading index: {args.index_file}")
    with open(args.index_file, "r", encoding="utf-8") as f:
        index = json.load(f)
    total_steps = len(index)
    print(f"Loaded {total_steps} edit steps")
    checkpoint_set = set(args.checkpoint_list)
    valid_checkpoints = [c for c in args.checkpoint_list if 1 <= c <= total_steps]
    max_checkpoint = max(valid_checkpoints, default=0)
    print(f"Checkpoints: {valid_checkpoints}, max_checkpoint={max_checkpoint}")
    print(f"实验将在第 {max_checkpoint} 个编辑后结束")

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
    #  断点续跑
    # ═══════════════════════════════════════════════════════════════

    resume_path = Path(args.resume_from) if args.resume_from else output_dir / "resume.pt"
    edit_state: Dict[int, int] = {}  # {case_id: applied_target_count}
    checkpoint_results: List[Dict[str, Any]] = []
    heat_vectors: Dict[int, torch.Tensor] = {}
    start_step = 0

    if args.resume_from or args.resume:
        if resume_path.exists():
            data_resume = torch.load(
                resume_path, map_location="cpu", weights_only=True,
            )
            for layer in edit_layers:
                w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                saved_w = data_resume["edit_layer_weights"][f"layer_{layer}"]
                with torch.no_grad():
                    nethook.get_parameter(model, w_name)[...] = saved_w.to(
                        next(model.parameters()).device, saved_w.dtype,
                    )
            cache_c = data_resume["cache_c"].to(cache_c.device)
            checkpoint_results = data_resume["checkpoint_results"]
            edit_state = data_resume["edit_state"]
            start_step = data_resume.get("next_step", 0)
            heat_vectors = data_resume.get("heat_vectors", {})
            print(f"从 {resume_path.name} 恢复：{len(edit_state)} 条知识已编辑，"
                  f"已执行 {start_step} 步")
        else:
            print(f"{resume_path.name} 不存在，无法恢复")
            return
    else:
        print("Starting fresh run.")

    # ═══════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════

    for step_idx in range(total_steps):
        if step_idx < start_step:
            continue

        entry = index[step_idx]
        case_id = entry["case_id"]
        target_idx = entry["target_idx"]
        record = records_map[case_id]
        rewrite = record["requested_rewrite"]

        current_target_str = rewrite["target_new"][target_idx]["str"]
        subject = rewrite["subject"]
        prompt = rewrite["prompt"]
        prompt_str = prompt.format(subject)
        target_true_str = rewrite["target_true"]["str"]

        request = {
            "prompt": prompt,
            "subject": subject,
            "target_new": {"str": current_target_str},
            "target_true": {"str": target_true_str},
        }

        print(f"\n{'=' * 60}")
        print(f"Step {step_idx + 1:3d}/{total_steps} | "
              f"case_id={case_id} | target[{target_idx}]={current_target_str}")
        print(f"  Prompt: {prompt_str}")
        print(f"{'=' * 60}")

        step_start = time.time()

        # 执行编辑
        _, new_heat_vectors = do_single_edit(
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


        # 更新编辑状态
        edit_state[case_id] = target_idx + 1

        step_time = time.time() - step_start
        print(f"Step completed in {step_time:.2f}s. "
              f"Edit state: {sum(edit_state.values())} edits applied "
              f"across {len(edit_state)} records")

        # ── Checkpoint 评估 ──
        if (step_idx + 1) in checkpoint_set:
            print(f"\n{'#' * 60}")
            print(f"Checkpoint at {step_idx + 1} edits")
            print(f"{'#' * 60}")

            ckpt_results: Dict[int, Dict[str, Any]] = {}

            for cid, applied_count in sorted(edit_state.items()):
                rec = records_map[cid]
                case_metrics = evaluate_multitarget(
                    model, tok, rec, edit_state, cid,
                )
                ckpt_results[cid] = case_metrics
                subject_name = rec["requested_rewrite"]["subject"]
                print(f"  case_id={cid} ({subject_name}): "
                      f"rewrite={case_metrics.get('rewrite_success', float('nan')):.3f}, "
                      f"paraphrase={case_metrics.get('paraphrase_success', float('nan')):.3f}, "
                      f"neigh={case_metrics.get('neighborhood_success', float('nan')):.3f}")

            # 保存细粒度结果
            ckpt_dir = output_dir / f"checkpoint_{step_idx + 1:03d}_edits"
            ckpt_dir.mkdir(exist_ok=True)
            for cid, cm in ckpt_results.items():
                with open(ckpt_dir / f"case_{cid}.json", "w") as f:
                    json.dump({"case_id": cid, **cm}, f, indent=2)

            # 汇总
            agg_metrics: Dict[str, float] = {}
            for key in ["rewrite_success", "paraphrase_success",
                        "neighborhood_success", "score"]:
                vals = [
                    cm.get(key, float("nan"))
                    for cm in ckpt_results.values()
                ]
                valid = [v for v in vals if not math.isnan(v)]
                agg_metrics["avg_" + key] = float(np.mean(valid)) if valid else float("nan")

            entry_result: Dict[str, Any] = {
                "checkpoint": step_idx + 1,
                "n_edited_records": len(edit_state),
                "n_total_edits": step_idx + 1,
                "step_time_seconds": round(step_time, 2),
            }
            entry_result.update(agg_metrics)
            checkpoint_results.append(entry_result)

            print(f"\n  Aggregated: "
                  f"rewrite={agg_metrics.get('avg_rewrite_success', float('nan')):.3f}, "
                  f"paraphrase={agg_metrics.get('avg_paraphrase_success', float('nan')):.3f}, "
                  f"neighborhood={agg_metrics.get('avg_neighborhood_success', float('nan')):.3f}, "
                  f"score={agg_metrics.get('avg_score', float('nan')):.3f}")

            # 保存结果
            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(checkpoint_results, f, indent=2, ensure_ascii=False)

        # ── 断点保存（每次 checkpoint 后） ──
        if (step_idx + 1) in checkpoint_set:
            edit_layer_weights = {}
            for layer in edit_layers:
                name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                edit_layer_weights[f"layer_{layer}"] = (
                    nethook.get_parameter(model, name).detach().cpu()
                )
            torch.save({
                "edit_layer_weights": edit_layer_weights,
                "cache_c": cache_c.cpu(),
                "checkpoint_results": checkpoint_results,
                "edit_state": edit_state,
                "next_step": step_idx + 1,
                "heat_vectors": heat_vectors,
            }, resume_path)
            # 保存带编号的 checkpoint 副本
            ckpt_resume_path = output_dir / f"resume_checkpoint_{step_idx + 1:04d}.pt"
            torch.save({
                "edit_layer_weights": edit_layer_weights,
                "cache_c": cache_c.cpu(),
                "checkpoint_results": checkpoint_results,
                "edit_state": edit_state,
                "next_step": step_idx + 1,
                "heat_vectors": heat_vectors,
            }, ckpt_resume_path)
            print(f"Checkpoint saved to {resume_path} and {ckpt_resume_path}")

            # 到达最大 checkpoint 后提前结束，不再继续编辑
            if max_checkpoint > 0 and (step_idx + 1) >= max_checkpoint:
                print(f"已到达最大 checkpoint ({max_checkpoint})，提前结束")
                break

    # ═══════════════════════════════════════════════════════════════
    #  最终恢复原始权重
    # ═══════════════════════════════════════════════════════════════

    try:
        if original_params_path.exists():
            orig_params = torch.load(
                original_params_path, map_location="cpu", weights_only=True,
            )
            for layer_name, orig_w in orig_params.items():
                layer = int(layer_name.split("_")[1])
                w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                with torch.no_grad():
                    nethook.get_parameter(model, w_name)[...] = orig_w.to(
                        next(model.parameters()).device, orig_w.dtype,
                    )
            print("Original weights restored.")
    except Exception as e:
        print(f"Warning: failed to restore original weights: {e}")

    # ── 绘图 ──
    if checkpoint_results:
        plot_results(checkpoint_results, output_dir)
        print(f"Plots saved to {output_dir / 'plots'}")
        print(f"Final results: {len(checkpoint_results)} checkpoints")

    print("Done.")


if __name__ == "__main__":
    main()
