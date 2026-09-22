"""
MMLU 通用能力评测脚本 —— 评测反复编辑实验前后模型的通用能力衰减。

功能:
  1. 评测原始模型的 MMLU 分数作为基线
  2. 从多个实验结果目录还原编辑后的模型权重，分别评测 MMLU
  3. 支持断点续跑，实时保存结果

用法:
  python eval_mmlu.py \
      --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
      --exp_dirs ./results/repeated_alphaedit ./results/repeated_cool \
      --labels alphaedit cool \
      --output_dir ./mmlu_eval_results \
      --num_tests 500 --resume

权重还原策略:
  1. resume 文件（可通过 --resume_name 指定，默认 resume.pt）存在 → 直接加载 edit_layer_weights 写入模型
  2. resume 文件不存在 → 跳过该实验，记录错误
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef

# ── 路径 ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent


from easyeditor import BaseEditor  # noqa: E402
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams  # noqa: E402
from easyeditor.util import nethook  # noqa: E402

# ═══════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════

FEW_SHOT_POOL_SIZE = 10  # 前 N 条作为 few-shot 候选池（与 BLUE 一致：FEW_SHOT_TEST_SPLIT=10）

MODEL_NAME_TO_MAX_CONTEXT_LENGTH = {
    "gpt2-xl": 1024,
    "llama-2-7b-hf": 4096,
    "llama3-8b-instruct": 4096,
    "meta-llama-3-8b-instruct": 4096,
    "meta-llama-3.1-8b": 131072,
    "meta-llama-3.1-8b-instruct": 131072,
    "llama3-8b": 4096,
    "eleutherai_gpt-j-6b": 2048,
    "gpt-j-6b": 2048,
    "gpt2-large": 1024,
    "gpt2-medium": 1024,
}


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════

def load_mmlu_data(
    pkl_path: Path,
    num_few_shots: int = 5,
    num_tests: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """加载 MMLU 数据集，分离 few-shot 示例和评测集。

    前 FEW_SHOT_POOL_SIZE 条用作 few-shot 候选池（与评测集不重叠），
    其余为评测集。从候选池中取前 num_few_shots 条作为 few-shot 示例。
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    few_shot_pool = data[:FEW_SHOT_POOL_SIZE]
    test_pool = data[FEW_SHOT_POOL_SIZE:]

    if num_tests is not None:
        test_pool = test_pool[:num_tests]

    few_shots = few_shot_pool[:num_few_shots]
    return few_shots, test_pool


# ═══════════════════════════════════════════════════════════════════
#  MMLU 评测核心（提取自 BLUE/glue_eval/mmlu_eval.py）
# ═══════════════════════════════════════════════════════════════════

def _get_label(idx: int) -> str:
    """将答案索引转为选项字母。"""
    return {0: "A", 1: "B", 2: "C", 3: "D"}.get(idx, "?")


def _parse_answer(text: str) -> int:
    """从模型生成文本中解析 A/B/C/D 答案。

    模型通常在 "Answer:" 后生成如 " A\n"、"B\n"、"A." 等文本。
    返回 0-3 表示 A-D，-1 表示无法识别。
    """
    t = text.lower()
    # 与 BLUE 一致：匹配 "a\n" / "b\n" / "c\n" / "d\n"
    if "a\n" in t:
        return 0
    elif "b\n" in t:
        return 1
    elif "c\n" in t:
        return 2
    elif "d\n" in t:
        return 3
    # 降级：匹配开头的单字母（后接换行 / 句点 / 右括号 / 空格 / 逗号，或整串仅该字母）
    t_stripped = t.strip()
    for i, letter in enumerate(["a", "b", "c", "d"]):
        if t_stripped.startswith(letter) and (
            len(t_stripped) == 1 or t_stripped[1] in "\n.) ,"
        ):
            return i
    return -1


def _build_few_shot_context(few_shots: List[Dict]) -> List[str]:
    """为每条 few-shot 示例构建格式化的上下文字符串。"""
    contexts = []
    for fs in few_shots:
        q = fs["question"]
        choices_str = "".join(
            f"({_get_label(i)}) {c}\n" for i, c in enumerate(fs["choices"])
        )
        contexts.append(
            f"Question: {q}\n{choices_str}Answer: {_get_label(fs['answer'])}\n"
        )
    return contexts


def evaluate_mmlu(
    model,
    tokenizer,
    eval_dataset: List[Dict],
    few_shots: List[Dict],
    gen_len: int = 5,
    print_logs: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """对给定数据集评测 MMLU。

    两种方法同时评估:
      - generation: 用模型 greedy decode 后从文本解析答案
      - suffix prob: 计算 " A"/" B"/" C"/" D" 的平均 neg log-prob，选概率最高者

    Returns:
        result_dict: 汇总指标
        stored_generations: 逐题详细记录
    """
    few_shot_contexts = _build_few_shot_context(few_shots)

    # 模型信息
    model_name = model.config._name_or_path.lower().split("/")[-1]
    max_context = MODEL_NAME_TO_MAX_CONTEXT_LENGTH.get(model_name, 4096)
    is_llama = "llama" in model_name
    device = next(model.parameters()).device

    # 设置 tokenizer pad_token（LLaMA 没有默认 pad token，eos_token_id 也可能为 None）
    # 使用 model.config.eos_token_id 作为可靠来源，tokenizer 的 eos_token_id 在 LLaMA 3 上可能是 None
    eos_id = model.config.eos_token_id or tokenizer.eos_token_id
    if eos_id is not None:
        tokenizer.pad_token_id = eos_id
        model.generation_config.pad_token_id = eos_id
        model.generation_config.eos_token_id = eos_id
    else:
        tokenizer.pad_token_id = 0
        model.generation_config.pad_token_id = 0

    # 覆盖 generation_config 中残留 temperature/top_p，避免与 do_sample=False 冲突
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    # 获取 A/B/C/D token ids（前置空格，适配 LLaMA 分词器）
    def _get_tok_ids(letter: str):
        ids = tokenizer(f" {letter}")["input_ids"]
        if is_llama:
            ids = ids[1:]  # 去掉 BOS
        return ids

    a_tok, b_tok, c_tok, d_tok = map(_get_tok_ids, ["A", "B", "C", "D"])
    suffixes = {
        0: ["A", a_tok, len(a_tok)],
        1: ["B", b_tok, len(b_tok)],
        2: ["C", c_tok, len(c_tok)],
        3: ["D", d_tok, len(d_tok)],
    }

    # 统计
    correct_gen = 0
    incorrect_gen = 0
    invalid_gen = 0
    correct_prob = 0
    incorrect_prob = 0

    predictions_gen = []
    predictions_prob = []
    labels = []
    stored_generations = []

    start_time = time.time()

    for idx, example in enumerate(eval_dataset):
        # ── 构建 prompt ──
        q = example["question"]
        choices_str = "".join(
            f"({_get_label(i)}) {c}\n" for i, c in enumerate(example["choices"])
        )
        question_part = f"Question: {q}\n{choices_str}Answer:"

        # 动态拼接 few-shot 上下文，控制总 token 数
        question_tok_len = len(tokenizer(question_part)["input_ids"])
        remaining = max_context - question_tok_len - gen_len

        actual_few_shot = ""
        for ctx in few_shot_contexts:
            ctx_len = len(tokenizer(ctx)["input_ids"])
            remaining -= ctx_len
            if remaining < 0:
                break
            actual_few_shot += ctx

        input_prompt = actual_few_shot + question_part
        label = example["answer"]

        if print_logs:
            print(input_prompt)

        input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
        input_prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        prefix_tok_len = len(tokenizer(input_prompt)["input_ids"])
        if is_llama:
            prefix_tok_len -= 1

        # ── 方法1: Greedy 生成 → 解析答案 ──
        max_len = input_ids.shape[1] + gen_len
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                do_sample=False,
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_suffix = generated_text.replace(input_prompt_text, "")
        answer_gen = _parse_answer(generated_suffix)

        predictions_gen.append(answer_gen)
        labels.append(label)

        if answer_gen == -1:
            invalid_gen += 1
        elif answer_gen == label:
            correct_gen += 1
        else:
            incorrect_gen += 1

        # ── 方法2: 后缀概率法 ──
        probs = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            prompt_tok = tokenizer(
                [f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                logits = model(**prompt_tok).logits
            if is_llama:
                logits = logits[:, 1:, :]
            cur_len = suffixes[i][2]
            for j in range(cur_len):
                cur_tok = suffixes[i][1][j]
                probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                )[cur_tok].item()
            probs[i] /= cur_len

        prob_a = np.exp(-probs[0])
        prob_b = np.exp(-probs[1])
        prob_c = np.exp(-probs[2])
        prob_d = np.exp(-probs[3])

        answer_prob = max(range(4), key=lambda x: [prob_a, prob_b, prob_c, prob_d][x])
        predictions_prob.append(answer_prob)

        if answer_prob == label:
            correct_prob += 1
        else:
            incorrect_prob += 1

        stored_generations.append({
            "question": q,
            "subject": example.get("subject", ""),
            "choices": example["choices"],
            "true_answer": _get_label(label),
            "generated_answer": _get_label(answer_gen) if answer_gen != -1 else "INVALID",
            "prob_answer": _get_label(answer_prob),
            "correct_gen": answer_gen == label,
            "correct_prob": answer_prob == label,
            "prob_a": float(prob_a),
            "prob_b": float(prob_b),
            "prob_c": float(prob_c),
            "prob_d": float(prob_d),
            "input_prompt": input_prompt_text,
            "generated_text": generated_suffix,
        })

        if print_logs:
            total_sofar = correct_gen + incorrect_gen + invalid_gen
            acc = correct_gen / total_sofar if total_sofar > 0 else 0
            print(
                f"  [{idx + 1}/{len(eval_dataset)}] "
                f"ACC(gen)={acc:.4f} | Gen:{_get_label(answer_gen)} "
                f"True:{_get_label(label)}"
            )

    elapsed = time.time() - start_time
    total = correct_gen + incorrect_gen + invalid_gen

    # 计算指标
    if len(set(labels)) > 1 and len(labels) > 1:
        mcc = matthews_corrcoef(labels, predictions_gen)
        f1 = f1_score(labels, predictions_gen, average="weighted")
        f1_prob = f1_score(labels, predictions_prob, average="weighted")
    else:
        mcc = 0.0
        f1 = 0.0
        f1_prob = 0.0

    result = {
        "correct_gen": correct_gen,
        "incorrect_gen": incorrect_gen,
        "invalid_gen": invalid_gen,
        "accuracy_gen": correct_gen / total if total > 0 else 0,
        "correct_prob": correct_prob,
        "incorrect_prob": incorrect_prob,
        "accuracy_prob": correct_prob / len(eval_dataset) if eval_dataset else 0,
        "total": len(eval_dataset),
        "f1": f1,
        "f1_prob": f1_prob,
        "mcc": mcc,
        "time_seconds": round(elapsed, 1),
    }

    return result, stored_generations


# ═══════════════════════════════════════════════════════════════════
#  权重还原
# ═══════════════════════════════════════════════════════════════════

def apply_edits_to_model(
    model, hparams, exp_dir: Path, edit_layers: List[int], resume_name: str
) -> Optional[int]:
    """从实验目录的 resume 文件恢复编辑后的权重到模型。

    返回 resume 文件中记录的已完成编辑迭代数（仅用于打印统计）；
    resume 文件不存在则返回 None。
    """
    resume_path = exp_dir / resume_name

    if not resume_path.exists():
        print(f"  [错误] resume 文件不存在: {resume_path}")
        return None

    resume_data = torch.load(resume_path, map_location="cpu", weights_only=True)
    edit_weights = resume_data.get("edit_layer_weights", {})
    for layer in edit_layers:
        key = f"layer_{layer}"
        if key not in edit_weights:
            print(f"  [警告] resume 文件中缺少 layer_{layer}")
            continue
        w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        saved_w = edit_weights[key]
        with torch.no_grad():
            nethook.get_parameter(model, w_name)[...] = saved_w.to(
                next(model.parameters()).device, saved_w.dtype
            )
    num_edits = len(resume_data.get("completed_iterations", []))
    print(f"  [{resume_name}] 恢复完成，{num_edits} 次编辑")
    return num_edits


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MMLU 通用能力评测")
    parser.add_argument(
        "--hparams", type=str, required=True, help="AlphaEdit 超参数 yaml 路径"
    )
    parser.add_argument(
        "--exp_dirs", type=str, nargs="+", required=True, help="实验目录列表"
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="实验标签（默认使用目录名）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./mmlu_eval_results", help="输出目录"
    )
    parser.add_argument(
        "--mmlu_data", type=str, default="./data/glue_eval/mmlu.pkl",
        help="MMLU 数据集 (pkl) 路径",
    )
    parser.add_argument(
        "--num_tests", type=int, default=None, help="评测题目数（默认全部）"
    )
    parser.add_argument(
        "--num_few_shots", type=int, default=5, help="Few-shot 示例数"
    )
    parser.add_argument(
        "--gen_len", type=int, default=5, help="生成最大 token 数"
    )
    parser.add_argument(
        "--resume_name", type=str, default="resume.pt",
        help="各实验目录中的 resume 文件名（默认 resume.pt）",
    )
    parser.add_argument(
        "--resume", action="store_true", help="从 checkpoint 断点续跑"
    )
    parser.add_argument(
        "--print_logs", action="store_true", help="打印逐题日志"
    )
    args = parser.parse_args()

    # ── 路径准备 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mmlu_data_path = Path(args.mmlu_data)
    if not mmlu_data_path.is_absolute():
        mmlu_data_path = SCRIPT_DIR / mmlu_data_path

    exp_dirs = [
        Path(d) if Path(d).is_absolute() else SCRIPT_DIR / d for d in args.exp_dirs
    ]
    labels = args.labels if args.labels else [d.name for d in exp_dirs]

    if len(labels) != len(exp_dirs):
        print("错误: --labels 数量与 --exp_dirs 不匹配")
        sys.exit(1)

    checkpoint_path = output_dir / "checkpoint.json"

    # ── 加载断点信息 ──
    completed: set = set()
    baseline_done = False
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed = set(ckpt.get("completed", []))
        baseline_done = ckpt.get("baseline_done", False)
        print(
            f"断点续跑: baseline={'已完成' if baseline_done else '未完成'}, "
            f"已完成实验: {completed}"
        )

    def save_checkpoint():
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "baseline_done": baseline_done,
                "completed": sorted(completed),
            }, f, indent=2, ensure_ascii=False)

    # ── 加载 MMLU 数据 ──
    print(f"加载 MMLU 数据: {mmlu_data_path}")
    few_shots, eval_dataset = load_mmlu_data(
        mmlu_data_path, args.num_few_shots, args.num_tests
    )
    print(f"  Few-shot: {len(few_shots)} 条, 评测集: {len(eval_dataset)} 条")

    # ── 加载模型 ──
    print(f"加载超参数: {args.hparams}")
    hparams = AlphaEditHyperParams.from_hparams(args.hparams)
    editor = BaseEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    edit_layers = hparams.layers
    print(f"模型加载完成，编辑层: {edit_layers}")

    # ── 1. Baseline 评测 ──
    baseline_path = output_dir / "baseline.json"
    baseline_result = None

    if baseline_done and baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline_result = json.load(f)
        print(
            f"\nBaseline 已完成，跳过。"
            f"ACC(gen)={baseline_result['accuracy_gen']:.4f}, "
            f"ACC(prob)={baseline_result['accuracy_prob']:.4f}"
        )
    else:
        print("\n" + "=" * 60)
        print("评测 Baseline（原始模型）")
        print("=" * 60)
        baseline_result, baseline_details = evaluate_mmlu(
            model, tok, eval_dataset, few_shots,
            gen_len=args.gen_len, print_logs=args.print_logs,
        )
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_result, f, indent=2, ensure_ascii=False)
        with open(output_dir / "baseline_details.json", "w", encoding="utf-8") as f:
            json.dump(baseline_details, f, indent=2, ensure_ascii=False)
        print(f"  ACC(gen):  {baseline_result['accuracy_gen']:.4f}")
        print(f"  ACC(prob): {baseline_result['accuracy_prob']:.4f}")
        print(f"  耗时: {baseline_result['time_seconds']:.1f}s")
        baseline_done = True
        save_checkpoint()

    # ── 2. 逐实验评测 ──
    results_summary = {"baseline": baseline_result, "experiments": {}}

    for label, exp_dir in zip(labels, exp_dirs):
        result_path = output_dir / f"{label}.json"

        if label in completed and result_path.exists():
            print(f"\n{'─' * 60}")
            print(f"[{label}] 已完成，跳过")
            with open(result_path, encoding="utf-8") as f:
                er = json.load(f)
            results_summary["experiments"][label] = er
            if "accuracy_gen" in er:
                delta = er["accuracy_gen"] - baseline_result["accuracy_gen"]
                print(f"  ACC(gen): {er['accuracy_gen']:.4f} (Δ={delta:+.4f})")
            continue

        print(f"\n{'=' * 60}")
        print(f"评测实验: {label}")
        print(f"  目录: {exp_dir}")
        print(f"{'=' * 60}")

        if not exp_dir.exists():
            print(f"  [跳过] 目录不存在")
            results_summary["experiments"][label] = {"error": "目录不存在"}
            completed.add(label)
            save_checkpoint()
            continue

        num_edits = apply_edits_to_model(
            model, hparams, exp_dir, edit_layers, args.resume_name
        )
        if num_edits is None:
            results_summary["experiments"][label] = {"error": "无法还原编辑权重"}
            completed.add(label)
            save_checkpoint()
            continue

        # 评测 MMLU
        exp_result, exp_details = evaluate_mmlu(
            model, tok, eval_dataset, few_shots,
            gen_len=args.gen_len, print_logs=args.print_logs,
        )
        exp_result["num_edits"] = num_edits
        exp_result["experiment_dir"] = str(exp_dir)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(exp_result, f, indent=2, ensure_ascii=False)
        with open(output_dir / f"{label}_details.json", "w", encoding="utf-8") as f:
            json.dump(exp_details, f, indent=2, ensure_ascii=False)

        dg = exp_result["accuracy_gen"] - baseline_result["accuracy_gen"]
        dp = exp_result["accuracy_prob"] - baseline_result["accuracy_prob"]
        print(f"  ACC(gen):  {exp_result['accuracy_gen']:.4f} (Δ={dg:+.4f})")
        print(f"  ACC(prob): {exp_result['accuracy_prob']:.4f} (Δ={dp:+.4f})")
        print(f"  编辑次数: {num_edits}, 耗时: {exp_result['time_seconds']:.1f}s")

        results_summary["experiments"][label] = exp_result
        completed.add(label)
        save_checkpoint()

    # ── 3. 汇总输出 ──
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(
        f"{'实验':<20} {'ACC(gen)':>10} {'Δ(gen)':>10} "
        f"{'ACC(prob)':>10} {'Δ(prob)':>10} {'编辑次数':>10}"
    )
    print("-" * 70)
    bl_gen = baseline_result["accuracy_gen"]
    bl_prob = baseline_result["accuracy_prob"]
    print(f"{'baseline':<20} {bl_gen:>10.4f} {'-':>10} {bl_prob:>10.4f} {'-':>10} {'-':>10}")
    for label in labels:
        er = results_summary["experiments"].get(label, {})
        if "error" in er:
            print(f"{label:<20} {'ERROR':>10}: {er['error'][:40]}")
        else:
            dg = er["accuracy_gen"] - bl_gen
            dp = er["accuracy_prob"] - bl_prob
            ne = er.get("num_edits", "?")
            print(
                f"{label:<20} {er['accuracy_gen']:>10.4f} {dg:>+10.4f} "
                f"{er['accuracy_prob']:>10.4f} {dp:>+10.4f} {str(ne):>10}"
            )

    print(f"\n结果已保存到: {output_dir.resolve()}")
    print(f"  baseline.json         原始模型 MMLU 结果")
    for label in labels:
        print(f"  {label}.json           评测结果")
        print(f"  {label}_details.json   逐题详细记录")
    print(f"  summary.json           汇总对比")
    print(f"  checkpoint.json        断点信息")


if __name__ == "__main__":
    main()
