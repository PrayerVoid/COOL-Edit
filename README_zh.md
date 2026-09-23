# COOL-Edit

[English](README.md) | 中文

> **含附录的论文完整版。** 论文出版版本受页数限制，未包含附录。包含全部附录的完整版已放在本仓库中：
> [`COOL-Edit-with-appendix.pdf`](COOL-Edit-with-appendix.pdf)。

论文 **COOL-Edit: Preventing Neuron Overheating for Serial Lifelong Knowledge Editing** 的代码。

大语言模型有时需要高频地更新**同一条**知识。在串行终身知识编辑（sLKE）设定下，AlphaEdit 等主流方法
会反复覆盖同一小块局部神经元，累积的扰动会严重干扰模型并最终导致崩溃，我们把这一现象称为**神经元过热
（neuron overheating）**。COOL-Edit 为每个神经元维护一个累积**热度（heat）**，热度越高被更新的概率越
低，从而把连续的编辑分散到整个模型中。

COOL-Edit 是叠加在稀疏掩码编辑（AlphaEdit）之上的神经元选择机制。本仓库的四种编辑模式与论文比较的四
种方法一一对应：

| `--mode`    | 论文中的方法                                                           |
| ------------- | ---------------------------------------------------------------------- |
| `alphaedit` | AlphaEdit（不做掩码）                                                  |
| `random`    | 随机稀疏掩码基线                                                       |
| `nmke`      | NMKE                                                                   |
| `heat`      | **COOL-Edit**（热度引导的 softmax 采样 + 伯努利掩码 + 热度衰减） |

## 目录结构

```
COOL-Edit/
├── run_repeated_edit.py          # 反复编辑 / 顺序编辑，四种模式
├── run_slke_benchmark.py         # 2000 步串行终身知识编辑基准
├── eval_mmlu.py                  # MMLU 通用能力评测
├── analyze_concentration.py      # 顺序编辑 + 逐次更新的集中度指标
├── analyze_overheating.py        # 离线分析：集中度 / 方向一致性 / 变形报告
├── analyze_norm_consistency.py   # 离线分析：逐神经元更新范数一致性与变形
├── analyze_neuron_restore.py     # 神经元回退实验
├── requirements.txt
├── easyeditor/                   # EasyEdit 框架（原样拷贝，未做修改）
├── hparams/
│   └── AlphaEdit/                # AlphaEdit 超参数配置
├── data/
│   ├── sLKE/                     # CounterFact-freq 数据集与编辑序列
│   └── glue_eval/mmlu.pkl        # MMLU 评测数据
├── COOL-Edit-with-appendix.pdf   # 含附录的论文完整版
└── README.md / README_zh.md
```

`easyeditor/` 是 [EasyEdit](https://github.com/zjunlp/EasyEdit) 框架的原样拷贝，未改动任何框架文件。所
有编辑方法仍然通过它调用（`BaseEditor`、`AlphaEditHyperParams`、`compute_z`、`compute_ks`、`nethook`），
神经元选择机制全部实现在顶层的实验脚本里。

## 环境准备

```bash
pip install -r requirements.txt
```

**模型权重。** `hparams/AlphaEdit/` 中的配置通过 `model_name` 指向本地路径
（`hparams/AlphaEdit/llama3.1-8b.yaml` → `./hugging_cache/llama-3.1-8b-instruct`）。请把对应的
HuggingFace 权重下载到该位置，或修改 `model_name` 指向自己的副本。

**投影矩阵 `P`。** AlphaEdit 需要零空间投影矩阵，由 `P_loc` 指定（LLaMA 为 `./null_space_project.pt`，
GPT-2 为 `./P_matrix/gpt2-xl-null-space.pt`）。若文件不存在，框架会在首次运行时自动计算并在本地缓存。

**数据。** `data/sLKE/` 与 `data/glue_eval/mmlu.pkl` 随仓库提供。CounterFact-freq 的造数 prompt 见论文
附录。

## 脚本与对应实验

| 脚本                          | 论文对应                                            | 说明                                                   |
| ----------------------------- | --------------------------------------------------- | ------------------------------------------------------ |
| `run_repeated_edit.py`      | 图 1、4，图 A.3                                     | 200 步反复编辑（以及作为对照的顺序编辑），覆盖四种模式 |
| `analyze_overheating.py`    | 图 2、5，图 A.1、A.2；表 A.1、A.2                   | 离线统计已保存的更新矩阵：集中度、方向一致性、参数变形、与 Specificity 的相关性 |
| `analyze_norm_consistency.py` | 图 3（top-*k* 排序依据）；图 5                    | 离线统计逐神经元更新范数的一致性与变形；产出回退实验所需的累积更新量 |
| `analyze_neuron_restore.py` | 图 3                                                | 在编辑后的模型上做神经元回退实验                       |
| `eval_mmlu.py`              | 图 6；表 II（MMLU 列）                              | 编辑前后的 MMLU 准确率                                 |
| `run_slke_benchmark.py`     | 表 I、II（耗时列）；图 7                            | 2000 步 CounterFact-freq 串行终身编辑基准              |
| `analyze_concentration.py`  | 图 2、A.1、A.2 的数据来源                           | 边做顺序编辑边记录逐次更新的集中度指标                 |

机制类的图（图 2、A.1、A.2）两侧的数据来源不同：反复编辑侧的更新矩阵来自 `run_repeated_edit.py`，
顺序编辑侧来自 `analyze_concentration.py`（或 `run_repeated_edit.py --edit_type sequential`），
统计与绘图由 `analyze_overheating.py` 完成。图 1、图 4 中的累积均值曲线由各次运行写出的
`results.json` 计算得到。

### `run_repeated_edit.py` —— 反复编辑（图 1、4，图 A.3）

反复编辑「Danielle Darrieux 的母语是法语」这条知识，目标值在 English 与 Spanish 之间交替
（`--edit_type repeated_3way`）；也可以对数据集前 *N* 条知识各编辑一次（`--edit_type sequential`）。
指标采用 N 路比较——指定目标必须优于**所有**其他候选目标——与论文一致。

```bash
# 反复编辑，AlphaEdit 基线
python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --mode alphaedit --edit_type repeated_3way --num_iterations 200 \
    --output_dir ./results/repeated_alphaedit

# COOL-Edit（tau=1、gamma=0.9、rho=0.8，即论文设置的超参数）
python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --mode heat --neuron_ratio 0.8 --heat_decay 0.9 --heat_temperature 1.0 \
    --edit_type repeated_3way --num_iterations 200 \
    --output_dir ./results/repeated_cool

# NMKE / 随机基线：--mode nmke / --mode random

# 顺序编辑（对照实验）
python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --mode alphaedit --edit_type sequential \
    --dataset data/counterfact.json --num_iterations 200 \
    --output_dir ./results/sequential_alphaedit

# GPT-2-XL（图 A.3 / A.4 的 repeated 一侧；顺序编辑一侧用上面的命令换同一 --hparams
# 和 --edit_type sequential）
python run_repeated_edit.py --hparams hparams/AlphaEdit/gpt2-xl.yaml \
    --mode alphaedit --edit_type repeated_3way --num_iterations 200 \
    --output_dir ./results/gpt2_repeated
```

每个实验目录下的输出：

- `original_params.pt` —— 编辑前各编辑层的权重
- `updates/iter_*.pt` —— 每次编辑的更新矩阵 `ΔW`（按层保存）
- `resume.pt` —— 保存当前权重、`cache_c` 与结果的断点（用 `--resume` 续跑）
- `results.json` —— 逐次迭代的指标
- `plots/` —— 指标曲线；heat 模式还会额外保存 `heat_snapshots/heat_history.pt`

### `analyze_concentration.py` —— 顺序编辑 + 逐次集中度（图 2、A.1、A.2 的数据来源）

对数据集前 *N* 条记录依次编辑，每次编辑后统计更新矩阵在神经元维度上的集中程度：基尼系数、归一化熵、
整体幅值以及 top-1/2/5/10/20/50 % 的质量占比。同时导出每次的 `ΔW`，这是绘制 Lorenz 曲线、计算更新
向量两两余弦相似度、以及绘制「累积更新量 vs 最终余弦相似度」逐神经元散点图的输入。

```bash
python analyze_concentration.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --dataset data/counterfact.json --num_edits 200 \
    --output_dir ./results/concentration
```

输出：`updates/iter_*.pt`、`metrics/layer_<l>_metrics.json`、`results.json`、`resume.pt`、
`original_params.pt`。

### `analyze_overheating.py` —— 更新模式报告（图 2、5，图 A.1、A.2；表 A.1、A.2）

离线分析：只读取各实验目录中的 `updates/iter_*.pt`、`original_params.pt` 与 `results.json`，不需要
模型和 GPU。对每个编辑层计算：

- **集中度与累积更新量**：每次迭代更新的基尼系数、归一化熵、top-1/2/5/10/20/50 % 质量占比，
  以及各神经元的累积更新量（Lorenz 曲线与分布直方图）；
- **方向一致性**（脚本里叫 tug of war）：相邻迭代（`repeated_3way` 中目标方向每次都翻转）与同向
  迭代（隔一次）之间更新向量余弦相似度的均值与中位数；反向迭代取相邻序列中后一次迭代编号为奇数
  的那一半；
- **参数变形**：编辑后权重与原始权重的逐神经元余弦相似度，及其与累积更新量的关系
  （即图 2(d) 的散点，以及图 5 的逐方法对比）；
- **相关性**：Specificity 与集中度、变形指标的相关性；
- 跨实验对比与总览 dashboard。

```bash
python analyze_overheating.py \
    --exp_dirs results/repeated_alphaedit results/repeated_random \
               results/repeated_nmke results/repeated_cool \
    --labels alphaedit random nmke cool \
    --output_dir ./results/analysis_report
```

输出：`scalars/<label>/layer_<l>_<analysis>.json`、`tensors/<label>/layer_<l>_<analysis>.npz`、
`cross_comparison.json`、`analysis_checkpoint.json`，以及 `concentration/`、`cumulative/`、`tug_of_war/`、
`deformation/`、`correlation/`、`cross_experiment/`、`dashboard/` 几个绘图目录（前五个按实验标签分子目录）。
加 `--resume` 可从断点继续。

### `analyze_norm_consistency.py` —— 范数一致性与变形（图 3 排序依据、图 5）

离线，对每个实验单遍扫描 `updates/iter_*.pt`。对每个神经元记录它在每次迭代中收到的更新的 L2 范数、
在全部迭代上的累加值（即**累积更新量**）、首个迭代的范数排序与后续每次迭代排序的 Spearman 秩相关
（衡量「热神经元集合」的稳定性），以及由此得到的参数变形。

```bash
python analyze_norm_consistency.py \
    --exp_dirs results/repeated_alphaedit results/repeated_cool \
    --labels alphaedit cool \
    --output_dir ./results/analysis_report/norm_consistency
```

输出：`norms/<label>_layer<l>_norms.npy`（逐迭代范数）、
`norms/<label>_layer<l>_cumulative.npy`（累积更新量）、
`consistency/consistency_stats_<label>.json`、`summary.json`，以及 `consistency_decay_*`、
`cumulative_hist_*`、`final_cos_sim_*`、`mag_vs_cos_*` 等图（分别在 `consistency/`、`histograms/`、
`deformation/` 子目录下）。

`--output_dir` 不能随意指定：`analyze_neuron_restore.py` 会在
`<exp_dir.parent>/analysis_report/norm_consistency/norms/` 下找累积更新量，因此把实验目录直接放在
`./results/` 下、并用 `--output_dir ./results/analysis_report/norm_consistency`，两个脚本才对得上。
GPT-2-XL 需加 `--layers 13 14 15 16 17`；`--skip_deformation` 只算范数与一致性部分。

### `analyze_neuron_restore.py` —— 神经元回退（图 3）

从编辑后的模型出发，按累积更新量从大到小（top-*k*）或随机地把神经元恢复为原始权重，测量 Specificity
随恢复比例的变化。支持两种选择范围：`global`（所有层一起排序）与 `per_layer`（每层各排各的）。

```bash
python analyze_neuron_restore.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --exp_dirs ./results/repeated_alphaedit ./results/repeated_cool \
    --labels alphaedit cool \
    --output_dir ./results/analysis_report/neuron_restore \
    --percentages "0,0.1,0.5,1,2,5,10,20,50,100" --random_trials 5
```

脚本从各实验目录读取 `original_params.pt` 与 `resume.pt`。排序依据是 `analyze_norm_consistency.py`
写出的累积更新量（`<norms_dir>/<label>_layer<l>_cumulative.npy`）；找不到时退化为用净权重变化
`‖W_orig − W_edited‖` 排序。

### `eval_mmlu.py` —— 通用能力（图 6；表 II）

先评测未编辑模型的 MMLU 作为基线，再从各实验的 `resume.pt` 还原编辑后的权重并分别评测，报告相对基线的
变化。采用相同的 few-shot 生成式评测设置。

```bash
python eval_mmlu.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --exp_dirs ./results/repeated_alphaedit ./results/repeated_nmke \
              ./results/repeated_random ./results/repeated_cool \
    --labels alphaedit nmke random cool \
    --output_dir ./mmlu_results --num_tests 500 --resume
```

输出：`baseline.json`、`baseline_details.json`、`<label>.json`、`<label>_details.json`、`summary.json`、
`checkpoint.json`。

### `run_slke_benchmark.py` —— 2000 步串行终身编辑（表 I、II；图 7）

执行 CounterFact-freq 的编辑序列：每一步把一条 subject–relation 知识编辑到它的下一个目标值，并在指定
checkpoint 用 N 路比较重新评测此前编辑过的全部知识。同时记录每一步的耗时，即表 II 中「平均单步耗时」列
的来源。

```bash
python run_slke_benchmark.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --dataset_dir data/sLKE --index_file data/sLKE/index_2000steps.json \
    --mode heat --neuron_ratio 0.8 --heat_decay 0.9 --heat_temperature 1.0 \
    --checkpoint_list 10 100 500 1000 1500 2000 \
    --output_dir ./results/slke_cool
```

输出：`checkpoint_<T>_edits/case_*.json`、`results.json`（各 checkpoint 的汇总指标与单步耗时）、
`resume.pt` 与 `resume_checkpoint_<T>.pt`、`plots/`。

## 数据

`data/sLKE/` 是本仓库的 **CounterFact-freq**，由 [CounterFact](https://rome.baulab.info/) 衍生而来。对每条
记录保留原有的 prompt、subject 与 ground truth，并生成若干语义一致的备选目标对象；在编辑流中只有目标
对象会变化。

| 文件                                                | 内容                                                                                         |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `counterfact_n5.json` … `counterfact_n50.json` | 按更新频率分组的记录——`counterfact_nK.json` 存放配对了 *K* 个备选目标的记录（5 → 50） |
| `index_2000steps.json`                            | 主实验使用的 2000 步编辑序列，每步一条（`dataset`、`case_id`、`target_idx`）           |
| `index_n50_4x50_seed42.json`                      | 规模较小的 200 步序列（4 条记录），供 `run_slke_benchmark.py` 快速试跑                        |

`data/glue_eval/mmlu.pkl` 是 `eval_mmlu.py` 使用的 MMLU 题目（5-shot，few-shot 池与评测集不重叠）。

顺序编辑相关的实验（`run_repeated_edit.py` 的 `--edit_type sequential` 以及 `analyze_concentration.py`）
读取标准 CounterFact 数据（`requested_rewrite` 记录，含 `target_new.str` / `target_true.str`）。该文件
未随仓库分发——请从 [EasyEdit](https://github.com/zjunlp/EasyEdit) 或
[ROME](https://rome.baulab.info/) 获取并放到 `data/counterfact.json`。

## 说明

- `hparams/` 中同时保留了 EasyEdit 自带的其它方法配置，本仓库只用到
  `hparams/AlphaEdit/llama3.1-8b.yaml` 与 `hparams/AlphaEdit/gpt2-xl.yaml`。
- 编辑类脚本写入按实验划分的 `output_dir`，带 `--resume` 传入同一目录即可续跑；
  `analyze_overheating.py` 与 `eval_mmlu.py` 也各自维护断点文件。
- 反复编辑对比（图 1、4、5）单个方法的完整流程：

  ```bash
  python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
      --mode heat --neuron_ratio 0.8 --heat_decay 0.9 --heat_temperature 1.0 \
      --edit_type repeated_3way --num_iterations 200 --output_dir ./results/repeated_cool
  python analyze_norm_consistency.py --exp_dirs results/repeated_cool --labels cool \
      --output_dir ./results/analysis_report/norm_consistency
  python analyze_overheating.py --exp_dirs results/repeated_cool --labels cool \
      --output_dir ./results/analysis_report
  python analyze_neuron_restore.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
      --exp_dirs ./results/repeated_cool --labels cool \
      --output_dir ./results/analysis_report/neuron_restore
  python eval_mmlu.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
      --exp_dirs ./results/repeated_cool --labels cool --output_dir ./results/mmlu
  ```

  对另外三种方法把第一条命令换成 `--mode alphaedit / random / nmke`（并换 `--output_dir`），
  然后把四个目录一起传给 `analyze_overheating.py` 与 `eval_mmlu.py`，即得到跨方法的对比结果。

## 引用

```bibtex
@article{cooledit,
  title  = {COOL-Edit: Preventing Neuron Overheating for Serial Lifelong Knowledge Editing},
  author = {Xiao, Zhibo and Duan, Haotong and Meng, Fan},
  year   = {2026},
  note   = {Accepted for publication}
}
```

## 致谢

本项目基于 [EasyEdit](https://github.com/zjunlp/EasyEdit) 与
[AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) 构建；数据集衍生自
[CounterFact](https://rome.baulab.info/)。
