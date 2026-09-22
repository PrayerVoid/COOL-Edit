# COOL-Edit

English | [中文](README_zh.md)

Code for **COOL-Edit: Preventing Neuron Overheating for Serial Lifelong Knowledge Editing**.

Large language models must sometimes be updated at high frequency on the *same* fact. Under serial
lifelong knowledge editing (sLKE), mainstream editors such as AlphaEdit repeatedly overwrite the same
localized subset of neurons; the accumulated perturbation — *neuron overheating* — causes severe
interference and eventually model collapse. COOL-Edit maintains a cumulative **heat** value per neuron
and lowers the update probability of hot neurons, spreading successive edits across the model.

COOL-Edit is a neuron-selection mechanism added on top of sparse-masked editing (AlphaEdit). The four
editing modes released here correspond directly to the four methods compared in the paper:

| `--mode`    | Method in paper                                                                     |
| ------------- | ----------------------------------------------------------------------------------- |
| `alphaedit` | AlphaEdit (no masking)                                                              |
| `random`    | Random sparse-mask baseline                                                         |
| `nmke`      | NMKE                                                                                |
| `heat`      | **COOL-Edit** (heat-guided softmax sampling + Bernoulli masking + heat decay) |

## Repository structure

```
COOL-Edit/
├── run_repeated_edit.py          # repeated / sequential editing, four modes
├── run_slke_benchmark.py         # 2000-step sLKE benchmark
├── eval_mmlu.py                  # MMLU general-capability evaluation
├── analyze_concentration.py      # sequential editing + per-edit concentration metrics
├── analyze_overheating.py        # offline: concentration / direction / deformation report
├── analyze_norm_consistency.py   # offline: per-neuron update-norm consistency + deformation
├── analyze_neuron_restore.py     # neuron-restoration experiment
├── requirements.txt
├── easyeditor/                   # EasyEdit framework (verbatim copy, unmodified)
├── hparams/
│   └── AlphaEdit/                # AlphaEdit hyper-parameter configs
├── data/
│   ├── sLKE/                     # CounterFact-freq + edit schedules
│   └── glue_eval/mmlu.pkl        # MMLU evaluation data
└── README.md / README_zh.md
```

`easyeditor/` is an unmodified copy of the [EasyEdit](https://github.com/zjunlp/EasyEdit) framework.
All knowledge-editing methods are still called through it (`BaseEditor`, `AlphaEditHyperParams`,
`compute_z`, `compute_ks`, `nethook`); the neuron-selection mechanisms live entirely in the top-level
scripts.

## Setup

```bash
pip install -r requirements.txt
```

**Model weights.** The configs in `hparams/AlphaEdit/` point at local paths via `model_name`
(`hparams/AlphaEdit/llama3.1-8b.yaml` → `./hugging_cache/llama-3.1-8b-instruct`). Download the
corresponding HuggingFace checkpoint there, or edit `model_name` to point at your own copy.

**Projection matrix `P`.** AlphaEdit needs a null-space projection matrix, referenced by `P_loc`
(`./null_space_project.pt` for LLaMA, `./P_matrix/gpt2-xl-null-space.pt` for GPT-2). If the file does
not exist, the framework computes it on the first run and caches it locally.

**Data.** `data/sLKE/` and `data/glue_eval/mmlu.pkl` ship with this repository. The dataset-generation
prompt for CounterFact-freq is given in the appendix of the paper.

## Scripts and the experiments they reproduce

| Script                        | Paper item                                                   | Summary                                                                                  |
| ----------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `run_repeated_edit.py`      | Fig. 1, 4, A.3                                               | 200-step repeated editing (and sequential editing for the contrast) under all four modes |
| `analyze_overheating.py`    | Fig. 2, 5, A.1, A.2; Table A.1, A.2                          | Offline report on the saved update matrices: concentration, direction consistency, deformation, correlation with Specificity |
| `analyze_norm_consistency.py` | Fig. 3 (the top-*k* ranking); Fig. 5                        | Offline per-neuron update-norm consistency and deformation; writes the cumulative norms consumed by the restoration experiment |
| `analyze_neuron_restore.py` | Fig. 3                                                       | Neuron-restoration experiment on an edited model                                         |
| `eval_mmlu.py`              | Fig. 6; Table II (MMLU columns)                              | MMLU accuracy before/after editing                                                       |
| `run_slke_benchmark.py`     | Table I, II (time column); Fig. 7                            | The 2000-step CounterFact-freq serial lifelong editing benchmark                         |
| `analyze_concentration.py`  | Data behind Fig. 2, A.1, A.2                                 | Sequential-editing runner that logs per-edit concentration metrics while editing          |

The mechanism figures (Fig. 2, A.1, A.2) combine both sides of the comparison: the repeated-editing
update matrices come from `run_repeated_edit.py`, the sequential-editing ones from
`analyze_concentration.py` (or from `run_repeated_edit.py --edit_type sequential`), and the analysis
and plots from `analyze_overheating.py`. The cumulative-mean metric curves in Fig. 1 and Fig. 4 are
computed from the `results.json` each run writes.

### `run_repeated_edit.py` — repeated editing (Fig. 1, 4; Fig. A.3)

Repeatedly edits the fact *"Danielle Darrieux's mother tongue is French"* while alternating the target
between English and Spanish (`--edit_type repeated_3way`), or edits the first *N* records of a dataset
once each (`--edit_type sequential`). Metrics are computed with an N-way comparison — the designated
target must dominate **all** other candidate targets — matching the paper.

```bash
# Repeated editing, AlphaEdit baseline
python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --mode alphaedit --edit_type repeated_3way --num_iterations 200 \
    --output_dir ./results/repeated_alphaedit

# COOL-Edit (tau=1, gamma=0.9, rho=0.8 — the paper's settings)
python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --mode heat --neuron_ratio 0.8 --heat_decay 0.9 --heat_temperature 1.0 \
    --edit_type repeated_3way --num_iterations 200 \
    --output_dir ./results/repeated_cool

# NMKE / random baselines: --mode nmke / --mode random

# Sequential editing (the contrast experiment)
python run_repeated_edit.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --mode alphaedit --edit_type sequential \
    --dataset data/counterfact.json --num_iterations 200 \
    --output_dir ./results/sequential_alphaedit

# GPT-2-XL (Repeated side of Fig. A.3 / A.4; the Sequential side uses the command above
# with the same --hparams and --edit_type sequential)
python run_repeated_edit.py --hparams hparams/AlphaEdit/gpt2-xl.yaml \
    --mode alphaedit --edit_type repeated_3way --num_iterations 200 \
    --output_dir ./results/gpt2_repeated
```

Outputs (per experiment directory):

- `original_params.pt` — edited-layer weights before editing
- `updates/iter_*.pt` — per-edit update matrix `ΔW` for every edited layer
- `resume.pt` — checkpoint with the current weights, `cache_c` and results (`--resume` to continue)
- `results.json` — per-iteration metrics
- `plots/` — metric curves; heat-mode runs also save `heat_snapshots/heat_history.pt`

### `analyze_concentration.py` — sequential editing with per-edit concentration metrics (data behind Fig. 2, A.1, A.2)

Edits the first *N* records sequentially and, after every edit, records how concentrated the update
matrix is over neurons: Gini coefficient, entropy (normalized), total magnitude and top-1/2/5/10/20/50 %
mass. It also dumps every `ΔW`, which is the input for the Lorenz curves, the pairwise cosine
similarity between update vectors, and the per-neuron scatter of cumulative update magnitude against
final cosine similarity.

```bash
python analyze_concentration.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --dataset data/counterfact.json --num_edits 200 \
    --output_dir ./results/concentration
```

Outputs: `updates/iter_*.pt`, `metrics/layer_<l>_metrics.json`, `results.json`, `resume.pt`,
`original_params.pt`.

### `analyze_overheating.py` — update-pattern report (Fig. 2, 5, A.1, A.2; Table A.1, A.2)

Offline: it reads `updates/iter_*.pt`, `original_params.pt` and `results.json` from each experiment
directory, so no model or GPU is needed. For every edited layer it computes

- **concentration and cumulative magnitude** — Gini, normalized entropy and top-1/2/5/10/20/50 % mass
  of each iteration's update, plus the per-neuron cumulative update magnitude (Lorenz curve and
  distribution histogram);
- **direction consistency** ("tug of war") — mean/median cosine similarity of the update vectors
  between adjacent iterations (in `repeated_3way` the target direction flips on every edit) and
  between same-direction iterations (one step apart); the opposite-direction series is the half
  of the adjacent series whose later iteration has an odd index;
- **deformation** — per-neuron cosine similarity between the final edited weights and the originals,
  and its relation to the cumulative update magnitude (the scatter behind Fig. 2(d), and the
  per-method comparison behind Fig. 5);
- **correlation** — Specificity against the concentration and deformation metrics;
- a cross-experiment comparison and a summary dashboard.

```bash
python analyze_overheating.py \
    --exp_dirs results/repeated_alphaedit results/repeated_random \
               results/repeated_nmke results/repeated_cool \
    --labels alphaedit random nmke cool \
    --output_dir ./results/analysis_report
```

Outputs: `scalars/<label>/layer_<l>_<analysis>.json`, `tensors/<label>/layer_<l>_<analysis>.npz`,
`cross_comparison.json`, `analysis_checkpoint.json`, and the `concentration/`, `cumulative/`,
`tug_of_war/`, `deformation/`, `correlation/`, `cross_experiment/` and `dashboard/` plot directories
(the first five are split per experiment label). Add `--resume` to continue from the checkpoint file.

### `analyze_norm_consistency.py` — norm consistency and deformation (Fig. 3 ranking, Fig. 5)

Offline, one pass over `updates/iter_*.pt` per experiment. For each neuron it records the L2 norm of
the update it receives at every iteration, the sum over iterations (the *cumulative update
magnitude*), the Spearman rank correlation between the first iteration's ranking and each later
iteration's (how stable the hot-neuron set is), and the resulting parameter deformation.

```bash
python analyze_norm_consistency.py \
    --exp_dirs results/repeated_alphaedit results/repeated_cool \
    --labels alphaedit cool \
    --output_dir ./results/analysis_report/norm_consistency
```

Outputs: `norms/<label>_layer<l>_norms.npy` (per-iteration norms),
`norms/<label>_layer<l>_cumulative.npy` (cumulative magnitudes),
`consistency/consistency_stats_<label>.json`, `summary.json`, and the `consistency_decay_*`,
`cumulative_hist_*`, `final_cos_sim_*`, `mag_vs_cos_*` plots (under `consistency/`, `histograms/`
and `deformation/` respectively).

The `--output_dir` here is not free-form: `analyze_neuron_restore.py` looks for the cumulative
magnitudes at `<exp_dir.parent>/analysis_report/norm_consistency/norms/`, so putting the experiment
directories directly under `./results/` and using `--output_dir ./results/analysis_report/norm_consistency`
makes the two scripts line up. Use `--layers 13 14 15 16 17` for GPT-2-XL, and `--skip_deformation`
to compute only the norm/consistency part.

### `analyze_neuron_restore.py` — neuron restoration (Fig. 3)

Starts from an edited model, then restores neurons to their original weights in descending order of
cumulative update magnitude (top-*k*), or at random, and measures Specificity as a function of the
restored fraction. Two selection scopes are supported: `global` (all layers ranked together) and
`per_layer` (each layer ranked separately).

```bash
python analyze_neuron_restore.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --exp_dirs ./results/repeated_alphaedit ./results/repeated_cool \
    --labels alphaedit cool \
    --output_dir ./results/analysis_report/neuron_restore \
    --percentages "0,0.1,0.5,1,2,5,10,20,50,100" --random_trials 5
```

It reads `original_params.pt` and `resume.pt` from each experiment directory. The ranking uses the
cumulative per-neuron update magnitudes written by `analyze_norm_consistency.py`
(`<norms_dir>/<label>_layer<l>_cumulative.npy`); when they are not found, the net weight change
`‖W_orig − W_edited‖` is used instead.

### `eval_mmlu.py` — general capability (Fig. 6; Table II)

Evaluates MMLU on the unedited model (baseline) and on each edited model restored from an experiment's
`resume.pt`, using the same N-shot generation setup, and reports the change relative to the baseline.

```bash
python eval_mmlu.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --exp_dirs ./results/repeated_alphaedit ./results/repeated_nmke \
              ./results/repeated_random ./results/repeated_cool \
    --labels alphaedit nmke random cool \
    --output_dir ./mmlu_results --num_tests 500 --resume
```

Outputs: `baseline.json`, `baseline_details.json`, `<label>.json`, `<label>_details.json`,
`summary.json`, `checkpoint.json`.

### `run_slke_benchmark.py` — 2000-step serial lifelong editing (Table I, II; Fig. 7)

Runs the CounterFact-freq editing schedule: each step edits one subject–relation pair to its next
target object, and at the requested checkpoints all knowledge edited so far is re-evaluated with the
N-way criterion. Records the per-step wall-clock time, which gives the "average step time" column of
Table II.

```bash
python run_slke_benchmark.py --hparams hparams/AlphaEdit/llama3.1-8b.yaml \
    --dataset_dir data/sLKE --index_file data/sLKE/index_2000steps.json \
    --mode heat --neuron_ratio 0.8 --heat_decay 0.9 --heat_temperature 1.0 \
    --checkpoint_list 10 100 500 1000 1500 2000 \
    --output_dir ./results/slke_cool
```

Outputs: `checkpoint_<T>_edits/case_*.json`, `results.json` (aggregated metrics and step time per
checkpoint), `resume.pt` and `resume_checkpoint_<T>.pt`, `plots/`.

## Data

`data/sLKE/` holds **CounterFact-freq**, derived from [CounterFact](https://rome.baulab.info/). For
each record, the original prompt, subject and ground truth are kept and a set of plausible alternative
target objects is generated; only the target object changes over the editing stream.

| File                                                | Contents                                                                                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `counterfact_n5.json` … `counterfact_n50.json` | Records grouped by update frequency —`counterfact_nK.json` holds the records paired with *K* alternative targets (5 → 50) |
| `index_2000steps.json`                            | The 2000-step editing schedule used in the main experiment: one entry per step (`dataset`, `case_id`, `target_idx`)       |
| `index_n50_4x50_seed42.json`                      | A small 200-step schedule over 4 records, for smoke-testing `run_slke_benchmark.py`                                            |

`data/glue_eval/mmlu.pkl` contains the MMLU questions used by `eval_mmlu.py` (5-shot, with a held-out
few-shot pool).

The sequential-editing runs (`--edit_type sequential` in `run_repeated_edit.py`, and
`analyze_concentration.py`) read the standard CounterFact release
(`requested_rewrite` records with `target_new.str` / `target_true.str`). That file is not
redistributed here — obtain it from [EasyEdit](https://github.com/zjunlp/EasyEdit) or
[ROME](https://rome.baulab.info/) and place it at `data/counterfact.json`.

## Notes

- `hparams/` also carries the stock EasyEdit configs of many other editors; only
  `hparams/AlphaEdit/llama3.1-8b.yaml` and `hparams/AlphaEdit/gpt2-xl.yaml` are used here.
- The editing scripts write to a per-experiment `output_dir`; pass the same directory with `--resume`
  to continue an interrupted run. `analyze_overheating.py` and `eval_mmlu.py` keep their own
  checkpoints as well.
- A typical run for the repeated-editing comparison (Fig. 1, 4, 5), for one method:

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

  Repeat the first command with `--mode alphaedit / random / nmke` (and different `--output_dir`) for
  the other three methods, then pass all four directories to `analyze_overheating.py` and
  `eval_mmlu.py` at once to get the cross-method comparisons.

## Citation

```bibtex
@article{cooledit,
  title  = {COOL-Edit: Preventing Neuron Overheating for Serial Lifelong Knowledge Editing},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements

Built on [EasyEdit](https://github.com/zjunlp/EasyEdit) and [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit).
Custom knowledge-editing datasets in this repository derive from [CounterFact](https://rome.baulab.info/).
