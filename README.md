# ChildPlay LLM-Based Behavioral Analysis

[![tests](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/test.yml/badge.svg)](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/test.yml)
[![lint](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/lint.yml/badge.svg)](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/lint.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reproducible benchmark of **18 attention-classification systems** on the
ChildPlay-gaze dataset, evaluating whether Large Language Models (LLMs) can
serve as semantic reasoners over structured behavioral features — and where
they systematically fail.

> **Headline result.** A compact rule-based classifier reaches
> Cohen's κ = **0.706** on the 3-class consensus (focused / mix / others),
> exceeding the **inter-rater κ = 0.613**. Seven open-weight LLMs cluster
> tightly between 0.797 and 0.812 accuracy, but every system — LLM,
> NLI-zero-shot, and supervised — produces F1 = 0 on the `others` class,
> motivating a **hybrid LLM + rule-based framework** as the central
> contribution.

---

## Table of Contents

- [Motivation](#motivation)
- [Pipeline](#pipeline)
- [Evaluation Suite](#evaluation-suite)
- [Headline Results](#headline-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Paper Artifacts](#reproducing-paper-artifacts)
- [Key Outputs](#key-outputs)
- [Paper Documentation](#paper-documentation)
- [Citation](#citation)

---

## Motivation

Modern gaze-estimation systems solve the **perception** problem — extracting
gaze points from raw video. They do **not** solve the **interpretation**
problem: turning gaze trajectories into reliable, human-meaningful
attention categories at the clip level.

This repository operates entirely at the interpretation layer. Given a set
of pre-computed per-child behavioral features (occlusion, gaze stability,
on-screen ratio, etc.), it asks **whether modern LLMs can replace, augment,
or be replaced by classical rule-based or supervised classifiers** for
labeling attention as `focused`, `mix`, or `others` (unreliable signal).

The findings inform a hybrid architecture: LLMs handle semantic
descriptions, while symbolic rules enforce reliability constraints on
quantitative features.

---

## Pipeline

```
  ChildPlay annotations               (frame-level gaze + bbox)
            │
            ▼
  ┌────────────────────────┐
  │  Feature aggregation   │   src/project_llm/features.py
  │  (per child × clip)    │   src/project_llm/temporal.py
  └────────────────────────┘
            │   visible_ratio, occlusion_ratio, gaze_shift_ratio,
            │   max_occlusion_streak, attention_stability_score, …
            ▼
  ┌────────────────────────────────────────────────────────────┐
  │                     18 classifier backends                  │
  │                                                             │
  │   Rule-based         Zero-shot NLI       Supervised   LLMs  │
  │   ─────────────      ────────────────    ──────────   ───── │
  │   labels_3class.py   DeBERTa-MNLI        LogReg       7×    │
  │                      BART-MNLI           SVM          open  │
  │                      DistilBERT-MNLI     RandomForest weight│
  │                                          XGBoost      LLMs  │
  │                                          LightGBM           │
  │                                          Dummy×2 (baseline) │
  └────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌────────────────────────────────────────────┐
  │  Phase A evaluation                        │
  │  • Bootstrap 95% CI (1000 iterations)      │
  │  • McNemar pairwise testing (153 pairs)    │
  │  • Per-class metrics + confusion matrices  │
  └────────────────────────────────────────────┘
            │
            ▼
   THREECLASS_REPORT.md  +  PAPER_NOTES.md  +  publication-ready CSVs
```

---

## Evaluation Suite

Eighteen systems are evaluated under the same 3-class consensus
(`focused` / `mix` / `others`) on the test split of the ChildPlay-gaze
69-sample manual evaluation set.

| Family | Backends |
|---|---|
| **Rule-based** | Threshold-driven classifier on aggregated features |
| **Zero-shot NLI** | DeBERTa-v3-MNLI · BART-large-MNLI · DistilBERT-base-MNLI |
| **Supervised (OOF CV)** | LogisticRegression · LinearSVM · RandomForest · XGBoost · LightGBM · Dummy(majority) · Dummy(stratified) |
| **LLMs (open-weight, zero-shot)** | Qwen2.5-72B-Instruct · Qwen2.5-7B-Instruct · qwen-7b · Yi-1.5-9B-Chat · Llama-3.1-8B-Instruct · Mistral-7B-Instruct-v0.3 · Phi-4-mini-instruct |

Every backend is run **natively at 3-class** under the same input
features and the same 69-clip consensus test set.

---

## Headline Results

3-class accuracy and κ on the 69-sample consensus test set:

| Rank | System | Accuracy | 95% CI | κ | Macro-F1 |
|---:|---|---:|---|---:|---:|
| 1 | **rule_based** | **0.870** | [0.783, 0.942] | **0.706** | 0.575 |
| 2 | Qwen2.5-72B-Instruct (LLM) | 0.812 | [0.710, 0.899] | 0.581 | 0.530 |
| 3 | Yi-1.5-9B-Chat (LLM) | 0.812 | [0.725, 0.899] | 0.549 | 0.518 |
| 4 | Llama-3.1-8B-Instruct (LLM) | 0.797 | [0.710, 0.884] | 0.521 | 0.509 |
| 5 | RandomForest (sup.) | 0.797 | [0.696, 0.884] | 0.491 | 0.496 |
| 6 | Phi-4-mini-instruct (LLM) | 0.797 | [0.710, 0.884] | 0.529 | 0.509 |
| 7 | Mistral-7B-Instruct-v0.3 (LLM) | 0.797 | [0.696, 0.884] | 0.542 | 0.515 |
| … | … | … | … | … | … |
| 13 | Dummy(majority) baseline | 0.696 | [0.594, 0.797] | 0.000 | 0.274 |
| 18 | BART-MNLI | 0.261 | [0.159, 0.362] | 0.000 | 0.138 |

Full ranking, bootstrap intervals, and 153-pair McNemar significance
table: [`output/3class_eval/phase_a/THREECLASS_REPORT.md`](output/3class_eval/phase_a/THREECLASS_REPORT.md).

**Inter-rater agreement**: raw 82.61%, κ = 0.613 (substantial). The
rule-based κ of 0.706 sits **above the human ceiling**.

**Shared limitation**: every system — including the rule-based one —
yields F1 = 0 on the `others` class. With only 3 `others` samples in
the 69-sample test set the metric is fragile, but the systematic
failure of the LLMs on numerical-threshold cues
([`PAPER_NOTES.md` §2–§9](output/PAPER_NOTES.md)) is the main empirical
argument for the hybrid framework.

---

## Repository Structure

```
project.LLM/
├── pyproject.toml               # Build / lint / test config
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Pinned dev deps (pytest, flake8, black)
│
├── src/project_llm/             # Importable feature pipeline package
│   ├── dataset.py               # ChildPlay annotation loader
│   ├── features.py              # Per-(child × clip) 16-feature aggregation
│   ├── temporal.py              # Temporal segmentation
│   ├── interactions.py          # Interaction inference layer
│   ├── labels_3class.py         # 3-class schema + rule-based classifier
│   ├── supervised_baselines.py  # OOF-CV supervised feature definitions
│   └── io_utils_v2.py           # Safe write / read helpers
│
├── scripts/                     # Standalone analysis scripts
│   ├── run_3class_pipeline.py
│   ├── phase_a_3class.py        # Bootstrap CI + McNemar runner
│   ├── build_3class_aggregate.py
│   ├── run_llm_3class.py        # Zero-shot LLM inference (3-class)
│   ├── predict_supervised_all_69.py
│   ├── per_rater_analysis.py    # Per-rater κ + disagreement zone
│   ├── run_hrsf.py              # HRSF formula + α sweep
│   ├── generate_paper_figures.py    # Figures 1–4
│   ├── generate_hrsf_figures.py     # Figures 6–8
│   ├── run_local_hf_llm_inference.py
│   └── download_models.py
│
├── tests/                       # Pytest suite
│
├── data set/                    # ChildPlay-gaze annotations
│
└── output/                      # Tracked: reports, figures, CSVs
    ├── PAPER_DRAFT.md           ← complete first-pass paper draft
    ├── PAPER_NOTES.md           # 17-section talking-point outline
    ├── all_predictions_3class.csv
    ├── 3class_eval/             # Native 3-class predictions
    │   └── phase_a/
    │       ├── THREECLASS_REPORT.md      ← main results report
    │       ├── PER_RATER_REPORT.md
    │       ├── hrsf/HRSF_REPORT.md       ← HRSF formula evaluation
    │       ├── bootstrap_ci_3class.csv
    │       ├── mcnemar_pairwise_3class.csv
    │       ├── per_class_metrics_3class.csv
    │       ├── per_rater_kappa.csv
    │       ├── disagreement_zone_analysis.csv
    │       ├── confusion_matrices/
    │       └── figures/                  # 7 publication-ready PNGs
    └── paper_eval/              # 69-sample manual annotation pack
```

Excluded from the repo (see [`.gitignore`](.gitignore)): the local
HuggingFace model cache (`models/`, ≈218 GB), `backup/`, the Python
virtualenv (`.venv/`), and the source-video files under
`output/paper_eval/manual_labeling_package/source_videos/`.

---

## Installation

Requires **Python 3.10+**.

```bash
git clone git@github.com:aazaadef/Behavioral_Modeling_LLM.git
cd Behavioral_Modeling_LLM

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For local HuggingFace LLM inference (Qwen, Llama, Yi, Mistral, Phi),
install the optional ML stack:

```bash
pip install transformers accelerate torch huggingface_hub
```

A HuggingFace token with **read** access is required to download gated
models (Llama, Mistral). Export it before running LLM inference:

```bash
export HUGGINGFACE_HUB_TOKEN=hf_...
```

---

## Usage

Run every backend natively at 3-class and regenerate the Phase A
report:

```bash
PYTHONPATH=src python scripts/run_3class_pipeline.py     # all 18 backends
PYTHONPATH=src python scripts/phase_a_3class.py          # bootstrap CI + McNemar
PYTHONPATH=src python scripts/build_3class_aggregate.py  # → all_predictions_3class.csv
PYTHONPATH=src python scripts/per_rater_analysis.py      # per-rater κ + disagreement zone
PYTHONPATH=src python scripts/run_hrsf.py                # HRSF formula + α sweep
PYTHONPATH=src python scripts/generate_paper_figures.py  # figs 1–4
PYTHONPATH=src python scripts/generate_hrsf_figures.py   # figs 6–8
```

### LLM inference (HuggingFace, local)

```bash
python scripts/run_local_hf_llm_inference.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --output-dir output/3class_eval
```

### Supervised baselines (OOF CV)

```bash
python scripts/predict_supervised_all_69.py
```

---

## Development

Install development dependencies (pytest, flake8, black):

```bash
pip install -r requirements-dev.txt
```

Run the dataset-free unit suite (this is what CI runs on every push):

```bash
pytest -m "not slow"
```

Run the heavier dataset-dependent tests (requires
`data set/ChildPlay-gaze/` on disk):

```bash
pytest -m slow
```

Lint and format:

```bash
flake8 src tests          # static checks
black --check src tests   # format check (no rewrites)
black src tests           # auto-format
```

CI runs `pytest -m "not slow"` on Python 3.10 / 3.11 / 3.12, plus
`flake8` and `black --check`. See [`.github/workflows/`](.github/workflows/).

---

## Reproducing Paper Artifacts

```bash
# Phase A statistics (bootstrap CI + McNemar)
python scripts/phase_a_3class.py

# Per-rater analysis (κ vs r1, r2, consensus + disagreement zone)
python scripts/per_rater_analysis.py

# HRSF formula α-sweep
python scripts/run_hrsf.py

# Regenerate paper figures
python scripts/generate_paper_figures.py
python scripts/generate_hrsf_figures.py
```

---

## Key Outputs

| File | Purpose |
|---|---|
| [`output/3class_eval/phase_a/THREECLASS_REPORT.md`](output/3class_eval/phase_a/THREECLASS_REPORT.md) | Primary results report — ranking, CIs, McNemar pairwise table, per-class metrics |
| [`output/3class_eval/phase_a/bootstrap_ci_3class.csv`](output/3class_eval/phase_a/bootstrap_ci_3class.csv) | Per-system accuracy / κ / F1 with 1000-iteration bootstrap CI |
| [`output/3class_eval/phase_a/mcnemar_pairwise_3class.csv`](output/3class_eval/phase_a/mcnemar_pairwise_3class.csv) | All 153 pairwise McNemar tests |
| [`output/3class_eval/phase_a/per_class_metrics_3class.csv`](output/3class_eval/phase_a/per_class_metrics_3class.csv) | Per-class precision / recall / F1 |
| [`output/all_predictions_3class.csv`](output/all_predictions_3class.csv) | 69 × 18 model prediction matrix + ground truth |
| [`output/3class_eval/phase_a/PER_RATER_REPORT.md`](output/3class_eval/phase_a/PER_RATER_REPORT.md) | Per-rater κ matrix, alignment fault-line, disagreement zone |
| [`output/3class_eval/phase_a/hrsf/HRSF_REPORT.md`](output/3class_eval/phase_a/hrsf/HRSF_REPORT.md) | HRSF formula, α-sweep, and per-class evaluation |
| [`output/3class_eval/phase_a/figures/`](output/3class_eval/phase_a/figures/) | Seven publication-ready PNG figures (300 DPI) |
| [`output/PAPER_NOTES.md`](output/PAPER_NOTES.md) | 17-section paper-writing talking points + figure inventory |
| [`output/PAPER_DRAFT.md`](output/PAPER_DRAFT.md) | Complete first-pass paper draft |

---

## Paper Documentation

The paper-writing notes are organized as 14 self-contained argument
sections: see [`output/PAPER_NOTES.md`](output/PAPER_NOTES.md). Highlights:

1. **§2 — Systematic LLM failure on the `others` class** (F1 = 0 across all 7 LLMs).
2. **§3 — Lexical bias**: LLMs anchor on categorical tokens like
   `dominant_gaze_class = outside_frame` rather than numerical features.
3. **§4 — Numerical-threshold reasoning failure**: LLMs ignore
   `occlusion_ratio` and `max_occlusion_streak` even when extreme.
4. **§5 — Paired case study** of two samples that contrast lexical vs
   numerical evidence, demonstrating §3 and §4 in the same dataset.
5. **§8 — Hybrid framework justification**: combines LLM semantic
   reasoning with symbolic rules for reliability constraints.
6. **§14 — Final framing**: this work bridges perception and
   interpretation rather than competing with video-based gaze models.

---

## Citation

Citation details will be added once the paper is publicly available.
For interim reference, please cite this repository:

```bibtex
@misc{aazaadef2026childplayllm,
  author = {Aazaadef},
  title  = {ChildPlay LLM-Based Behavioral Analysis: A Hybrid
            Framework for Gaze-Based Attention Classification},
  year   = {2026},
  url    = {https://github.com/aazaadef/Behavioral_Modeling_LLM}
}
```

---

## Acknowledgements

Built on the [ChildPlay-gaze](https://github.com/idiap/childplay) dataset.
LLM inference uses open-weight models hosted on the HuggingFace Hub:
Qwen, Yi, Llama, Mistral, and Phi families.
