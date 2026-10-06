# Attention Inference from Child Gaze Features: An Exploratory Benchmark of Rules, Machine Learning, and LLMs

[![tests](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/test.yml/badge.svg)](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/test.yml)
[![lint](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/lint.yml/badge.svg)](https://github.com/aazaadef/Behavioral_Modeling_LLM/actions/workflows/lint.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code, predictions and reports for an exploratory benchmark of **18 systems** that assign one of three attention categories (`focused`, `mix`, `others`) to the 69 child–clip sequences of the ChildPlay-gaze test split. The features are aggregated from ChildPlay's human gaze annotations. The systems are a rule-based threshold classifier, supervised tabular models, zero-shot NLI models and seven open-weight LLMs.

> **Summary.** On this sample, the rule-based classifier has the highest point estimates of agreement with the two-rater consensus (Cohen's κ = 0.706, 95% CI 0.55–0.87; inter-rater κ = 0.613, 95% CI 0.47–0.84). Its advantage over the strongest LLMs and a random forest is not statistically significant after correction for multiple comparisons, and no system reliably identifies the three `others` sequences. All findings are exploratory and concern this one small sample.

---

## Table of Contents

- [Pipeline](#pipeline)
- [Evaluation Suite](#evaluation-suite)
- [Results](#results)
- [Human Reference Labels](#human-reference-labels)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing the Paper](#reproducing-the-paper)
- [Citation](#citation)

---

## Pipeline

```
  ChildPlay annotations               (frame-level human gaze annotations + bbox)
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
  Evaluation against the two-rater consensus and each rater
  (original analysis: scripts/ and output/3class_eval/;
   revised analysis: revision/)
```

---

## Evaluation Suite

Eighteen systems are evaluated under the same three classes (`focused` / `mix` / `others`) on all 69 child–clip sequences of the ChildPlay-gaze test split.

| Family | Backends |
|---|---|
| **Rule-based** | Threshold classifier on aggregated features (thresholds fixed before labelling) |
| **Zero-shot NLI** | DeBERTa-v3 · BART-large-MNLI · DistilBERT-base-MNLI |
| **Supervised (out-of-fold CV)** | LogisticRegression · LinearSVM · RandomForest · XGBoost · LightGBM · Dummy(majority) · Dummy(stratified) |
| **LLMs (open-weight, zero-shot)** | Qwen2.5-72B-Instruct · Qwen2.5-7B-Instruct · qwen-7b (Qwen1.5-7B-Chat) · Yi-1.5-9B-Chat · Llama-3.1-8B-Instruct · Mistral-7B-Instruct-v0.3 · Phi-4-mini-instruct |

"Phase A" in file names denotes the stage in which all systems were run from scratch under the three-class schema; it is not a separate study.

---

## Results

The results of the revised analysis, with paired cluster-bootstrap intervals over YouTube channels, exact McNemar tests with Holm correction, grouped cross-validation, prompt ablations, out-of-fold few-shot prompting, NLI formulation sensitivity, LLM calibration and HRSF ablations, are in [`revision/`](revision/); start with [`revision/statistics/STATISTICS.md`](revision/statistics/STATISTICS.md).

- The rule-based classifier has the highest point estimates (accuracy 0.870, κ = 0.706). After Holm correction it differs significantly only from LightGBM, the linear SVM, NLI BART and the stratified dummy; against the strongest LLMs the paired difference is 5 versus 1 discordant sequences (exact p = 0.22, Holm-adjusted 0.75).
- Model-to-consensus κ and rater-to-rater κ describe different relationships and are not compared as a "ceiling".
- F1 for `others` is 0 for all 18 systems; with three `others` sequences no conclusion about this class is possible.
- Adding the rule's thresholds or three labelled examples to the LLM prompt lowered agreement for six of the seven LLMs.
- The exploratory HRSF analysis (a reliability gate followed by a vote between the rule and the LLMs) did not improve on the rule.

The original analysis (bootstrap over sequences, unadjusted McNemar tests) is kept in [`output/3class_eval/phase_a/`](output/3class_eval/phase_a/) for traceability.

---

## Human Reference Labels

Two authors labelled all 69 sequences independently from the video clips, without feature values or model outputs, using the guideline of the original five-category protocol ([`output/paper_eval/annotation_guidelines.md`](output/paper_eval/annotation_guidelines.md)). The five categories were then merged into the three classes; no sequence was relabelled. Disagreements were resolved by joint discussion between the two raters. See [`output/paper_eval/README.md`](output/paper_eval/README.md).

---

## Repository Structure

```
├── pyproject.toml               # Build / lint / test config
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Pinned dev deps (pytest, flake8, black)
│
├── src/project_llm/             # Importable feature pipeline package
│   ├── dataset.py               # ChildPlay annotation loader
│   ├── features.py              # Per-(child × clip) feature aggregation
│   ├── temporal.py              # Temporal segmentation
│   ├── interactions.py          # Interaction inference layer
│   ├── labels_3class.py         # 3-class schema, rule-based classifier, LLM prompt
│   ├── supervised_baselines.py  # Supervised feature definitions
│   └── io_utils_v2.py           # Safe write / read helpers
│
├── scripts/                     # Original analysis scripts
├── revision/                    # Revised analysis (see revision/README.md)
├── tests/                       # Pytest suite
├── data set/                    # ChildPlay-gaze annotations
│
└── output/                      # Tracked: reports, figures, CSVs
    ├── all_predictions_3class.csv    # 69 × 18 predictions + labels
    ├── 3class_eval/                  # Native 3-class predictions and original reports
    └── paper_eval/                   # Labelling guideline and label sheet
```

Excluded from the repository (see [`.gitignore`](.gitignore)): the local HuggingFace model cache (`models/`), `backup/`, virtual environments, the source videos, and any video frames. No images of children are included.

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

## Reproducing the Paper

Original analysis:

```bash
PYTHONPATH=src python scripts/phase_a_3class.py         # original statistics
PYTHONPATH=src python scripts/per_rater_analysis.py     # per-rater κ
PYTHONPATH=src python scripts/run_hrsf.py               # HRSF α sweep
```

Revised analysis (all tables, figures and supplementary files of the revised article): see [`revision/README.md`](revision/README.md).

---

## Citation

Citation details will be added once the paper is publicly available. For interim reference, please cite this repository:

```bibtex
@misc{faraji2026childattention,
  author = {Faraji, Aazaade and Norscia, Ivan and Cordoni, Giada and Pombo, Nuno},
  title  = {Attention Inference from Child Gaze Features: An Exploratory
            Benchmark of Rules, Machine Learning, and LLMs},
  year   = {2026},
  url    = {https://github.com/aazaadef/Behavioral_Modeling_LLM}
}
```

---

## Acknowledgements

Built on the ChildPlay-gaze dataset (Tafasca et al., ICCV 2023; [Zenodo record 8252535](https://zenodo.org/records/8252535)). LLM inference uses open-weight models from the Qwen, Yi, Llama, Mistral and Phi families.

---

## License

- **Code** in this repository is licensed under the [MIT License](LICENSE).
- **Data** under `data set/` (ChildPlay-gaze annotations) is third-party and licensed
  under **CC BY-NC 4.0** (non-commercial) — see [`data set/ATTRIBUTION.md`](data%20set/ATTRIBUTION.md).
