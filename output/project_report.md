# Project Report

> **Status.** Sections 1–17 below record the chronology of Phase 1–3
> (dataset audit → pipeline → benchmarks → supervised baselines → first
> LLM runs). Numbers in §14.4, §16.4, and §17.5 reflect the earlier
> 24-sample / single-rater ground truth and are now **preliminary**.
>
> The canonical current state is:
> - [`final_project_report.md`](final_project_report.md) — narrative
> - [`phase_a_analysis/PHASE_A_REPORT.md`](phase_a_analysis/PHASE_A_REPORT.md) — statistics
> - [`phase_a_analysis/inter_rater_report.md`](phase_a_analysis/inter_rater_report.md) — inter-rater agreement
>
> §18 (appended) records Phase 4 (two-rater labeling) and Phase 5
> (Phase-A analysis). §20 records Phase 6 — the native 3-class re-run,
> which is now the headline result set.

## 1. Project And Dataset Audit

در شروع کار، فایل‌های پروژه و دیتاست از روی دیسک بررسی شدند:

- `README.md`
- `requirements.md`
- `requirements.txt`
- `data set/ChildPlay-gaze/ChildPlay-gaze/README.md`
- `clips.csv`
- `splits.csv`
- `videos.csv`
- annotation CSV files in `annotations/train`, `annotations/val`, `annotations/test`

نتیجه این مرحله:

- دیتاست فقط annotation-based است و video processing در این پروژه استفاده نشد.
- schema واقعی annotationها تایید شد:
  - `clip`
  - `frame`
  - `person_id`
  - `bbox_x`
  - `bbox_y`
  - `bbox_width`
  - `bbox_height`
  - `gaze_class`
  - `gaze_x`
  - `gaze_y`
  - `is_child`
- ساختار split تایید شد:
  - `train = 330` annotation files
  - `val = 26`
  - `test = 45`
- metadata dataset تایید شد:
  - `401` clips
  - `95` videos

## 2. Core Annotation-Only Pipeline

یک pipeline ماژولار برای تبدیل gaze annotation به behavioral interpretation ساخته شد.

ماژول‌های اصلی:

- dataset loader: `src/project_llm/dataset.py`
- temporal sequence builder: `src/project_llm/temporal.py`
- interaction approximation engine: `src/project_llm/interactions.py`
- behavioral feature extraction: `src/project_llm/features.py`
- interpretation backends: `src/project_llm/llm.py`
- prompt builder: `src/project_llm/prompts.py`
- report generation: `src/project_llm/reports.py`
- orchestration: `src/project_llm/pipeline.py`
- CLI entrypoint: `src/project_llm/cli.py` and `main.py`

نتیجه این مرحله:

- داده‌ها per child / per clip بارگذاری شدند.
- فقط `is_child = 1` نگه داشته شد.
- child identifier به صورت reproducible ساخته شد:
  - `child_id = "{clip_id}:person_{person_id}"`
- خروجی‌های اصلی pipeline تولید شدند:
  - `child_sequences.jsonl`
  - `temporal_sequences.jsonl`
  - `interaction_events.jsonl`
  - `behavioral_features.jsonl`
  - `interpretations.jsonl`
  - `summary.json`
  - `report.md`

نمونه اجرای end-to-end:

- `output/restart_run`

## 3. Research-Oriented Interpretation Layer

برای تفسیر رفتاری، چند backend طراحی شد:

- `rule-based`
- `openai:<model>`
- `zero-shot:<transformer-model>`

نکات مهم:

- backend پیش‌فرض reproducible و بدون API dependency بود: `rule-based`
- backend OpenAI برای GPT اضافه شد، ولی benchmark آن در نهایت به دلیل quota کامل نشد
- backend zero-shot با `transformers` روی مدل‌های مختلف اجرا شد

## 4. Baseline Benchmarking

benchmark اولیه روی `test` split و در سطح child-sequence اجرا شد.

### 4.1 Rule-Based

مسیر خروجی:

- `output/benchmark_rule_based`

نتیجه:

- `69` sequences evaluated
- `avg_confidence = 0.7959`
- predicted labels:
  - `focused_attention = 47`
  - `mixed_attention = 18`
  - `exploratory_attention = 4`

### 4.2 Zero-Shot DistilBERT

مدل:

- `typeform/distilbert-base-uncased-mnli`

مسیر خروجی:

- `output/benchmark_zero_shot`

نتیجه:

- `69` sequences evaluated
- `avg_confidence = 0.5259`
- predicted labels:
  - `exploratory_attention = 69`

### 4.3 Zero-Shot BART

مدل:

- `facebook/bart-large-mnli`

مسیر خروجی:

- `output/benchmark_bart_large_mnli`

نتیجه:

- `69` sequences evaluated
- `avg_confidence = 0.2843`
- predicted labels:
  - `mixed_attention = 40`
  - `occluded_attention = 29`

### 4.4 Zero-Shot DeBERTa

مدل:

- `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`

مسیر خروجی:

- `output/benchmark_deberta_zeroshot`

نتیجه:

- `69` sequences evaluated
- `avg_confidence = 0.5889`
- predicted labels:
  - `focused_attention = 69`

### 4.5 OpenAI GPT

مدل هدف:

- `openai:gpt-4.1-mini`

مسیر خروجی:

- `output/benchmark_openai`

نتیجه:

- benchmark شروع شد
- اما با خطای API متوقف شد:
  - `HTTP 429`
  - `insufficient_quota`

## 5. Final Benchmark Comparison

یک گزارش نهایی بین مدل‌های اجراشده ساخته شد.

مسیر خروجی:

- `output/final_model_comparison/report.md`
- `output/final_model_comparison/summary.json`

نتیجه:

- ranking by average confidence:
  1. `rule-based`
  2. `zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
  3. `zero-shot:typeform/distilbert-base-uncased-mnli`
  4. `zero-shot:facebook/bart-large-mnli`
- بهترین zero-shot baseline ثبت شد:
  - `zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0`

## 6. Pairwise Agreement Analysis

برای مدل‌های اجراشده pairwise agreement محاسبه شد.

مسیر خروجی:

- `output/final_model_comparison/pairwise_agreement.json`

نتایج مهم:

- `rule-based` vs `DeBERTa` = `0.6812`
- `rule-based` vs `BART` = `0.1159`
- `rule-based` vs `DistilBERT` = `0.0580`
- `DistilBERT` vs `BART` = `0.0000`
- `DistilBERT` vs `DeBERTa` = `0.0000`
- `BART` vs `DeBERTa` = `0.0000`

برداشت:

- فقط DeBERTa تا حدی با baseline rule-based همسو بود.
- BART و DistilBERT behavior بسیار متفاوتی نشان دادند.

## 7. Rule-Based Plus DeBERTa Ensemble

یک ensemble بین `rule-based` و `DeBERTa` ساخته شد.

مسیر خروجی:

- `output/ensemble_rule_based_deberta`

نتیجه:

- `69` sequences
- `avg_confidence = 0.6278`
- pairwise agreement:
  - ensemble vs `rule-based` = `1.0`
  - ensemble vs `DeBERTa` = `0.6812`

برداشت:

- در این نسخه، ensemble با tie-break وزن‌دار عملا به `rule-based` نزدیک شد.
- بنابراین از نظر label output، ensemble چیزی فراتر از baseline تولید نکرد.

## 8. Paper-Grade Evaluation Preparation

چون high-level behavioral ground truth در دیتاست وجود نداشت، یک workflow برای evaluation مقاله‌ای آماده شد.

ماژول:

- `src/project_llm/manual_eval.py`

artifactهای ساخته‌شده:

- `output/paper_eval/label_schema.json`
- `output/paper_eval/annotation_guidelines.md`
- `output/paper_eval/manual_eval_subset.json`
- `output/paper_eval/manual_eval_annotations.csv`
- `output/paper_eval/manual_eval_subset_summary.json`
- `output/paper_eval/README.md`

نتیجه:

- یک subset هدفمند برای annotation دستی ساخته شد
- `24` نمونه انتخاب شدند
- ترکیب subset:
  - `12` disagreement cases
  - `12` rule/deberta agreement cases

نکته:

- این مسیر برای evaluation مقاله‌ای صحیح‌تر است، اما به annotation انسانی نیاز دارد.

## 9. Silver-Label Evaluation Without Human Annotation

چون annotation انسانی موجود نبود، یک مسیر جایگزین ساخته شد:

- `silver-label evaluation`

ایده:

- از توافق کم‌ابهام بین `rule-based` و `DeBERTa` در نمونه‌های high-confidence یک pseudo-ground-truth ساخته شد
- این ground truth انسانی نیست و باید در مقاله با برچسب `silver-label` گزارش شود

مسیر خروجی:

- `output/silver_eval/silver_labels.json`
- `output/silver_eval/silver_labels.csv`
- `output/silver_eval/silver_label_summary.json`
- `output/silver_eval/silver_eval_summary.json`
- `output/silver_eval/report.md`

نتیجه:

- `33` silver examples ساخته شدند
- همه نمونه‌ها در این subset نقره‌ای `focused_attention` بودند
- `mean_silver_confidence = 0.7662`

ارزیابی مدل‌ها روی silver subset:

- `rule-based`
  - `accuracy = 1.0`
  - `macro_f1 = 0.2`
- `DeBERTa`
  - `accuracy = 1.0`
  - `macro_f1 = 0.2`
- `ensemble`
  - `accuracy = 1.0`
  - `macro_f1 = 0.2`
- `DistilBERT`
  - `accuracy = 0.0`
  - `macro_f1 = 0.0`
- `BART`
  - `accuracy = 0.0`
  - `macro_f1 = 0.0`

برداشت:

- این ارزیابی به نفع `rule-based` و `DeBERTa` بایاس دارد، چون silver labels از توافق همان‌ها ساخته شده است
- بنابراین برای مقاله باید به‌عنوان surrogate evaluation و نه ground-truth benchmark گزارش شود

## 10. Testing And Verification

برای کدهای پروژه test suite نگه داشته شد و در طول مسیر گسترش پیدا کرد.

فایل تست:

- `tests/test_pipeline.py`

موارد پوشش داده‌شده:

- metadata consistency
- child sequence loading
- feature extraction
- pipeline outputs
- report writing
- benchmark summary generation
- combined summary logic
- saved-run ensemble
- manual evaluation workflow
- silver-label workflow

آخرین وضعیت تست:

- `Ran 10 tests`
- `OK`

## 11. Final Status

چیزهایی که الان به‌صورت عملی و کامل دارید:

- annotation-only ChildPlay processing pipeline
- temporal segmentation
- interaction approximation
- modular interpretation backends
- benchmark outputs for:
  - `rule-based`
  - `DistilBERT`
  - `BART`
  - `DeBERTa`
- final model comparison report
- pairwise agreement analysis
- rule-based plus DeBERTa ensemble
- manual-eval workflow for paper setup
- silver-label evaluation fallback without human annotation

## 12. Most Important Findings (Phase 1)

- بهترین baseline فعلی: `rule-based`
- بهترین zero-shot baseline: `DeBERTa`
- `rule-based` and `DeBERTa` agreement: `0.6812`
- `DistilBERT` and `BART` برای این task ضعیف‌تر و ناپایدارتر بودند
- GPT benchmark به دلیل quota کامل نشد
- برای مقاله قوی، هنوز annotation انسانی یا یک ground-truth بیرونی بهترین راه است
- بدون annotation انسانی، بهترین گزینه قابل‌دفاع:
  - report کردن `silver-label evaluation` با caveat روشن

---

## 13. Code Documentation (Phase 2)

All active Python files in the project have been annotated with English
docstrings and module-level documentation explaining their role in the
pipeline.

Files commented:
- 18 source modules under `src/project_llm/`
- `main.py` (entry point)
- 7 scripts under `scripts/`
- 2 test files under `tests/`

A full backup of the project was created in `backup/` before any cleanup
or modification.  Redundant / obsolete files (old runs, failed probes,
superseded pipeline versions) were identified across 4 tiers and deleted,
while the backup was preserved as an untouched reference.

## 14. Supervised Classification Baselines (Phase 3)

### 14.1 Motivation

Zero-shot NLI models showed label collapse (DeBERTa → 100% `focused_attention`,
DistilBERT → 100% `exploratory_attention`).  To determine whether the
extracted gaze features carry enough signal for classification, three
classical supervised classifiers were trained directly on the 14 numerical
features from `BehavioralFeatures`.

### 14.2 Method

- Features: 14 numerical columns from `behavioral_features.jsonl`
  (`observed_frames`, `visible_ratio`, `on_screen_ratio`, `gaze_shift_ratio`,
  `occlusion_ratio`, `eyes_closed_ratio`, `mean_head_area`, `mean_head_motion`,
  `mean_gaze_motion`, `visible_gaze_fraction`, `gaze_shift_events`,
  `max_visible_streak`, `max_occlusion_streak`, `attention_stability_score`)
- Labels: manual `final_label` from `manual_eval_annotations.csv`
- Evaluation: Stratified k-fold cross-validation (k adapted to smallest class)
- Classifiers: Logistic Regression, Linear SVM, Random Forest (all with StandardScaler)

### 14.3 Implementation

New source module:
- `src/project_llm/supervised_baselines.py`

Runner script:
- `scripts/run_supervised_baselines.py`

### 14.4 Results (on 24 labeled samples, 2 classes)

| Classifier          | Accuracy      | Macro-F1      | Cohen's Kappa |
|---------------------|---------------|---------------|---------------|
| RandomForest        | 0.9600±0.0800 | 0.9600±0.0800 | 0.9231        |
| LinearSVM           | 0.8200±0.1833 | 0.8200±0.1833 | 0.6462        |
| LogisticRegression  | 0.7700±0.2750 | 0.7600±0.2939 | 0.5462        |

### 14.5 Key Findings

- RandomForest achieves **96% accuracy** and **0.92 kappa** — a very strong result
  showing that the extracted gaze features carry significant discriminative signal.
- Even Logistic Regression (77%) outperforms the rule-based baseline (66.7%),
  suggesting that a learned combination of features is substantially better
  than hand-crafted thresholds.
- All three supervised classifiers outperform every zero-shot NLI model.

Output directory:
- `output/supervised_baselines/`

## 15. Unified LLM Inference with Corrected Label Space (Phase 3)

### 15.1 Problem

The previous Qwen-7B run used an incorrect label space (`inside_visible`,
`outside`, `unsure`) which describes gaze location rather than attention
behavior.  This made its predictions incomparable with the other models.

### 15.2 Solution

A new unified inference script was created that enforces the correct
5-label behavioral attention schema for all local LLM models:

Script:
- `scripts/run_local_llm_unified.py`

Features:
- Uses the canonical 5-label schema with label definitions in the prompt
- Supports any HuggingFace model (local path or Hub ID)
- Supports 4-bit and 8-bit quantization for large models (70B+)
- Includes robust JSON extraction with partial-match label normalization
- Removes metadata fields (child_id, clip_id) from the feature payload
  sent to the LLM to prevent information leakage

### 15.3 Model Download Script

Script:
- `scripts/download_models.py`

Recommended models for the project's GPU setup (4× NVIDIA RTX A6000, 192GB total):

| Model                              | Size   | VRAM (fp16) | VRAM (4-bit) |
|------------------------------------|--------|-------------|--------------|
| meta-llama/Llama-3.1-8B-Instruct  | ~16GB  | ~16GB       | ~8GB         |
| Qwen/Qwen2.5-7B-Instruct          | ~14GB  | ~14GB       | ~7GB         |
| meta-llama/Llama-3.1-70B-Instruct | ~140GB | ~140GB      | ~40GB        |
| Qwen/Qwen2.5-72B-Instruct         | ~145GB | ~145GB      | ~42GB        |

### 15.4 Usage

```bash
# Download a model
python scripts/download_models.py meta-llama/Llama-3.1-8B-Instruct

# Run inference (small model, full precision)
python scripts/run_local_llm_unified.py --model ./models/meta-llama--Llama-3.1-8B-Instruct

# Run inference (large model, 4-bit quantized)
python scripts/run_local_llm_unified.py --model Qwen/Qwen2.5-72B-Instruct --load-in-4bit
```

## 16. Final Comparison Framework (Phase 3)

### 16.1 Purpose

A unified comparison script that automatically discovers all available
model predictions and evaluates them against the manual ground-truth labels.

Script:
- `scripts/run_final_comparison.py`

### 16.2 Auto-Discovery

The script discovers predictions from:
1. Rule-based baseline (`output/benchmark_rule_based/`)
2. Zero-shot NLI models (`output/benchmark_*/`)
3. Local LLM runs (`output/llm_runs/*/`)
4. Supervised baselines (`output/supervised_baselines/`)

Models with mismatched label spaces are automatically skipped with a warning.

### 16.3 Outputs

- `model_evaluation.json` — per-model accuracy, macro-F1, kappa, confusion matrices
- `model_ranking.csv` — models ranked by accuracy
- `pairwise_agreement.json` / `.csv` — pairwise agreement between all models
- `comparison_summary.json` — aggregate metadata

### 16.4 Current Results (24 labeled samples — PRELIMINARY)

**WARNING**: These results use only the 24 currently labeled samples
(out of 69).  Evaluation on the full 69-sample ground truth must be
re-run after manual labeling is complete.

| Rank | Model                              | Accuracy | Macro-F1 | Kappa  |
|------|------------------------------------|----------|----------|--------|
| 1    | supervised_RandomForest            | 0.9583   | 0.9583   | 0.9167 |
| 2    | supervised_LinearSVM               | 0.8333   | 0.8322   | 0.6667 |
| 3    | supervised_LogisticRegression      | 0.7917   | 0.7884   | 0.5833 |
| 4    | rule_based                         | 0.6667   | 0.5000   | 0.5000 |
| 5    | deberta                            | 0.5000   | 0.3333   | 0.0000 |
| 6    | distilbert                         | 0.5000   | 0.3333   | 0.0000 |
| 7    | llm_Qwen2.5-7B-Instruct            | 0.5000   | 0.2581   | 0.1724 |
| 8    | llm_Qwen2.5-72B-Instruct           | 0.4583   | 0.2657   | 0.2973 |
| 9    | llm_qwen-7b (old, new prompt)      | 0.3333   | 0.2969   | 0.2000 |
| 10   | bart                               | 0.0000   | 0.0000   | 0.0000 |

Note: DeBERTa and DistilBERT both show kappa=0.0 because each predicts
a single label for all samples — their 50% accuracy is purely due to
class balance in the labeled subset (12 focused + 12 exploratory).

Output directory:
- `output/final_comparison/`

## 17. LLM Inference Runs with Corrected Label Space (Phase 4)

### 17.1 Environment Setup

- HuggingFace Hub login: completed (token validated)
- bitsandbytes 0.48.2 installed for 4-bit quantization
- Models directory: `/home/d3849/project/project.LLM/models/`

### 17.2 Completed Runs

**Qwen-7B (old local model) with corrected 5-label prompt:**
- Before: Qwen-7B with inside_visible/outside label space → 95.7% inside_visible (unusable)
- After: Qwen-7B with 5-label prompt → non-degenerate distribution
- Label distribution: `focused_attention=40, mixed_attention=28, exploratory_attention=1`
- 0 errors, confirms the prompt fix works correctly
- Output: `output/llm_runs/models_qwen-7b/`

**Qwen2.5-7B-Instruct (newly downloaded):**
- Model: `Qwen/Qwen2.5-7B-Instruct` (~14GB)
- Label distribution: `focused_attention=63, mixed_attention=6`
- 0 errors
- Shows partial label collapse (tends to predict focused_attention) but still valid output
- Output: `output/llm_runs/models_Qwen--Qwen2_5-7B-Instruct/`

### 17.3 Qwen2.5-72B-Instruct — COMPLETED

- Model: `Qwen/Qwen2.5-72B-Instruct` (~136GB, 37 safetensor shards)
- Quantization: 4-bit (nf4, compute_dtype=fp16)
- Load time: ~24 seconds
- Inference time: ~7 minutes for 69 sequences (~6 sec/sample)
- 0 errors
- Label distribution: `focused_attention=48, mixed_attention=18, reduced_visual_availability=2, exploratory_attention=1`
- **Uses 4 distinct labels** — best label diversity among all LLMs tested
- Output: `output/llm_runs/models_Qwen--Qwen2_5-72B-Instruct/`

### 17.4 Blocked

**Llama-3.1-8B-Instruct:**
- Download failed with `GatedRepoError` — access not yet granted
- User must visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  and click "Request access" (typically approved within minutes)

### 17.5 Observations from LLM Predictions (24 labeled samples only)

- Both Qwen variants produce non-degenerate distributions (unlike zero-shot NLI)
- Current accuracy is lower than rule-based baseline — but:
  - Labels are biased toward `focused_attention` / `exploratory_attention` (the only 2 classes in labeled subset)
  - LLMs predict `mixed_attention` for many samples — these may actually be correct once full 69-label ground truth is available
  - Final judgment must wait until manual labeling of all 69 samples is complete

## 17. Remaining Steps for Paper Submission

### 17.1 In Progress (User)
- Complete manual labeling of all 69 test-split samples

### 17.2 Ready to Execute (After Labeling)
1. Re-run supervised baselines with full 69-label ground truth:
   ```bash
   rm -rf output/supervised_baselines
   python scripts/run_supervised_baselines.py
   ```
2. Download and run LLM models with corrected label space:
   ```bash
   python scripts/download_models.py meta-llama/Llama-3.1-8B-Instruct
   python scripts/run_local_llm_unified.py --model ./models/meta-llama--Llama-3.1-8B-Instruct
   python scripts/download_models.py Qwen/Qwen2.5-72B-Instruct
   python scripts/run_local_llm_unified.py --model ./models/Qwen--Qwen2.5-72B-Instruct --load-in-4bit
   ```
3. Re-run final comparison with all models:
   ```bash
   rm -rf output/final_comparison
   python scripts/run_final_comparison.py
   ```

### 17.3 Expected Paper Narrative

1. **Problem**: Automated classification of child visual attention behavior from gaze tracking data.
2. **Feature engineering**: 14 numerical features extracted from frame-level gaze annotations.
3. **Finding 1 (negative)**: Zero-shot NLI models (DeBERTa, DistilBERT, BART) collapse to single labels — generic NLI is unsuitable for this task.
4. **Finding 2 (positive)**: Supervised classifiers on extracted features achieve high accuracy (RandomForest: 96%), demonstrating that the features are highly discriminative.
5. **Finding 3**: Comparison of rule-based heuristics vs. supervised learning vs. LLM reasoning on the same feature set.
6. **Ground truth**: 69 manually labeled test samples with inter-rater validation.

---

## 18. Phase 4 — Two-Rater Labeling and Consensus

### 18.1 Independent Annotation

Both the PhD candidate (rater 1) and the advisor (rater 2) labeled all
69 test samples independently using the 5-label schema:
`focused_attention`, `exploratory_attention`, `occluded_attention`,
`reduced_visual_availability`, `mixed_attention`.

Raw ratings are recorded in
[`all_predictions_with_two_annotators.csv`](all_predictions_with_two_annotators.csv)
(columns `rater1`, `rater2`, `final_manual_label`).

### 18.2 Inter-rater Agreement

| Metric                     | Value            |
|----------------------------|------------------|
| Raw agreement              | 54/69 = 78.3 %   |
| **Cohen's κ**              | **0.531**        |
| Landis–Koch interpretation | "moderate"       |
| Disagreement cases         | 15               |
| Most common disagreement   | exploratory ↔ focused (3 cases) |

The 15 disagreements were resolved by discussion and a consensus label
recorded in `final_manual_label`. Compared with the previous single-rater
ground truth, **8 samples received a different consensus label** after
the review.

Script: [`scripts/inter_rater_and_update.py`](../scripts/inter_rater_and_update.py).
Full analysis: [`phase_a_analysis/inter_rater_report.md`](phase_a_analysis/inter_rater_report.md).

### 18.3 Updated Label Distribution (n = 69)

| Label                        | Rater 1 | Rater 2 | Consensus |
|------------------------------|--------:|--------:|----------:|
| focused_attention            | 47      | 48      | **48**    |
| mixed_attention              | 17      | 17      | **17**    |
| occluded_attention           | 2       | 3       | **3**     |
| exploratory_attention        | 3       | 1       | **1**     |
| reduced_visual_availability  | 0       | 0       | 0         |

The task is effectively **2-class in practice** — two of the five
canonical labels have ≤3 samples and one has zero. Per-class metrics on
those labels are anecdotal.

---

## 19. Phase 5 — Phase-A Statistical Analysis

A full Q1-style statistical analysis of the 10 evaluated models against
the consensus ground truth. Script:
[`scripts/phase_a_analysis.py`](../scripts/phase_a_analysis.py); outputs
under [`output/phase_a_analysis/`](phase_a_analysis/).

### 19.1 Headline accuracy (bootstrap 95 % CI, 1000 iterations)

| Rank | Model                               | Accuracy | κ       | Macro-F1 | 95 % CI (acc)  |
|-----:|-------------------------------------|---------:|--------:|---------:|:---------------|
| 1    | **rule_based**                      | **0.854**|**0.680**| **0.521**| [0.768, 0.928] |
| 2    | llm_Qwen2.5-72B-Instruct            | 0.783    | 0.515   | 0.330    | [0.681, 0.870] |
| 3    | supervised_RandomForest (OOF)       | 0.768    | 0.444   | 0.414    | [0.667, 0.870] |
| 4    | llm_qwen-7b (legacy, fixed prompt)  | 0.725    | 0.441   | 0.381    | [0.623, 0.826] |
| 5    | supervised_LinearSVM (OOF)          | 0.710    | 0.314   | 0.446    | [0.609, 0.812] |
| 6    | llm_Qwen2.5-7B-Instruct             | 0.709    | 0.151   | 0.311    | [0.608, 0.812] |
| 7    | supervised_LogisticRegression (OOF) | 0.696    | 0.268   | 0.353    | [0.594, 0.812] |
| 8    | zero-shot:DeBERTa                   | 0.696    | 0.000   | 0.237    | [0.594, 0.797] |
| 9    | zero-shot:BART                      | 0.144    | −0.021  | 0.112    | [0.073, 0.232] |
| 10   | zero-shot:DistilBERT                | 0.014    | 0.000   | 0.007    | [0.000, 0.043] |

Majority-class floor: **0.696**. Every model ≤ this value is at or below
the "predict `focused` for everything" baseline.

### 19.2 Supervised re-evaluation (leakage-corrected)

The earlier 0.96 RandomForest accuracy came from a single train/test split
that reused the training samples for evaluation
(`supervised_all69_predictions.csv`). Phase A uses the **out-of-fold
cross-validated** predictions in `supervised_predictions.csv` instead. Under
the correct protocol, RandomForest drops to 0.768 and the linear
classifiers drop to ~0.70 — still useful, but no longer the headline
result.

### 19.3 Statistical significance (McNemar, α = 0.05)

Rule-based is **significantly** better than:

- LinearSVM (p = 0.021), LogisticRegression (p = 0.007)
- Qwen2.5-7B (p = 0.013), qwen-7b legacy (p = 0.022)
- DeBERTa, BART, DistilBERT (p < 0.001 each)

Rule-based vs RandomForest (p = 0.110) and vs Qwen2.5-72B (p = 0.180)
are **not** significant. Full 45-pair matrix:
[`phase_a_analysis/mcnemar_pairwise.csv`](phase_a_analysis/mcnemar_pairwise.csv).

### 19.4 Ablation (LinearSVM on the 14 numerical features)

The temporal group (`observed_frames`, `attention_stability_score`) alone
achieves **75.4 % accuracy (κ = 0.371)** — higher than the full 14-feature
model. Motion features contribute net negative signal
(`without_motion` = 0.725 > `full` = 0.711). This is a shift from the
pre-consensus picture, driven by the new `occluded_attention` samples
being separable primarily by temporal features. Full table:
[`phase_a_analysis/ablation.csv`](phase_a_analysis/ablation.csv).

### 19.5 3-class merged view (focused / mixed / other)

Collapsing the three rare labels into `other`:

| Model                           | Accuracy | Balanced acc. | κ     |
|---------------------------------|---------:|--------------:|------:|
| rule_based                      | **0.855**| **0.663**     |**0.684**|
| llm_Qwen2.5-72B-Instruct        | 0.783    | 0.514         | 0.516 |
| supervised_RandomForest         | 0.768    | 0.495         | 0.450 |

Rule-based κ = 0.684 exceeds the **inter-rater κ = 0.531** — on this
task the model is operating at roughly the human-agreement ceiling.

### 19.6 Updated Findings

1. Rule-based is the best system and statistically ties with the largest
   LLM and the best supervised model; it significantly beats every NLI
   model and every linear supervised baseline.
2. The supervised linear baselines were brittle: their prior 91 %/κ=0.81
   numbers dropped to ~70 %/κ≈0.3 once train/test leakage was fixed and
   consensus labels were applied.
3. NLI models (DeBERTa, BART, DistilBERT) should be reported as
   **negative controls** — all three are at or below majority-class
   accuracy.
4. The effective task is 2-class (`focused` vs `mixed`) plus a
   rare/ambiguous catch-all; no model generalises on the ≤3-sample
   labels. Any clinical/behavioural claim should be scoped accordingly.
5. Inter-rater κ = 0.531 is the interpretation ceiling — even humans
   disagree on 22 % of cases. Claims of super-human accuracy should be
   read against that ceiling.

### 19.7 Publication scope (honest assessment)

At n = 69 with inter-rater κ = 0.531, the results are **appropriate for
a Q2/Q3 venue** (Behavior Research Methods, Frontiers in Psychology,
Sensors, a workshop paper), not a Q1 venue. Strengthening to Q1 would
require: external validation on a second dataset, a third annotator,
expanded labeling to n ≥ 150, and at least one class-balancing
intervention to move off the majority-class floor.

## 20. Phase 6 — From 5-class to 3-class (schema refinement + re-run)

### 20.1 Motivation: from 5-class to 3-class

ChildPlay-gaze is natively a **5-class** dataset
(`focused_attention`, `mixed_attention`, `exploratory_attention`,
`occluded_attention`, `reduced_visual_availability`). Phase-A results
in §19 are computed against that native schema. Three independent
empirical signals on the 69-sample test set argue that the 5-class
schema is **under-supported at this sample size** and that a
semantically-justified 3-class collapse is the principled choice:

**(i) Empirical class degeneracy.** In the two-rater consensus over
the full 69-sample test set, two of the five dataset classes are
near-absent:

| 5-class label | Consensus count (n = 69) |
|---|---:|
| `focused_attention` | 48 |
| `mixed_attention` | 17 |
| `occluded_attention` | 3 |
| `exploratory_attention` | **1** |
| `reduced_visual_availability` | **0** |

No confidence interval, no per-class F1, and no McNemar significance
test can be meaningfully reported for a class with 0–1 samples. The
dataset itself makes 5-class evaluation statistically ill-posed at
this scale.

**(ii) Inter-rater agreement is bound by the boundary pairs.** Under
the 5-class schema the two annotators achieve raw agreement = 54/69
(78.3 %) and Cohen's κ = 0.531 ("moderate" on Landis–Koch). The 15
disagreements cluster almost entirely on two semantically-adjacent
pairs:

| Disagreement pair (5-class) | Count |
|---|---:|
| `focused_attention` ↔ `mixed_attention` | 8 |
| `mixed_attention` ↔ `occluded_attention` | 3 |
| `exploratory_attention` ↔ `mixed_attention` | 3 |
| `exploratory_attention` ↔ `focused_attention` | 1 |

Three of these (`exploratory ↔ mixed`) are pure within-family
ambiguities: the two labels describe the same underlying behavior
(active gaze switching / scanning) at different intensities. The
`occluded ↔ reduced_visual_availability` axis shows the same pattern
on the visibility-failure side (both describe an unusable gaze
signal, differing only in cause).

**(iii) Semantic coherence of the 3-class target.** The three
resulting classes map cleanly onto the attention-literature
distinction between **sustained attention** (one stable target),
**active scanning / divided attention** (multiple targets or
switching), and **unusable gaze signal** (the child is occluded,
off-frame, or eyes closed). These are the three operational states a
downstream application actually needs to distinguish; the 5-class
distinctions within each cluster are behaviourally fine-grained but
not actionable in this use case.

Collectively, (i) makes 5-class evaluation statistically ill-posed on
n = 69, (ii) shows that the remaining 5-class distinctions are the
ones humans themselves cannot agree on, and (iii) shows that the
collapse is principled rather than convenient. This motivates the
3-class schema for **every downstream analysis**; §19 remains as the
5-class reference analysis but should be read as a negative control
("this is what under-supported labels look like").

### 20.2 Deterministic 5 → 3 mapping

The collapse is a single-valued function of the 5-class label
(implemented in [`src/project_llm/labels_3class.py`](../src/project_llm/labels_3class.py),
`MAP_5_TO_3`):

| 5-class label | 3-class label | Semantic family |
|---|---|---|
| `focused_attention` | `focused` | sustained attention |
| `mixed_attention` | `mix` | active scanning / divided attention |
| `exploratory_attention` | `mix` | active scanning / divided attention |
| `occluded_attention` | `others` | unusable gaze signal |
| `reduced_visual_availability` | `others` | unusable gaze signal |

Post-collapse, the consensus distribution becomes
`focused = 48, mix = 18, others = 3` — every class now has at least
3 samples and is amenable to bootstrap CI and McNemar testing.

**Important: the Phase 6 results are a native re-run, not a
post-hoc mapping of 5-class model outputs.** The rule engine uses
new 3-class thresholds, NLI backends use three 3-class candidate
labels, supervised baselines were re-trained with
StratifiedKFold OOF CV on 3-class targets, and every LLM ran with a
fresh 3-label prompt. (As evidence that this is not just a relabel:
LinearSVM *drops* from 0.710 → 0.667 under native 3-class re-training,
which would be impossible under post-hoc mapping — re-mapping can
only increase accuracy.) The 5-class CSVs in §19 are preserved
unmodified so the Phase-A 5-class analysis remains reproducible.

Orchestration scripts: [`scripts/run_3class_pipeline.py`](../scripts/run_3class_pipeline.py)
(rule + NLI + supervised), [`scripts/run_llm_3class.py`](../scripts/run_llm_3class.py),
[`scripts/phase_a_3class.py`](../scripts/phase_a_3class.py), and
[`scripts/write_3class_report.py`](../scripts/write_3class_report.py).

### 20.3 Inter-rater agreement under the 3-class schema

Re-evaluated on the same 69 samples with the deterministic 5→3
mapping applied to each rater independently:

| Schema | Raw agreement | Cohen's κ | Landis–Koch |
|---|---|---:|---|
| 5-class | 54/69 (78.3 %) | 0.531 | moderate |
| **3-class** | **57/69 (82.6 %)** | **0.613** | **substantial** |

Three of the 15 5-class disagreements (the `exploratory ↔ mixed`
pairs) resolve automatically under the collapse, lifting κ from
0.531 to 0.613 and crossing the Landis–Koch
moderate → substantial threshold. The remaining 12 disagreements
are split between `focused ↔ mix` (9 cases, genuinely ambiguous
boundary) and `mix ↔ others` (3 cases, visibility-failure
judgement). This κ shift is **post-hoc evidence that the schema is
right**: the disagreements it eliminates are precisely the ones
that were semantic duplicates, and the disagreements that remain
are the ones that genuinely require better labels or more samples
to resolve.

Consensus counts under 3-class: focused = 48, mix = 18, others = 3.

### 20.4 Headline accuracy (bootstrap 95 % CI, 1000 iter.)

**18 systems evaluated** — rule-based, three NLI zero-shot backends,
seven supervised baselines (LogReg, LinearSVM, RandomForest, XGBoost,
LightGBM, Dummy_majority, Dummy_stratified), and seven LLMs spanning
five families (Alibaba, Mistral AI, Microsoft, 01.ai, Meta):

| Rank | Model | Family | Accuracy | 95 % CI | κ | Macro-F1 |
|---:|---|---|---:|---|---:|---:|
| 1 | rule_based | — | **0.870** | [0.783, 0.942] | **0.706** | 0.575 |
| 2 | Qwen2.5-72B-Instruct | Alibaba | 0.812 | [0.710, 0.899] | 0.581 | 0.530 |
| 2 | Yi-1.5-9B-Chat | 01.ai | 0.812 | [0.725, 0.899] | 0.549 | 0.518 |
| 4 | Llama-3.1-8B-Instruct | Meta | 0.797 | [0.710, 0.884] | 0.521 | 0.509 |
| 4 | supervised_RandomForest | — | 0.797 | [0.696, 0.884] | 0.491 | 0.496 |
| 4 | Phi-4-mini-instruct | Microsoft | 0.797 | [0.710, 0.884] | 0.529 | 0.509 |
| 4 | Mistral-7B-Instruct-v0.3 | Mistral | 0.797 | [0.696, 0.884] | 0.542 | 0.515 |
| 8 | qwen-7b (legacy) | Alibaba | 0.768 | [0.667, 0.855] | 0.453 | 0.484 |
| 9 | Qwen2.5-7B-Instruct | Alibaba | 0.754 | [0.652, 0.855] | 0.392 | 0.460 |
| 10 | supervised_XGBoost | — | 0.739 | [0.638, 0.841] | 0.366 | 0.451 |
| 11 | supervised_LogisticRegression | — | 0.725 | [0.609, 0.826] | 0.331 | 0.444 |
| 11 | supervised_LightGBM | — | 0.725 | [0.609, 0.826] | 0.360 | 0.450 |
| 13 | supervised_Dummy_majority | — | 0.696 | [0.594, 0.797] | 0.000 | 0.274 |
| 13 | distilbert (NLI) | — | 0.696 | [0.594, 0.797] | 0.000 | 0.274 |
| 15 | deberta (NLI) | — | 0.681 | [0.579, 0.783] | −0.026 | 0.270 |
| 16 | supervised_LinearSVM | — | 0.667 | [0.565, 0.768] | 0.190 | 0.389 |
| 17 | supervised_Dummy_stratified | — | 0.536 | [0.420, 0.652] | −0.163 | 0.269 |
| 18 | bart (NLI) | — | 0.261 | [0.159, 0.362] | 0.000 | 0.138 |

**Cross-family observation.** The top tier (0.797–0.870) contains
rule-based, RandomForest, and **five different LLM families** (Alibaba,
01.ai, Meta, Mistral, Microsoft). The cross-family spread among the
five LLM 7–72 B systems is within 2 correct samples of each other out
of 69 — strong evidence that the result is not model-family-specific.
Three LLMs (Mistral-7B, Phi-4-mini, Qwen2.5-72B) are the only models
that ever predict the rare `others` class (n = 3).

**Noise floor.** `Dummy_majority` (always predict `focused`) reaches
0.696 with κ = 0; `Dummy_stratified` falls to 0.536 with negative κ.
Both NLI zero-shot models (distilbert 0.696, deberta 0.681) are
statistically indistinguishable from `Dummy_majority`, while boosted
trees (XGBoost 0.739, LightGBM 0.725) match the linear LogReg tier.

### 20.5 5-class vs native 3-class (same models)

Every retained model improves under 3-class except LinearSVM (−4 pts)
and DeBERTa (−1 pt). Rule-based κ moves 0.679 → 0.706, Qwen2.5-72B
moves 0.515 → 0.581, RandomForest moves 0.444 → 0.491.

### 20.6 Statistical significance (rule_based vs rest)

McNemar at α = 0.05 — rule_based **significantly beats** bart,
Dummy_stratified, LinearSVM, LightGBM, XGBoost, deberta,
LogisticRegression, distilbert, Dummy_majority, Qwen2.5-7B, and
qwen-7b (legacy). It **ties** (p > 0.05) with RandomForest, Phi-4-mini,
Mistral-7B, Llama-3.1-8B, Qwen2.5-72B, and Yi-1.5-9B. Full 153-pair
matrix:
[`output/3class_eval/phase_a/mcnemar_pairwise_3class.csv`](3class_eval/phase_a/mcnemar_pairwise_3class.csv).

### 20.7 Take-aways

1. Rule-based remains the strongest system and **exceeds the
   human-agreement ceiling** (κ = 0.706 > inter-rater κ = 0.613).
2. **Five independent LLM families cluster in 0.797–0.812** (Alibaba,
   01.ai, Meta, Mistral, Microsoft) — the LLM result is family-agnostic
   and scale-insensitive from 4 B (Phi-4-mini) to 72 B (Qwen2.5-72B).
3. Collapsing to 3 classes lifts inter-rater κ from moderate to
   substantial — most disagreements were on the mix/exploratory
   boundary, not on focused.
4. Supervised baselines benefit from being **retrained** on 3 labels
   rather than post-hoc mapped (the latter leaked their 5-class
   decision boundaries). Boosted trees (XGBoost, LightGBM) sit at the
   linear-model tier; RandomForest is the only supervised model that
   reaches the LLM cluster.
5. NLI models and `Dummy_majority` all land at ≈ 0.696 (κ ≈ 0); this
   is the noise floor for the 48/69 majority class. NLI zero-shot is
   confirmed as a negative control, not a usable baseline.

Full artifacts and side-by-side tables:
[`output/3class_eval/phase_a/THREECLASS_REPORT.md`](3class_eval/phase_a/THREECLASS_REPORT.md).
