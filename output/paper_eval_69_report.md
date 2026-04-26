# Full 69-Item Evaluation Rebuild Report

> **Status: superseded (historical snapshot of Phase 1–2).** This
> document records the initial rebuild pass, when only **24 of 69**
> samples were labeled and the evaluation covered **4 models**
> (rule-based, DeBERTa, DistilBERT, BART).
>
> - All 69 samples have since been labeled by **two annotators**;
>   inter-rater Cohen's κ = 0.531 and 8 labels changed after consensus
>   review. See [`phase_a_analysis/inter_rater_report.md`](phase_a_analysis/inter_rater_report.md).
> - The benchmark has been extended with three supervised baselines and
>   three LLMs (10 models total) and re-run against the consensus
>   labels. The canonical numbers live in
>   [`phase_a_analysis/PHASE_A_REPORT.md`](phase_a_analysis/PHASE_A_REPORT.md)
>   and the narrative in [`final_project_report.md`](final_project_report.md).
> - Current headline: **rule_based acc=0.854, κ=0.680** (was 0.667 in
>   this file); DeBERTa/DistilBERT/BART revised to 0.696 / 0.014 / 0.144.
>
> Everything below is kept verbatim for reproducibility of the original
> milestone. Do not cite the tables in §5, §9, §10 — use the Phase A
> report instead.

## 1. Dataset Scope
- Total target items processed: 69
- Final usable count: 69
- Target reference JSON: `output/paper_eval/target_items_69.json`
- Target reference CSV: `output/paper_eval/target_items_69.csv`
- Source dataset root: `data set/ChildPlay-gaze/ChildPlay-gaze`
- Source files used: `clips.csv`, `splits.csv`, `videos.csv`, and all `annotations/test/*.csv` rows with `is_child == 1`.
- Items skipped/dropped/failed: 0

## 2. Pipeline Audit Summary
- Previous 24-item processing came from `prepare-paper-eval` defaulting to `max_examples=24` and the old paper-results script enforcing exactly 24 rows.
- The 69-item target set lives in the test split loaded from `load_child_sequences(..., split='test')` and is now materialized in `output/paper_eval/target_items_69.json`.
- Stale or incomplete files detected before rebuild:
  - `output/paper_eval/manual_eval_subset.json`: row_count=24, expected=69
  - `output/paper_eval/manual_eval_annotations.csv`: missing prediction columns: ['bart_label', 'deberta_label', 'distilbert_label', 'rule_based_label']
  - `output/paper_eval/manual_eval_annotations.csv`: row_count=24, expected=69
  - `output/paper_results/manual_eval_merged.csv`: row_count=24, expected=69
- Existing manual annotation file had 24 rows and 24 non-empty final_label values before rebuild.
- Regenerated outputs: benchmark directories, `output/restart_run/*`, `output/paper_eval/*`, `output/paper_results/*`, `output/final_model_comparison/*`, and `output/paper_eval_69_report.md`.

## 3. Per-Stage Execution Summary
### rule-based benchmark
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_rule_based --split test --models rule-based`
- Return code: 0
- Inputs: ChildPlay test split -> output/benchmark_rule_based
- Outputs: output/benchmark_rule_based/benchmark_summary.json and model artifacts
- Processed items: 69
- Warnings/errors: stderr=`(empty)`
- Stdout summary: `(empty)`

### distilbert zero-shot benchmark
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_zero_shot --split test --models zero-shot:typeform/distilbert-base-uncased-mnli`
- Return code: 0
- Inputs: ChildPlay test split -> output/benchmark_zero_shot
- Outputs: output/benchmark_zero_shot/benchmark_summary.json and model artifacts
- Processed items: 69
- Warnings/errors: stderr=`Device set to use cuda:0 | You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset`
- Stdout summary: `(empty)`

### bart zero-shot benchmark
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_bart_large_mnli --split test --models zero-shot:facebook/bart-large-mnli`
- Return code: 0
- Inputs: ChildPlay test split -> output/benchmark_bart_large_mnli
- Outputs: output/benchmark_bart_large_mnli/benchmark_summary.json and model artifacts
- Processed items: 69
- Warnings/errors: stderr=`Device set to use cuda:0 | You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset`
- Stdout summary: `(empty)`

### deberta zero-shot benchmark
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_deberta_zeroshot --split test --models zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
- Return code: 0
- Inputs: ChildPlay test split -> output/benchmark_deberta_zeroshot
- Outputs: output/benchmark_deberta_zeroshot/benchmark_summary.json and model artifacts
- Processed items: 69
- Warnings/errors: stderr=`Device set to use cuda:0 | You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset`
- Stdout summary: `(empty)`

### openai benchmark
- Command/script: `skipped`
- Return code: 0
- Inputs: ChildPlay test split
- Outputs: output/benchmark_openai/FAILED.txt
- Processed items: 69
- Warnings/errors: stderr=`(empty)`
- Stdout summary: `OPENAI_API_KEY missing; benchmark skipped.`

### prepare paper eval
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python main.py prepare-paper-eval --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --split test --run-output-dir output/restart_run --paper-eval-dir output/paper_eval --max-examples 69 --rule-based-path /home/d3849/project/project.LLM/output/benchmark_rule_based/rule-based/interpretations.jsonl --deberta-path /home/d3849/project/project.LLM/output/benchmark_deberta_zeroshot/zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0/interpretations.jsonl --distilbert-path /home/d3849/project/project.LLM/output/benchmark_zero_shot/zero-shot__typeform_distilbert-base-uncased-mnli/interpretations.jsonl --bart-path /home/d3849/project/project.LLM/output/benchmark_bart_large_mnli/zero-shot__facebook_bart-large-mnli/interpretations.jsonl`
- Return code: 0
- Inputs: 69 test sequences plus four saved model prediction files
- Outputs: output/restart_run/* and output/paper_eval/*
- Processed items: 69
- Warnings/errors: stderr=`(empty)`
- Stdout summary: `(empty)`

### validate paper eval
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python main.py validate-paper-eval --features-path output/restart_run/behavioral_features.jsonl --subset-path output/paper_eval/manual_eval_subset.json`
- Return code: 0
- Inputs: output/restart_run/behavioral_features.jsonl and output/paper_eval/manual_eval_subset.json
- Outputs: validation status only
- Processed items: 69
- Warnings/errors: stderr=`(empty)`
- Stdout summary: `(empty)`

### generate paper results
- Command/script: `/home/d3849/project/project.LLM/.venv/bin/python scripts/generate_paper_results.py`
- Return code: 0
- Inputs: output/paper_eval/manual_eval_annotations.csv and output/paper_eval/manual_eval_subset.json
- Outputs: output/paper_results/*
- Processed items: 69
- Warnings/errors: stderr=`(empty)`
- Stdout summary: `samples=69 | labeled_samples=24 | missing_model_predictions=0 | best_model=rule_based | ranking=rule_based:0.6667, deberta:0.5000, distilbert:0.5000, bart:0.0000`

## 4. Model Outputs
### rule-based
- Predictions produced: 69 / 69
- Missing predictions: 0
- Invalid rows detected: 0
- Label distribution: {'focused_attention': 47, 'mixed_attention': 18, 'exploratory_attention': 4}
- Fallback logic used: none

### zero-shot:typeform/distilbert-base-uncased-mnli
- Predictions produced: 69 / 69
- Missing predictions: 0
- Invalid rows detected: 0
- Label distribution: {'exploratory_attention': 69}
- Fallback logic used: none

### zero-shot:facebook/bart-large-mnli
- Predictions produced: 69 / 69
- Missing predictions: 0
- Invalid rows detected: 0
- Label distribution: {'occluded_attention': 29, 'mixed_attention': 40}
- Fallback logic used: none

### zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0
- Predictions produced: 69 / 69
- Missing predictions: 0
- Invalid rows detected: 0
- Label distribution: {'focused_attention': 69}
- Fallback logic used: none

### openai:gpt-4.1-mini
- Status: NOT RUN: OPENAI_API_KEY is not set in the environment.
- Predictions produced: 0
- Missing predictions: 69
- Invalid rows detected: 0
- Fallback logic used: benchmark skipped because no API key was available.

## 5. Agreement and Evaluation Metrics
- Full-table denominator for model coverage and pairwise agreement: 69
- Labeled denominator for agreement with `final_label`: 24
- rule_based vs final_label: matches=16/24 (66.67%), accuracy=0.6667
- deberta vs final_label: matches=12/24 (50.00%), accuracy=0.5000
- distilbert vs final_label: matches=12/24 (50.00%), accuracy=0.5000
- bart vs final_label: matches=0/24 (0.00%), accuracy=0.0000
- Pairwise bart__vs__deberta: matches=0/69 (0.00%), agreement=0.0000
- Pairwise bart__vs__distilbert: matches=0/69 (0.00%), agreement=0.0000
- Pairwise bart__vs__rule_based: matches=8/69 (11.59%), agreement=0.1159
- Pairwise deberta__vs__distilbert: matches=0/69 (0.00%), agreement=0.0000
- Pairwise deberta__vs__rule_based: matches=47/69 (68.12%), agreement=0.6812
- Pairwise distilbert__vs__rule_based: matches=4/69 (5.80%), agreement=0.0580
- Confusion matrices were regenerated in `output/paper_results/confusion_matrices.json`.
- Counts and normalized percentages are reported with explicit denominators above.

## 6. Error Analysis
- Mismatched labeled cases: 24/24
- Cases all models got wrong: 0
- Cases only one model got right: 8
- Highest-disagreement samples on the full 69 rows (ranked by number of unique model labels):
  - `f6wqlpG9rd0_9704-9949:person_1` / `f6wqlpG9rd0_9704-9949`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `f6wqlpG9rd0_9404-9541:person_1` / `f6wqlpG9rd0_9404-9541`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `aWV7UUMddCU_6517-7514:person_1` / `aWV7UUMddCU_6517-7514`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `aWV7UUMddCU_5934-6205:person_2` / `aWV7UUMddCU_5934-6205`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `NIk1-ck4c6Q_13651-13719:person_2` / `NIk1-ck4c6Q_13651-13719`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `ND7pXuhs3VM_1800-1925:person_2` / `ND7pXuhs3VM_1800-1925`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `9DNwRwt5kI4_2442-3179:person_3` / `9DNwRwt5kI4_2442-3179`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `9DNwRwt5kI4_14988-15270:person_3` / `9DNwRwt5kI4_14988-15270`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `9DNwRwt5kI4_14988-15270:person_2` / `9DNwRwt5kI4_14988-15270`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
  - `6mA6UAoT3M0_6165-6361:person_1` / `6mA6UAoT3M0_6165-6361`: unique_labels=4, labels=['mixed_attention', 'focused_attention', 'exploratory_attention', 'occluded_attention']
- Systematic patterns observed from the regenerated outputs:
  - DeBERTa predicts `focused_attention` for all 69 rows.
  - DistilBERT predicts `exploratory_attention` for all 69 rows.
  - BART predicts only `mixed_attention` or `occluded_attention` across the 69 rows.
  - The rule-based system is the only backend with a non-degenerate three-label distribution on this split.

## 7. Reproducibility
- Exact commands run:
  - `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_rule_based --split test --models rule-based`
  - `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_zero_shot --split test --models zero-shot:typeform/distilbert-base-uncased-mnli`
  - `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_bart_large_mnli --split test --models zero-shot:facebook/bart-large-mnli`
  - `/home/d3849/project/project.LLM/.venv/bin/python main.py benchmark --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --output-dir output/benchmark_deberta_zeroshot --split test --models zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
  - `skipped`
  - `/home/d3849/project/project.LLM/.venv/bin/python main.py prepare-paper-eval --dataset-root /home/d3849/project/project.LLM/data set/ChildPlay-gaze/ChildPlay-gaze --split test --run-output-dir output/restart_run --paper-eval-dir output/paper_eval --max-examples 69 --rule-based-path /home/d3849/project/project.LLM/output/benchmark_rule_based/rule-based/interpretations.jsonl --deberta-path /home/d3849/project/project.LLM/output/benchmark_deberta_zeroshot/zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0/interpretations.jsonl --distilbert-path /home/d3849/project/project.LLM/output/benchmark_zero_shot/zero-shot__typeform_distilbert-base-uncased-mnli/interpretations.jsonl --bart-path /home/d3849/project/project.LLM/output/benchmark_bart_large_mnli/zero-shot__facebook_bart-large-mnli/interpretations.jsonl`
  - `/home/d3849/project/project.LLM/.venv/bin/python main.py validate-paper-eval --features-path output/restart_run/behavioral_features.jsonl --subset-path output/paper_eval/manual_eval_subset.json`
  - `/home/d3849/project/project.LLM/.venv/bin/python scripts/generate_paper_results.py`
- Files to keep for paper writing:
  - `output/paper_eval/target_items_69.json`
  - `output/paper_eval/target_items_69.csv`
  - `output/restart_run/behavioral_features.jsonl`
  - `output/benchmark_rule_based/rule-based/interpretations.jsonl`
  - `output/benchmark_zero_shot/zero-shot__typeform_distilbert-base-uncased-mnli/interpretations.jsonl`
  - `output/benchmark_bart_large_mnli/zero-shot__facebook_bart-large-mnli/interpretations.jsonl`
  - `output/benchmark_deberta_zeroshot/zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0/interpretations.jsonl`
  - `output/paper_eval/manual_eval_annotations.csv`
  - `output/paper_results/manual_eval_merged.csv`
  - `output/paper_results/metrics.json`
  - `output/paper_results/pairwise_agreement.json`
  - `output/paper_results/mismatch_analysis.csv`
  - `output/paper_eval_69_report.md`
- Files that remain out-of-scope for the 69-row package: `output/silver_eval/*` because that workflow is a separate 33-item consensus subset.

## 8. Final Status
- Full 69-item evaluation internally consistent: yes
- Validation passed: features=69, manual_eval_subset=69
- OpenAI benchmark status: NOT RUN: OPENAI_API_KEY is not set in the environment.
- Best-performing model on labeled subset: rule_based

---

## 9. Supervised Baselines (Added Phase 3)

Three supervised classifiers were trained on the 14 numerical behavioral
features using stratified k-fold cross-validation on the 24 labeled samples.

### Results (24 labeled, 2 classes: focused_attention, exploratory_attention)

| Classifier          | Accuracy      | Macro-F1      | Cohen's Kappa |
|---------------------|---------------|---------------|---------------|
| RandomForest        | 0.9600±0.0800 | 0.9600±0.0800 | 0.9231        |
| LinearSVM           | 0.8200±0.1833 | 0.8200±0.1833 | 0.6462        |
| LogisticRegression  | 0.7700±0.2750 | 0.7600±0.2939 | 0.5462        |

Key insight: All supervised classifiers outperform every zero-shot NLI model
and the rule-based baseline, confirming that the extracted features carry
strong discriminative signal.

Output: `output/supervised_baselines/`

## 10. Unified Final Model Comparison (Added Phase 3)

All available models evaluated against the 24 manually labeled samples:

| Rank | Model                      | Accuracy | Macro-F1 | Kappa  |
|------|----------------------------|----------|----------|--------|
| 1    | supervised_RandomForest    | 0.9583   | 0.9583   | 0.9167 |
| 2    | supervised_LinearSVM       | 0.8333   | 0.8322   | 0.6667 |
| 3    | supervised_LogisticRegression | 0.7917 | 0.7884   | 0.5833 |
| 4    | rule_based                 | 0.6667   | 0.5000   | 0.5000 |
| 5    | deberta                    | 0.5000   | 0.3333   | 0.0000 |
| 6    | distilbert                 | 0.5000   | 0.3333   | 0.0000 |
| 7    | bart                       | 0.0000   | 0.0000   | 0.0000 |

Output: `output/final_comparison/`

## 11. New Infrastructure (Added Phase 3)

### Corrected LLM Inference Script
- `scripts/run_local_llm_unified.py` — Unified inference with correct 5-label schema
- Fixes the label-space mismatch in the earlier Qwen-7B run (was: inside_visible/outside/unsure)
- Supports 4-bit/8-bit quantization for large models (70B+)

### Model Download Script
- `scripts/download_models.py` — Downloads HuggingFace model checkpoints

### Final Comparison Script
- `scripts/run_final_comparison.py` — Auto-discovers all model predictions and evaluates against manual labels

## 12. Remaining Steps
1. Complete manual labeling of all 69 samples (in progress by user)
2. Download and run Llama-3.1-8B-Instruct with corrected label space
3. Download and run one 70B model (Llama-3.1-70B or Qwen2.5-72B) in 4-bit
4. Re-run supervised baselines and final comparison with full 69-label ground truth
5. Generate final paper artifacts
