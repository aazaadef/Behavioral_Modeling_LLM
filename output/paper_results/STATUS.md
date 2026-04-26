# STATUS — Deprecated snapshot

The files in this directory are a **frozen snapshot** of an earlier
evaluation pass (Phase 1–2). They should not be used for any paper or
analysis work — the numbers are stale in two respects:

1. **Ground truth has changed.** All files in this folder were computed
   when only 24 of the 69 test samples were labeled, and by a single
   rater. The current ground truth is the 69-sample two-rater consensus
   in `output/paper_eval/manual_eval_annotations.csv`
   (inter-rater κ = 0.531; 8 labels changed after consensus review).
2. **Model coverage is incomplete.** Only four models are represented
   (rule-based, DeBERTa, DistilBERT, BART). The current evaluation
   covers 10 models (adding three supervised baselines and three LLMs).

## Canonical replacements

| Old file (here)                | Replaced by                                                                               |
|--------------------------------|-------------------------------------------------------------------------------------------|
| `metrics.json`, `summary.txt`  | [`../phase_a_analysis/bootstrap_ci.csv`](../phase_a_analysis/bootstrap_ci.csv)            |
| `accuracy_table.csv`, `paper_ready_table.txt` | [`../phase_a_analysis/PHASE_A_REPORT.md`](../phase_a_analysis/PHASE_A_REPORT.md) §1 |
| `confusion_matrices.json`      | [`../phase_a_analysis/confusion_matrices/`](../phase_a_analysis/confusion_matrices/)      |
| `mismatch_analysis.csv`        | [`../phase_a_analysis/per_sample_errors.csv`](../phase_a_analysis/per_sample_errors.csv)  |
| `pairwise_agreement.json`      | [`../phase_a_analysis/mcnemar_pairwise.csv`](../phase_a_analysis/mcnemar_pairwise.csv)    |
| `manual_eval_merged.csv`       | [`../paper_eval/manual_eval_annotations.csv`](../paper_eval/manual_eval_annotations.csv)  |
| `model_predictions.{csv,json}` | [`../all_predictions_with_two_annotators.csv`](../all_predictions_with_two_annotators.csv)|

## Key headline numbers (old vs current)

| Model       | Old (24 labeled, 1 rater) | Current (69 labeled, consensus) |
|-------------|--------------------------:|--------------------------------:|
| rule_based  | acc=0.667                 | **acc=0.854, κ=0.680**          |
| deberta     | acc=0.500                 | acc=0.696, κ=0.000              |
| distilbert  | acc=0.500                 | acc=0.014, κ=0.000              |
| bart        | acc=0.000                 | acc=0.144, κ=−0.021             |

The files here are kept only for reproducibility of the earlier
milestone. Do not cite them.
