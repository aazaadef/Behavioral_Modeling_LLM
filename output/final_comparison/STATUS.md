# STATUS — Deprecated snapshot

The files in this directory are a **frozen snapshot** from the
single-annotator / pre-LLM evaluation and are no longer authoritative.

## Why stale

- Ground truth has been replaced by the 69-sample **two-rater consensus**
  labels in `output/paper_eval/manual_eval_annotations.csv`. 8 labels
  changed after consensus review; inter-rater κ = 0.531.
- The supervised numbers here
  (`supervised_RandomForest: acc=0.9583, κ=0.9167`,
  `supervised_LinearSVM: acc=0.8333, κ=0.6667`,
  `supervised_LogisticRegression: acc=0.7917, κ=0.5833`) were computed
  from `supervised_all69_predictions.csv`, which trains on all 69 samples
  and predicts on the same 69 (train/test leakage). The current
  evaluation uses the out-of-fold CV predictions in
  `supervised_baselines/supervised_predictions.csv` instead.
- Several models evaluated more recently (Qwen2.5-7B, Qwen2.5-72B,
  qwen-7b legacy) are missing from this snapshot.

## Canonical replacements

| Old file (here)                | Replaced by                                                                         |
|--------------------------------|-------------------------------------------------------------------------------------|
| `model_ranking.csv`            | [`../phase_a_analysis/bootstrap_ci.csv`](../phase_a_analysis/bootstrap_ci.csv)      |
| `model_evaluation.json`        | [`../phase_a_analysis/all_results.json`](../phase_a_analysis/all_results.json)      |
| `comparison_summary.json`      | [`../phase_a_analysis/PHASE_A_REPORT.md`](../phase_a_analysis/PHASE_A_REPORT.md)    |
| `pairwise_agreement.{csv,json}`| [`../phase_a_analysis/mcnemar_pairwise.csv`](../phase_a_analysis/mcnemar_pairwise.csv) |

## Correct current numbers (top 3)

| Rank | Model                          | Accuracy | Cohen κ |
|-----:|--------------------------------|---------:|--------:|
| 1    | rule_based                     | 0.854    | 0.680   |
| 2    | llm_Qwen2.5-72B-Instruct       | 0.783    | 0.515   |
| 3    | supervised_RandomForest (OOF)  | 0.768    | 0.444   |

See [`../final_model_comparison/report.md`](../final_model_comparison/report.md)
for the full 10-model ranking and significance testing.

Do not cite the files in this folder.
