# ChildPlay Model Comparison

Evaluated split: test (69 child sequences, two-rater consensus labels).

> **Note — superseded by 3-class analysis.** The numbers below are the
> 5-class ranking over 10 models. The paper-facing analysis has since
> been extended to the **canonical 3-class schema over 18 models**
> (seven supervised baselines including XGBoost/LightGBM/Dummy, seven
> LLMs across five families — Alibaba, Mistral, Microsoft, 01.ai,
> Meta — plus rule-based and three NLI backends). See
> [output/3class_eval/phase_a/THREECLASS_REPORT.md](../3class_eval/phase_a/THREECLASS_REPORT.md)
> and the narrative write-up in
> [output/final_project_report.md §11](../final_project_report.md).
>
> The 5-class statistical report is still
> [output/phase_a_analysis/PHASE_A_REPORT.md](../phase_a_analysis/PHASE_A_REPORT.md).

## Ground truth

- 69 samples labeled independently by two human annotators.
- Inter-rater Cohen's κ = 0.531 (raw agreement 78.3%, "moderate").
- Post-consensus label distribution:
  `focused_attention=48, mixed_attention=17, occluded_attention=3, exploratory_attention=1, reduced_visual_availability=0`.

## Ranking by accuracy (all 10 evaluated models)

| Rank | Model                               | Accuracy | Cohen κ | Macro-F1 | 95% CI (acc)  |
|-----:|-------------------------------------|---------:|--------:|---------:|:--------------|
| 1    | rule_based                          | **0.854**| **0.680**| **0.521**| [0.768, 0.928]|
| 2    | llm_Qwen2.5-72B-Instruct            | 0.783    | 0.515   | 0.330    | [0.681, 0.870]|
| 3    | supervised_RandomForest (OOF)       | 0.768    | 0.444   | 0.414    | [0.667, 0.870]|
| 4    | llm_qwen-7b (legacy, fixed prompt)  | 0.725    | 0.441   | 0.381    | [0.623, 0.826]|
| 5    | supervised_LinearSVM (OOF)          | 0.710    | 0.314   | 0.446    | [0.609, 0.812]|
| 6    | llm_Qwen2.5-7B-Instruct             | 0.709    | 0.151   | 0.311    | [0.608, 0.812]|
| 7    | supervised_LogisticRegression (OOF) | 0.696    | 0.268   | 0.353    | [0.594, 0.812]|
| 8    | zero-shot:DeBERTa                   | 0.696    | 0.000   | 0.237    | [0.594, 0.797]|
| 9    | zero-shot:BART                      | 0.144    | −0.021  | 0.112    | [0.073, 0.232]|
| 10   | zero-shot:DistilBERT                | 0.014    | 0.000   | 0.007    | [0.000, 0.043]|

`Dummy_majority` accuracy is 0.696 on this split — every model at or below
that floor (DeBERTa, BART, DistilBERT, and LogReg at the boundary) should
be treated as a negative control, not a system.

## Statistical significance (McNemar, α = 0.05)

Rule-based is **significantly better** than:
- supervised_LinearSVM (p = 0.021)
- supervised_LogisticRegression (p = 0.007)
- llm_Qwen2.5-7B (p = 0.013)
- llm_qwen-7b (p = 0.022)
- all three NLI models (p < 0.001 each)

Rule-based vs supervised_RandomForest (p = 0.110) and
rule-based vs Qwen2.5-72B (p = 0.180) are **not** significant — the honest
claim is that rule-based ties with the best learned systems on n = 69.

Full 45-pair matrix: [phase_a_analysis/mcnemar_pairwise.csv](../phase_a_analysis/mcnemar_pairwise.csv).

## Pairwise agreement between models

Highlights (full tables in [phase_a_analysis/](../phase_a_analysis/)):

- rule_based vs DeBERTa ≈ 0.68 (both lean on focused_attention)
- rule_based vs BART ≈ 0.12 (BART label space incompatible)
- rule_based vs DistilBERT ≈ 0.06 (DistilBERT collapsed to exploratory)
- DistilBERT vs BART / DeBERTa ≈ 0.00 (disjoint label collapses)

## Recommendation

- **Best system:** `rule_based` — highest accuracy (0.854) and κ (0.680),
  statistically significant wins over every NLI and linear-supervised model.
- **Best learned system:** `supervised_RandomForest` when linear models
  fail to fit rare labels; `Qwen2.5-72B-Instruct` when prompt-only
  inference is required.
- **Do not use:** DeBERTa, BART, DistilBERT zero-shot — all below the
  majority-class floor.

## Supersedes

This report supersedes:
- The earlier 4-model confidence ranking (rule-based, DistilBERT, BART,
  DeBERTa) computed before LLM and supervised baselines were added.
- The 24-sample, 2-class preliminary comparison in
  `paper_results/` and `final_comparison/` (see each folder's
  `STATUS.md`).
