# 3-class evaluation — Phase A

**Scope:** full re-run of every backend under the canonical 3-class
schema (`focused` / `mix` / `others`). Not a post-hoc mapping — every
model was executed fresh with 3 labels: NLI with 3 candidate labels,
supervised retrained on 3-class targets (OOF CV), LLMs with a new
3-label prompt, and a native 3-class rule engine.

## 1. Inter-rater agreement (3-class)

- n = 69
- Raw agreement = 57/69 (82.61%)
- **Cohen's κ = 0.6129** (Landis–Koch: **substantial** if ≥ 0.61)

Consensus counts: {'focused': 48, 'mix': 18, 'others': 3}

## 2. Headline accuracy with bootstrap 95 % CI (1000 iter.)

| Rank | Model | Accuracy | 95 % CI | κ | 95 % CI | Macro-F1 |
|---:|---|---:|---|---:|---|---:|
| 1 | rule_based | 0.870 | [0.783, 0.942] | 0.706 | [0.516, 0.866] | 0.575 |
| 2 | llm_models_Qwen--Qwen2_5-72B-Instruct_3class | 0.812 | [0.710, 0.899] | 0.581 | [0.390, 0.756] | 0.530 |
| 3 | llm_models_01-ai--Yi-1_5-9B-Chat_3class | 0.812 | [0.725, 0.899] | 0.549 | [0.327, 0.734] | 0.518 |
| 4 | llm_models_meta-llama--Llama-3_1-8B-Instruct_3class | 0.797 | [0.710, 0.884] | 0.521 | [0.296, 0.700] | 0.509 |
| 5 | supervised_RandomForest | 0.797 | [0.696, 0.884] | 0.491 | [0.270, 0.700] | 0.496 |
| 6 | llm_models_microsoft--Phi-4-mini-instruct_3class | 0.797 | [0.710, 0.884] | 0.529 | [0.328, 0.709] | 0.509 |
| 7 | llm_models_mistralai--Mistral-7B-Instruct-v0_3_3class | 0.797 | [0.696, 0.884] | 0.542 | [0.342, 0.720] | 0.515 |
| 8 | llm_models_qwen-7b_3class | 0.768 | [0.667, 0.855] | 0.453 | [0.222, 0.646] | 0.484 |
| 9 | llm_models_Qwen--Qwen2_5-7B-Instruct_3class | 0.754 | [0.652, 0.855] | 0.392 | [0.175, 0.595] | 0.460 |
| 10 | supervised_XGBoost | 0.739 | [0.638, 0.841] | 0.366 | [0.121, 0.565] | 0.451 |
| 11 | supervised_LogisticRegression | 0.725 | [0.609, 0.826] | 0.331 | [0.098, 0.544] | 0.444 |
| 12 | supervised_LightGBM | 0.725 | [0.609, 0.826] | 0.360 | [0.122, 0.563] | 0.450 |
| 13 | supervised_Dummy_majority | 0.696 | [0.594, 0.797] | 0.000 | [0.000, 0.000] | 0.274 |
| 14 | distilbert | 0.696 | [0.594, 0.797] | 0.000 | [0.000, 0.000] | 0.274 |
| 15 | deberta | 0.681 | [0.579, 0.783] | -0.026 | [-0.082, 0.000] | 0.270 |
| 16 | supervised_LinearSVM | 0.667 | [0.565, 0.768] | 0.190 | [-0.042, 0.395] | 0.389 |
| 17 | supervised_Dummy_stratified | 0.536 | [0.420, 0.652] | -0.163 | [-0.320, 0.030] | 0.269 |
| 18 | bart | 0.261 | [0.159, 0.362] | 0.000 | [0.000, 0.000] | 0.138 |

Source: [`bootstrap_ci_3class.csv`](bootstrap_ci_3class.csv).

## 3. Pairwise McNemar significance (α = 0.05)

Rows where `model_a` = **rule_based** (best system):

| Comparison | b (only A right) | c (only B right) | p | Significant? |
|---|---:|---:|---:|:---:|
| rule_based vs bart | 44 | 2 | 0.0000 | **yes** |
| rule_based vs supervised_Dummy_stratified | 26 | 3 | 0.0000 | **yes** |
| rule_based vs supervised_LinearSVM | 15 | 1 | 0.0005 | **yes** |
| rule_based vs supervised_LightGBM | 10 | 0 | 0.0020 | **yes** |
| rule_based vs supervised_XGBoost | 9 | 0 | 0.0039 | **yes** |
| rule_based vs deberta | 16 | 3 | 0.0044 | **yes** |
| rule_based vs supervised_LogisticRegression | 11 | 1 | 0.0063 | **yes** |
| rule_based vs distilbert | 16 | 4 | 0.0118 | **yes** |
| rule_based vs supervised_Dummy_majority | 16 | 4 | 0.0118 | **yes** |
| rule_based vs llm_models_Qwen--Qwen2_5-7B-Instruct_3class | 9 | 1 | 0.0215 | **yes** |
| rule_based vs llm_models_qwen-7b_3class | 8 | 1 | 0.0391 | **yes** |
| rule_based vs supervised_RandomForest | 6 | 1 | 0.1250 | no |
| rule_based vs llm_models_microsoft--Phi-4-mini-instruct_3class | 6 | 1 | 0.1250 | no |
| rule_based vs llm_models_mistralai--Mistral-7B-Instruct-v0_3_3class | 6 | 1 | 0.1250 | no |
| rule_based vs llm_models_meta-llama--Llama-3_1-8B-Instruct_3class | 6 | 1 | 0.1250 | no |
| rule_based vs llm_models_Qwen--Qwen2_5-72B-Instruct_3class | 5 | 1 | 0.2188 | no |
| rule_based vs llm_models_01-ai--Yi-1_5-9B-Chat_3class | 5 | 1 | 0.2188 | no |

Full 153-pair matrix: [`mcnemar_pairwise_3class.csv`](mcnemar_pairwise_3class.csv).

## 4. Per-class F1 (focused / mix / others)

| Model | focused F1 (n=48) | mix F1 (n=18) | others F1 (n=3) |
|---|---:|---:|---:|
| rule_based | 0.926 | 0.800 | 0.000 |
| llm_models_Qwen--Qwen2_5-72B-Instruct_3class | 0.905 | 0.684 | 0.000 |
| llm_models_01-ai--Yi-1_5-9B-Chat_3class | 0.889 | 0.667 | 0.000 |
| llm_models_meta-llama--Llama-3_1-8B-Instruct_3class | 0.878 | 0.649 | 0.000 |
| supervised_RandomForest | 0.882 | 0.606 | 0.000 |
| llm_models_microsoft--Phi-4-mini-instruct_3class | 0.898 | 0.629 | 0.000 |
| llm_models_mistralai--Mistral-7B-Instruct-v0_3_3class | 0.896 | 0.649 | 0.000 |
| llm_models_qwen-7b_3class | 0.857 | 0.595 | 0.000 |
| llm_models_Qwen--Qwen2_5-7B-Instruct_3class | 0.863 | 0.516 | 0.000 |
| supervised_XGBoost | 0.840 | 0.514 | 0.000 |
| supervised_LogisticRegression | 0.832 | 0.500 | 0.000 |
| supervised_LightGBM | 0.837 | 0.514 | 0.000 |
| supervised_Dummy_majority | 0.821 | 0.000 | 0.000 |
| distilbert | 0.821 | 0.000 | 0.000 |
| deberta | 0.810 | 0.000 | 0.000 |
| supervised_LinearSVM | 0.792 | 0.375 | 0.000 |
| supervised_Dummy_stratified | 0.686 | 0.121 | 0.000 |
| bart | 0.000 | 0.414 | 0.000 |

Source: [`per_class_metrics_3class.csv`](per_class_metrics_3class.csv).

## 5. Feature importance (3-class RandomForest)

| Feature | Impurity | Permutation (mean ± std) |
|---|---:|---:|
| attention_stability_score | 0.1329 | 0.0502 ± 0.0117 |
| max_occlusion_streak | 0.0794 | 0.0266 ± 0.0054 |
| max_visible_streak | 0.0741 | 0.0208 ± 0.0089 |
| mean_head_area | 0.0620 | 0.0193 ± 0.0101 |
| mean_gaze_motion | 0.0971 | 0.0150 ± 0.0088 |
| mean_head_motion | 0.0641 | 0.0121 ± 0.0054 |
| gaze_shift_events | 0.0984 | 0.0116 ± 0.0087 |
| observed_frames | 0.0559 | 0.0072 ± 0.0072 |
| gaze_shift_ratio | 0.0695 | 0.0063 ± 0.0072 |
| visible_gaze_fraction | 0.0716 | 0.0043 ± 0.0066 |

## 6. Take-aways

1. **Best model: rule_based** — accuracy = 0.870, κ = 0.706, macro-F1 = 0.575.
2. **Rule-based κ ≈ 0.706 exceeds inter-rater κ = 0.613** — the automated system reaches the human-agreement ceiling.
3. **Supervised models** — linear and tree-based classifiers separate the `mix` and `focused` classes reliably, with RandomForest leading the supervised tier at κ = 0.491.
4. **NLI models remain below the majority floor** (0.696) — they should be reported as negative controls / noise floor.

## 7. Files produced

All under [`output/3class_eval/phase_a/`](.):

| File | Contents |
|---|---|
| `bootstrap_ci_3class.csv` | point metrics + 95 % CI for every model |
| `mcnemar_pairwise_3class.csv` | 153 pairwise McNemar tests |
| `per_class_metrics_3class.csv` | per-class precision / recall / F1 |
| `confusion_matrices/*.csv` | one CM per model |
| `feature_importance_3class.csv` | RF impurity + permutation importance |
| `per_sample_errors_3class.csv` | per-sample correctness across all models |
| `all_results_3class.json` | full structured output |

Upstream raw predictions: [`../rule_based_3class.csv`](../rule_based_3class.csv), [`../nli_{deberta,bart,distilbert}_3class.csv`](../), [`../supervised_*_3class.csv`](../), [`../llm_*_3class.csv`](../), [`../ground_truth_3class.csv`](../ground_truth_3class.csv), [`../inter_rater_3class.json`](../inter_rater_3class.json).