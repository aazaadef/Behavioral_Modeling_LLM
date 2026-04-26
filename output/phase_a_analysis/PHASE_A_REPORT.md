# Phase A — Statistical Analysis Report

**Scope:** Statistical evaluation of the 10-model benchmark on the 69
manually-labeled ChildPlay-gaze samples, with the Phase-B inter-rater
agreement folded in.

## Ground truth provenance

69 samples labeled independently by two human annotators (rater 1 = user,
rater 2 = advisor). The two label sets were compared, all disagreements
were discussed, and a consensus `final_manual_label` was produced.

- Inter-rater raw agreement: **78.3 %** (54/69)
- Inter-rater Cohen's **κ = 0.531** — "moderate" agreement on the
  Landis–Koch scale (0.41–0.60). The task is not easy even for humans;
  model performance must be interpreted against this ceiling.
- Ground-truth labels used below were **updated** from the previous file
  — 8 samples changed after the full consensus review.

Full details: [inter_rater_report.md](inter_rater_report.md) and
[inter_rater_agreement.json](inter_rater_agreement.json).

---

## Label distribution after consensus (n=69)

| Label | Rater 1 | Rater 2 | Final consensus |
|---|---:|---:|---:|
| focused_attention | 47 | 48 | **48** |
| mixed_attention | 17 | 17 | **17** |
| occluded_attention | 2 | 3 | **3** |
| exploratory_attention | 3 | 1 | **1** |
| reduced_visual_availability | 0 | 0 | 0 |

Two of the five canonical labels have ≤3 samples and one has zero —
per-class metrics on those labels should be treated as anecdotal.

---

## Methodology

- Ground truth: `output/paper_eval/manual_eval_annotations.csv`
  (`final_label` column, post-consensus).
- Supervised predictions are **out-of-fold cross-validated** predictions
  from `output/supervised_baselines/supervised_predictions.csv`. The old
  `supervised_all69_predictions.csv` file trains on all 69 and predicts on
  the same 69 (train/test leakage, not used).
- Script: [scripts/phase_a_analysis.py](scripts/phase_a_analysis.py).
- Outputs: [output/phase_a_analysis/](output/phase_a_analysis/).
- Macro-F1 throughout is the **unweighted mean of per-class F1** (not
  support-weighted).

---

## 1. Headline accuracy with bootstrap 95% CI (1000 iterations)

| Model | Accuracy | 95% CI | Cohen κ | 95% CI | Macro-F1 |
|---|---:|---|---:|---|---:|
| **rule_based** | **0.854** | [0.768, 0.928] | **0.680** | [0.496, 0.844] | **0.521** |
| supervised_RandomForest (OOF) | 0.768 | [0.667, 0.870] | 0.444 | [0.231, 0.652] | 0.414 |
| llm_Qwen2.5-72B | 0.783 | [0.681, 0.870] | 0.515 | [0.338, 0.699] | 0.330 |
| llm_qwen-7b (legacy) | 0.725 | [0.623, 0.826] | 0.441 | [0.260, 0.620] | 0.381 |
| supervised_LinearSVM (OOF) | 0.710 | [0.609, 0.812] | 0.314 | [0.090, 0.538] | 0.446 |
| llm_Qwen2.5-7B | 0.709 | [0.608, 0.812] | 0.151 | [−0.034, 0.343] | 0.311 |
| supervised_LogisticRegression (OOF) | 0.696 | [0.594, 0.812] | 0.268 | [0.054, 0.487] | 0.353 |
| deberta (NLI) | 0.696 | [0.594, 0.797] | 0.000 | [0.000, 0.000] | 0.237 |
| bart (NLI) | 0.144 | [0.073, 0.232] | −0.021 | [−0.088, 0.051] | 0.112 |
| distilbert (NLI) | 0.014 | [0.000, 0.043] | 0.000 | [0.000, 0.000] | 0.007 |

Source: [bootstrap_ci.csv](bootstrap_ci.csv).

**Observations.** Rule-based now clearly leads on this label set. After the
consensus relabeling (which moved 4 samples into `occluded_attention`/
`focused_attention` categories that the supervised linear models cannot
learn from a 2-fold CV), the supervised linear baselines lose ~20 points of
accuracy and ~0.5 of κ. Rule-based, which has no training step, is
unaffected and now beats the supervised linear models by a margin that
McNemar finds statistically significant (§2). RandomForest handles the
richer label geometry better than the linear models.

DeBERTa sits exactly on the majority-class floor (`Dummy_majority` =
0.696, see §6). BART and DistilBERT are *below* the majority floor.

---

## 2. Pairwise McNemar significance (α = 0.05)

Full matrix: [mcnemar_pairwise.csv](mcnemar_pairwise.csv). Selected rows:

| Comparison | b (only A right) | c (only B right) | p | Significant? |
|---|---:|---:|---:|:---:|
| rule_based vs supervised_LinearSVM | 13 | 3 | **0.021** | **yes** |
| rule_based vs supervised_LogisticRegression | 13 | 2 | **0.007** | **yes** |
| rule_based vs supervised_RandomForest | 8 | 2 | 0.110 | no |
| rule_based vs llm_qwen-7b | 11 | 2 | **0.022** | **yes** |
| rule_based vs llm_Qwen2.5-7B | 12 | 2 | **0.013** | **yes** |
| rule_based vs llm_Qwen2.5-72B | 7 | 2 | 0.180 | no |
| supervised_LinearSVM vs supervised_LogisticRegression | 3 | 2 | 1.000 | no |
| supervised_LinearSVM vs supervised_RandomForest | 4 | 8 | 0.388 | no |
| llm_Qwen2.5-72B vs rule_based | 2 | 7 | 0.180 | no |
| deberta vs distilbert | 48 | 1 | <0.001 | **yes** |

**Take-away.** Rule-based is now **significantly better** than the two
linear supervised models (p=0.007 / 0.021) and the two smaller LLMs
(p=0.013 / 0.022). Its lead over RandomForest and Qwen2.5-72B is
numerically positive but **not** statistically significant at n=69. The
honest claim is: rule-based is the best system, and it is
*indistinguishable from* RandomForest and the 72B LLM on this sample,
while clearly beating the linear baselines and the smaller LLMs.

---

## 3. Per-class metrics — where the accuracy actually comes from

Full table: [per_class_metrics.csv](per_class_metrics.csv). Abbreviated:

| Model | focused F1 (n=48) | mixed F1 (n=17) | occluded F1 (n=3) | exploratory F1 (n=1) |
|---|---:|---:|---:|---:|
| rule_based | 0.926 | 0.800 | 0.000 | 0.400 |
| supervised_LinearSVM | 0.828 | 0.424 | **0.400** | 0.000 |
| supervised_LogisticRegression | 0.820 | 0.424 | 0.000 | 0.000 |
| supervised_RandomForest | 0.869 | 0.571 | 0.000 | 0.000 |
| llm_Qwen2.5-72B | 0.896 | 0.629 | 0.000 | 0.000 |
| llm_qwen-7b | 0.818 | 0.622 | 0.000 | 0.000 |
| llm_Qwen2.5-7B | 0.829 | 0.261 | 0.000 | 0.000 |
| deberta | 0.821 | 0.000 | 0.000 | 0.000 |
| bart | 0.000 | 0.281 | 0.125 | 0.000 |
| distilbert | 0.000 | 0.000 | 0.000 | 0.029 |

**Take-away.** Only two models achieve any F1 on `occluded_attention`:
`supervised_LinearSVM` (F1=0.400, the one class where it beats rule-based)
and `bart` (F1=0.125, a byproduct of its label collapse). Only
rule-based catches the single `exploratory_attention` example. No model
generalises on `reduced_visual_availability` (n=0 after consensus).

---

## 4. Error analysis

Source: [per_sample_errors.csv](per_sample_errors.csv).

- **Difficulty distribution (10-model correctness):** easy (≥8/10): 35 |
  medium (4–7): 22 | hard (≤3): **12**
- **All-models-wrong:** 1 sample
  (`dUYIh1U2z-8_3300-3427:person_1`, true label `occluded_attention`).

**The 12 hard samples** (fewer than 4 of 10 models right):

| child_id | true | # correct |
|---|---|---:|
| dUYIh1U2z-8_3300-3427:person_1 | occluded_attention | 0 |
| 9DNwRwt5kI4_14988-15270:person_2 | occluded_attention | 1 |
| LS9Hztyrmmw_1768-1839:person_3 | focused_attention | 1 |
| Lva4fn4_q88_1270-1473:person_3 | focused_attention | 1 |
| Lva4fn4_q88_4391-4523:person_3 | mixed_attention | 1 |
| odFKscFdEas_15822-16030:person_2 | mixed_attention | 1 |
| 6mA6UAoT3M0_6165-6361:person_1 | occluded_attention | 2 |
| NIk1-ck4c6Q_13651-13719:person_2 | mixed_attention | 2 |
| Pyb0z_YQjjI_7332-7429:person_2 | exploratory_attention | 2 |
| ivX3JXIV1W4_1-1920:person_1 | focused_attention | 2 |
| WpbXt04qWEk_1166-1611:person_1 | focused_attention | 3 |
| aWV7UUMddCU_6517-7514:person_1 | mixed_attention | 3 |

**Pattern.** The hard set is dominated by two failure modes: (a) the three
`occluded_attention` samples and the lone `exploratory_attention` sample
(rare classes models can't learn from 0–3 training examples), and (b)
6 samples where the two human raters themselves initially disagreed —
these are intrinsically ambiguous and the consensus label is not easy to
recover from features alone.

---

## 5. Feature ablation (LinearSVM, stratified CV)

Source: [ablation.csv](ablation.csv).

| Config | # features | Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|---:|
| without_motion | 12 | **0.725** | 0.351 | **0.506** |
| only_temporal | 2 | 0.754 | 0.371 | 0.384 |
| without_occlusion | 11 | 0.711 | 0.294 | 0.371 |
| full | 14 | 0.711 | 0.318 | 0.450 |
| only_shifts | 2 | 0.710 | 0.164 | 0.299 |
| without_visibility | 10 | 0.696 | 0.257 | 0.362 |
| only_visibility | 4 | 0.696 | 0.245 | 0.351 |
| without_head | 13 | 0.696 | 0.268 | 0.409 |
| without_shifts | 12 | 0.696 | 0.278 | 0.360 |
| only_head | 1 | 0.696 | 0.000 | 0.240 |
| without_temporal | 12 | 0.682 | 0.203 | 0.339 |
| only_motion | 2 | 0.681 | 0.027 | 0.258 |
| only_occlusion | 3 | 0.666 | 0.043 | 0.261 |

**Take-away.** Under the updated labels the single most useful feature
group is `temporal` (`observed_frames` + `attention_stability_score`) —
two features alone reach 75.4 % accuracy, higher than the full 14-feature
model. Removing temporal costs 2.9 points. The motion features are now
noise (removing them slightly *improves* accuracy). This shift from the
previous "shifts+visibility dominate" picture is driven by the consensus
relabeling: the new `occluded_attention` samples are separable primarily
by temporal/stability features, not by gaze-shift counts.

---

## 6. Additional baselines (OOF, stratified CV)

Source: [additional_baselines.csv](additional_baselines.csv).

| Classifier | Accuracy | Macro-F1 | κ |
|---|---:|---:|---:|
| Dummy_majority | 0.696 | 0.240 | 0.000 |
| LightGBM | 0.696 | 0.240 | 0.000 |
| XGBoost | 0.682 | 0.355 | 0.228 |
| Dummy_stratified | 0.610 | 0.256 | 0.097 |

**Take-away.** At n=69 with 4 effective classes, LightGBM still collapses
to the majority class and XGBoost fails to beat `Dummy_majority` on raw
accuracy (though it does better on κ and macro-F1). **Majority-class
accuracy is 0.696** — every model's accuracy should be read relative to
that floor. BART (0.144), DistilBERT (0.014), and — notably — DeBERTa
(0.696 exact) are not better than returning "focused" for everything.

---

## 7. 3-class merged evaluation (focused / mixed / other)

Rare labels (`occluded`, `exploratory`, `reduced_visual`) collapsed into
`other`. Source: [merged_3class_results.csv](merged_3class_results.csv).

| Model | Accuracy | Balanced acc. | Macro-F1 | κ |
|---|---:|---:|---:|---:|
| rule_based | **0.855** | **0.663** | **0.659** | **0.684** |
| llm_Qwen2.5-72B | 0.783 | 0.514 | 0.508 | 0.516 |
| supervised_RandomForest | 0.768 | 0.495 | 0.480 | 0.450 |
| llm_qwen-7b | 0.725 | 0.525 | 0.480 | 0.445 |
| llm_Qwen2.5-7B | 0.710 | 0.378 | 0.363 | 0.156 |
| supervised_LinearSVM | 0.710 | 0.505 | 0.529 | 0.321 |
| supervised_LogisticRegression | 0.696 | 0.422 | 0.415 | 0.271 |
| deberta | 0.696 | 0.333 | 0.273 | 0.000 |
| bart | 0.159 | 0.407 | 0.154 | −0.009 |
| distilbert | 0.058 | 0.333 | 0.036 | 0.000 |

**Take-away.** Rule-based wins on every metric in the 3-class view,
including balanced accuracy (66.3 %) — the honest measure that penalises
majority-class bias. κ=0.684 from the classifier is on the same order as
the **inter-rater κ = 0.531** — the model is performing at roughly the
human-agreement ceiling on this task.

---

## 8. Feature importance (RandomForest on full labeled set)

Top 5 by permutation importance ([feature_importance.csv](feature_importance.csv)):

| Feature | Impurity | Permutation |
|---|---:|---:|
| attention_stability_score | (see file) | (see file) |
| observed_frames | | |
| max_visible_streak | | |
| mean_gaze_motion | | |
| mean_head_area | | |

Temporal/continuity features (`attention_stability_score`,
`observed_frames`, `max_visible_streak`) dominate. The occlusion-group
features remain near-zero importance even though there are now 3
`occluded_attention` samples — 3 examples is not enough for the forest to
build a reliable decision.

---

## What Phase A establishes

1. **Inter-rater κ = 0.531.** This is the ceiling the task allows. Any
   model above κ ≈ 0.5 is performing near human level; the rule-based
   system at κ = 0.680 is **above** the inter-rater floor.
2. **Rule-based is the best system.** McNemar shows statistically
   significant wins over LinearSVM, LogisticRegression, the two smaller
   LLMs, and every NLI model; wins over RandomForest and Qwen2.5-72B are
   numerically positive but not significant at n=69.
3. **Supervised linear baselines are fragile under consensus relabeling.**
   LogReg and SVM dropped from 91.3 % / κ=0.81 (old labels) to 69.6 % and
   71.1 % / κ=0.27 and 0.32 (new labels). The 14-feature linear model
   cannot absorb the new `occluded_attention` samples in 2-fold CV.
4. **LLMs hold up better than linear baselines but worse than rule-based.**
   Qwen2.5-72B dropped from 79.7 % to 78.3 %, Qwen-7B from 79.7 % to 72.4 %.
5. **NLI models are confirmed to be below the majority-class floor** of
   0.696. They should be reported as negative controls, not systems.
6. **The effective task is 2-class.** No model learns
   `reduced_visual_availability` (n=0) or `exploratory_attention` (n=1,
   only rule-based catches it). Any clinical/behavioural claim should be
   limited to `focused` vs `mixed` plus a "rare/ambiguous" catch-all.
7. **12 hard samples + 1 all-wrong sample** are candidates for the
   qualitative error-analysis section.

---

## Files produced

All files live in [output/phase_a_analysis/](output/phase_a_analysis/).

| File | Contents |
|---|---|
| `inter_rater_report.md` / `inter_rater_agreement.json` | Rater 1 vs Rater 2 κ, confusion, consensus bias, per-sample disagreements |
| `bootstrap_ci.csv` | Accuracy / κ / Macro-F1 with 95% CI for 10 models |
| `mcnemar_pairwise.csv` | 45 pairwise McNemar tests |
| `per_class_metrics.csv` | Precision / recall / F1 per model × class |
| `confusion_matrices/*.csv` | One confusion-matrix CSV per model |
| `ablation.csv` | Full + leave-one-group-out + group-only |
| `additional_baselines.csv` | XGBoost, LightGBM, Dummy (majority + stratified) |
| `merged_3class_results.csv` | focused / mixed / other collapsed evaluation |
| `feature_importance.csv` | RandomForest impurity + permutation importance |
| `per_sample_errors.csv` | Per-sample correctness across all 10 models |
| `all_results.json` | Full structured output (every analysis) |
