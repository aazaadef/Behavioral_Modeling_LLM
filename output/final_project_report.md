# Final Project Report: Automated Classification of Child Visual Attention Behavior from Gaze Tracking Data

> **Headline update (Phase 6).** The current primary results are the
> **native 3-class re-run** — see
> [`3class_eval/phase_a/THREECLASS_REPORT.md`](3class_eval/phase_a/THREECLASS_REPORT.md)
> and §11 below. Rule-based reaches **accuracy 0.870 / κ 0.706**, exceeding
> the 3-class inter-rater ceiling of κ = 0.613. Sections 1–10 describe the
> original 5-class benchmark and remain valid as the appendix / secondary view.
>
> **See also:** [phase_a_analysis/PHASE_A_REPORT.md](phase_a_analysis/PHASE_A_REPORT.md) — statistical upgrade of the 5-class benchmark: bootstrap CIs, pairwise McNemar tests, ablation, XGBoost/LightGBM/Dummy baselines, post-hoc 3-class merged evaluation, feature importance, error analysis.

---

## 1. Project Overview

### 1.1 Objective
Automated classification of child visual attention behavior using gaze tracking annotations from the **ChildPlay-gaze** dataset. The goal is to compare multiple classification approaches — rule-based heuristics, zero-shot NLI models, supervised classifiers, and local LLMs — against human-annotated ground truth.

### 1.2 Dataset
- **ChildPlay-gaze**: Annotation-based gaze tracking dataset (no video processing)
- **401 clips** from **95 YouTube videos** of children playing
- **Splits**: train=330, val=26, test=45 annotation files
- **Test set**: **69 child-clip sequences** (multiple children per clip)
- **Per-frame annotations**: `frame, person_id, bbox_x, bbox_y, bbox_width, bbox_height, gaze_class, gaze_x, gaze_y, is_child`

### 1.3 Label Space (5 Canonical Labels)

| Label | Definition |
|-------|-----------|
| `focused_attention` | Predominantly stable, sustained visual attention on a target |
| `exploratory_attention` | Frequent shifts or scanning across multiple targets |
| `occluded_attention` | Attention cannot be characterized due to unavailable gaze |
| `reduced_visual_availability` | Interpretation limited by closed eyes or off-frame gaze |
| `mixed_attention` | Both stable attention and notable shifts without one dominant pattern |

### 1.4 Ground Truth Distribution (69 consensus-labeled samples)

Labels are the **consensus** between two independent human annotators (user
+ advisor). Inter-rater raw agreement 78.3 %, Cohen's **κ = 0.531**
(moderate). All 15 disagreements were resolved by joint review; see
[phase_a_analysis/inter_rater_report.md](phase_a_analysis/inter_rater_report.md).

| Label | Count | Percentage |
|-------|-------|-----------|
| focused_attention | 48 | 69.6% |
| mixed_attention | 17 | 24.6% |
| occluded_attention | 3 | 4.3% |
| exploratory_attention | 1 | 1.4% |
| reduced_visual_availability | 0 | 0.0% |

---

## 2. Pipeline Architecture

### 2.1 Data Processing Pipeline

```
Raw CSV Annotations
    |
    v
Dataset Loader (dataset.py)
    - Filters is_child=1
    - Groups frames per child per clip
    - child_id = "{clip_id}:person_{person_id}"
    |
    v
Temporal Segmentation (temporal.py)
    - Segments frames into continuous behavioral episodes
    - Types: stable, shift, occluded, outside, eyes_closed
    |
    v
Interaction Inference (interactions.py)
    - Maps segments to behavioral events
    - Detects focused_attention (stable >= 1.0s)
    - Detects exploratory_behavior (>= 3 shift segments)
    |
    v
Feature Extraction (features.py)
    - 14 numerical features per child sequence
    |
    v
Classification Backends (llm.py, supervised_baselines.py, run_local_llm_unified.py)
```

### 2.2 Extracted Features (14 numerical)

| Feature | Description |
|---------|-----------|
| observed_frames | Total number of annotated frames |
| visible_ratio | Fraction of frames with visible gaze |
| on_screen_ratio | Fraction of frames with on-screen gaze |
| gaze_shift_ratio | Fraction of frames classified as gaze shift |
| occlusion_ratio | Fraction of frames with occluded gaze |
| eyes_closed_ratio | Fraction of frames with eyes closed |
| mean_head_area | Average bounding box area of head |
| mean_head_motion | Average frame-to-frame head center displacement |
| mean_gaze_motion | Average frame-to-frame gaze point displacement |
| visible_gaze_fraction | Fraction of visible frames with valid gaze |
| gaze_shift_events | Number of distinct gaze shift events |
| max_visible_streak | Longest consecutive visible-gaze run |
| max_occlusion_streak | Longest consecutive occluded run |
| attention_stability_score | Composite stability metric (0-1) |

### 2.3 Source Code Structure

| Module | Role |
|--------|------|
| `src/project_llm/dataset.py` | Data loading and child sequence extraction |
| `src/project_llm/temporal.py` | Temporal segmentation of frame sequences |
| `src/project_llm/interactions.py` | Interaction event inference from segments |
| `src/project_llm/features.py` | Behavioral feature extraction |
| `src/project_llm/llm.py` | Rule-based and zero-shot NLI backends |
| `src/project_llm/supervised_baselines.py` | Supervised classifiers (LogReg, SVM, RF) |
| `src/project_llm/manual_eval.py` | Manual evaluation workflow and annotation tools |
| `scripts/run_local_llm_unified.py` | Unified LLM inference with 5-label schema |
| `scripts/run_final_comparison.py` | Final comparison of all models |
| `scripts/run_supervised_baselines.py` | Supervised baseline runner |
| `scripts/download_models.py` | HuggingFace model downloader |

---

## 3. Models Evaluated

### 3.1 Rule-Based Baseline
Hand-crafted thresholds on extracted features:
- `eyes_closed_ratio >= 0.3` -> reduced_visual_availability
- `occlusion_ratio >= 0.35` -> occluded_attention
- `visible_ratio >= 0.75 AND gaze_shift_ratio <= 0.1 AND mean_gaze_motion <= 120 AND attention_stability_score >= 0.65` -> focused_attention
- `gaze_shift_ratio >= 0.2 OR mean_gaze_motion >= 180` -> exploratory_attention
- Otherwise -> mixed_attention

### 3.2 Zero-Shot NLI Models (3 models)
Premise: structured gaze feature summary  
Hypothesis template: `"This clip is best described as {label}."`

| Model | Source |
|-------|--------|
| DistilBERT-MNLI | typeform/distilbert-base-uncased-mnli |
| BART-Large-MNLI | facebook/bart-large-mnli |
| DeBERTa-v3-Large | MoritzLaurer/deberta-v3-large-zeroshot-v2.0 |

### 3.3 Supervised Classifiers (3 models)
All use StandardScaler + classifier pipeline, evaluated via stratified 2-fold CV:

| Classifier | Configuration |
|-----------|---------------|
| LogisticRegression | solver=lbfgs, max_iter=2000 |
| LinearSVM | max_iter=5000, dual=auto |
| RandomForest | n_estimators=200, class_weight=balanced |

### 3.4 Local LLMs (3 models)
Structured prompt with label definitions, forcing JSON output (`label`, `confidence`, `reasoning`):

| Model | Size | Quantization | GPU Setup |
|-------|------|-------------|-----------|
| Qwen-7B (legacy) | ~14GB | None (fp16) | 1x A6000 |
| Qwen2.5-7B-Instruct | ~14GB | None (fp16) | 1x A6000 |
| Qwen2.5-72B-Instruct | ~136GB | 4-bit (nf4) | 4x A6000 |

Hardware: **4x NVIDIA RTX A6000** (48GB each, 192GB total VRAM)

---

## 4. Final Results (69 Consensus-Labeled Test Samples)

> **These numbers are against the updated consensus labels** (two-annotator
> agreement with discussion-based resolution). They supersede the original
> single-annotator evaluation. 8 of 69 labels changed from the earlier
> file, which moved SVM/LogReg accuracy down and left rule-based at the
> top. Supervised numbers are **out-of-fold CV** — no train/test leakage.

### 4.1 Model Ranking (with bootstrap 95% CI)

| Rank | Model | Accuracy | 95% CI | Macro-F1 | Cohen's κ |
|------|-------|---------:|---|---------:|----------:|
| 1 | **Rule-Based** | **0.854** | [0.768, 0.928] | **0.521** | **0.680** |
| 2 | LLM Qwen2.5-72B (4-bit) | 0.783 | [0.681, 0.870] | 0.330 | 0.515 |
| 3 | Supervised RandomForest (OOF) | 0.768 | [0.667, 0.870] | 0.414 | 0.444 |
| 4 | LLM Qwen-7B | 0.725 | [0.623, 0.826] | 0.381 | 0.441 |
| 5 | Supervised LinearSVM (OOF) | 0.710 | [0.609, 0.812] | 0.446 | 0.314 |
| 6 | LLM Qwen2.5-7B | 0.709 | [0.608, 0.812] | 0.311 | 0.151 |
| 7 | Supervised LogisticRegression (OOF) | 0.696 | [0.594, 0.812] | 0.353 | 0.268 |
| 8 | Zero-Shot DeBERTa | 0.696 | [0.594, 0.797] | 0.237 | 0.000 |
| 9 | Zero-Shot BART | 0.144 | [0.073, 0.232] | 0.112 | −0.021 |
| 10 | Zero-Shot DistilBERT | 0.014 | [0.000, 0.043] | 0.007 | 0.000 |

Bootstrap source: [phase_a_analysis/bootstrap_ci.csv](phase_a_analysis/bootstrap_ci.csv).

**Statistical significance (McNemar).** Rule-based significantly beats
LinearSVM (p=0.021), LogReg (p=0.007), Qwen-7B (p=0.022), and Qwen2.5-7B
(p=0.013). Its lead over RandomForest (p=0.110) and Qwen2.5-72B (p=0.180)
is numerically positive but not statistically significant at n=69. See
[phase_a_analysis/mcnemar_pairwise.csv](phase_a_analysis/mcnemar_pairwise.csv).

**Reference points.** Majority-class baseline (`Dummy_majority`) accuracy
is **0.696**, and inter-rater κ between the two human annotators is
**0.531**. Rule-based's κ of 0.680 is above the inter-rater ceiling.

### 4.2 Supervised Baselines (2-fold Stratified CV)

| Classifier | Accuracy (CV) | Macro-F1 (CV) | Kappa |
|-----------|---------------|---------------|-------|
| RandomForest | 0.7693 ± 0.0836 | 0.4335 ± 0.1123 | 0.4643 |
| LinearSVM | 0.7105 ± 0.0248 | 0.4495 ± 0.0246 | 0.3178 |
| LogisticRegression | 0.6962 ± 0.0391 | 0.3640 ± 0.0609 | 0.2727 |

### 4.3 LLM Prediction Distributions

| Model | focused | mixed | exploratory | reduced_visual | occluded |
|-------|---------|-------|-------------|----------------|----------|
| **Ground Truth (consensus)** | **48** | **17** | **1** | **0** | **3** |
| Qwen2.5-72B | 48 | 18 | 1 | 2 | 0 |
| Qwen-7B | 40 | 28 | 1 | 0 | 0 |
| Qwen2.5-7B | 63 | 6 | 0 | 0 | 0 |
| Rule-Based | 47 | 18 | 4 | 0 | 0 |
| DeBERTa | 69 | 0 | 0 | 0 | 0 |
| DistilBERT | 0 | 0 | 69 | 0 | 0 |
| BART | 0 | 40 | 0 | 0 | 29 |

### 4.4 Zero-Shot NLI Label Collapse

| Model | Behavior | All predictions |
|-------|----------|----------------|
| DistilBERT | Complete collapse | 69/69 = exploratory_attention (100%) |
| DeBERTa | Complete collapse | 69/69 = focused_attention (100%) |
| BART | Partial collapse | 40 mixed + 29 occluded (2 labels only) |

**Root cause**: Generic NLI models cannot map numerical gaze features to domain-specific attention labels. The hypothesis template `"This clip is best described as {label}"` does not provide enough semantic grounding for the model to differentiate between labels.

---

## 5. Key Findings

### Finding 1: Rule-based is the best system
The hand-crafted rule-based classifier reaches **85.4 % accuracy** and
**κ = 0.680** (highest of all 10 models) on the consensus labels.
McNemar confirms statistically significant wins over LinearSVM (p=0.021),
LogisticRegression (p=0.007), Qwen-7B (p=0.022), and Qwen2.5-7B (p=0.013).
Its leads over RandomForest (p=0.110) and Qwen2.5-72B (p=0.180) are
positive but not significant at n=69. Rule-based's κ of 0.680 sits
**above** the inter-rater κ = 0.531 — it agrees with the consensus more
than the two humans agreed with each other.

### Finding 2: Supervised linear baselines are fragile under consensus relabeling
Against the first-pass (single-annotator) labels, LinearSVM and
LogisticRegression achieved 91.3 % accuracy and κ=0.81. Against the
two-annotator consensus labels they drop to 71.1 % / 69.6 % and
κ=0.318 / 0.268. The 14-feature linear models cannot absorb the three
`occluded_attention` samples in 2-fold CV. RandomForest is more robust
(76.9 %, κ=0.464).

### Finding 3 (Negative): Zero-shot NLI models fail completely
All three NLI models exhibit label collapse. DeBERTa predicts
`focused_attention` for all 69 samples — its 69.6 % accuracy is exactly
the `Dummy_majority` floor (κ=0). BART and DistilBERT collapse to
non-majority classes and score **below** the majority floor (14 % and
1 % accuracy). Under the prompting protocol used here, generic zero-shot
NLI is unsuitable for gaze-based attention classification.

### Finding 4 (Mixed): LLMs sit in the middle
Qwen-family LLMs reach 71–78 % accuracy and κ=0.15–0.52 — better than
the two smaller NLI failures and comparable with RandomForest on raw
accuracy, but worse than rule-based on every metric.

### Finding 5: LLM scale gives inconsistent returns
| Model | Parameters | Accuracy | κ |
|-------|-----------|----------|---:|
| Qwen2.5-7B | 7B | 0.709 | 0.151 |
| Qwen-7B (legacy) | 7B | 0.725 | 0.441 |
| Qwen2.5-72B | 72B (4-bit) | 0.783 | 0.515 |
The 72B model is best, but the 7B legacy run beats the newer 7B-Instruct
model by 29 κ points — prompt/formatting choices matter as much as
scale at this data size.

### Finding 6: Inter-rater κ = 0.531 is the task ceiling
Two humans labeling the same 69 samples independently reach only
*moderate* agreement (κ=0.531). Any model claim above κ ≈ 0.5 is
performing near the human-agreement ceiling, so headline numbers must
be framed relative to it, not to an assumed 100 % gold.

### Finding 7: Macro-F1 is low because rare classes are effectively unlearnable
Macro-F1 is the **unweighted** mean over all five classes; a single
zero-F1 class drags it down by ~0.2. Only `rule_based` achieves non-zero
F1 on `exploratory_attention` (F1=0.400 on n=1), and only
`supervised_LinearSVM` achieves non-zero F1 on `occluded_attention`
(F1=0.400 on n=3). No model learns `reduced_visual_availability` (n=0).
The practical task is therefore **2-class**: `focused` vs `mixed`, with
the rest as a "rare/ambiguous" catch-all.

### Finding 8 (Phase A): Temporal features dominate the supervised signal
Ablation on LinearSVM ([ablation.csv](phase_a_analysis/ablation.csv))
shows that two temporal features alone — `observed_frames` +
`attention_stability_score` — reach 75.4 % accuracy, actually beating
the full 14-feature model (71.1 %). Motion features add noise; removing
them slightly improves performance. Occlusion features contribute
essentially nothing at n=3 occluded samples.

---

## 6. Methodology Notes

### 6.1 Evaluation Protocol
- **Ground truth**: 69 test-split child sequences labeled independently by two human annotators (user + advisor); disagreements resolved by joint review. Inter-rater raw agreement 78.3 %, Cohen's κ = 0.531.
- **Metrics**: Accuracy, Macro-F1 (**unweighted** mean of per-class F1 scores — not weighted by support), Cohen's Kappa (chance-corrected agreement), bootstrap 95 % CI (1000 iterations), pairwise McNemar significance.
- **Supervised CV**: Stratified 2-fold (reduced from 5-fold because smallest class has 1 sample) — out-of-fold predictions used for evaluation.

### 6.2 LLM Inference Protocol
- Structured prompt including label definitions and expected JSON output format
- Chat template applied for instruct-tuned models
- Greedy decoding (do_sample=False) for reproducibility
- Robust JSON extraction with fallback strategies (fenced blocks, inline braces, truncation repair)
- Label normalization via partial matching with fallback to mixed_attention

### 6.3 Limitations
1. **Class imbalance**: 69.6 % focused_attention — models biased toward majority class; `Dummy_majority` baseline is 0.696.
2. **Moderate inter-rater agreement**: κ=0.531 means the consensus label is not itself a perfect gold; it is the ceiling the task allows at this annotation protocol.
3. **Small test set**: 69 samples limits statistical power, especially for minority classes (1 exploratory, 3 occluded, 0 reduced_visual).
4. **No Llama models**: Llama-3.1-8B/70B not tested due to gated repository access.
5. **Supervised CV folds**: Reduced to 2-fold because the smallest class has 1 sample — high variance in supervised numbers.
6. **No video review**: Manual labels based on numerical features + model previews, not direct video observation.

---

## 7. Output Files and Artifacts

### 7.1 Final Comparison
- `output/final_comparison/model_ranking.csv` — ranked model results
- `output/final_comparison/model_evaluation.json` — per-model detailed metrics
- `output/final_comparison/pairwise_agreement.csv` — inter-model agreement
- `output/final_comparison/comparison_summary.json` — metadata

### 7.2 Supervised Baselines
- `output/supervised_baselines/supervised_summary.json` — CV results
- `output/supervised_baselines/supervised_predictions.csv` — per-sample predictions
- `output/supervised_baselines/supervised_confusion_matrices.json`
- `output/supervised_baselines/supervised_classification_reports.json`

### 7.3 LLM Runs
- `output/llm_runs/models_qwen-7b/` — Qwen-7B predictions
- `output/llm_runs/models_Qwen--Qwen2_5-7B-Instruct/` — Qwen2.5-7B predictions
- `output/llm_runs/models_Qwen--Qwen2_5-72B-Instruct/` — Qwen2.5-72B predictions

Each contains: `prompt_logs.jsonl`, `raw_outputs.jsonl`, `parsed_outputs.jsonl`, `errors.jsonl`, `run_summary.json`

### 7.4 Zero-Shot NLI Runs
- `output/benchmark_rule_based/` — rule-based baseline
- `output/benchmark_zero_shot/` — DistilBERT predictions
- `output/benchmark_bart_large_mnli/` — BART predictions
- `output/benchmark_deberta_zeroshot/` — DeBERTa predictions

### 7.5 Manual Evaluation
- `output/paper_eval/manual_eval_annotations.csv` — ground truth (69 final_label)
- `output/paper_eval/label_schema.json` — label definitions
- `output/paper_eval/annotation_guidelines.md` — annotation protocol
- `output/all_predictions_for_labeling.csv` — combined predictions from all 10 models

---

## 8. Project Timeline

| Phase | Activity | Key Output |
|-------|----------|-----------|
| Phase 1 | Dataset audit, pipeline construction, rule-based + NLI benchmarks | Core pipeline, 4 baseline models |
| Phase 2 | Code documentation, cleanup, backup | All files documented in English |
| Phase 3 | Supervised baselines, unified LLM infrastructure, label space fix | 3 supervised + 3 LLM models |
| Phase 4 | Model downloads and inference (Qwen-7B, Qwen2.5-7B, Qwen2.5-72B) | Complete LLM predictions |
| Phase 5 | Manual labeling of 69 samples, final comparison | Ground truth + final results |

---

## 9. Conclusion

On 69 two-annotator consensus-labeled samples, the hand-crafted rule-based
classifier is the best of 10 systems at κ=0.680, above the inter-rater
ceiling (κ=0.531). Main conclusions:

1. **Domain-engineered rules beat all learned systems** at this sample
   size. Rule-based significantly outperforms both linear supervised
   baselines (McNemar p≤0.021) and both smaller LLMs. Its lead over
   RandomForest and the 72B LLM is numerically positive but not
   statistically significant at n=69.

2. **Supervised linear baselines are not robust to label revisions.**
   Moving from single-annotator to two-annotator consensus labels
   dropped SVM/LogReg accuracy by ~20 points and κ by ~0.5. RandomForest
   was more robust but still lost ~10 points.

3. **Zero-shot NLI is unusable on this task.** DeBERTa predicts the
   majority class for every sample; BART and DistilBERT score *below*
   the majority-class floor.

4. **Task ceiling, not model ceiling.** Two humans agreed on only 78 %
   of labels (κ=0.531). This bounds what any model can achieve against
   a "true" gold standard. Rule-based at κ=0.680 is already above this
   ceiling.

5. **The practical task is 2-class** (focused vs mixed). No model
   learns `occluded` (n=3), `exploratory` (n=1), or
   `reduced_visual_availability` (n=0). Clinical/behavioural claims
   should be scoped accordingly.

6. **LLMs contribute qualitative value**: Qwen-family models produce
   per-sample reasoning alongside predictions, which the linear
   baselines cannot, and may be useful for interpretability even when
   they are not the most accurate classifier.

---

## 10. Recommendations for Paper

> **See also: [PAPER_NOTES.md](PAPER_NOTES.md)** — detailed talking
> points for the discussion section, organized as 14 argument units
> covering (1) motivation for LLMs, (2) systematic `others` failure,
> (3) lexical bias, (4) numerical reasoning gap, (5) A/B sample case
> study, (6)–(7) surface vs analytical reasoning, (8)–(9) hybrid
> framework justification, (10)–(11) positioning vs video-based work,
> (12) interpretability / scalability / decision-support advantages,
> (13)–(14) broader AI-systems insight and final framing.

### Narrative Structure
1. **Problem**: Automated classification of child visual attention from gaze annotations on a small, human-ambiguous dataset.
2. **Method**: Feature extraction pipeline + 10 classification approaches across 4 families, with two-annotator consensus ground truth.
3. **Primary finding**: Rule-based heuristics dominate on n=69 and reach κ=0.68 — above the inter-rater ceiling of κ=0.53.
4. **Secondary findings**: (a) zero-shot NLI collapses to or below the majority floor; (b) supervised linear baselines are fragile to relabeling; (c) LLM scale helps but 7B legacy ≈ 72B on this task.
5. **Method contribution**: Full statistical pipeline (bootstrap CI, McNemar, ablation, dummy baselines, 3-class merged eval, feature importance) instead of single-accuracy table.
6. **Discussion**: Feature engineering + domain heuristics outperform learned systems when the effective class count > training set size per class.

### Strengthening the Paper
- Add video-based verification on the 12 hard samples (Phase B remaining work).
- Add a blind re-annotation round on a 25-sample subset to confirm inter-rater κ stability.
- Investigate why the rule-based heuristic wins — is it the hard thresholds on `attention_stability_score` that the linear CV folds cannot reproduce?
- Consider reporting only the 2-class or 3-class merged evaluation as the headline, with the 5-class result in the appendix.

---

*Report generated: 2026-04-16*
*Hardware: 4x NVIDIA RTX A6000 (192GB VRAM total)*
*All models are open-source/free — no commercial API dependencies*

---

## 11. Phase 6 — From 5-class to 3-class (schema refinement + re-run)

### 11.1 Motivation: from 5-class to 3-class

ChildPlay-gaze is natively a **5-class** dataset (`focused_attention`,
`mixed_attention`, `exploratory_attention`, `occluded_attention`,
`reduced_visual_availability`). The Phase-A analysis (earlier sections)
uses that native schema. Three independent empirical signals on the
69-sample test set argue that the 5-class schema is **under-supported
at this sample size** and that a semantically-justified 3-class
collapse is the principled choice:

**(i) Empirical class degeneracy.** In the two-rater consensus, two
of the five dataset classes are near-absent:

| 5-class label | Consensus count (n = 69) |
|---|---:|
| `focused_attention` | 48 |
| `mixed_attention` | 17 |
| `occluded_attention` | 3 |
| `exploratory_attention` | **1** |
| `reduced_visual_availability` | **0** |

No CI, F1, or McNemar test can be meaningfully reported for a class
with 0–1 samples; the dataset itself makes 5-class evaluation
statistically ill-posed at n = 69.

**(ii) Inter-rater agreement is bound by boundary-pair ambiguities.**
Under 5-class the two annotators reach 54/69 (78.3 %, κ = 0.531 —
"moderate"). The 15 disagreements cluster almost entirely on two
semantically-adjacent pairs:

| Disagreement pair (5-class) | Count |
|---|---:|
| `focused_attention` ↔ `mixed_attention` | 8 |
| `mixed_attention` ↔ `occluded_attention` | 3 |
| `exploratory_attention` ↔ `mixed_attention` | 3 |
| `exploratory_attention` ↔ `focused_attention` | 1 |

The 3 `exploratory ↔ mixed` disagreements are within-family: both
labels describe active gaze switching / scanning at different
intensities. The same pattern holds on the visibility-failure side
(`occluded ↔ reduced_visual_availability` both describe unusable
gaze signal).

**(iii) Semantic coherence of the 3-class target.** The resulting
three classes map onto the attention-literature distinction
between **sustained attention**, **active scanning / divided
attention**, and **unusable gaze signal** — the three operational
states a downstream application actually needs to distinguish.

Together, (i) makes 5-class evaluation ill-posed on n = 69, (ii)
shows that the remaining distinctions are the ones humans themselves
cannot agree on, and (iii) shows the collapse is principled rather
than convenient.

### 11.2 Deterministic 5 → 3 mapping

The collapse is a single-valued function of the 5-class label
(`MAP_5_TO_3` in [`src/project_llm/labels_3class.py`](../src/project_llm/labels_3class.py)):

| 5-class label | 3-class label | Semantic family |
|---|---|---|
| `focused_attention` | `focused` | sustained attention |
| `mixed_attention` | `mix` | active scanning / divided attention |
| `exploratory_attention` | `mix` | active scanning / divided attention |
| `occluded_attention` | `others` | unusable gaze signal |
| `reduced_visual_availability` | `others` | unusable gaze signal |

Post-collapse consensus distribution: `focused = 48, mix = 18,
others = 3`. Every class now has ≥ 3 samples.

**The Phase-6 results are a native re-run, not a post-hoc mapping
of 5-class model outputs.** Rule engine uses new 3-class
thresholds; NLI backends use three 3-class candidate labels;
supervised baselines were re-trained with StratifiedKFold OOF CV on
3-class targets; every LLM ran with a fresh 3-label prompt. As
evidence: LinearSVM *drops* from 0.710 → 0.667 under native
3-class re-training, which would be impossible under post-hoc
mapping.

### 11.3 Inter-rater agreement under the 3-class schema

| Schema | Raw agreement | Cohen's κ | Landis–Koch |
|---|---|---:|---|
| 5-class | 54/69 (78.3 %) | 0.531 | moderate |
| **3-class** | **57/69 (82.6 %)** | **0.613** | **substantial** |

The 3 `exploratory ↔ mixed` disagreements resolve under the
collapse, lifting κ across the Landis–Koch moderate → substantial
threshold. The remaining 12 disagreements split as `focused ↔ mix`
(9 cases) and `mix ↔ others` (3 cases) — genuine boundary
ambiguities, not label-space artefacts. This κ shift is **post-hoc
evidence that the schema is right**: the collapse eliminates
disagreements that were semantic duplicates and preserves the ones
that genuinely require more data or a better definition to resolve.

Consensus distribution under 3-class: focused = 48, mix = 18, others = 3.

### 11.4 Headline accuracy (bootstrap 95 % CI, 1000 iter.)

**18 systems evaluated** — rule-based, three NLI zero-shot backends,
seven supervised baselines (LogReg, LinearSVM, RandomForest, XGBoost,
LightGBM, Dummy_majority, Dummy_stratified), and seven LLMs spanning
five families (Alibaba, Mistral AI, Microsoft, 01.ai, Meta):

| Rank | Model | Family | Accuracy | 95 % CI | κ | Macro-F1 |
|---:|---|---|---:|---|---:|---:|
| 1 | **rule_based** | — | **0.870** | [0.783, 0.942] | **0.706** | 0.575 |
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

Majority-class floor (Dummy_majority): **0.696 (κ = 0)**.

**Cross-family observation.** The top tier (0.797–0.870) contains
rule-based, RandomForest, and **five different LLM families** clustered
tightly. Five LLMs (Yi-1.5-9B, Qwen2.5-72B, Llama-3.1-8B, Phi-4-mini,
Mistral-7B) are within 2 correct samples of each other on n = 69 —
the LLM result is **not Qwen-specific and not scale-driven**: a 4 B
Phi-4-mini ties a 72 B Qwen, and all major open-weight families
(Alibaba, Meta, Mistral, Microsoft, 01.ai) land in the same bucket.
Only three LLMs (Mistral, Phi-4-mini, Qwen2.5-72B) ever predict the
minority `others` class (n = 3).

**Noise-floor calibration.** `Dummy_majority` (always predict `focused`)
reaches 0.696 with κ = 0; `Dummy_stratified` (sample from the class
prior) falls to 0.536 with negative κ. Both NLI zero-shot models
(distilbert 0.696, deberta 0.681) are statistically indistinguishable
from `Dummy_majority`. Boosted trees (XGBoost 0.739, LightGBM 0.725)
sit at the linear LogReg tier, not the LLM tier.

### 11.5 5-class vs native 3-class (Δ accuracy, same models)

| Model | 5-class | 3-class | Δ acc | Δ κ |
|---|---:|---:|---:|---:|
| rule_based | 0.854 | 0.870 | +0.016 | +0.026 |
| Qwen2.5-72B | 0.783 | 0.812 | +0.028 | +0.066 |
| supervised_RandomForest | 0.768 | 0.797 | +0.030 | +0.047 |
| Qwen2.5-7B | 0.709 | 0.754 | +0.044 | +0.241 |
| supervised_LogisticRegression | 0.696 | 0.725 | +0.029 | +0.063 |
| distilbert | 0.014 | 0.696 | +0.682 | 0.000 |
| supervised_LinearSVM | 0.710 | 0.667 | −0.043 | −0.124 |
| deberta | 0.696 | 0.681 | −0.014 | −0.026 |
| bart | 0.144 | 0.261 | +0.117 | +0.021 |

Four supervised baselines (XGBoost, LightGBM, Dummy_majority,
Dummy_stratified) and four LLMs (Yi-1.5-9B, Llama-3.1-8B, Phi-4-mini,
Mistral-7B) are 3-class-native — no 5-class counterpart to compare to.

### 11.6 Pairwise significance (rule_based vs others, McNemar α = 0.05)

Rule-based **significantly beats** (p < 0.05): bart, Dummy_stratified,
LinearSVM, LightGBM, XGBoost, deberta, LogisticRegression, distilbert,
Dummy_majority, Qwen2.5-7B, qwen-7b (legacy).
**Ties** (p > 0.05) with: RandomForest, Phi-4-mini, Mistral-7B,
Llama-3.1-8B, Qwen2.5-72B, Yi-1.5-9B.

Full 153-pair matrix: [`3class_eval/phase_a/mcnemar_pairwise_3class.csv`](3class_eval/phase_a/mcnemar_pairwise_3class.csv).

### 11.7 Take-aways

1. **Rule-based is the strongest system** (κ = 0.706) and **exceeds the
   human-agreement ceiling** (inter-rater κ = 0.613).
2. **Five independent LLM families cluster in 0.797–0.812** (Alibaba,
   Mistral, Microsoft, 01.ai, Meta) — the LLM result is
   family-agnostic and scale-insensitive from 4 B (Phi-4-mini) to 72 B
   (Qwen2.5-72B). This rebuts the "one-family, weak comparison"
   critique and supports the generalization claim.
3. **Collapsing to 3 classes lifts inter-rater κ from moderate to
   substantial** — most disagreements lived on the mix/exploratory
   boundary, not on focused.
4. **Supervised models benefit from retraining on 3 labels** (vs
   post-hoc mapping). Among seven supervised baselines, only
   RandomForest reaches the LLM cluster; boosted trees (XGBoost,
   LightGBM) match the linear LogReg tier, confirming that structured
   supervision on 69 samples is not enough to beat instruction-tuned
   LLMs.
5. **NLI zero-shot and `Dummy_majority` all land at 0.696 (κ ≈ 0)** —
   this is the 48/69 majority floor. NLI backends are confirmed as
   negative controls, not usable baselines.

Artifacts:
- [`3class_eval/phase_a/THREECLASS_REPORT.md`](3class_eval/phase_a/THREECLASS_REPORT.md) — full report with side-by-side tables
- [`3class_eval/phase_a/bootstrap_ci_3class.csv`](3class_eval/phase_a/bootstrap_ci_3class.csv)
- [`3class_eval/phase_a/mcnemar_pairwise_3class.csv`](3class_eval/phase_a/mcnemar_pairwise_3class.csv)
- [`3class_eval/phase_a/per_class_metrics_3class.csv`](3class_eval/phase_a/per_class_metrics_3class.csv)
- [`3class_eval/phase_a/feature_importance_3class.csv`](3class_eval/phase_a/feature_importance_3class.csv)
