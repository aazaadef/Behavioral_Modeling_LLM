# Cross-Dataset Validation Plan

A concrete, phased plan to replicate the 18-system benchmark and the
HRSF formula on a second gaze dataset, in order to verify that the
findings reported in [`PAPER_DRAFT.md`](PAPER_DRAFT.md) generalise
beyond ChildPlay-gaze.

This plan does **not** rely on cross-validation tricks — it is a true
external replication: a different dataset, fresh annotations, the
same 18 systems, a fresh α sweep for HRSF.

---

## 1. What "validation on another dataset" means precisely

We replicate **everything** on a held-out external dataset:

  * The same 16 behavioural feature pipeline.
  * The same 18 systems (rule-based, 3 NLI, 7 supervised, 7 LLMs).
  * The same Phase A protocol (bootstrap CI, McNemar pairwise).
  * The same HRSF formula with the **same α grid** {0.0, 0.1, …, 1.0}
    and the **same domain-driven thresholds** τ_v = 0.50, τ_s = 30,
    τ_e = 0.30 — *not* re-tuned.
  * A fresh two-rater consensus on a reasonable test subset.

If HRSF still wins (κ above the new dataset's inter-rater ceiling,
F1(`others`) > 0, alignment fault line still visible), the
contribution is robust. If it doesn't, we have a fair, falsifiable
generalisation claim with a clear failure mode to discuss.

---

## 2. Candidate datasets (ranked by feasibility)

The ChildPlay-gaze annotation schema includes
{`inside_visible`, `outside_frame`, `gaze_shift`, `inside_occluded`,
`inside_uncertain`, `eyes_closed`}. The dataset's own README notes
that `inside_visible` and `outside_frame` map directly to the
standard inside/outside flags in GazeFollow and VideoAttentionTarget.
This compatibility is the basis for everything below.

### 2.1 VideoAttentionTarget — **best fit, recommended target**

  * **Reference:** Chong et al., "Detecting Attended Visual Targets
    in Video", CVPR 2020.
  * **URL:** `http://chong.in/research/social-gaze`
  * **Size:** ~1,300 video segments from 50 different shows;
    ~109 K annotated frames.
  * **Annotation format per frame:** head bounding box, gaze target
    point (when in-frame), and an "out-of-frame" binary flag.
  * **Population:** mostly adult (children present in some shows).
  * **License:** academic use; videos are extracted from publicly
    aired media.

  **Why it fits:**
    * The `inside / outside` flag maps directly to ChildPlay's
      `inside_visible` / `outside_frame`, as the ChildPlay paper
      itself notes.
    * Per-frame annotations enable our 16-feature aggregation.
    * Video-level (not single-image), so our temporal features
      (occlusion streak, gaze-shift events) are computable.

  **Adaptation cost:**
    * `gaze_shift`, `inside_occluded`, `inside_uncertain`,
      `eyes_closed` are **not** annotated. We can derive proxies:
      - `inside_occluded` ← head bbox missing while a gaze point is
        expected.
      - `gaze_shift` ← inferred from temporal jumps in gaze target
        coordinates (we already do this in
        [`src/project_llm/temporal.py`](../src/project_llm/temporal.py)).
      - `eyes_closed` ← cannot be derived, set to 0 throughout
        (a documented limitation that we report explicitly).
    * `dominant_gaze_class` will be one of a smaller set; the
      lexical-anchoring case study still applies because
      `outside_frame` is preserved.
    * The 3-class schema (focused/mix/others) needs new manual
      labels — see §3 below.

  **Verdict:** ✅ **Best target.** ~3–5 weeks of adaptation work,
  ~4–6 weeks of fresh annotation.

### 2.2 GazeFollow

  * **Reference:** Recasens et al., NeurIPS 2015.
  * **Size:** ~130 K images, single-frame.
  * **Annotation:** gaze target (or "outside frame" flag) for each
    person.
  * **Population:** adult.

  **Why it does *not* fit well:** single-image. Our pipeline assumes
  per-clip aggregation across many frames (visibility ratio,
  occlusion streak, gaze-shift events). On a single image these
  collapse to trivial values and the rule-based / threshold features
  become uninformative. We could pool by person across the
  dataset's tracked individuals, but the conceptual fit breaks.

  **Verdict:** 🔴 Skip for primary replication; could be used in a
  reduced "single-frame APRH" ablation as supplementary material.

### 2.3 GazeFollow360 / Gaze360

  * **Reference:** Kellnhofer et al., ICCV 2019.
  * **Annotation:** continuous 3D gaze direction, no behavioural
    classes.

  **Verdict:** 🔴 Different problem (gaze estimation, not gaze
  *behaviour interpretation*). Skip.

### 2.4 DAiSEE — student engagement

  * **Reference:** Gupta et al., 2016 (preprint).
  * **Size:** ~9,000 video clips, 10 seconds each.
  * **Annotation:** four affective states (boredom, confusion,
    engagement, frustration), four levels each.
  * **Population:** university students (adults).

  **Why it could fit:** behavioural-state classification at clip
  level, similar granularity to ChildPlay 3-class.
  **Why it doesn't fit perfectly:** no gaze-target annotation, no
  visibility/occlusion classes. Our 16-feature pipeline would need
  major surgery (need a gaze tracker as a preprocessor).

  **Verdict:** 🟡 Possible for an extension paper if combined with
  an off-the-shelf gaze estimator; not feasible for the present
  paper without significant new infrastructure.

### 2.5 EYEDIAP

  * **Reference:** Funes Mora et al., ETRA 2014.
  * **Annotation:** gaze direction in lab settings.

  **Verdict:** 🔴 Lab-controlled adult gaze; far from ChildPlay's
  uncontrolled-play domain. Skip.

### 2.6 Re-annotated YouTube subset

  * **Source:** scrape ~100 short clips of children playing from
    publicly available YouTube videos (CreativeCommons-licensed,
    or with explicit permission).
  * **Annotation:** apply our existing two-rater protocol with the
    3-class schema, plus the ChildPlay-style frame-level
    `gaze_class` annotation.

  **Why it could fit:** maximum control; we get exactly the
  annotation we need. **Cost:** the highest of any option (IRB
  review if vulnerable populations, ethics approval, ~6–8 weeks of
  rater work).

  **Verdict:** 🟡 Best for an extension paper, not for this one.

---

## 3. Recommended path — VideoAttentionTarget (VAT)

We recommend a phased plan targeting VideoAttentionTarget:

### Phase 1 — Feasibility study (1–2 weeks)

  1. Download VAT annotations (videos can be obtained via the
     official scripts).
  2. Run `scripts/adapt_external_dataset.py` (scaffolded — see
     [`scripts/adapt_external_dataset.py`](../scripts/adapt_external_dataset.py))
     to convert a 50-clip pilot subset into the ChildPlay-compatible
     CSV format.
  3. Run the rule-based and one supervised system on the pilot to
     confirm the feature pipeline produces reasonable values.
  4. Spot-check the aggregated features against the original VAT
     annotations.
  5. **Go/no-go decision** — if the rule-based feature distribution
     looks pathological (e.g., visible_ratio always 1.0 because no
     occlusion class), abandon VAT and try GazeFollow + temporal
     reconstruction.

### Phase 2 — Annotation pass (4–6 weeks)

  1. Sample 100 clips from VAT (stratified across the source shows
     to avoid single-show bias).
  2. The same two raters who labelled the ChildPlay test subset
     apply the 3-class schema (`focused` / `mix` / `others`).
  3. Compute the inter-rater κ on this new set. This becomes the
     external dataset's "human ceiling" — and it must be computed
     *before* running any system, to avoid post-hoc anchoring.

### Phase 3 — Full system replication (1–2 weeks)

  1. Run all 18 systems on the 100 VAT clips.
  2. Run HRSF with the **frozen** thresholds and α grid from this
     paper. *No threshold or α refit on VAT.* This is what makes
     it a true cross-dataset replication.
  3. Re-run the per-rater κ analysis on VAT.
  4. Re-derive the alignment fault line on VAT — does the LLM /
     rule-tree split persist?

### Phase 4 — Comparison and write-up (2–4 weeks)

  1. Build a side-by-side comparison table (ChildPlay vs VAT) for
     each headline metric.
  2. Build a side-by-side fault-line figure: same Δ-bar chart,
     ChildPlay top, VAT bottom.
  3. If results replicate: a new section "External validation on
     VideoAttentionTarget" enters the paper and the title can be
     strengthened to claim cross-dataset robustness.
  4. If results partly replicate (e.g., HRSF still wins on κ but
     fault line dissolves): a new section discusses *which* of the
     three claims (above-ceiling κ, F1(`others`) > 0, alignment
     fault line) survive on VAT and which do not. This is a
     publishable result either way.

**Total realistic timeline:** 8–14 weeks of work for a research
group; ~3–4 months end-to-end for one researcher with concurrent
duties.

---

## 4. Pre-registration of the validation

Because the temptation to fish for the right α / threshold on a new
dataset is real, we recommend **pre-registering** the following
before the VAT run:

  * α grid: {0.0, 0.1, 0.2, …, 1.0} — frozen.
  * Best α: the value that maximises κ on the VAT consensus.
    No fold-by-fold tuning, no tweaking.
  * Thresholds τ_v, τ_s, τ_e: identical to those reported here
    (0.50, 30, 0.30).
  * Per-rater κ ceiling: computed before any system is run.
  * Reporting: every system's 3-class metrics with bootstrap CI,
    in the same format as
    [`bootstrap_ci_3class.csv`](3class_eval/phase_a/bootstrap_ci_3class.csv).

A simple way to pre-register is to commit the empty results
template (`vat_eval/`) to the public repository **before** starting
the VAT runs, so the git history makes the protocol falsifiable.

---

## 5. What success looks like

A successful external replication produces all four of the
following on VAT:

  1. Headline κ for HRSF above VAT's inter-rater ceiling.
  2. F1(`others`) > 0 for HRSF, F1(`others`) = 0 for the 18
     baseline systems (or at least: HRSF dominates baselines on
     this metric).
  3. The alignment fault line (LLMs aligning with one rater,
     rule/tree systems with the other) is still visible.
  4. The optimal α on VAT lies within the [0.4, 0.8] band where
     the ChildPlay sensitivity curve plateaus.

If 1–3 hold but 4 doesn't (e.g., VAT's optimal α is 0.2), this is
a *useful* finding: it suggests the rater-style fault line is
domain-dependent. The paper would acknowledge this and the next
extension would model α as a learned function of dataset
properties.

---

## 6. What it would cost in this paper

If the VAT run is folded into this paper rather than the next, the
section structure becomes:

  * `§5 Results (ChildPlay)` — current Section 5
  * `§6 Per-rater analysis (ChildPlay)` — current Section 6
  * `§7 Discussion (ChildPlay)` — current Section 7
  * `§8 The HRSF formula (ChildPlay)` — current Section 8
  * **`§9 External validation on VideoAttentionTarget` — NEW**
  * `§10 Limitations` — current Section 9
  * `§11 Conclusion` — current Section 10

Page budget: a typical Q1 paper has 8–10 pages of main text. The
current draft is well under that, so VAT could fit without
displacing existing content. Realistically, plan for an extra
2 pages.

---

## 7. What it would *not* cost

Importantly, the existing ChildPlay results are not invalidated
by the VAT run. Every figure and table stays. Even if VAT shows
the framework only partially replicates, the paper's main
contributions (per-rater fault line, F1(`others`) > 0 via HRSF,
formal hybrid framework) remain on solid empirical ground for
ChildPlay. The VAT section becomes "scope" rather than "make or
break".

---

## 8. Pragmatic recommendation

If the goal is **the present paper**, two options:

  * **Option A (slow but safe).** Hold the paper until VAT is done.
    Best ceiling on review outcome but adds 3–4 months.
  * **Option B (fast and honest).** Submit the paper as is, with
    [§9 (this plan, summarised)](PAPER_DRAFT.md#9-limitations) as
    the "Future Work" subsection. If accepted with major revisions,
    the VAT run is the obvious revision package; if accepted as is,
    the VAT run becomes the basis for the extension paper.

Recommendation: **Option B.** The current ChildPlay paper is
publishable on its own merits; running VAT in parallel as a
"v2 manuscript" is the cleanest division of labour.

---

## 9. Pointer to the adapter scaffold

[`scripts/adapt_external_dataset.py`](../scripts/adapt_external_dataset.py)
contains a thin scaffold that reads a parquet/CSV of
external-dataset annotations in a documented schema and emits the
ChildPlay-compatible per-clip annotation files our pipeline
expects. The script is intentionally minimal — fill in the
`extract_*` functions when the target dataset format is final.
