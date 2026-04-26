"""Build two 3-class aggregate CSVs:

1. output/all_predictions_3class.csv
   - 69 rows, one per child_id
   - final_manual_label_3class + 18 per-model prediction columns
     (label + optional confidence where available)

2. output/paper_eval/manual_eval_annotations_3class.csv
   - 69 rows with full metadata (source_video_url, bbox hint, etc.)
   - All 18 model predictions under 3-class schema
   - Empty annotator_1_label_3class / annotator_2_label_3class columns
     for a fresh labeling pass
   - suggested_reference_label_3class taken from current consensus
   - agreement_pattern_3class derived from all-model-agreement check

No file in this script touches existing 5-class CSVs; the 5-class
versions are preserved so the Phase A 5-class analysis remains
reproducible.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "3class_eval"
OUT_AGG = ROOT / "output" / "all_predictions_3class.csv"
OUT_MANUAL = ROOT / "output" / "paper_eval" / "manual_eval_annotations_3class.csv"


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(EVAL_DIR / "ground_truth_3class.csv")
    return gt.set_index("child_id")


def load_model_predictions() -> dict[str, pd.DataFrame]:
    """Return {model_name: DataFrame indexed on child_id} for every 3-class CSV."""
    preds: dict[str, pd.DataFrame] = {}

    fixed = {
        "rule_based": "rule_based_3class.csv",
        "nli_deberta": "nli_deberta_3class.csv",
        "nli_bart": "nli_bart_3class.csv",
        "nli_distilbert": "nli_distilbert_3class.csv",
    }
    for name, fname in fixed.items():
        df = pd.read_csv(EVAL_DIR / fname).set_index("child_id")
        preds[name] = df

    for sup_csv in sorted(EVAL_DIR.glob("supervised_*_3class.csv")):
        if sup_csv.stem.endswith("_summary_3class"):
            continue
        name = sup_csv.stem.replace("_3class", "")
        preds[name] = pd.read_csv(sup_csv).set_index("child_id")

    for llm_csv in sorted(EVAL_DIR.glob("llm_*_3class.csv")):
        name = llm_csv.stem.replace("_3class", "")
        preds[name] = pd.read_csv(llm_csv).set_index("child_id")

    return preds


def build_aggregate(gt: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = pd.DataFrame(index=gt.index)
    out.index.name = "child_id"
    out["clip_id"] = gt["clip_id"]
    out["final_manual_label_3class"] = gt["final_3class"]
    out["rater1_3class"] = gt["rater1_3"]
    out["rater2_3class"] = gt["rater2_3"]

    for name, df in preds.items():
        out[f"{name}_label"] = df["pred_3class"].reindex(gt.index)
        if "confidence" in df.columns:
            out[f"{name}_confidence"] = df["confidence"].reindex(gt.index)
    return out.reset_index()


def build_manual_eval(
    gt: pd.DataFrame,
    preds: dict[str, pd.DataFrame],
    existing_manual: pd.DataFrame,
) -> pd.DataFrame:
    """Master annotation file for a fresh 3-class labeling pass."""
    meta_cols = [
        "child_id",
        "clip_id",
        "split",
        "source_video_url",
        "target_person_and_time",
        "target_person_bbox_hint",
    ]
    out = existing_manual[meta_cols].copy()
    out = out.set_index("child_id")

    out["suggested_reference_label_3class"] = gt["final_3class"].reindex(out.index)

    # Per-model label columns.
    for name, df in preds.items():
        out[f"{name}_label_3class"] = df["pred_3class"].reindex(out.index)

    # Agreement pattern: do all 18 models agree with each other?
    pred_cols = [c for c in out.columns if c.endswith("_label_3class") and c != "suggested_reference_label_3class"]
    models_only = out[pred_cols]
    n_unique_per_row = models_only.nunique(axis=1)
    out["agreement_pattern_3class"] = n_unique_per_row.map(
        lambda n: "all_agree" if n == 1 else f"split_{n}_ways"
    )

    # Empty annotator columns to be filled in.
    out["annotator_1_label_3class"] = pd.NA
    out["annotator_2_label_3class"] = pd.NA
    out["final_label_3class"] = gt["final_3class"].reindex(out.index)
    out["notes"] = pd.NA

    return out.reset_index()


def main() -> None:
    gt = load_ground_truth()
    preds = load_model_predictions()
    print(f"Loaded {len(preds)} models, {len(gt)} samples.")

    agg = build_aggregate(gt, preds)
    agg.to_csv(OUT_AGG, index=False)
    print(f"Wrote {OUT_AGG} ({len(agg)} rows, {len(agg.columns)} cols)")

    existing_manual = pd.read_csv(ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv")
    manual = build_manual_eval(gt, preds, existing_manual)
    manual.to_csv(OUT_MANUAL, index=False)
    print(f"Wrote {OUT_MANUAL} ({len(manual)} rows, {len(manual.columns)} cols)")

    # Sanity: confirm no "exploratory" anywhere in either output.
    for path in (OUT_AGG, OUT_MANUAL):
        df = pd.read_csv(path)
        hits = 0
        for col in df.columns:
            if df[col].dtype == "object":
                hits += df[col].astype(str).str.contains("exploratory", case=False, na=False).sum()
        print(f"{path.name}: exploratory_hits={hits}")


if __name__ == "__main__":
    main()
