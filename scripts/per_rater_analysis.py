"""Per-rater agreement analysis for the 18-system 3-class evaluation.

Computes Cohen's kappa for every system against:
  * rater 1 only (rater1_3)
  * rater 2 only (rater2_3)
  * the consensus label (final_3class)

with paired bootstrap 95% confidence intervals (1000 iterations), and
analyzes the 12 samples where the two raters disagreed to see how each
system behaved in those ambiguity zones.

Outputs:
  output/3class_eval/phase_a/per_rater_kappa.csv
  output/3class_eval/phase_a/disagreement_zone_analysis.csv

The script does NOT regenerate any model predictions; it only re-uses
the saved per-system 3-class CSVs and the inter-rater ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "3class_eval"
PHASE_A_DIR = EVAL_DIR / "phase_a"
GT_PATH = EVAL_DIR / "ground_truth_3class.csv"

N_BOOTSTRAP = 1000
RNG_SEED = 42


def _bootstrap_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = N_BOOTSTRAP
) -> tuple[float, float]:
    """Return the 2.5th / 97.5th percentile of bootstrap kappa estimates."""
    rng = np.random.default_rng(RNG_SEED)
    samples = []
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        # Skip degenerate resamples where one rater has only one class.
        yt, yp = y_true[idx], y_pred[idx]
        if len(set(yt)) < 2 or len(set(yp)) < 2:
            continue
        samples.append(cohen_kappa_score(yt, yp))
    if not samples:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _system_name(csv_path: Path) -> str:
    name = csv_path.stem
    if name.endswith("_3class"):
        name = name[: -len("_3class")]
    return name


def load_model_predictions() -> dict[str, pd.DataFrame]:
    """Load every per-system 3-class CSV keyed by child_id."""
    out: dict[str, pd.DataFrame] = {}
    for csv in sorted(EVAL_DIR.glob("*_3class.csv")):
        # Skip the ground truth file and any per-family summary tables.
        if csv.name in {"ground_truth_3class.csv"} or "summary" in csv.stem:
            continue
        df = pd.read_csv(csv)
        if "pred_3class" not in df.columns or "child_id" not in df.columns:
            continue
        out[_system_name(csv)] = df.set_index("child_id")[["pred_3class"]]
    return out


def per_rater_kappa(gt: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the headline per-rater kappa table for every system."""
    rows = []
    for system, df in preds.items():
        merged = gt.join(df, how="inner")
        # Drop any rows where any of the four labels is missing.
        merged = merged.dropna(subset=["rater1_3", "rater2_3", "final_3class", "pred_3class"])
        y_pred = merged["pred_3class"].to_numpy()
        for ref_name, ref_col in [
            ("rater1", "rater1_3"),
            ("rater2", "rater2_3"),
            ("consensus", "final_3class"),
        ]:
            y_ref = merged[ref_col].to_numpy()
            kappa = cohen_kappa_score(y_ref, y_pred)
            ci_lo, ci_hi = _bootstrap_kappa(y_ref, y_pred)
            rows.append(
                {
                    "system": system,
                    "reference": ref_name,
                    "kappa": kappa,
                    "ci_lo_2_5": ci_lo,
                    "ci_hi_97_5": ci_hi,
                    "n_samples": len(merged),
                }
            )
    return pd.DataFrame(rows)


def disagreement_zone_table(gt: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each rater-disagreement sample, list every model's prediction.

    Adds counts of how many systems sided with rater1, with rater2,
    or produced a third label.
    """
    disagreements = gt[gt["rater1_3"] != gt["rater2_3"]].copy()
    rows = []
    for child_id, row in disagreements.iterrows():
        record = {
            "child_id": child_id,
            "clip_id": row["clip_id"],
            "rater1": row["rater1_3"],
            "rater2": row["rater2_3"],
            "consensus": row["final_3class"],
        }
        sided_r1 = sided_r2 = third = 0
        for system, df in preds.items():
            if child_id in df.index:
                pred = df.loc[child_id, "pred_3class"]
                record[system] = pred
                if pred == row["rater1_3"]:
                    sided_r1 += 1
                elif pred == row["rater2_3"]:
                    sided_r2 += 1
                else:
                    third += 1
            else:
                record[system] = ""
        record["sided_rater1"] = sided_r1
        record["sided_rater2"] = sided_r2
        record["third_label"] = third
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    PHASE_A_DIR.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT_PATH).set_index("child_id")
    preds = load_model_predictions()
    print(f"Loaded {len(preds)} system CSVs against {len(gt)} consensus rows.")

    kappa_df = per_rater_kappa(gt, preds)
    kappa_path = PHASE_A_DIR / "per_rater_kappa.csv"
    kappa_df.to_csv(kappa_path, index=False)
    print(f"Wrote {kappa_path} ({len(kappa_df)} rows)")

    dz_df = disagreement_zone_table(gt, preds)
    dz_path = PHASE_A_DIR / "disagreement_zone_analysis.csv"
    dz_df.to_csv(dz_path, index=False)
    print(f"Wrote {dz_path} ({len(dz_df)} disagreement samples)")

    # Quick console summary.
    pivot = kappa_df.pivot(index="system", columns="reference", values="kappa").round(3)
    pivot = pivot[["rater1", "rater2", "consensus"]].sort_values("consensus", ascending=False)
    print("\nPer-rater kappa (sorted by consensus kappa):")
    print(pivot.to_string())

    print("\nDisagreement zone per-system breakdown:")
    print(f"  Total samples where rater1 != rater2: {len(dz_df)}")
    if len(dz_df):
        sums = dz_df[["sided_rater1", "sided_rater2", "third_label"]].sum()
        total = sums.sum()
        for col in ["sided_rater1", "sided_rater2", "third_label"]:
            pct = 100.0 * sums[col] / total if total else 0
            print(f"  {col}: {sums[col]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
