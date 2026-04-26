"""Phase-A style statistical analysis on 3-class predictions.

Reads the prediction CSVs produced by scripts/run_3class_pipeline.py
(rule-based, NLI, supervised) and scripts/run_llm_3class.py (LLMs),
aligns them to the 3-class consensus ground truth, and runs:

  - Bootstrap 95% CI (1000 iterations) for accuracy / kappa / macro-F1
  - Pairwise McNemar significance tests (45 comparisons for 10 models)
  - Per-class precision/recall/F1
  - Per-sample difficulty (correctness across all models)
  - RandomForest (3-class) feature importance — impurity + permutation

Outputs land in output/3class_eval/phase_a/.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.labels_3class import LABELS_3CLASS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

EVAL_DIR = ROOT / "output" / "3class_eval"
OUT_DIR = EVAL_DIR / "phase_a"


# ═════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════

def load_predictions() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load GT and every available model prediction keyed by child_id."""
    gt = pd.read_csv(EVAL_DIR / "ground_truth_3class.csv").set_index("child_id")
    predictions: dict[str, pd.Series] = {}

    fixed_files = {
        "rule_based": "rule_based_3class.csv",
        "deberta": "nli_deberta_3class.csv",
        "bart": "nli_bart_3class.csv",
        "distilbert": "nli_distilbert_3class.csv",
    }
    for name, fname in fixed_files.items():
        path = EVAL_DIR / fname
        if path.exists():
            df = pd.read_csv(path).set_index("child_id")
            predictions[name] = df["pred_3class"].reindex(gt.index)

    # Supervised predictions — discover every supervised_*_3class.csv.
    # Excludes summary files which don't carry per-sample predictions.
    for sup_csv in sorted(EVAL_DIR.glob("supervised_*_3class.csv")):
        if sup_csv.stem.endswith("_summary_3class"):
            continue
        name = sup_csv.stem.replace("_3class", "")
        df = pd.read_csv(sup_csv).set_index("child_id")
        predictions[name] = df["pred_3class"].reindex(gt.index)

    # LLM predictions — discover any llm_*_3class.csv file.
    for llm_csv in sorted(EVAL_DIR.glob("llm_*_3class.csv")):
        name = llm_csv.stem  # e.g. llm__models_Qwen--Qwen2_5-72B-Instruct_3class
        df = pd.read_csv(llm_csv).set_index("child_id")
        predictions[name] = df["pred_3class"].reindex(gt.index)

    logger.info("Loaded %d models", len(predictions))
    return gt, predictions


# ═════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, f1_score, balanced_accuracy_score,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS_3CLASS)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=LABELS_3CLASS)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=LABELS_3CLASS)),
    }


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, kappas, f1s = [], [], []
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, f1_score,
    )
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        kappas.append(cohen_kappa_score(yt, yp, labels=LABELS_3CLASS))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0, labels=LABELS_3CLASS))
    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_ci": ci(accs),
        "kappa_mean": float(np.mean(kappas)),
        "kappa_ci": ci(kappas),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_ci": ci(f1s),
    }


def mcnemar_pair(y_true: np.ndarray, yp_a: np.ndarray, yp_b: np.ndarray) -> dict:
    """Two-sided mid-p McNemar test with exact binomial."""
    from scipy.stats import binomtest

    a_right = yp_a == y_true
    b_right = yp_b == y_true
    b_count = int(np.sum(a_right & ~b_right))  # only A right
    c_count = int(np.sum(~a_right & b_right))  # only B right
    n = b_count + c_count
    if n == 0:
        return {"b": b_count, "c": c_count, "p": 1.0}
    p = float(binomtest(min(b_count, c_count), n, 0.5).pvalue)
    return {"b": b_count, "c": c_count, "p": p}


# ═════════════════════════════════════════════════════════════════════
# Per-class metrics and confusion matrices
# ═════════════════════════════════════════════════════════════════════

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_3CLASS, zero_division=0
    )
    return {
        lbl: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(s[i]),
        }
        for i, lbl in enumerate(LABELS_3CLASS)
    }


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_3CLASS)
    return pd.DataFrame(cm, index=LABELS_3CLASS, columns=LABELS_3CLASS)


# ═════════════════════════════════════════════════════════════════════
# Feature importance for 3-class RandomForest
# ═════════════════════════════════════════════════════════════════════

def rf_feature_importance(gt: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
    from project_llm.supervised_baselines import FEATURE_COLUMNS

    features_path = (
        ROOT / "output" / "benchmark_rule_based" / "rule-based"
        / "behavioral_features.jsonl"
    )
    feats_by_id: dict[str, dict] = {}
    with features_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            feats_by_id[r["child_id"]] = r
    X_rows, y_list = [], []
    for cid, row in gt.iterrows():
        feat = feats_by_id[cid]
        X_rows.append([float(feat[c]) for c in FEATURE_COLUMNS])
        y_list.append(row["final_3class"])
    X = np.array(X_rows)
    y = np.array(y_list)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    rf = RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight="balanced",
    ).fit(Xs, y)
    perm = permutation_importance(rf, Xs, y, n_repeats=30, random_state=42)
    df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "impurity_importance": rf.feature_importances_,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    return df


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt, predictions = load_predictions()
    y_true = gt["final_3class"].to_numpy()

    # Drop rows with any missing prediction
    all_rows = gt.index.tolist()
    for name, series in predictions.items():
        missing = series[series.isna()].index.tolist()
        if missing:
            logger.warning("%s has %d missing predictions; will skip those", name, len(missing))

    # Bootstrap CI + point metrics for every model.
    rows = []
    for name, pred in predictions.items():
        yp = pred.to_numpy()
        mask = ~pd.isna(pred)
        y_true_m = y_true[mask]
        yp_m = yp[mask]
        m = compute_metrics(y_true_m, yp_m)
        boot = bootstrap_ci(y_true_m, yp_m)
        rows.append({
            "model": name, "n": int(mask.sum()),
            **{k: round(v, 4) for k, v in m.items()},
            "accuracy_ci_low": round(boot["accuracy_ci"][0], 4),
            "accuracy_ci_high": round(boot["accuracy_ci"][1], 4),
            "kappa_ci_low": round(boot["kappa_ci"][0], 4),
            "kappa_ci_high": round(boot["kappa_ci"][1], 4),
            "macro_f1_ci_low": round(boot["macro_f1_ci"][0], 4),
            "macro_f1_ci_high": round(boot["macro_f1_ci"][1], 4),
        })
    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary.to_csv(OUT_DIR / "bootstrap_ci_3class.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))

    # McNemar pairwise tests.
    mcnemar_rows = []
    model_names = list(predictions.keys())
    for a, b in combinations(model_names, 2):
        ya = predictions[a].to_numpy()
        yb = predictions[b].to_numpy()
        mask = ~(pd.isna(predictions[a]) | pd.isna(predictions[b]))
        result = mcnemar_pair(y_true[mask], ya[mask], yb[mask])
        mcnemar_rows.append({
            "model_a": a, "model_b": b,
            "b_only_a_right": result["b"],
            "c_only_b_right": result["c"],
            "p_value": round(result["p"], 6),
            "significant": result["p"] < 0.05,
        })
    mc = pd.DataFrame(mcnemar_rows).sort_values("p_value")
    mc.to_csv(OUT_DIR / "mcnemar_pairwise_3class.csv", index=False)

    # Per-class metrics + confusion matrices.
    pc_rows = []
    cm_dir = OUT_DIR / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    for name, pred in predictions.items():
        mask = ~pd.isna(pred)
        pcm = per_class_metrics(y_true[mask], pred.to_numpy()[mask])
        for lbl, m in pcm.items():
            pc_rows.append({"model": name, "label": lbl, **{k: round(v, 4) for k, v in m.items()}})
        cm = confusion_matrix_df(y_true[mask], pred.to_numpy()[mask])
        cm.to_csv(cm_dir / f"{name}.csv")
    pd.DataFrame(pc_rows).to_csv(OUT_DIR / "per_class_metrics_3class.csv", index=False)

    # Per-sample correctness.
    psamp_rows = []
    for cid in gt.index:
        true_label = gt.loc[cid, "final_3class"]
        correct_count = 0
        preds_row = {}
        for name, pred in predictions.items():
            p = pred.loc[cid] if cid in pred.index else None
            preds_row[name] = p
            if p == true_label:
                correct_count += 1
        psamp_rows.append({
            "child_id": cid, "true_3class": true_label,
            "num_correct": correct_count,
            "num_models": len(predictions),
            **preds_row,
        })
    per_samp = pd.DataFrame(psamp_rows).sort_values("num_correct")
    per_samp.to_csv(OUT_DIR / "per_sample_errors_3class.csv", index=False)

    # Feature importance (RF 3-class).
    fi = rf_feature_importance(gt)
    fi.to_csv(OUT_DIR / "feature_importance_3class.csv", index=False)

    # All-in-one JSON.
    all_results = {
        "summary": summary.to_dict(orient="records"),
        "mcnemar": mc.to_dict(orient="records"),
        "per_class": pc_rows,
        "feature_importance": fi.to_dict(orient="records"),
        "n_samples": len(gt),
        "labels": LABELS_3CLASS,
        "label_distribution": dict(Counter(y_true.tolist())),
    }
    (OUT_DIR / "all_results_3class.json").write_text(
        json.dumps(all_results, indent=2, default=float, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Done. Outputs in %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
