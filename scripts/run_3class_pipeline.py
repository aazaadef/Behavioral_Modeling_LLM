"""Re-run every non-LLM component in the 3-class canonical schema.

Steps performed:

  1. Map two-annotator consensus ground truth to the 3-class schema and
     recompute inter-rater Cohen's kappa in the 3-class view.
  2. Re-run the rule-based classifier natively in 3-class.
  3. Re-run NLI (DeBERTa, BART, DistilBERT) with exactly 3 candidate
     labels.
  4. Retrain supervised baselines (LogReg / LinearSVM / RandomForest /
     XGBoost / LightGBM / Dummy) on the 3-class labels under out-of-fold
     stratified cross-validation.

LLM inference is run separately by scripts/run_llm_3class.py because it
needs the GPU and longer wall-clock time.

All outputs land in output/3class_eval/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.labels_3class import (
    LABELS_3CLASS,
    MAP_5_TO_3,
    map_5_to_3,
    rule_based_3class,
    nli_candidate_labels_3class,
    nli_hypothesis_template_3class,
)
from project_llm.supervised_baselines import FEATURE_COLUMNS

OUTPUT_DIR = ROOT / "output" / "3class_eval"
ANNOTATIONS_CSV = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"
TWO_RATER_CSV = ROOT / "output" / "all_predictions_with_two_annotators.csv"
FEATURES_JSONL = (
    ROOT / "output" / "benchmark_rule_based" / "rule-based" / "behavioral_features.jsonl"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Step 1 — Ground truth and inter-rater kappa in 3-class
# ═════════════════════════════════════════════════════════════════════

def step_1_ground_truth() -> pd.DataFrame:
    """Build the 3-class ground-truth table and inter-rater stats."""
    logger.info("Step 1: Ground truth + inter-rater kappa")

    two = pd.read_csv(TWO_RATER_CSV)
    # Map rater1, rater2, final_manual_label to 3-class.
    two["rater1_3"] = two["rater1"].apply(map_5_to_3)
    two["rater2_3"] = two["rater2"].apply(map_5_to_3)
    two["final_3class"] = two["final_manual_label"].apply(map_5_to_3)

    # Recompute kappa under 3-class. Use sklearn to match Phase-B.
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    r1 = two["rater1_3"].tolist()
    r2 = two["rater2_3"].tolist()
    kappa_3 = float(cohen_kappa_score(r1, r2))
    raw_agree = int(sum(a == b for a, b in zip(r1, r2)))
    n = len(two)

    label_counts = {
        "rater1": {k: int(v) for k, v in two["rater1_3"].value_counts().sort_index().items()},
        "rater2": {k: int(v) for k, v in two["rater2_3"].value_counts().sort_index().items()},
        "consensus": {k: int(v) for k, v in two["final_3class"].value_counts().sort_index().items()},
    }
    # Ensure all 3 labels appear, even if zero.
    for key in label_counts:
        for lbl in LABELS_3CLASS:
            label_counts[key].setdefault(lbl, 0)

    cm = confusion_matrix(r1, r2, labels=LABELS_3CLASS).tolist()

    stats = {
        "n": n,
        "raw_agreement": raw_agree,
        "raw_agreement_pct": round(raw_agree / n * 100, 2),
        "cohens_kappa_3class": round(kappa_3, 4),
        "label_counts": label_counts,
        "confusion_rater1_vs_rater2": {
            "labels": LABELS_3CLASS,
            "matrix": cm,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "inter_rater_3class.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Ground-truth CSV (one row per child_id).
    gt = two[["child_id", "clip_id", "rater1_3", "rater2_3", "final_3class"]].copy()
    gt.to_csv(OUTPUT_DIR / "ground_truth_3class.csv", index=False)

    logger.info(
        "  inter-rater kappa (3-class) = %.4f  | raw agreement = %d/%d (%.1f%%)",
        kappa_3, raw_agree, n, raw_agree / n * 100,
    )
    logger.info("  consensus counts = %s", label_counts["consensus"])
    return gt


# ═════════════════════════════════════════════════════════════════════
# Step 2 — Rule-based in 3-class
# ═════════════════════════════════════════════════════════════════════

def step_2_rule_based(gt: pd.DataFrame) -> pd.DataFrame:
    """Run the native 3-class rule-based classifier on all 69 samples."""
    logger.info("Step 2: Rule-based 3-class")

    from project_llm.features import BehavioralFeatures

    features_by_id: dict[str, BehavioralFeatures] = {}
    with FEATURES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            features_by_id[row["child_id"]] = BehavioralFeatures(**row)

    rows = []
    for _, r in gt.iterrows():
        feats = features_by_id[r["child_id"]]
        label, conf = rule_based_3class(feats)
        rows.append({
            "child_id": r["child_id"],
            "true_3class": r["final_3class"],
            "pred_3class": label,
            "confidence": conf,
            "correct": label == r["final_3class"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "rule_based_3class.csv", index=False)

    acc = df["correct"].mean()
    logger.info("  rule-based 3-class accuracy = %.4f (%d/%d)", acc, df["correct"].sum(), len(df))
    return df


# ═════════════════════════════════════════════════════════════════════
# Step 3 — NLI in 3-class
# ═════════════════════════════════════════════════════════════════════

def step_3_nli(gt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Re-run DeBERTa/BART/DistilBERT with exactly 3 candidate labels."""
    logger.info("Step 3: NLI 3-class")

    from transformers import pipeline
    from project_llm.features import BehavioralFeatures

    features_by_id: dict[str, BehavioralFeatures] = {}
    with FEATURES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            features_by_id[row["child_id"]] = BehavioralFeatures(**row)

    def premise(feats: BehavioralFeatures) -> str:
        return (
            "Child gaze behavior summary. "
            f"visible_ratio={feats.visible_ratio:.3f}, "
            f"gaze_shift_ratio={feats.gaze_shift_ratio:.3f}, "
            f"occlusion_ratio={feats.occlusion_ratio:.3f}, "
            f"eyes_closed_ratio={feats.eyes_closed_ratio:.3f}, "
            f"mean_gaze_motion={feats.mean_gaze_motion:.2f}, "
            f"attention_stability_score={feats.attention_stability_score:.3f}, "
            f"max_visible_streak={feats.max_visible_streak}."
        )

    models = {
        "deberta": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "bart": "facebook/bart-large-mnli",
        "distilbert": "typeform/distilbert-base-uncased-mnli",
    }
    candidate_labels = nli_candidate_labels_3class()
    template = nli_hypothesis_template_3class()

    outputs: dict[str, pd.DataFrame] = {}
    for short, hf_id in models.items():
        logger.info("  %s → %s", short, hf_id)
        clf = pipeline("zero-shot-classification", model=hf_id, device=0)
        rows = []
        for _, r in gt.iterrows():
            feats = features_by_id[r["child_id"]]
            result = clf(
                premise(feats),
                candidate_labels=candidate_labels,
                hypothesis_template=template,
                multi_label=False,
            )
            top_label = str(result["labels"][0])
            top_score = float(result["scores"][0])
            rows.append({
                "child_id": r["child_id"],
                "true_3class": r["final_3class"],
                "pred_3class": top_label,
                "confidence": round(top_score, 4),
                "correct": top_label == r["final_3class"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / f"nli_{short}_3class.csv", index=False)
        outputs[short] = df
        acc = df["correct"].mean()
        dist = dict(df["pred_3class"].value_counts())
        logger.info("    accuracy=%.4f  distribution=%s", acc, dist)
        # Free GPU between models.
        del clf
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return outputs


# ═════════════════════════════════════════════════════════════════════
# Step 4 — Supervised retrain on 3-class (OOF CV + additional baselines)
# ═════════════════════════════════════════════════════════════════════

def step_4_supervised(gt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Retrain supervised baselines on the 3-class labels using OOF CV."""
    logger.info("Step 4: Supervised retrain (3-class, OOF)")

    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import LinearSVC

    # Load features and assemble (X, y).
    features_by_id: dict[str, dict] = {}
    with FEATURES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            features_by_id[row["child_id"]] = row

    X_rows, y_list, cids = [], [], []
    for _, r in gt.iterrows():
        feat = features_by_id[r["child_id"]]
        X_rows.append([float(feat[c]) for c in FEATURE_COLUMNS])
        y_list.append(r["final_3class"])
        cids.append(r["child_id"])
    X = np.array(X_rows)
    y = np.array(y_list)

    # Pick fold count adaptive to smallest class.
    _, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min())
    n_folds = min(5, min_class)
    if n_folds < 2:
        n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    logger.info("  n_folds = %d (min class = %d)", n_folds, min_class)

    # Build classifier zoo.
    classifiers: dict[str, tuple[object, bool]] = {
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
            ]),
            False,  # needs_label_encode
        ),
        "LinearSVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(max_iter=5000, dual="auto", random_state=42)),
            ]),
            False,
        ),
        "RandomForest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced",
                )),
            ]),
            False,
        ),
        "Dummy_majority": (
            DummyClassifier(strategy="most_frequent", random_state=42),
            False,
        ),
        "Dummy_stratified": (
            DummyClassifier(strategy="stratified", random_state=42),
            False,
        ),
    }
    # Optional boosters.
    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = (
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            ),
            True,  # needs label encoding
        )
    except ImportError:
        logger.warning("  xgboost not installed — skipping")
    try:
        from lightgbm import LGBMClassifier
        classifiers["LightGBM"] = (
            LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1,
            ),
            False,
        )
    except ImportError:
        logger.warning("  lightgbm not installed — skipping")

    results: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for name, (model, needs_le) in classifiers.items():
        all_true: list[str] = []
        all_pred: list[str] = []
        all_cids: list[str] = []
        fold_acc, fold_f1, fold_kappa = [], [], []

        for tr, te in skf.split(X, y):
            clf = clone(model)
            if needs_le:
                le_fold = LabelEncoder()
                y_tr_enc = le_fold.fit_transform(y[tr])
                clf.fit(X[tr], y_tr_enc)
                pred_raw = clf.predict(X[te])
                pred = le_fold.inverse_transform(pred_raw)
            else:
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
            y_true_fold = y[te]
            all_true.extend(y_true_fold.tolist())
            all_pred.extend(pred.tolist())
            all_cids.extend([cids[i] for i in te])
            fold_acc.append(accuracy_score(y_true_fold, pred))
            fold_f1.append(f1_score(y_true_fold, pred, average="macro", zero_division=0))
            fold_kappa.append(cohen_kappa_score(y_true_fold, pred))

        df = pd.DataFrame({
            "child_id": all_cids,
            "true_3class": all_true,
            "pred_3class": all_pred,
            "correct": [t == p for t, p in zip(all_true, all_pred)],
        })
        df.to_csv(OUTPUT_DIR / f"supervised_{name}_3class.csv", index=False)
        results[name] = df

        acc_overall = accuracy_score(all_true, all_pred)
        f1_overall = f1_score(all_true, all_pred, average="macro", zero_division=0)
        kappa_overall = cohen_kappa_score(all_true, all_pred)
        summary_rows.append({
            "classifier": name,
            "num_samples": len(all_true),
            "accuracy": round(acc_overall, 4),
            "macro_f1": round(f1_overall, 4),
            "kappa": round(kappa_overall, 4),
            "per_fold_acc_mean": round(float(np.mean(fold_acc)), 4),
            "per_fold_acc_std": round(float(np.std(fold_acc)), 4),
        })
        logger.info(
            "  %-22s acc=%.4f  macroF1=%.4f  kappa=%.4f",
            name, acc_overall, f1_overall, kappa_overall,
        )

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "supervised_summary_3class.csv", index=False
    )
    return results


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = step_1_ground_truth()
    step_2_rule_based(gt)
    step_3_nli(gt)
    step_4_supervised(gt)
    logger.info("Done. Outputs in %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
