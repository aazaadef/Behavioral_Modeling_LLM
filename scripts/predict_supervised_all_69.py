"""Fit supervised baselines on all labeled samples and predict every
child_id in the behavioral features file (including unlabeled ones).

Unlike ``run_supervised_baselines.py`` which only produces out-of-fold
predictions for the labeled subset, this script trains each classifier
on ALL currently labeled samples and then predicts the label for every
of the 69 test sequences — so the combined predictions CSV can show a
supervised prediction for every row, not just the labeled ones.

For LogisticRegression and RandomForest a ``predict_proba`` confidence
(max class probability) is also written.  LinearSVC has no probability
output, so its confidence column stays empty.

Outputs:
    output/supervised_baselines/supervised_all69_predictions.csv
        columns: child_id, classifier, predicted_label, confidence

Usage:
    python scripts/predict_supervised_all_69.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.supervised_baselines import (
    FEATURE_COLUMNS,
    LABEL_NAMES,
    _build_classifiers,
    _load_features_and_labels,
)

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "output_experiments_v2" / "hf_comparison_v1" / "behavioral_features.jsonl",
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / "supervised_baselines" / "supervised_all69_predictions.csv",
    )
    args = parser.parse_args()

    import numpy as np

    # Training data: labeled subset.
    X_train, y_train, _, _ = _load_features_and_labels(args.features, args.annotations)
    logger.info("Training on %d labeled samples", len(y_train))

    # Prediction data: every child_id in the features file.
    all_feat: list[dict] = []
    with args.features.open("r", encoding="utf-8") as fh:
        for line in fh:
            all_feat.append(json.loads(line))
    logger.info("Predicting for %d total samples", len(all_feat))

    X_all = np.array([[float(row[c]) for c in FEATURE_COLUMNS] for row in all_feat])
    cids_all = [row["child_id"] for row in all_feat]

    classifiers = _build_classifiers()
    rows_out: list[dict[str, str]] = []

    for name, pipe in classifiers:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_all)

        confs: list[str] = [""] * len(preds)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            probas = pipe.predict_proba(X_all)
            max_p = probas.max(axis=1)
            confs = [f"{p:.4f}" for p in max_p]

        for cid, pred, conf in zip(cids_all, preds, confs):
            rows_out.append({
                "child_id": cid,
                "classifier": name,
                "predicted_label": pred,
                "confidence": conf,
            })
        logger.info("%s — predicted %d samples", name, len(preds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["child_id", "classifier", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
