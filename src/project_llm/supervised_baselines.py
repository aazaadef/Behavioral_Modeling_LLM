"""Supervised classification baselines using scikit-learn.

Trains Logistic Regression, Linear SVM, and Random Forest classifiers on
the 11 numerical features from :class:`BehavioralFeatures` to predict one
of the five behavioral attention labels.  Uses stratified k-fold
cross-validation when training data is limited (e.g. 69 labeled samples)
and reports per-fold and aggregate metrics (accuracy, macro-F1, per-class
accuracy).

These baselines serve as an upper-bound reference for what a simple
classifier can achieve on the extracted gaze features — without any LLM
or NLI reasoning.  If a lightweight classifier matches or outperforms
LLMs, that is itself a significant finding for the paper.
"""

from __future__ import annotations

import json
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# ── Feature columns used for classification ──────────────────────────
# These must match the numerical fields of BehavioralFeatures exactly.
FEATURE_COLUMNS: list[str] = [
    "observed_frames",
    "visible_ratio",
    "on_screen_ratio",
    "gaze_shift_ratio",
    "occlusion_ratio",
    "eyes_closed_ratio",
    "mean_head_area",
    "mean_head_motion",
    "mean_gaze_motion",
    "visible_gaze_fraction",
    "gaze_shift_events",
    "max_visible_streak",
    "max_occlusion_streak",
    "attention_stability_score",
]

# The five canonical behavioral-attention labels.
LABEL_NAMES: list[str] = [
    "focused_attention",
    "exploratory_attention",
    "occluded_attention",
    "reduced_visual_availability",
    "mixed_attention",
]


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ClassifierResult:
    """Aggregate result for one classifier across all CV folds."""

    classifier_name: str
    num_folds: int
    num_samples: int
    per_fold_accuracy: list[float]
    per_fold_macro_f1: list[float]
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_kappa: float
    # Aggregated predictions across all folds (each sample predicted once).
    all_true: list[str] = field(default_factory=list)
    all_pred: list[str] = field(default_factory=list)
    all_child_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "classifier_name": self.classifier_name,
            "num_folds": self.num_folds,
            "num_samples": self.num_samples,
            "per_fold_accuracy": self.per_fold_accuracy,
            "per_fold_macro_f1": self.per_fold_macro_f1,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.std_accuracy, 4),
            "mean_macro_f1": round(self.mean_macro_f1, 4),
            "std_macro_f1": round(self.std_macro_f1, 4),
            "mean_kappa": round(self.mean_kappa, 4),
        }


# ── Feature extraction helpers ───────────────────────────────────────


def _load_features_and_labels(
    features_path: Path,
    annotations_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load feature vectors and ground-truth labels from saved files.

    Only samples that have a non-empty ``final_label`` in the annotations
    CSV are included.  Returns (X, y, child_ids, clip_ids).
    """
    # Read behavioral features keyed by child_id.
    features_by_id: dict[str, dict[str, Any]] = {}
    with features_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            features_by_id[row["child_id"]] = row

    # Read manual annotations and filter to labeled rows.
    labeled_rows: list[dict[str, str]] = []
    with annotations_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = row.get("final_label", "").strip()
            if label and label in LABEL_NAMES:
                labeled_rows.append(row)

    if not labeled_rows:
        raise ValueError("No labeled samples found in annotations CSV")

    # Build aligned arrays.
    X_rows: list[list[float]] = []
    y_list: list[str] = []
    child_ids: list[str] = []
    clip_ids: list[str] = []

    for row in labeled_rows:
        cid = row["child_id"]
        if cid not in features_by_id:
            logger.warning("child_id %s has a label but no features — skipping", cid)
            continue
        feat = features_by_id[cid]
        X_rows.append([float(feat[col]) for col in FEATURE_COLUMNS])
        y_list.append(row["final_label"])
        child_ids.append(cid)
        clip_ids.append(row["clip_id"])

    return np.array(X_rows), np.array(y_list), child_ids, clip_ids


# ── Classifier definitions ───────────────────────────────────────────


def _build_classifiers() -> list[tuple[str, Pipeline]]:
    """Return a list of (name, sklearn_pipeline) pairs for evaluation.

    Each pipeline includes StandardScaler so that features with different
    magnitudes (e.g. mean_head_area vs visible_ratio) are handled fairly.
    """
    return [
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "LinearSVM",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LinearSVC(
                            max_iter=5000,
                            dual="auto",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=None,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
    ]


# ── Cross-validation runner ──────────────────────────────────────────


def run_supervised_cv(
    X: np.ndarray,
    y: np.ndarray,
    child_ids: list[str],
    n_folds: int = 5,
) -> list[ClassifierResult]:
    """Run stratified k-fold CV for all classifiers and return results.

    When fewer than ``n_folds`` samples exist per class the number of
    folds is automatically reduced to avoid sklearn errors.
    """
    # Determine safe number of folds based on smallest class count.
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())
    effective_folds = min(n_folds, min_class_count)
    if effective_folds < 2:
        effective_folds = 2
    if effective_folds != n_folds:
        logger.info(
            "Reduced CV folds from %d to %d (smallest class has %d samples)",
            n_folds,
            effective_folds,
            min_class_count,
        )

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
    classifiers = _build_classifiers()
    results: list[ClassifierResult] = []

    for clf_name, pipeline in classifiers:
        fold_acc: list[float] = []
        fold_f1: list[float] = []
        fold_kappa: list[float] = []
        all_true: list[str] = []
        all_pred: list[str] = []
        all_cids: list[str] = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_f1.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
            fold_kappa.append(cohen_kappa_score(y_test, y_pred))

            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())
            all_cids.extend([child_ids[i] for i in test_idx])

        results.append(
            ClassifierResult(
                classifier_name=clf_name,
                num_folds=effective_folds,
                num_samples=len(y),
                per_fold_accuracy=fold_acc,
                per_fold_macro_f1=fold_f1,
                mean_accuracy=float(np.mean(fold_acc)),
                std_accuracy=float(np.std(fold_acc)),
                mean_macro_f1=float(np.mean(fold_f1)),
                std_macro_f1=float(np.std(fold_f1)),
                mean_kappa=float(np.mean(fold_kappa)),
                all_true=all_true,
                all_pred=all_pred,
                all_child_ids=all_cids,
            )
        )
        logger.info(
            "%s — acc=%.4f±%.4f  macro-F1=%.4f±%.4f  kappa=%.4f",
            clf_name,
            results[-1].mean_accuracy,
            results[-1].std_accuracy,
            results[-1].mean_macro_f1,
            results[-1].std_macro_f1,
            results[-1].mean_kappa,
        )

    return results


# ── Output writers ───────────────────────────────────────────────────


def save_supervised_results(
    output_dir: Path,
    results: list[ClassifierResult],
    label_names: list[str] | None = None,
) -> None:
    """Write all supervised baseline outputs to *output_dir*.

    Produces:
    - ``supervised_summary.json`` — aggregate metrics per classifier
    - ``supervised_predictions.csv`` — per-sample predictions across folds
    - ``supervised_confusion_matrices.json`` — confusion matrices
    - ``supervised_classification_reports.json`` — sklearn classification reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if label_names is None:
        label_names = LABEL_NAMES

    # 1. Summary JSON.
    summary = {
        "classifiers": [r.to_dict() for r in results],
        "feature_columns": FEATURE_COLUMNS,
        "label_names": label_names,
    }
    (output_dir / "supervised_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Per-sample predictions CSV.
    with (output_dir / "supervised_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "child_id",
                "true_label",
                "classifier",
                "predicted_label",
                "correct",
            ],
        )
        writer.writeheader()
        for result in results:
            for cid, true, pred in zip(result.all_child_ids, result.all_true, result.all_pred):
                writer.writerow(
                    {
                        "child_id": cid,
                        "true_label": true,
                        "classifier": result.classifier_name,
                        "predicted_label": pred,
                        "correct": "yes" if true == pred else "no",
                    }
                )

    # 3. Confusion matrices.
    cm_data: dict[str, Any] = {}
    for result in results:
        present_labels = sorted(set(result.all_true) | set(result.all_pred))
        cm = confusion_matrix(result.all_true, result.all_pred, labels=present_labels)
        cm_data[result.classifier_name] = {
            "labels": present_labels,
            "matrix": cm.tolist(),
        }
    (output_dir / "supervised_confusion_matrices.json").write_text(
        json.dumps(cm_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. Classification reports.
    report_data: dict[str, Any] = {}
    for result in results:
        report_data[result.classifier_name] = classification_report(
            result.all_true,
            result.all_pred,
            zero_division=0,
            output_dict=True,
        )
    (output_dir / "supervised_classification_reports.json").write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Public entry point ───────────────────────────────────────────────


def run_supervised_baselines(
    features_path: Path,
    annotations_path: Path,
    output_dir: Path,
    n_folds: int = 5,
) -> list[ClassifierResult]:
    """End-to-end supervised baseline evaluation.

    Loads features and labels, runs cross-validated classifiers, saves
    all outputs, and returns the list of :class:`ClassifierResult` objects.
    """
    X, y, child_ids, clip_ids = _load_features_and_labels(features_path, annotations_path)
    logger.info("Loaded %d labeled samples with %d features each", len(y), X.shape[1])

    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info("  %s: %d samples", label, count)

    results = run_supervised_cv(X, y, child_ids, n_folds=n_folds)
    save_supervised_results(output_dir, results)
    return results
