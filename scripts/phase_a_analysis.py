"""Phase A: Complete statistical analysis pipeline for Q1 paper readiness.

Produces every analysis required to upgrade the benchmark report to a
scientific paper:

1. Confusion matrices for every model
2. Per-class precision/recall/F1
3. Balanced accuracy + weighted-F1
4. Bootstrap 95% confidence intervals (accuracy, kappa)
5. McNemar pairwise significance tests
6. Error analysis (all-wrong samples, per-class errors, duration correlation)
7. Ablation study over feature groups
8. Additional baselines: XGBoost, LightGBM, Dummy
9. 3-class merged evaluation for robustness
10. RandomForest feature importance

All outputs are written to output/phase_a_analysis/.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.supervised_baselines import (
    FEATURE_COLUMNS,
    LABEL_NAMES,
    _load_features_and_labels,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output" / "phase_a_analysis"

# ── Feature groups for ablation ──────────────────────────────────────
FEATURE_GROUPS: dict[str, list[str]] = {
    "visibility": [
        "visible_ratio", "on_screen_ratio",
        "visible_gaze_fraction", "max_visible_streak",
    ],
    "motion": ["mean_head_motion", "mean_gaze_motion"],
    "shifts": ["gaze_shift_ratio", "gaze_shift_events"],
    "occlusion": [
        "occlusion_ratio", "eyes_closed_ratio", "max_occlusion_streak",
    ],
    "temporal": ["observed_frames", "attention_stability_score"],
    "head": ["mean_head_area"],
}

# ── 3-class merged label mapping ─────────────────────────────────────
LABEL_MERGE_3CLASS: dict[str, str] = {
    "focused_attention": "focused",
    "mixed_attention": "mixed",
    "exploratory_attention": "other",
    "occluded_attention": "other",
    "reduced_visual_availability": "other",
}


# ── Data loading ─────────────────────────────────────────────────────

def load_ground_truth(annotations_path: Path) -> dict[str, str]:
    gt: dict[str, str] = {}
    with annotations_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("final_label") or "").strip()
            if label:
                gt[row["child_id"]] = label
    return gt


def load_all_predictions() -> dict[str, dict[str, str]]:
    models: dict[str, dict[str, str]] = {}

    rb = ROOT / "output" / "benchmark_rule_based" / "rule-based" / "interpretations.jsonl"
    if rb.exists():
        models["rule_based"] = _load_jsonl_labels(rb, "interaction_type")

    zs = {
        "distilbert": ROOT / "output" / "benchmark_zero_shot" / "zero-shot__typeform_distilbert-base-uncased-mnli" / "interpretations.jsonl",
        "bart": ROOT / "output" / "benchmark_bart_large_mnli" / "zero-shot__facebook_bart-large-mnli" / "interpretations.jsonl",
        "deberta": ROOT / "output" / "benchmark_deberta_zeroshot" / "zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0" / "interpretations.jsonl",
    }
    for name, path in zs.items():
        if path.exists():
            models[name] = _load_jsonl_labels(path, "interaction_type")

    llm_dir = ROOT / "output" / "llm_runs"
    if llm_dir.exists():
        for run in sorted(llm_dir.iterdir()):
            p = run / "parsed_outputs.jsonl"
            if p.exists():
                models[f"llm_{run.name}"] = _load_jsonl_labels(p, "label")

    # Use out-of-fold CV predictions (supervised_predictions.csv) — NOT
    # the train-on-all/predict-on-all file which leaks the training labels
    # into evaluation and trivially yields 100% for RandomForest.
    sup_oof = ROOT / "output" / "supervised_baselines" / "supervised_predictions.csv"
    if sup_oof.exists():
        for clf, preds in _load_supervised_oof(sup_oof).items():
            models[f"supervised_{clf}"] = preds

    return models


def _load_jsonl_labels(path: Path, key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["child_id"]] = row[key]
    return out


def _load_supervised_oof(path: Path) -> dict[str, dict[str, str]]:
    by_clf: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            by_clf.setdefault(row["classifier"], {})[row["child_id"]] = row["predicted_label"]
    return by_clf


# ── Analysis 1: Confusion matrices + per-class metrics ───────────────

def compute_confusion_and_per_class(
    models: dict[str, dict[str, str]],
    gt: dict[str, str],
) -> dict[str, Any]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    child_ids = sorted(gt.keys())
    y_true = [gt[cid] for cid in child_ids]
    label_set = sorted(set(y_true) | {lbl for m in models.values() for lbl in m.values()})

    results: dict[str, Any] = {}
    for name, preds in sorted(models.items()):
        y_pred = [preds.get(cid, "") for cid in child_ids]
        cm = confusion_matrix(y_true, y_pred, labels=label_set)
        report = classification_report(
            y_true, y_pred, labels=label_set, zero_division=0, output_dict=True,
        )
        results[name] = {
            "confusion_labels": label_set,
            "confusion_matrix": cm.tolist(),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "classification_report": report,
        }
    return results


# ── Analysis 2: Bootstrap CI ─────────────────────────────────────────

def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accs, kappas, macro_f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        accs.append(accuracy_score(yt, yp))
        # Guard against degenerate samples with single label.
        try:
            kappas.append(cohen_kappa_score(yt, yp))
        except Exception:
            kappas.append(np.nan)
        macro_f1s.append(f1_score(yt, yp, average="macro", zero_division=0))

    def ci(values):
        values = np.array([v for v in values if not np.isnan(v)])
        lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return round(float(np.mean(values)), 4), round(float(lo), 4), round(float(hi), 4)

    return {
        "accuracy": ci(accs),
        "kappa": ci(kappas),
        "macro_f1": ci(macro_f1s),
    }


def compute_bootstrap_all(
    models: dict[str, dict[str, str]],
    gt: dict[str, str],
) -> dict[str, Any]:
    child_ids = sorted(gt.keys())
    y_true = [gt[cid] for cid in child_ids]
    out: dict[str, Any] = {}
    for name, preds in sorted(models.items()):
        y_pred = [preds.get(cid, "") for cid in child_ids]
        out[name] = bootstrap_ci(y_true, y_pred)
    return out


# ── Analysis 3: McNemar significance test ────────────────────────────

def mcnemar_test(
    y_true: list[str],
    y_pred_a: list[str],
    y_pred_b: list[str],
) -> dict[str, float]:
    """McNemar's test for paired model predictions."""
    from scipy.stats import binomtest

    b = sum(1 for t, a, c in zip(y_true, y_pred_a, y_pred_b) if a == t and c != t)
    c = sum(1 for t, a, c in zip(y_true, y_pred_a, y_pred_b) if a != t and c == t)

    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "note": "models agree on all samples"}

    # Exact binomial test (robust for small n).
    result = binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided")
    return {"b": b, "c": c, "p_value": round(float(result.pvalue), 6)}


def compute_pairwise_mcnemar(
    models: dict[str, dict[str, str]],
    gt: dict[str, str],
) -> list[dict[str, Any]]:
    child_ids = sorted(gt.keys())
    y_true = [gt[cid] for cid in child_ids]
    names = sorted(models.keys())
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            ya = [models[a].get(cid, "") for cid in child_ids]
            yb = [models[b_name].get(cid, "") for cid in child_ids]
            test = mcnemar_test(y_true, ya, yb)
            rows.append({
                "model_a": a,
                "model_b": b_name,
                "b_only_a_correct": test["b"],
                "c_only_b_correct": test["c"],
                "p_value": test["p_value"],
                "significant_at_0.05": test["p_value"] < 0.05,
            })
    return rows


# ── Analysis 4: Error analysis ───────────────────────────────────────

def error_analysis(
    models: dict[str, dict[str, str]],
    gt: dict[str, str],
) -> dict[str, Any]:
    child_ids = sorted(gt.keys())
    model_names = sorted(models.keys())

    per_sample_errors: list[dict[str, Any]] = []
    for cid in child_ids:
        true_label = gt[cid]
        correct_count = 0
        wrong_models: list[str] = []
        predictions: dict[str, str] = {}
        for m in model_names:
            pred = models[m].get(cid, "")
            predictions[m] = pred
            if pred == true_label:
                correct_count += 1
            else:
                wrong_models.append(m)
        per_sample_errors.append({
            "child_id": cid,
            "true_label": true_label,
            "num_correct": correct_count,
            "num_wrong": len(wrong_models),
            "difficulty": "hard" if correct_count <= 3 else ("medium" if correct_count <= 7 else "easy"),
            "wrong_models": wrong_models,
            "predictions": predictions,
        })

    all_wrong = [r for r in per_sample_errors if r["num_correct"] == 0]
    all_right = [r for r in per_sample_errors if r["num_correct"] == len(model_names)]

    # Per-class error rate per model.
    per_class_errors: dict[str, dict[str, dict[str, int]]] = {}
    for m in model_names:
        by_true: dict[str, dict[str, int]] = {}
        for cid in child_ids:
            true_label = gt[cid]
            pred = models[m].get(cid, "")
            by_true.setdefault(true_label, {"total": 0, "wrong": 0})
            by_true[true_label]["total"] += 1
            if pred != true_label:
                by_true[true_label]["wrong"] += 1
        per_class_errors[m] = by_true

    return {
        "num_samples": len(child_ids),
        "num_models": len(model_names),
        "hard_samples_all_wrong": len(all_wrong),
        "easy_samples_all_right": len(all_right),
        "all_wrong_details": all_wrong,
        "per_sample_errors": per_sample_errors,
        "per_class_error_rates": per_class_errors,
    }


# ── Analysis 5: Ablation study ───────────────────────────────────────

def run_ablation(
    features_path: Path,
    annotations_path: Path,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    # Load features + labels (labeled subset only).
    X_full, y, child_ids, _ = _load_features_and_labels(features_path, annotations_path)

    # Run LinearSVM with all features then with each group removed.
    def _evaluate(cols: list[str]) -> dict[str, float]:
        col_idx = [FEATURE_COLUMNS.index(c) for c in cols]
        X_sub = X_full[:, col_idx]

        unique, counts = np.unique(y, return_counts=True)
        n_folds = min(5, int(counts.min()))
        if n_folds < 2:
            n_folds = 2

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accs, kappas, f1s = [], [], []
        for tr, te in skf.split(X_sub, y):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(max_iter=5000, dual="auto", random_state=42)),
            ])
            pipe.fit(X_sub[tr], y[tr])
            yp = pipe.predict(X_sub[te])
            accs.append(accuracy_score(y[te], yp))
            kappas.append(cohen_kappa_score(y[te], yp))
            f1s.append(f1_score(y[te], yp, average="macro", zero_division=0))
        return {
            "accuracy": round(float(np.mean(accs)), 4),
            "kappa": round(float(np.mean(kappas)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "num_features": len(cols),
            "feature_set": cols,
        }

    results: dict[str, Any] = {
        "full": _evaluate(FEATURE_COLUMNS),
    }
    for group_name, group_cols in FEATURE_GROUPS.items():
        remaining = [c for c in FEATURE_COLUMNS if c not in group_cols]
        results[f"without_{group_name}"] = _evaluate(remaining)

    # Each group alone.
    for group_name, group_cols in FEATURE_GROUPS.items():
        results[f"only_{group_name}"] = _evaluate(group_cols)

    return results


# ── Analysis 6: Additional baselines ─────────────────────────────────

def run_additional_baselines(
    features_path: Path,
    annotations_path: Path,
) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    X_full, y, child_ids, _ = _load_features_and_labels(features_path, annotations_path)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    unique, counts = np.unique(y, return_counts=True)
    n_folds = min(5, int(counts.min()))
    if n_folds < 2:
        n_folds = 2

    classifiers: list[tuple[str, Any]] = [
        ("Dummy_majority", DummyClassifier(strategy="most_frequent", random_state=42)),
        ("Dummy_stratified", DummyClassifier(strategy="stratified", random_state=42)),
        ("XGBoost", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                objective="multi:softmax", eval_metric="mlogloss",
                verbosity=0, random_state=42,
            )),
        ])),
        ("LightGBM", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42, verbose=-1,
            )),
        ])),
    ]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results: dict[str, Any] = {}

    for name, clf in classifiers:
        # XGBoost requires contiguous label ids starting from 0 on the
        # training fold — fit a fold-local LabelEncoder so folds that miss
        # a rare class (e.g. exploratory_attention n=1) don't crash.
        use_encoded = name in {"XGBoost", "LightGBM"}
        all_true, all_pred, all_cids = [], [], []
        accs, kappas, f1s = [], [], []
        for tr, te in skf.split(X_full, y):
            target_te = y[te]
            from sklearn.base import clone
            clf_copy = clone(clf)
            if use_encoded:
                le_fold = LabelEncoder()
                target_tr = le_fold.fit_transform(y[tr])
                clf_copy.fit(X_full[tr], target_tr)
                pred_raw = clf_copy.predict(X_full[te])
                pred = le_fold.inverse_transform(pred_raw)
            else:
                clf_copy.fit(X_full[tr], y[tr])
                pred = clf_copy.predict(X_full[te])
            all_true.extend(target_te.tolist())
            all_pred.extend(pred.tolist())
            all_cids.extend([child_ids[i] for i in te])
            accs.append(accuracy_score(target_te, pred))
            kappas.append(cohen_kappa_score(target_te, pred))
            f1s.append(f1_score(target_te, pred, average="macro", zero_division=0))

        results[name] = {
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_macro_f1": round(float(np.mean(f1s)), 4),
            "mean_kappa": round(float(np.mean(kappas)), 4),
            "num_folds": n_folds,
            "all_child_ids": all_cids,
            "all_true": all_true,
            "all_pred": all_pred,
            "classification_report": classification_report(
                all_true, all_pred, zero_division=0, output_dict=True,
            ),
        }

    return results


# ── Analysis 7: 3-class merged evaluation ────────────────────────────

def merged_3class_evaluation(
    models: dict[str, dict[str, str]],
    gt: dict[str, str],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
    )

    child_ids = sorted(gt.keys())
    y_true = [LABEL_MERGE_3CLASS.get(gt[cid], gt[cid]) for cid in child_ids]

    out: dict[str, Any] = {}
    for name, preds in sorted(models.items()):
        y_pred = [LABEL_MERGE_3CLASS.get(preds.get(cid, ""), preds.get(cid, "")) for cid in child_ids]
        out[name] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        }
    return out


# ── Analysis 8: Feature importance ───────────────────────────────────

def feature_importance_analysis(
    features_path: Path,
    annotations_path: Path,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _, _ = _load_features_and_labels(features_path, annotations_path)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, random_state=42, class_weight="balanced",
        )),
    ])
    pipe.fit(X, y)
    rf = pipe.named_steps["clf"]

    # Impurity-based.
    importances = rf.feature_importances_.tolist()

    # Permutation-based (more reliable).
    perm = permutation_importance(pipe, X, y, n_repeats=30, random_state=42)

    return {
        "feature_names": FEATURE_COLUMNS,
        "impurity_importance": [round(v, 4) for v in importances],
        "permutation_importance_mean": [round(v, 4) for v in perm.importances_mean.tolist()],
        "permutation_importance_std": [round(v, 4) for v in perm.importances_std.tolist()],
        "ranked_by_permutation": sorted(
            zip(FEATURE_COLUMNS, perm.importances_mean.tolist()),
            key=lambda x: -x[1],
        ),
    }


# ── Output writers ───────────────────────────────────────────────────

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


def write_summary_csvs(output_dir: Path, all_results: dict[str, Any]) -> None:
    # Confusion matrix CSVs (one per model).
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    for name, res in all_results["confusion_and_per_class"].items():
        labels = res["confusion_labels"]
        matrix = res["confusion_matrix"]
        with (cm_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["true\\predicted"] + labels)
            for i, row in enumerate(matrix):
                w.writerow([labels[i]] + row)

    # Per-class metrics CSV.
    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model", "class", "precision", "recall", "f1", "support",
        ])
        for name, res in all_results["confusion_and_per_class"].items():
            report = res["classification_report"]
            for cls, metrics in report.items():
                if cls in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                if isinstance(metrics, dict):
                    writer.writerow([
                        name, cls,
                        round(metrics["precision"], 4),
                        round(metrics["recall"], 4),
                        round(metrics["f1-score"], 4),
                        int(metrics["support"]),
                    ])

    # Bootstrap CI CSV.
    with (output_dir / "bootstrap_ci.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model",
            "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high",
            "kappa_mean", "kappa_ci_low", "kappa_ci_high",
            "macro_f1_mean", "macro_f1_ci_low", "macro_f1_ci_high",
        ])
        for name, ci in all_results["bootstrap_ci"].items():
            writer.writerow([
                name,
                *ci["accuracy"], *ci["kappa"], *ci["macro_f1"],
            ])

    # McNemar CSV.
    with (output_dir / "mcnemar_pairwise.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "model_a", "model_b", "b_only_a_correct",
            "c_only_b_correct", "p_value", "significant_at_0.05",
        ])
        writer.writeheader()
        for row in all_results["mcnemar"]:
            writer.writerow(row)

    # Ablation CSV.
    with (output_dir / "ablation.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "num_features", "accuracy", "kappa", "macro_f1"])
        for cfg, m in all_results["ablation"].items():
            writer.writerow([cfg, m["num_features"], m["accuracy"], m["kappa"], m["macro_f1"]])

    # Additional baselines CSV.
    with (output_dir / "additional_baselines.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["classifier", "accuracy", "macro_f1", "kappa", "num_folds"])
        for name, m in all_results["additional_baselines"].items():
            writer.writerow([
                name, m["mean_accuracy"], m["mean_macro_f1"],
                m["mean_kappa"], m["num_folds"],
            ])

    # 3-class merged CSV.
    with (output_dir / "merged_3class_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model", "accuracy", "balanced_accuracy",
            "macro_f1", "weighted_f1", "kappa",
        ])
        for name, m in all_results["merged_3class"].items():
            writer.writerow([
                name, m["accuracy"], m["balanced_accuracy"],
                m["macro_f1"], m["weighted_f1"], m["kappa"],
            ])

    # Feature importance CSV.
    fi = all_results["feature_importance"]
    with (output_dir / "feature_importance.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "impurity_importance", "permutation_mean", "permutation_std"])
        for i, f in enumerate(fi["feature_names"]):
            writer.writerow([
                f,
                fi["impurity_importance"][i],
                fi["permutation_importance_mean"][i],
                fi["permutation_importance_std"][i],
            ])

    # Per-sample errors CSV.
    with (output_dir / "per_sample_errors.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "child_id", "true_label", "num_correct",
            "num_wrong", "difficulty", "wrong_models",
        ])
        for row in all_results["error_analysis"]["per_sample_errors"]:
            writer.writerow([
                row["child_id"], row["true_label"], row["num_correct"],
                row["num_wrong"], row["difficulty"],
                "|".join(row["wrong_models"]),
            ])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotations_path = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"
    features_path = ROOT / "output_experiments_v2" / "hf_comparison_v1" / "behavioral_features.jsonl"

    gt = load_ground_truth(annotations_path)
    models = load_all_predictions()
    logger.info("Loaded %d ground-truth labels and %d models", len(gt), len(models))

    all_results: dict[str, Any] = {}

    logger.info("[1/8] Confusion matrices + per-class metrics")
    all_results["confusion_and_per_class"] = compute_confusion_and_per_class(models, gt)

    logger.info("[2/8] Bootstrap 95% CI (1000 iterations)")
    all_results["bootstrap_ci"] = compute_bootstrap_all(models, gt)

    logger.info("[3/8] McNemar pairwise significance tests")
    all_results["mcnemar"] = compute_pairwise_mcnemar(models, gt)

    logger.info("[4/8] Error analysis")
    all_results["error_analysis"] = error_analysis(models, gt)

    logger.info("[5/8] Ablation study (feature groups)")
    all_results["ablation"] = run_ablation(features_path, annotations_path)

    logger.info("[6/8] Additional baselines (XGBoost, LightGBM, Dummy)")
    all_results["additional_baselines"] = run_additional_baselines(features_path, annotations_path)

    logger.info("[7/8] 3-class merged evaluation")
    all_results["merged_3class"] = merged_3class_evaluation(models, gt)

    logger.info("[8/8] Feature importance (RandomForest)")
    all_results["feature_importance"] = feature_importance_analysis(features_path, annotations_path)

    logger.info("Writing outputs to %s", OUTPUT_DIR)
    write_json(OUTPUT_DIR / "all_results.json", all_results)
    write_summary_csvs(OUTPUT_DIR, all_results)

    print(f"\n{'='*80}\nPhase A Analysis Complete\n{'='*80}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files written:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            print(f"  {f.name}")
        elif f.is_dir():
            print(f"  {f.name}/ ({len(list(f.glob('*')))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
