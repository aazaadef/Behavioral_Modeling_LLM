"""Final comparison: evaluate all models against manual ground-truth labels.

Reads the completed manual_eval_annotations.csv (with final_label column)
and compares every available model's predictions against it.  Produces
a comprehensive comparison report including accuracy, macro-F1, Cohen's
kappa, per-class metrics, confusion matrices, and pairwise agreement
between all models.

This script is designed to be run AFTER:
    1. Manual labeling of all 69 test samples is complete
    2. All model inference runs have been saved
    3. Supervised baselines have been computed

It automatically discovers available model outputs from known directories.

Usage:
    python scripts/run_final_comparison.py
    python scripts/run_final_comparison.py --output-dir output/final_comparison_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.io_utils_v2 import iso_timestamp, write_json

logger = logging.getLogger(__name__)

# ── Canonical label space ────────────────────────────────────────────
ALLOWED_LABELS = [
    "focused_attention",
    "exploratory_attention",
    "occluded_attention",
    "reduced_visual_availability",
    "mixed_attention",
]


# ── Model discovery ──────────────────────────────────────────────────

def _discover_model_predictions() -> dict[str, dict[str, str]]:
    """Discover all available model predictions from known output paths.

    Returns a dict of {model_name: {child_id: predicted_label}}.
    """
    models: dict[str, dict[str, str]] = {}

    # 1. Rule-based baseline.
    rb_path = ROOT / "output" / "benchmark_rule_based" / "rule-based" / "interpretations.jsonl"
    if rb_path.exists():
        models["rule_based"] = _load_interpretations(rb_path)
        logger.info("Loaded rule_based: %d predictions", len(models["rule_based"]))

    # 2. Zero-shot NLI models.
    zs_paths = {
        "distilbert": ROOT / "output" / "benchmark_zero_shot" / "zero-shot__typeform_distilbert-base-uncased-mnli" / "interpretations.jsonl",
        "bart": ROOT / "output" / "benchmark_bart_large_mnli" / "zero-shot__facebook_bart-large-mnli" / "interpretations.jsonl",
        "deberta": ROOT / "output" / "benchmark_deberta_zeroshot" / "zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0" / "interpretations.jsonl",
    }
    for name, path in zs_paths.items():
        if path.exists():
            models[name] = _load_interpretations(path)
            logger.info("Loaded %s: %d predictions", name, len(models[name]))

    # 3. Local LLM runs (unified script outputs).
    llm_dir = ROOT / "output" / "llm_runs"
    if llm_dir.exists():
        for run_dir in sorted(llm_dir.iterdir()):
            parsed_path = run_dir / "parsed_outputs.jsonl"
            if parsed_path.exists():
                model_name = f"llm_{run_dir.name}"
                models[model_name] = _load_parsed_outputs(parsed_path)
                logger.info("Loaded %s: %d predictions", model_name, len(models[model_name]))

    # 4. Legacy HF-LLM runs (v5 Qwen with old label space — skip if labels don't match).
    hf_legacy = ROOT / "output_experiments_v2" / "hf_llm_local_runs_v5" / "parsed_outputs.jsonl"
    if hf_legacy.exists():
        preds = _load_parsed_outputs(hf_legacy)
        # Check if labels are in the correct space.
        sample_labels = set(preds.values())
        if sample_labels <= set(ALLOWED_LABELS):
            models["qwen7b_legacy"] = preds
            logger.info("Loaded qwen7b_legacy: %d predictions", len(preds))
        else:
            logger.warning(
                "Skipping qwen7b_legacy — label space mismatch: %s",
                sample_labels - set(ALLOWED_LABELS),
            )

    # 5. Supervised baselines (from CV predictions).
    sup_path = ROOT / "output" / "supervised_baselines" / "supervised_predictions.csv"
    if sup_path.exists():
        sup_models = _load_supervised_predictions(sup_path)
        for name, preds in sup_models.items():
            models[name] = preds
            logger.info("Loaded %s: %d predictions", name, len(preds))

    return models


def _load_interpretations(path: Path) -> dict[str, str]:
    """Load predictions from a V1-style interpretations.jsonl file."""
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            predictions[row["child_id"]] = row["interaction_type"]
    return predictions


def _load_parsed_outputs(path: Path) -> dict[str, str]:
    """Load predictions from a parsed_outputs.jsonl file (LLM runs)."""
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            predictions[row["child_id"]] = row["label"]
    return predictions


def _load_supervised_predictions(path: Path) -> dict[str, dict[str, str]]:
    """Load supervised baseline predictions grouped by classifier name."""
    by_clf: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            clf_name = f"supervised_{row['classifier']}"
            by_clf.setdefault(clf_name, {})[row["child_id"]] = row["predicted_label"]
    return by_clf


# ── Evaluation metrics ───────────────────────────────────────────────

def _compute_metrics(
    true_labels: list[str],
    pred_labels: list[str],
    label_names: list[str],
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, Cohen's kappa, and per-class metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    present_labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=present_labels)
    report = classification_report(
        true_labels, pred_labels, zero_division=0, output_dict=True
    )

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "cohen_kappa": round(kappa, 4),
        "num_samples": len(true_labels),
        "label_distribution_true": dict(Counter(true_labels)),
        "label_distribution_pred": dict(Counter(pred_labels)),
        "confusion_matrix_labels": present_labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _compute_pairwise_agreement(
    models: dict[str, dict[str, str]],
    child_ids: list[str],
) -> list[dict[str, Any]]:
    """Compute pairwise agreement rates between all model pairs."""
    names = sorted(models.keys())
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            matches = sum(
                1 for cid in child_ids
                if models[left].get(cid) == models[right].get(cid)
            )
            rows.append({
                "model_a": left,
                "model_b": right,
                "agreement": round(matches / len(child_ids), 4) if child_ids else 0.0,
                "matches": matches,
                "total": len(child_ids),
            })
    return rows


# ── Main comparison ──────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Final comparison of all models against manual labels")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "final_comparison")
    parser.add_argument("--annotations", type=Path, default=ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv")
    args = parser.parse_args()

    # Load ground-truth labels.
    if not args.annotations.exists():
        logger.error("Annotations file not found: %s", args.annotations)
        return 1

    ground_truth: dict[str, str] = {}
    all_child_ids: list[str] = []
    with args.annotations.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = row["child_id"]
            all_child_ids.append(cid)
            label = row.get("final_label", "").strip()
            if label and label in ALLOWED_LABELS:
                ground_truth[cid] = label

    logger.info("Ground truth: %d labeled out of %d total", len(ground_truth), len(all_child_ids))
    if not ground_truth:
        logger.error("No labeled samples found — complete manual labeling first!")
        return 1

    # Discover all available model predictions.
    models = _discover_model_predictions()
    if not models:
        logger.error("No model predictions found!")
        return 1

    logger.info("Found %d models: %s", len(models), ", ".join(sorted(models.keys())))

    # Evaluate each model against ground truth.
    labeled_ids = sorted(ground_truth.keys())
    model_results: dict[str, Any] = {}

    for model_name, predictions in sorted(models.items()):
        true_labels = []
        pred_labels = []
        for cid in labeled_ids:
            if cid in predictions:
                true_labels.append(ground_truth[cid])
                pred_labels.append(predictions[cid])

        if not true_labels:
            logger.warning("Model %s has no overlap with labeled samples — skipping", model_name)
            continue

        metrics = _compute_metrics(true_labels, pred_labels, ALLOWED_LABELS)
        model_results[model_name] = metrics
        logger.info(
            "%s — acc=%.4f  macro-F1=%.4f  kappa=%.4f  (n=%d)",
            model_name, metrics["accuracy"], metrics["macro_f1"],
            metrics["cohen_kappa"], metrics["num_samples"],
        )

    # Pairwise agreement (on all 69 samples, not just labeled ones).
    pairwise = _compute_pairwise_agreement(models, all_child_ids)

    # Save outputs.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.output_dir / "model_evaluation.json", model_results)
    write_json(args.output_dir / "pairwise_agreement.json", pairwise)
    write_json(args.output_dir / "comparison_summary.json", {
        "num_labeled_samples": len(ground_truth),
        "num_total_samples": len(all_child_ids),
        "num_models_evaluated": len(model_results),
        "models": sorted(model_results.keys()),
        "label_distribution_ground_truth": dict(Counter(ground_truth.values())),
        "timestamp": iso_timestamp(),
    })

    # Write ranking table as CSV.
    ranking = sorted(
        model_results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    with (args.output_dir / "model_ranking.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "rank", "model", "accuracy", "macro_f1", "cohen_kappa", "num_samples",
        ])
        writer.writeheader()
        for rank, (name, metrics) in enumerate(ranking, 1):
            writer.writerow({
                "rank": rank,
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "cohen_kappa": metrics["cohen_kappa"],
                "num_samples": metrics["num_samples"],
            })

    # Write pairwise agreement as CSV.
    with (args.output_dir / "pairwise_agreement.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model_a", "model_b", "agreement", "matches", "total"])
        writer.writeheader()
        for row in pairwise:
            writer.writerow(row)

    # Print summary table.
    print(f"\n{'='*80}")
    print(f"FINAL MODEL COMPARISON — {len(ground_truth)} labeled samples")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Model':<30}{'Accuracy':>10}{'Macro-F1':>10}{'Kappa':>10}")
    print("-" * 80)
    for rank, (name, metrics) in enumerate(ranking, 1):
        print(f"{rank:<6}{name:<30}{metrics['accuracy']:>10.4f}{metrics['macro_f1']:>10.4f}{metrics['cohen_kappa']:>10.4f}")
    print("-" * 80)
    print(f"Output saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
