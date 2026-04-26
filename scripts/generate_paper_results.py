"""Generate final paper results from manual annotations.

Loads the annotation CSV and subset predictions, merges model labels,
computes accuracy / per-class accuracy / confusion matrices on the
labeled subset, pairwise agreement on the full set, and writes all
metrics, tables, and mismatch analysis to ``output/paper_results/``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ANNOTATIONS_PATH = Path("output/paper_eval/manual_eval_annotations.csv")
SUBSET_PATH = Path("output/paper_eval/manual_eval_subset.json")
OUTPUT_DIR = Path("output/paper_results")

MODEL_COLUMNS = {
    "rule_based": "rule_based_label",
    "deberta": "deberta_label",
    "distilbert": "distilbert_label",
    "bart": "bart_label",
}
CONFIDENCE_COLUMNS = [
    "rule_based_confidence",
    "deberta_confidence",
]
EVAL_LABELS = [
    "focused_attention",
    "exploratory_attention",
    "occluded_attention",
    "no_attention",
]
LABEL_MAP = {
    "focused_attention": "focused_attention",
    "exploratory_attention": "exploratory_attention",
    "occluded_attention": "occluded_attention",
    "mixed_attention": "no_attention",
    "reduced_visual_availability": "no_attention",
    "no_attention": "no_attention",
}


def _load_annotations(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _load_subset_predictions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["child_id"], row["clip_id"]): row
        for row in rows
    }


def _normalize_label(label: str) -> str:
    if label not in LABEL_MAP:
        raise ValueError(f"Unsupported label encountered during evaluation: {label}")
    return LABEL_MAP[label]


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return round(sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / len(y_true), 4) if y_true else 0.0


def _per_class_accuracy(y_true: list[str], y_pred: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label in EVAL_LABELS:
        indices = [index for index, gold in enumerate(y_true) if gold == label]
        if not indices:
            result[label] = None
            continue
        correct = sum(1 for index in indices if y_pred[index] == y_true[index])
        result[label] = round(correct / len(indices), 4)
    return result


def _confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {
        gold: {pred: 0 for pred in EVAL_LABELS}
        for gold in EVAL_LABELS
    }
    for gold, pred in zip(y_true, y_pred):
        matrix[gold][pred] += 1
    return matrix


def _pairwise_agreement(rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    items = sorted(MODEL_COLUMNS.items())
    for index, (left_name, left_column) in enumerate(items):
        left_values = [row[left_column].strip() for row in rows if row.get(left_column, "").strip()]
        for right_name, right_column in items[index + 1 :]:
            shared = [
                (row[left_column].strip(), row[right_column].strip())
                for row in rows
                if row.get(left_column, "").strip() and row.get(right_column, "").strip()
            ]
            matches = sum(1 for left, right in shared if left == right)
            denominator = len(shared)
            result[f"{left_name}__vs__{right_name}"] = {
                "matches": matches,
                "denominator": denominator,
                "agreement": round(matches / denominator, 4) if denominator else 0.0,
            }
    return result


def main() -> int:
    rows, fieldnames = _load_annotations(ANNOTATIONS_PATH)
    if not rows:
        raise ValueError(f"No rows found in {ANNOTATIONS_PATH}")

    subset_predictions = _load_subset_predictions(SUBSET_PATH)
    csv_missing_model_columns = [column for column in MODEL_COLUMNS.values() if column not in fieldnames]
    subset_missing_model_columns = [
        column
        for column in MODEL_COLUMNS.values()
        if column not in next(iter(subset_predictions.values()))
    ]
    if csv_missing_model_columns and subset_missing_model_columns:
        raise ValueError(
            "Model prediction columns are missing from both the annotation CSV and manual_eval_subset.json: "
            + ", ".join(sorted(set(csv_missing_model_columns) & set(subset_missing_model_columns)))
        )

    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row["child_id"], row["clip_id"])
        merged = dict(row)
        if key not in subset_predictions:
            raise ValueError(f"Missing prediction row for annotated sample: {key[0]} / {key[1]}")
        for column in MODEL_COLUMNS.values():
            if column not in merged or not merged[column]:
                merged[column] = subset_predictions[key][column]
        for column in CONFIDENCE_COLUMNS:
            if column in subset_predictions[key] and (column not in merged or not merged[column]):
                merged[column] = str(subset_predictions[key][column])
        enriched_rows.append(merged)

    labeled_rows = [row for row in enriched_rows if row.get("final_label", "").strip()]
    invalid_gold = sorted({
        row["final_label"].strip()
        for row in labeled_rows
        if row["final_label"].strip() not in EVAL_LABELS
    })
    if invalid_gold:
        raise ValueError(f"Found unsupported final_label values: {invalid_gold}")

    missing_model_labels: dict[str, int] = {}
    for column in MODEL_COLUMNS.values():
        missing_count = sum(1 for row in enriched_rows if not row.get(column, "").strip())
        if missing_count:
            missing_model_labels[column] = missing_count
    if missing_model_labels:
        raise ValueError(f"Missing model labels after merge: {missing_model_labels}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_fieldnames = [
        "child_id",
        "clip_id",
        "split",
        "agreement_pattern",
        "suggested_reference_label",
        "final_label",
        "rule_based_label",
        "deberta_label",
        "distilbert_label",
        "bart_label",
        "rule_based_confidence",
        "deberta_confidence",
        "annotator_1_label",
        "annotator_2_label",
        "notes",
    ]
    with (OUTPUT_DIR / "manual_eval_merged.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in enriched_rows:
            writer.writerow({field: row.get(field, "") for field in merged_fieldnames})

    with (OUTPUT_DIR / "model_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["child_id", "clip_id", "model", "label"])
        for row in enriched_rows:
            for model_name, column in MODEL_COLUMNS.items():
                writer.writerow([row["child_id"], row["clip_id"], model_name, row[column].strip()])

    model_predictions_json = {
        model_name: [
            {
                "child_id": row["child_id"],
                "clip_id": row["clip_id"],
                "label": row[column].strip(),
            }
            for row in enriched_rows
        ]
        for model_name, column in MODEL_COLUMNS.items()
    }
    (OUTPUT_DIR / "model_predictions.json").write_text(
        json.dumps(model_predictions_json, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    pairwise_agreement = _pairwise_agreement(enriched_rows)

    metrics: dict[str, object] = {
        "total_rows": len(enriched_rows),
        "labeled_rows": len(labeled_rows),
        "unlabeled_rows": len(enriched_rows) - len(labeled_rows),
        "coverage_by_model": {
            model_name: {
                "predictions": sum(1 for row in enriched_rows if row[column].strip()),
                "missing_predictions": sum(1 for row in enriched_rows if not row[column].strip()),
            }
            for model_name, column in MODEL_COLUMNS.items()
        },
        "pairwise_agreement": pairwise_agreement,
    }
    confusion_matrices: dict[str, dict[str, dict[str, int]]] = {}

    if labeled_rows:
        y_true = [row["final_label"].strip() for row in labeled_rows]
        accuracy = {}
        per_class_accuracy = {}
        for model_name, column in MODEL_COLUMNS.items():
            raw_labels = [row[column].strip() for row in labeled_rows]
            y_pred = [_normalize_label(label) for label in raw_labels]
            accuracy[model_name] = _accuracy(y_true, y_pred)
            per_class_accuracy[model_name] = _per_class_accuracy(y_true, y_pred)
            confusion_matrices[model_name] = _confusion_matrix(y_true, y_pred)
        metrics["accuracy"] = accuracy
        metrics["per_class_accuracy"] = per_class_accuracy
        metrics["gold_label_counts"] = dict(Counter(y_true))

    disagreement_count = sum(1 for row in enriched_rows if row["agreement_pattern"] == "disagreement")
    agreement_count = len(enriched_rows) - disagreement_count
    metrics["agreement_stats"] = {
        "total_samples": len(enriched_rows),
        "disagreement_count": disagreement_count,
        "agreement_count": agreement_count,
        "disagreement_percentage": round(disagreement_count / len(enriched_rows), 4),
    }

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "pairwise_agreement.json").write_text(
        json.dumps(pairwise_agreement, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    ranking = []
    if "accuracy" in metrics:
        ranking = sorted(metrics["accuracy"].items(), key=lambda item: (-item[1], item[0]))  # type: ignore[arg-type]
        with (OUTPUT_DIR / "accuracy_table.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Model", "Accuracy", "Labeled Denominator"])
            for model_name, value in ranking:
                writer.writerow([model_name, f"{value:.4f}", len(labeled_rows)])

        table_lines = [
            "Model                     Accuracy   LabeledDenominator",
            "------------------------------------------------------",
        ]
        for model_name, value in ranking:
            table_lines.append(f"{model_name:<25}{value:.4f}     {len(labeled_rows)}")
        (OUTPUT_DIR / "paper_ready_table.txt").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    mismatch_rows: list[dict[str, str]] = []
    for row in labeled_rows:
        normalized = {
            model_name: _normalize_label(row[column].strip())
            for model_name, column in MODEL_COLUMNS.items()
        }
        correct_models = [model_name for model_name, label in normalized.items() if label == row["final_label"].strip()]
        mismatch_rows.append(
            {
                "child_id": row["child_id"],
                "clip_id": row["clip_id"],
                "final_label": row["final_label"].strip(),
                "num_correct_models": str(len(correct_models)),
                "correct_models": "|".join(correct_models),
                **{f"{model_name}_label": label for model_name, label in normalized.items()},
            }
        )
    with (OUTPUT_DIR / "mismatch_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "child_id",
            "clip_id",
            "final_label",
            "num_correct_models",
            "correct_models",
            "rule_based_label",
            "deberta_label",
            "distilbert_label",
            "bart_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in mismatch_rows:
            writer.writerow(row)

    summary_lines = [
        f"Samples merged: {len(enriched_rows)}",
        f"Labeled samples: {len(labeled_rows)}",
        f"Unlabeled samples: {len(enriched_rows) - len(labeled_rows)}",
        f"Disagreement cases: {disagreement_count}/{len(enriched_rows)} ({metrics['agreement_stats']['disagreement_percentage']:.2%})",
    ]
    if ranking:
        best_model, best_accuracy = ranking[0]
        summary_lines.append(f"Best model on labeled subset: {best_model} ({best_accuracy:.4f}, denominator={len(labeled_rows)})")
        summary_lines.append("Ranking: " + " > ".join(f"{name} ({value:.4f})" for name, value in ranking))
    else:
        summary_lines.append("Best model on labeled subset: not available because no final_label values are present.")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"samples={len(enriched_rows)}")
    print(f"labeled_samples={len(labeled_rows)}")
    print("missing_model_predictions=0")
    if ranking:
        print(f"best_model={ranking[0][0]}")
        print("ranking=" + ", ".join(f"{name}:{value:.4f}" for name, value in ranking))
    else:
        print("best_model=NA")
        print("ranking=NA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
