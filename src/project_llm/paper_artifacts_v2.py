"""Paper-ready artifact generation from v2 experiment results.

Produces: model comparison table (CSV + Markdown), per-class accuracy
table, pairwise agreement table, confusion matrices (JSON + CSV),
prompt examples, error analysis cases, and case study examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project_llm.io_utils_v2 import write_csv, write_json, write_text


def _confusion_matrix(
    labels: list[str], y_true: list[str], y_pred: list[str]
) -> dict[str, dict[str, int]]:
    # JSON-friendly confusion matrix keyed by gold then predicted label.
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in zip(y_true, y_pred):
        matrix[gold][pred] += 1
    return matrix


def _write_markdown_table(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    write_text(path, "\n".join(lines) + "\n")


def generate_paper_artifacts(
    output_dir: Path,
    merged_rows: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    model_metadata: list[dict[str, Any]],
) -> None:
    # Emit the comparison tables and qualitative exports that are most useful in paper drafting.
    labels = sorted(
        {row["final_label"] for row in merged_rows if row.get("final_label")}
        | {
            row[key]
            for row in merged_rows
            for key in row
            if key.endswith("__label") and row.get(key)
        }
    )
    labeled_rows = [row for row in merged_rows if row.get("final_label")]

    comparison_rows = []
    per_class_rows = []
    confusion_payload: dict[str, Any] = {}
    confusion_csv_rows: list[dict[str, Any]] = []
    agreement_rows = pairwise_rows

    for summary in model_summaries:
        comparison_rows.append(
            {
                "model_name": summary["model_name"],
                "backend_family": summary["backend_family"],
                "num_samples": summary["num_samples"],
                "overall_accuracy": summary.get("overall_accuracy", ""),
                "average_confidence": summary["average_confidence"],
                "notable_failure_modes": " | ".join(summary["notable_failure_modes"]),
            }
        )
        for label, value in summary.get("per_class_accuracy", {}).items():
            per_class_rows.append(
                {
                    "model_name": summary["model_name"],
                    "label": label,
                    "per_class_accuracy": value,
                }
            )

        if labeled_rows:
            model_column = summary["prediction_column"]
            y_true = [row["final_label"] for row in labeled_rows]
            y_pred = [row.get(model_column, "") for row in labeled_rows]
            confusion = _confusion_matrix(labels, y_true, y_pred)
            confusion_payload[summary["model_name"]] = confusion
            for gold, preds in confusion.items():
                for pred, value in preds.items():
                    confusion_csv_rows.append(
                        {
                            "model_name": summary["model_name"],
                            "gold_label": gold,
                            "predicted_label": pred,
                            "count": value,
                        }
                    )

    write_csv(
        output_dir / "model_comparison_table.csv",
        comparison_rows,
        [
            "model_name",
            "backend_family",
            "num_samples",
            "overall_accuracy",
            "average_confidence",
            "notable_failure_modes",
        ],
    )
    _write_markdown_table(
        output_dir / "model_comparison_table.md",
        [
            "model_name",
            "backend_family",
            "num_samples",
            "overall_accuracy",
            "average_confidence",
            "notable_failure_modes",
        ],
        [
            [
                row["model_name"],
                row["backend_family"],
                row["num_samples"],
                row["overall_accuracy"],
                row["average_confidence"],
                row["notable_failure_modes"],
            ]
            for row in comparison_rows
        ],
    )
    write_csv(
        output_dir / "per_class_accuracy_table.csv",
        per_class_rows,
        ["model_name", "label", "per_class_accuracy"],
    )
    write_csv(
        output_dir / "agreement_table.csv",
        agreement_rows,
        ["left_model", "right_model", "agreement", "matches", "denominator"],
    )
    write_json(output_dir / "confusion_matrices.json", confusion_payload)
    write_csv(
        output_dir / "confusion_matrices.csv",
        confusion_csv_rows,
        ["model_name", "gold_label", "predicted_label", "count"],
    )

    # Keep prompt examples compact so they are easy to inspect first.
    prompt_examples_rows = []
    for row in merged_rows[:25]:
        prompt_examples_rows.append(
            {
                "child_id": row["child_id"],
                "clip_id": row["clip_id"],
                "split": row["split"],
                "feature_summary": row["feature_summary"],
                "prompt_texts_json": row["prompt_texts_json"],
            }
        )
    write_csv(
        output_dir / "prompt_examples.csv",
        prompt_examples_rows,
        ["child_id", "clip_id", "split", "feature_summary", "prompt_texts_json"],
    )

    error_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for row in merged_rows:
        agrees = row["models_agree"] == "yes"
        any_correct = row.get("num_correct_models", "") not in {"", "0"}
        if row.get("final_label"):
            if agrees and row.get("all_models_correct") == "yes":
                case_rows.append({**row, "case_bucket": "all_models_agree_and_correct"})
            elif agrees and row.get("all_models_correct") == "no":
                error_rows.append({**row, "case_bucket": "all_models_agree_and_wrong"})
            elif row.get("disagreement_flag") == "yes":
                case_rows.append({**row, "case_bucket": "strong_disagreement_case"})
            elif not any_correct:
                error_rows.append({**row, "case_bucket": "no_model_correct"})
        else:
            if row.get("disagreement_flag") == "yes":
                case_rows.append({**row, "case_bucket": "unlabeled_disagreement_case"})

    for model_column in (
        [key for key in merged_rows[0].keys() if key.endswith("__label")] if merged_rows else []
    ):
        model_name = model_column[: -len("__label")]
        for row in merged_rows:
            if not row.get("final_label"):
                continue
            pred = row.get(model_column, "")
            if pred == row["final_label"]:
                others = [
                    value
                    for key, value in row.items()
                    if key.endswith("__label") and key != model_column and value
                ]
                if others and any(other != row["final_label"] for other in others):
                    case_rows.append({**row, "case_bucket": f"{model_name}_wins"})

    export_fields = [
        "case_bucket",
        "child_id",
        "clip_id",
        "split",
        "feature_summary",
        "final_label",
        "models_agree",
        "disagreement_flag",
        "num_correct_models",
        "prompt_texts_json",
        "predictions_json",
        "confidences_json",
    ]
    write_csv(output_dir / "examples_for_error_analysis.csv", error_rows, export_fields)
    write_csv(output_dir / "examples_for_case_study.csv", case_rows, export_fields)
    write_json(output_dir / "model_metadata.json", model_metadata)
