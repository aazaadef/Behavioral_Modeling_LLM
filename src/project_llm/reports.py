"""Report generation for v1 pipeline outputs.

Produces summary JSON and markdown reports aggregating statistics
from pipeline runs and model comparisons.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from project_llm.features import BehavioralFeatures
from project_llm.interactions import InteractionEvent
from project_llm.llm import InterpretationResult


def build_summary(
    features: list[BehavioralFeatures],
    interactions: list[InteractionEvent],
    interpretations: list[InterpretationResult],
) -> dict[str, object]:
    interaction_counts = Counter(item.interaction_type for item in interpretations)
    event_counts = Counter(item.interaction_type for item in interactions)
    split_counts = Counter(item.split for item in features)
    split_confidences: dict[str, list[float]] = defaultdict(list)
    for feature, interpretation in zip(features, interpretations):
        split_confidences[feature.split].append(interpretation.confidence)

    mean_confidence_by_split = {
        split: round(sum(values) / len(values), 4) for split, values in split_confidences.items()
    }
    mean_attention_stability = (
        round(sum(item.attention_stability_score for item in features) / len(features), 4)
        if features
        else 0.0
    )

    return {
        "num_child_sequences": len(features),
        "split_counts": dict(split_counts),
        "interaction_type_counts": dict(interaction_counts),
        "interaction_event_counts": dict(event_counts),
        "mean_confidence_by_split": mean_confidence_by_split,
        "mean_attention_stability_score": mean_attention_stability,
        "average_attention_event_duration": round(
            sum(
                item.duration_seconds
                for item in interactions
                if item.interaction_type == "focused_attention"
            )
            / max(
                1, sum(1 for item in interactions if item.interaction_type == "focused_attention")
            ),
            4,
        ),
    }


def write_reports(
    output_dir: Path,
    features: list[BehavioralFeatures],
    interactions: list[InteractionEvent],
    interpretations: list[InterpretationResult],
) -> None:
    summary = build_summary(features, interactions, interpretations)
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    lines = [
        "# ChildPlay Behavioral Report",
        "",
        f"Child sequences analyzed: {summary['num_child_sequences']}",
        f"Mean attention stability score: {summary['mean_attention_stability_score']}",
        "",
        "## Split Counts",
    ]
    for split, count in sorted(summary["split_counts"].items()):
        lines.append(f"- {split}: {count}")

    lines.extend(["", "## Interaction Types"])
    for label, count in sorted(summary["interaction_type_counts"].items()):
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Interaction Events"])
    for label, count in sorted(summary["interaction_event_counts"].items()):
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Mean Confidence By Split"])
    for split, value in sorted(summary["mean_confidence_by_split"].items()):
        lines.append(f"- {split}: {value}")

    lines.extend(["", "## Aggregate Statistics"])
    lines.append(
        f"- average_attention_event_duration: {summary['average_attention_event_duration']}"
    )

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_comparison_reports(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    lines = [
        "# ChildPlay Model Comparison",
        "",
        f"Evaluated split: {summary['evaluated_split']}",
        f"Child sequences evaluated: {summary['num_sequences']}",
        "",
        "## Completed Models",
    ]
    for model in summary["completed_models"]:
        lines.append(f"- {model['spec']}: avg_confidence={model['avg_confidence']}")

    lines.extend(["", "## Failed Models"])
    if summary["failed_models"]:
        for model in summary["failed_models"]:
            lines.append(f"- {model['spec']}: {model['reason']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Ranking By Average Confidence"])
    for index, spec in enumerate(summary["ranking_by_avg_confidence"], start=1):
        lines.append(f"{index}. {spec}")

    lines.extend(["", "## Recommendation"])
    lines.append(f"- best_current_model: {summary['recommendation']['best_current_model']}")
    lines.append(
        f"- best_zero_shot_baseline: {summary['recommendation']['best_zero_shot_baseline']}"
    )
    lines.append(f"- note: {summary['recommendation']['note']}")

    if summary.get("pairwise_agreement"):
        lines.extend(["", "## Pairwise Agreement"])
        for pair, value in sorted(summary["pairwise_agreement"].items()):
            lines.append(f"- {pair}: {value}")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
