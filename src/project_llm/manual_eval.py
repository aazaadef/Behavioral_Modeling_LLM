"""Manual evaluation workflow.

Handles the complete lifecycle of human annotation: building the
evaluation subset (with stratified sampling by agreement pattern),
writing annotation templates, evaluating against manual and silver
labels (accuracy, macro-F1, confusion matrix, Cohen's kappa), and
generating the inter-annotator agreement statistics.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from project_llm.benchmark import _load_interpretations
from project_llm.dataset import ChildClipSequence, load_child_sequences
from project_llm.features import BehavioralFeatures
from project_llm.llm import InterpretationResult
from project_llm.pipeline import run_pipeline, save_artifacts
from project_llm.reports import write_comparison_reports


# The five behavioral labels with their human-readable descriptions.
LABEL_SCHEMA = {
    "focused_attention": "Predominantly stable, sustained visual attention on a target within the clip.",
    "exploratory_attention": "Frequent shifts or scanning across multiple targets without one stable focus.",
    "occluded_attention": "Attention cannot be characterized confidently because visible gaze is often unavailable or uncertain.",
    "reduced_visual_availability": "Interpretation is limited by eyes closed, outside-frame gaze, or similar loss of visual evidence.",
    "mixed_attention": "The clip contains both stable attention and notable shifts without one dominant pattern.",
}


@dataclass(frozen=True)
class ManualEvalExample:
    child_id: str
    clip_id: str
    split: str
    source_video_url: str
    target_person_and_time: str
    target_person_bbox_hint: str
    rule_based_label: str
    deberta_label: str
    distilbert_label: str
    bart_label: str
    agreement_pattern: str
    rule_based_confidence: float
    deberta_confidence: float

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SilverLabelExample:
    child_id: str
    clip_id: str
    split: str
    silver_label: str
    rule_based_confidence: float
    deberta_confidence: float
    silver_confidence: float
    support: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def load_behavioral_features(path: Path) -> list[BehavioralFeatures]:
    results: list[BehavioralFeatures] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            results.append(BehavioralFeatures(**payload))
    return results


def _key(sequence: ChildClipSequence | InterpretationResult) -> tuple[str, str]:
    return (sequence.child_id, sequence.clip_id)


def _feature_key(feature: BehavioralFeatures) -> tuple[str, str]:
    return (feature.child_id, feature.clip_id)


def _index_by_key(items: list[InterpretationResult]) -> dict[tuple[str, str], InterpretationResult]:
    return {_key(item): item for item in items}


def write_label_schema(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "label_schema.json"
    guide_path = output_dir / "annotation_guidelines.md"
    with schema_path.open("w", encoding="utf-8") as handle:
        json.dump(LABEL_SCHEMA, handle, ensure_ascii=True, indent=2)

    lines = [
        "# Manual Evaluation Guidelines",
        "",
        "Use one label per child-sequence.",
        "Base decisions only on the structured gaze evidence and clip-level summaries.",
        "Do not infer diagnosis, emotion, or social intent beyond the visible evidence.",
        "",
        "## Labels",
    ]
    for label, description in LABEL_SCHEMA.items():
        lines.append(f"- `{label}`: {description}")
    lines.extend(
        [
            "",
            "## Annotation Notes",
            "- Prefer `occluded_attention` when visual evidence is frequently unavailable.",
            "- Prefer `reduced_visual_availability` when eyes are closed or gaze is persistently outside the frame.",
            "- Use `mixed_attention` only when no single pattern clearly dominates.",
        ]
    )
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manual_eval_subset(
    sequences: list[ChildClipSequence],
    rule_based_path: Path,
    deberta_path: Path,
    distilbert_path: Path,
    bart_path: Path,
    max_examples: int | None = None,
) -> list[ManualEvalExample]:
    rule_based = _index_by_key(_load_interpretations(rule_based_path))
    deberta = _index_by_key(_load_interpretations(deberta_path))
    distilbert = _index_by_key(_load_interpretations(distilbert_path))
    bart = _index_by_key(_load_interpretations(bart_path))

    examples: list[ManualEvalExample] = []
    for sequence in sequences:
        key = _key(sequence)
        if key not in rule_based or key not in deberta or key not in distilbert or key not in bart:
            continue
        labels = {
            "rule_based": rule_based[key].interaction_type,
            "deberta": deberta[key].interaction_type,
            "distilbert": distilbert[key].interaction_type,
            "bart": bart[key].interaction_type,
        }
        unique_labels = sorted(set(labels.values()))
        agreement_pattern = "full_agreement" if len(unique_labels) == 1 else (
            "rule_deberta_agree" if labels["rule_based"] == labels["deberta"] else "disagreement"
        )
        clip_suffix = sequence.clip_id.rsplit("_", 1)[1]
        frame_range = clip_suffix.replace("-downsampled", "")
        start_frame_text, end_frame_text = frame_range.split("-", 1)
        start_frame = int(start_frame_text)
        end_frame = int(end_frame_text)
        source_start_seconds = round((start_frame - 1) / sequence.fps, 3)
        source_end_seconds = round(end_frame / sequence.fps, 3)
        clip_duration_seconds = round(len(sequence.frames) / sequence.fps, 3)
        first_frame = sequence.frames[0]
        middle_frame = sequence.frames[len(sequence.frames) // 2]
        last_frame = sequence.frames[-1]
        target_person_bbox_hint = (
            f"track person_{sequence.person_id}; "
            f"clip_frame_{first_frame.frame}=("
            f"{first_frame.bbox_x:.1f},{first_frame.bbox_y:.1f},{first_frame.bbox_width:.1f},{first_frame.bbox_height:.1f}"
            f"); "
            f"clip_frame_{middle_frame.frame}=("
            f"{middle_frame.bbox_x:.1f},{middle_frame.bbox_y:.1f},{middle_frame.bbox_width:.1f},{middle_frame.bbox_height:.1f}"
            f"); "
            f"clip_frame_{last_frame.frame}=("
            f"{last_frame.bbox_x:.1f},{last_frame.bbox_y:.1f},{last_frame.bbox_width:.1f},{last_frame.bbox_height:.1f}"
            f")"
        )
        examples.append(
            ManualEvalExample(
                child_id=sequence.child_id,
                clip_id=sequence.clip_id,
                split=sequence.split,
                source_video_url=f"https://www.youtube.com/watch?v={sequence.video_id}",
                target_person_and_time=(
                    f"target=person_{sequence.person_id}; "
                    f"source_video_seconds={source_start_seconds:.3f}-{source_end_seconds:.3f}; "
                    f"clip_seconds=0.000-{clip_duration_seconds:.3f}"
                ),
                target_person_bbox_hint=target_person_bbox_hint,
                rule_based_label=labels["rule_based"],
                deberta_label=labels["deberta"],
                distilbert_label=labels["distilbert"],
                bart_label=labels["bart"],
                agreement_pattern=agreement_pattern,
                rule_based_confidence=rule_based[key].confidence,
                deberta_confidence=deberta[key].confidence,
            )
        )

    examples.sort(key=lambda item: (item.split, item.clip_id, item.child_id))
    if max_examples is None or max_examples >= len(examples):
        return examples

    groups: dict[str, list[ManualEvalExample]] = defaultdict(list)
    for example in examples:
        groups[example.agreement_pattern].append(example)

    for group in groups.values():
        group.sort(
            key=lambda item: (
                abs(item.rule_based_confidence - item.deberta_confidence),
                item.child_id,
            ),
            reverse=True,
        )

    selected: list[ManualEvalExample] = []
    quotas = {
        "disagreement": max_examples // 2,
        "rule_deberta_agree": max_examples // 3,
        "full_agreement": max_examples - (max_examples // 2) - (max_examples // 3),
    }
    for label in ["disagreement", "rule_deberta_agree", "full_agreement"]:
        selected.extend(groups[label][: quotas[label]])

    if len(selected) < max_examples:
        leftovers = [item for item in examples if item not in selected]
        leftovers.sort(key=lambda item: item.child_id)
        selected.extend(leftovers[: max_examples - len(selected)])

    return selected[:max_examples]


def write_manual_eval_subset(
    output_dir: Path,
    subset: list[ManualEvalExample],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_json = output_dir / "manual_eval_subset.json"
    template_csv = output_dir / "manual_eval_annotations.csv"
    summary_json = output_dir / "manual_eval_subset_summary.json"
    tracks_json = output_dir / "target_person_tracks.json"
    existing_rows: dict[tuple[str, str], dict[str, str]] = {}

    if template_csv.exists():
        with template_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                existing_rows[(row["child_id"], row["clip_id"])] = row

    with subset_json.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in subset], handle, ensure_ascii=True, indent=2)

    with tracks_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                item.child_id: {
                    "child_id": item.child_id,
                    "clip_id": item.clip_id,
                    "target_person_bbox_hint": item.target_person_bbox_hint,
                }
                for item in subset
            },
            handle,
            ensure_ascii=True,
            indent=2,
        )

    with template_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "child_id",
                "clip_id",
                "split",
                "source_video_url",
                "target_person_and_time",
                "target_person_bbox_hint",
                "agreement_pattern",
                "suggested_reference_label",
                "rule_based_label",
                "deberta_label",
                "distilbert_label",
                "bart_label",
                "rule_based_confidence",
                "deberta_confidence",
                "annotator_1_label",
                "annotator_2_label",
                "final_label",
                "notes",
            ],
        )
        writer.writeheader()
        for item in subset:
            existing = existing_rows.get((item.child_id, item.clip_id), {})
            writer.writerow(
                {
                    "child_id": item.child_id,
                    "clip_id": item.clip_id,
                    "split": item.split,
                    "source_video_url": item.source_video_url,
                    "target_person_and_time": item.target_person_and_time,
                    "target_person_bbox_hint": item.target_person_bbox_hint,
                    "agreement_pattern": item.agreement_pattern,
                    "suggested_reference_label": item.rule_based_label,
                    "rule_based_label": item.rule_based_label,
                    "deberta_label": item.deberta_label,
                    "distilbert_label": item.distilbert_label,
                    "bart_label": item.bart_label,
                    "rule_based_confidence": f"{item.rule_based_confidence:.4f}",
                    "deberta_confidence": f"{item.deberta_confidence:.4f}",
                    "annotator_1_label": existing.get("annotator_1_label", ""),
                    "annotator_2_label": existing.get("annotator_2_label", ""),
                    "final_label": existing.get("final_label", ""),
                    "notes": existing.get("notes", ""),
                }
            )

    summary = {
        "num_examples": len(subset),
        "agreement_pattern_counts": dict(Counter(item.agreement_pattern for item in subset)),
        "rule_based_label_counts": dict(Counter(item.rule_based_label for item in subset)),
        "deberta_label_counts": dict(Counter(item.deberta_label for item in subset)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def validate_manual_eval_subset_against_features(
    subset: list[ManualEvalExample],
    features: list[BehavioralFeatures],
) -> None:
    feature_keys = {_feature_key(feature) for feature in features}
    missing = [
        f"{item.child_id} ({item.clip_id})"
        for item in subset
        if (item.child_id, item.clip_id) not in feature_keys
    ]
    if missing:
        raise ValueError(
            "Manual evaluation subset references child sequences missing from behavioral features: "
            + ", ".join(missing[:5])
            + (f" ... (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        )


def validate_saved_manual_eval_consistency(
    features_path: Path,
    subset_path: Path,
) -> dict[str, int]:
    features = load_behavioral_features(features_path)
    subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))
    subset = [ManualEvalExample(**item) for item in subset_payload]
    validate_manual_eval_subset_against_features(subset, features)
    return {
        "behavioral_features": len(features),
        "manual_eval_subset": len(subset),
    }


def prepare_manual_eval_workflow(
    dataset_root: Path,
    run_output_dir: Path,
    paper_eval_output_dir: Path,
    rule_based_path: Path,
    deberta_path: Path,
    distilbert_path: Path,
    bart_path: Path,
    split: str = "test",
    max_examples: int | None = None,
    max_sequences: int | None = None,
) -> dict[str, int]:
    artifacts = run_pipeline(
        dataset_root=dataset_root,
        split=split,
        backend_name="rule-based",
        model="rule-based",
        max_sequences=max_sequences,
    )
    save_artifacts(artifacts, run_output_dir)

    sequences = load_child_sequences(dataset_root, split=split)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    subset = build_manual_eval_subset(
        sequences,
        rule_based_path=rule_based_path,
        deberta_path=deberta_path,
        distilbert_path=distilbert_path,
        bart_path=bart_path,
        max_examples=max_examples,
    )
    validate_manual_eval_subset_against_features(subset, artifacts.features)
    write_label_schema(paper_eval_output_dir)
    write_manual_eval_subset(paper_eval_output_dir, subset)
    validate_saved_manual_eval_consistency(
        run_output_dir / "behavioral_features.jsonl",
        paper_eval_output_dir / "manual_eval_subset.json",
    )
    return {
        "behavioral_features": len(artifacts.features),
        "manual_eval_subset": len(subset),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
    return round(sum(f1_values) / len(labels), 4) if labels else 0.0


def _cohen_kappa(labels_a: list[str], labels_b: list[str], labels: list[str]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    observed = _safe_divide(sum(1 for a, b in zip(labels_a, labels_b) if a == b), len(labels_a))
    a_counts = Counter(labels_a)
    b_counts = Counter(labels_b)
    expected = sum(_safe_divide(a_counts[label], len(labels_a)) * _safe_divide(b_counts[label], len(labels_b)) for label in labels)
    return round(_safe_divide(observed - expected, 1 - expected) if expected != 1 else 0.0, 4)


def evaluate_against_manual_labels(
    annotations_csv: Path,
    model_specs_to_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    labeled_rows = [row for row in rows if row.get("final_label")]
    if not labeled_rows:
        summary = {
            "num_labeled_examples": 0,
            "message": "No final_label values found. Fill manual_eval_annotations.csv before running evaluation.",
        }
        (output_dir / "paper_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        return summary

    gold = {(row["child_id"], row["clip_id"]): row["final_label"] for row in labeled_rows}
    labels = sorted(LABEL_SCHEMA)
    model_results: list[dict[str, object]] = []
    for spec, path in model_specs_to_paths.items():
        predictions = _index_by_key(_load_interpretations(path))
        y_true: list[str] = []
        y_pred: list[str] = []
        for key, label in gold.items():
            if key in predictions:
                y_true.append(label)
                y_pred.append(predictions[key].interaction_type)
        accuracy = round(_safe_divide(sum(1 for g, p in zip(y_true, y_pred) if g == p), len(y_true)), 4) if y_true else 0.0
        confusion = {
            gold_label: {pred_label: 0 for pred_label in labels}
            for gold_label in labels
        }
        for g, p in zip(y_true, y_pred):
            confusion[g][p] += 1
        model_results.append(
            {
                "spec": spec,
                "num_evaluated_examples": len(y_true),
                "accuracy": accuracy,
                "macro_f1": _macro_f1(y_true, y_pred, labels),
                "confusion_matrix": confusion,
            }
        )

    annotator_1 = [row["annotator_1_label"] for row in labeled_rows if row.get("annotator_1_label")]
    annotator_2 = [row["annotator_2_label"] for row in labeled_rows if row.get("annotator_2_label")]
    summary = {
        "num_labeled_examples": len(labeled_rows),
        "labels": labels,
        "inter_annotator_agreement": {
            "cohen_kappa": _cohen_kappa(annotator_1, annotator_2, labels)
            if len(annotator_1) == len(labeled_rows) and len(annotator_2) == len(labeled_rows)
            else None,
        },
        "models": model_results,
    }
    (output_dir / "paper_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def build_silver_label_subset(
    sequences: list[ChildClipSequence],
    rule_based_path: Path,
    deberta_path: Path,
    min_rule_confidence: float = 0.8,
    min_deberta_confidence: float = 0.55,
) -> list[SilverLabelExample]:
    rule_based = _index_by_key(_load_interpretations(rule_based_path))
    deberta = _index_by_key(_load_interpretations(deberta_path))
    subset: list[SilverLabelExample] = []
    for sequence in sequences:
        key = _key(sequence)
        if key not in rule_based or key not in deberta:
            continue
        left = rule_based[key]
        right = deberta[key]
        if left.interaction_type != right.interaction_type:
            continue
        if left.confidence < min_rule_confidence or right.confidence < min_deberta_confidence:
            continue
        subset.append(
            SilverLabelExample(
                child_id=sequence.child_id,
                clip_id=sequence.clip_id,
                split=sequence.split,
                silver_label=left.interaction_type,
                rule_based_confidence=left.confidence,
                deberta_confidence=right.confidence,
                silver_confidence=round((left.confidence + right.confidence) / 2.0, 4),
                support="rule_based_and_deberta_agreement",
            )
        )
    subset.sort(key=lambda item: (item.silver_confidence, item.child_id), reverse=True)
    return subset


def write_silver_label_subset(output_dir: Path, subset: list[SilverLabelExample]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "silver_labels.json"
    csv_path = output_dir / "silver_labels.csv"
    summary_path = output_dir / "silver_label_summary.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in subset], handle, ensure_ascii=True, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "child_id",
                "clip_id",
                "split",
                "silver_label",
                "rule_based_confidence",
                "deberta_confidence",
                "silver_confidence",
                "support",
            ],
        )
        writer.writeheader()
        for item in subset:
            writer.writerow(item.to_dict())

    summary = {
        "num_examples": len(subset),
        "label_counts": dict(Counter(item.silver_label for item in subset)),
        "mean_silver_confidence": round(
            sum(item.silver_confidence for item in subset) / len(subset), 4
        ) if subset else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def evaluate_against_silver_labels(
    silver_labels_json: Path,
    model_specs_to_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    silver_rows = json.loads(silver_labels_json.read_text(encoding="utf-8"))
    gold = {(row["child_id"], row["clip_id"]): row["silver_label"] for row in silver_rows}
    labels = sorted(LABEL_SCHEMA)
    model_results: list[dict[str, object]] = []
    for spec, path in model_specs_to_paths.items():
        predictions = _index_by_key(_load_interpretations(path))
        y_true: list[str] = []
        y_pred: list[str] = []
        for key, label in gold.items():
            if key in predictions:
                y_true.append(label)
                y_pred.append(predictions[key].interaction_type)
        accuracy = round(_safe_divide(sum(1 for g, p in zip(y_true, y_pred) if g == p), len(y_true)), 4) if y_true else 0.0
        model_results.append(
            {
                "spec": spec,
                "num_evaluated_examples": len(y_true),
                "accuracy": accuracy,
                "macro_f1": _macro_f1(y_true, y_pred, labels),
            }
        )
    summary = {
        "evaluation_type": "silver_label_consensus",
        "num_silver_examples": len(gold),
        "labels": labels,
        "models": model_results,
        "note": "Silver labels are consensus labels from rule-based and DeBERTa on high-confidence agreed examples; they are not human ground truth.",
    }
    (output_dir / "silver_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    write_comparison_reports(
        output_dir,
        {
            "evaluated_split": "test",
            "num_sequences": len(gold),
            "completed_models": [
                {
                    "spec": row["spec"],
                    "avg_confidence": row["accuracy"],
                    "num_sequences": row["num_evaluated_examples"],
                    "interaction_type_counts": {},
                }
                for row in model_results
            ],
            "failed_models": [],
            "ranking_by_avg_confidence": [
                row["spec"] for row in sorted(model_results, key=lambda item: item["accuracy"], reverse=True)
            ],
            "recommendation": {
                "best_current_model": max(model_results, key=lambda item: item["accuracy"])["spec"] if model_results else None,
                "best_zero_shot_baseline": "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                "note": summary["note"],
            },
            "pairwise_agreement": {},
        },
    )
    return summary
