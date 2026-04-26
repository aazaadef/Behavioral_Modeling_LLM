"""V2 extended experiment pipeline.

Runs multiple model backends over the same preprocessed data and produces
merged comparison tables, disagreement analysis, pairwise agreement
matrices, per-model artifacts, and paper-ready exports.  Supports
optional manual labels for accuracy computation.
"""

from __future__ import annotations

import json
import os
import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from project_llm.backends_v2 import BackendResponse, build_backend_v2
from project_llm.benchmark import parse_model_spec
from project_llm.dataset import ChildClipSequence, load_child_sequences
from project_llm.features import BehavioralFeatures, extract_behavioral_features
from project_llm.interactions import InteractionEvent, infer_interactions
from project_llm.io_utils_v2 import (
    build_reproducibility_notes,
    ensure_new_output_dir,
    iso_timestamp,
    sanitize_model_spec,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from project_llm.paper_artifacts_v2 import generate_paper_artifacts
from project_llm.prompt_logging import PromptLogEntry, build_feature_summary
from project_llm.temporal import TemporalSegment, build_temporal_segments


# Known zero-shot NLI model specs for reference and default experiments.
SUPPORTED_HF_MODELS = [
    "zero-shot:typeform/distilbert-base-uncased-mnli",
    "zero-shot:facebook/bart-large-mnli",
    "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
]


@dataclass(frozen=True)
class ModelRunArtifactsV2:
    # Per-model bundle used to assemble merged experiment and paper outputs.
    spec: str
    backend_name: str
    model_name: str
    backend_family: str
    run_dir: Path
    child_sequences: list[ChildClipSequence]
    temporal_sequences: list[list[TemporalSegment]]
    interaction_events: list[list[InteractionEvent]]
    behavioral_features: list[BehavioralFeatures]
    interpretations: list[dict[str, Any]]
    prompt_logs: list[PromptLogEntry]


T = TypeVar("T")


def _flatten(nested: list[list[T]]) -> list[T]:
    return [item for group in nested for item in group]


def _load_manual_labels(dataset_root: Path) -> dict[tuple[str, str], str]:
    # Reuse any existing manual labels when present, but do not require them.
    candidates = [
        Path("output/paper_eval/manual_eval_annotations.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        import csv

        labels: dict[tuple[str, str], str] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                final_label = row.get("final_label", "").strip()
                if final_label:
                    labels[(row["child_id"], row["clip_id"])] = final_label
        return labels
    return {}


def _build_setup_fix_text(spec: str, reason: str) -> str:
    if spec.startswith("zero-shot:"):
        model = spec.split(":", 1)[1]
        return (
            f"Model: {spec}\n"
            f"Failure: {reason}\n"
            "Why: Hugging Face zero-shot inference could not be initialized or executed in this environment.\n"
            "What to do:\n"
            "1. Activate the project venv.\n"
            "2. Install missing packages if needed.\n"
            f"3. Pre-download or verify access to the model `{model}`.\n"
            "Exact commands:\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements.txt transformers scikit-learn torch\n"
            f"python -c \"from transformers import pipeline; pipeline('zero-shot-classification', model='{model}')\"\n"
            "Afterward: re-running the same experiment command should be enough if the model is accessible.\n"
        )
    if spec.startswith("openai:"):
        model = spec.split(":", 1)[1]
        return (
            f"Model: {spec}\n"
            f"Failure: {reason}\n"
            "Why: OpenAI access is not configured or reachable from this environment.\n"
            "What to do:\n"
            "1. Install dependencies in the project venv.\n"
            "2. Export a valid OpenAI API key.\n"
            "Exact commands:\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements.txt\n"
            "export OPENAI_API_KEY=your_key_here\n"
            f"python main.py run-extended-experiments --models openai:{model} --execution-mode test-only --output-dir output_experiments_v2/openai_retry\n"
            "Afterward: re-running is enough.\n"
        )
    if spec.startswith("deepseek:"):
        model = spec.split(":", 1)[1]
        return (
            f"Model: {spec}\n"
            f"Failure: {reason}\n"
            "Why: DeepSeek uses an OpenAI-compatible API client and needs explicit credentials plus a reachable endpoint.\n"
            "What to do:\n"
            "1. Install dependencies in the project venv.\n"
            "2. Export `DEEPSEEK_API_KEY`.\n"
            "3. Optionally export `DEEPSEEK_BASE_URL` if your endpoint differs from https://api.deepseek.com.\n"
            "Exact commands:\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements.txt\n"
            "export DEEPSEEK_API_KEY=your_key_here\n"
            "export DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
            f"python main.py run-extended-experiments --models deepseek:{model} --execution-mode test-only --output-dir output_experiments_v2/deepseek_retry\n"
            "Afterward: re-running is enough if the API credentials and endpoint are valid.\n"
        )
    return (
        f"Model: {spec}\n"
        f"Failure: {reason}\n"
        "What to do: inspect the traceback or exception and then re-run the same command after fixing the environment.\n"
    )


def _write_setup_issues(output_dir: Path, issues: list[dict[str, str]]) -> None:
    if not issues:
        return
    lines = ["Setup issues encountered while building the extended experiment outputs.", ""]
    for issue in issues:
        lines.append(_build_setup_fix_text(issue["spec"], issue["reason"]))
        lines.append("")
    write_text(output_dir / "setup_issues_and_fixes.txt", "\n".join(lines))


def run_extended_experiments(
    *,
    dataset_root: Path,
    output_dir: Path,
    model_specs: list[str],
    execution_mode: str,
    max_sequences: int | None = None,
    prompt_version: str = "childplay-behavior-v2",
    command: str | None = None,
) -> dict[str, Any]:
    if execution_mode not in {"test-only", "full-analysis"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    # Shared deterministic preprocessing is done once, then reused across model backends.
    ensure_new_output_dir(output_dir)
    selected_split = "test" if execution_mode == "test-only" else None
    sequences = load_child_sequences(dataset_root, split=selected_split)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    temporal_sequences = [build_temporal_segments(sequence) for sequence in sequences]
    interaction_events = [infer_interactions(segments) for segments in temporal_sequences]
    behavioral_features = [extract_behavioral_features(sequence) for sequence in sequences]

    manual_labels = _load_manual_labels(dataset_root)
    issues: list[dict[str, str]] = []
    successful_runs: list[ModelRunArtifactsV2] = []

    write_jsonl(output_dir / "child_sequences.jsonl", [sequence.to_dict() for sequence in sequences])
    write_jsonl(output_dir / "temporal_sequences.jsonl", [segment.to_dict() for segment in _flatten(temporal_sequences)])
    write_jsonl(output_dir / "interaction_events.jsonl", [event.to_dict() for event in _flatten(interaction_events)])
    write_jsonl(output_dir / "behavioral_features.jsonl", [feature.to_dict() for feature in behavioral_features])

    for spec in model_specs:
        backend_name, model_name = parse_model_spec(spec)
        run_dir = output_dir / sanitize_model_spec(spec)
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            # Each backend writes into its own fresh directory so runs stay isolated.
            backend = build_backend_v2(backend_name, model_name)
            backend_responses: list[BackendResponse] = [
                backend.interpret_with_logging(feature, segments, events)
                for feature, segments, events in zip(behavioral_features, temporal_sequences, interaction_events)
            ]
            interpretations = [response.interpretation.to_dict() for response in backend_responses]
            prompt_logs = [response.prompt_log for response in backend_responses]

            write_jsonl(run_dir / "child_sequences.jsonl", [sequence.to_dict() for sequence in sequences])
            write_jsonl(run_dir / "temporal_sequences.jsonl", [segment.to_dict() for segment in _flatten(temporal_sequences)])
            write_jsonl(run_dir / "interaction_events.jsonl", [event.to_dict() for event in _flatten(interaction_events)])
            write_jsonl(run_dir / "behavioral_features.jsonl", [feature.to_dict() for feature in behavioral_features])
            write_jsonl(run_dir / "interpretations.jsonl", interpretations)
            write_jsonl(run_dir / "logged_prompts.jsonl", [log.to_dict() for log in prompt_logs])
            write_csv(
                run_dir / "logged_prompts.csv",
                [
                    {
                        **log.to_dict(),
                        "candidate_labels": json.dumps(log.candidate_labels, ensure_ascii=True),
                        "evidence": json.dumps(log.evidence, ensure_ascii=True),
                        "limitations": json.dumps(log.limitations, ensure_ascii=True),
                    }
                    for log in prompt_logs
                ],
                [
                    "child_id",
                    "clip_id",
                    "backend_name",
                    "model_name",
                    "split",
                    "premise_text",
                    "full_prompt_text",
                    "candidate_labels",
                    "predicted_label",
                    "confidence",
                    "evidence",
                    "limitations",
                    "timestamp",
                    "prompt_version",
                    "feature_summary",
                ],
            )
            successful_runs.append(
                ModelRunArtifactsV2(
                    spec=spec,
                    backend_name=backend.backend_name,
                    model_name=backend.model_name,
                    backend_family=backend.backend_family,
                    run_dir=run_dir,
                    child_sequences=sequences,
                    temporal_sequences=temporal_sequences,
                    interaction_events=interaction_events,
                    behavioral_features=behavioral_features,
                    interpretations=interpretations,
                    prompt_logs=prompt_logs,
                )
            )
        except Exception as exc:
            issues.append({"spec": spec, "reason": str(exc)})
            write_text(run_dir / "FAILED.txt", str(exc) + "\n")

    _write_setup_issues(output_dir, issues)

    merged_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []

    if successful_runs:
        base_features = {feature.child_id: feature for feature in behavioral_features}
        prompt_log_index: dict[tuple[str, str, str], PromptLogEntry] = {}
        for run in successful_runs:
            for log in run.prompt_logs:
                prompt_log_index[(run.spec, log.child_id, log.clip_id)] = log

        prediction_maps = {
            run.spec: {
                (row["child_id"], row["clip_id"]): row
                for row in run.interpretations
            }
            for run in successful_runs
        }

        keys = sorted(
            {
                (feature.child_id, feature.clip_id)
                for feature in behavioral_features
            }
        )

        for child_id, clip_id in keys:
            feature = base_features[child_id]
            row: dict[str, Any] = {
                "child_id": child_id,
                "clip_id": clip_id,
                "split": feature.split,
                "feature_summary": build_feature_summary(feature),
                "final_label": manual_labels.get((child_id, clip_id), ""),
            }
            predictions_json: dict[str, Any] = {}
            confidences_json: dict[str, Any] = {}
            prompt_texts_json: dict[str, Any] = {}
            labels: list[str] = []
            for run in successful_runs:
                pred = prediction_maps[run.spec][(child_id, clip_id)]
                label_col = f"{sanitize_model_spec(run.spec)}__label"
                conf_col = f"{sanitize_model_spec(run.spec)}__confidence"
                row[label_col] = pred["interaction_type"]
                row[conf_col] = pred["confidence"]
                predictions_json[run.spec] = pred["interaction_type"]
                confidences_json[run.spec] = pred["confidence"]
                log = prompt_log_index[(run.spec, child_id, clip_id)]
                prompt_texts_json[run.spec] = {
                    "premise_text": log.premise_text,
                    "full_prompt_text": log.full_prompt_text,
                }
                labels.append(pred["interaction_type"])

            row["models_agree"] = "yes" if len(set(labels)) == 1 else "no"
            row["disagreement_flag"] = "yes" if row["models_agree"] == "no" else "no"
            row["predictions_json"] = json.dumps(predictions_json, ensure_ascii=True, sort_keys=True)
            row["confidences_json"] = json.dumps(confidences_json, ensure_ascii=True, sort_keys=True)
            row["prompt_texts_json"] = json.dumps(prompt_texts_json, ensure_ascii=True, sort_keys=True)
            if row["final_label"]:
                num_correct = sum(1 for value in predictions_json.values() if value == row["final_label"])
                row["num_correct_models"] = num_correct
                row["all_models_correct"] = "yes" if num_correct == len(successful_runs) else "no"
            else:
                row["num_correct_models"] = ""
                row["all_models_correct"] = ""
            merged_rows.append(row)
            if row["disagreement_flag"] == "yes":
                disagreement_rows.append(
                    {
                        "child_id": child_id,
                        "clip_id": clip_id,
                        "split": feature.split,
                        "feature_summary": row["feature_summary"],
                        "final_label": row["final_label"],
                        "predictions_json": row["predictions_json"],
                        "confidences_json": row["confidences_json"],
                    }
                )

        for run in successful_runs:
            labels = [row["interaction_type"] for row in run.interpretations]
            confidences = [float(row["confidence"]) for row in run.interpretations]
            label_counts = Counter(labels)
            prediction_column = f"{sanitize_model_spec(run.spec)}__label"
            summary: dict[str, Any] = {
                "spec": run.spec,
                "model_name": run.model_name,
                "backend_name": run.backend_name,
                "backend_family": run.backend_family,
                "prediction_column": prediction_column,
                "num_samples": len(run.interpretations),
                "label_distribution": dict(label_counts),
                "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "per_class_accuracy": {},
                "overall_accuracy": None,
                "notable_failure_modes": [],
            }
            if manual_labels:
                labeled_rows = [row for row in merged_rows if row["final_label"]]
                y_true = [row["final_label"] for row in labeled_rows]
                y_pred = [row[prediction_column] for row in labeled_rows]
                if y_true:
                    summary["overall_accuracy"] = round(
                        sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / len(y_true), 4
                    )
                    for label in sorted(set(y_true)):
                        indices = [index for index, gold in enumerate(y_true) if gold == label]
                        summary["per_class_accuracy"][label] = round(
                            sum(1 for index in indices if y_pred[index] == y_true[index]) / len(indices), 4
                        )
                    wrong_labels = [row[prediction_column] for row in labeled_rows if row[prediction_column] != row["final_label"]]
                    summary["notable_failure_modes"] = [
                        f"most_common_errors={dict(Counter(wrong_labels).most_common(3))}"
                    ] if wrong_labels else ["no_observed_errors_on_labeled_subset"]
            if not summary["notable_failure_modes"]:
                summary["notable_failure_modes"] = ["manual labels unavailable for failure-mode estimation"]

            confidence_rows.append(
                {
                    "model_name": run.model_name,
                    "backend_family": run.backend_family,
                    "num_samples": summary["num_samples"],
                    "average_confidence": summary["average_confidence"],
                }
            )
            for label, count in sorted(label_counts.items()):
                per_class_rows.append(
                    {
                        "model_name": run.model_name,
                        "backend_family": run.backend_family,
                        "label": label,
                        "predicted_count": count,
                        "per_class_accuracy": summary["per_class_accuracy"].get(label, ""),
                    }
                )
            model_metadata.append(
                {
                    "spec": run.spec,
                    "backend_name": run.backend_name,
                    "backend_family": run.backend_family,
                    "model_name": run.model_name,
                    "run_dir": str(run.run_dir),
                    "prompt_version": prompt_version,
                    "num_samples": summary["num_samples"],
                    "label_distribution": summary["label_distribution"],
                    "average_confidence": summary["average_confidence"],
                    "overall_accuracy": summary["overall_accuracy"],
                }
            )
            run_dict = summary
            model_metadata[-1]["notable_failure_modes"] = run_dict["notable_failure_modes"]

        for index, left in enumerate(successful_runs):
            left_column = f"{sanitize_model_spec(left.spec)}__label"
            for right in successful_runs[index + 1 :]:
                right_column = f"{sanitize_model_spec(right.spec)}__label"
                shared = [(row[left_column], row[right_column]) for row in merged_rows]
                matches = sum(1 for a, b in shared if a == b)
                pairwise_rows.append(
                    {
                        "left_model": left.spec,
                        "right_model": right.spec,
                        "agreement": round(matches / len(shared), 4) if shared else 0.0,
                        "matches": matches,
                        "denominator": len(shared),
                    }
                )

        agreement_index: dict[str, list[dict[str, Any]]] = {}
        for row in pairwise_rows:
            agreement_index.setdefault(row["left_model"], []).append(
                {"other_model": row["right_model"], "agreement": row["agreement"]}
            )
            agreement_index.setdefault(row["right_model"], []).append(
                {"other_model": row["left_model"], "agreement": row["agreement"]}
            )
        for metadata in model_metadata:
            metadata["agreement_with_other_models"] = agreement_index.get(metadata["spec"], [])

        root_fields = [
            "child_id",
            "clip_id",
            "split",
            "feature_summary",
            "final_label",
            "models_agree",
            "disagreement_flag",
            "num_correct_models",
            "all_models_correct",
            "predictions_json",
            "confidences_json",
            "prompt_texts_json",
        ]
        dynamic_fields: list[str] = []
        for run in successful_runs:
            dynamic_fields.extend(
                [
                    f"{sanitize_model_spec(run.spec)}__label",
                    f"{sanitize_model_spec(run.spec)}__confidence",
                ]
            )

        aggregated_interpretations = []
        aggregated_prompt_logs = []
        for run in successful_runs:
            for interpretation in run.interpretations:
                aggregated_interpretations.append({**interpretation, "model_spec": run.spec})
            for log in run.prompt_logs:
                aggregated_prompt_logs.append({**log.to_dict(), "model_spec": run.spec})

        write_jsonl(output_dir / "interpretations.jsonl", aggregated_interpretations)
        write_jsonl(output_dir / "logged_prompts.jsonl", aggregated_prompt_logs)
        write_csv(
            output_dir / "logged_prompts.csv",
            [
                {
                    **row,
                    "candidate_labels": json.dumps(row["candidate_labels"], ensure_ascii=True),
                    "evidence": json.dumps(row["evidence"], ensure_ascii=True),
                    "limitations": json.dumps(row["limitations"], ensure_ascii=True),
                }
                for row in aggregated_prompt_logs
            ],
            [
                "model_spec",
                "child_id",
                "clip_id",
                "backend_name",
                "model_name",
                "split",
                "premise_text",
                "full_prompt_text",
                "candidate_labels",
                "predicted_label",
                "confidence",
                "evidence",
                "limitations",
                "timestamp",
                "prompt_version",
                "feature_summary",
            ],
        )
        write_csv(output_dir / "merged_predictions.csv", merged_rows, root_fields + dynamic_fields)
        write_json(output_dir / "merged_predictions.json", merged_rows)
        write_csv(
            output_dir / "disagreement_report.csv",
            disagreement_rows,
            ["child_id", "clip_id", "split", "feature_summary", "final_label", "predictions_json", "confidences_json"],
        )
        write_csv(
            output_dir / "confidence_summary.csv",
            confidence_rows,
            ["model_name", "backend_family", "num_samples", "average_confidence"],
        )
        write_csv(
            output_dir / "per_class_summary.csv",
            per_class_rows,
            ["model_name", "backend_family", "label", "predicted_count", "per_class_accuracy"],
        )

        generate_paper_artifacts(
            output_dir=output_dir,
            merged_rows=merged_rows,
            model_summaries=[
                {
                    **metadata,
                    "prediction_column": f"{sanitize_model_spec(metadata['spec'])}__label",
                    "per_class_accuracy": {
                        row["label"]: row["per_class_accuracy"]
                        for row in per_class_rows
                        if row["model_name"] == metadata["model_name"]
                    },
                }
                for metadata in model_metadata
            ],
            pairwise_rows=pairwise_rows,
            model_metadata=model_metadata,
        )
    else:
        write_jsonl(output_dir / "interpretations.jsonl", [])
        write_jsonl(output_dir / "logged_prompts.jsonl", [])
        write_csv(
            output_dir / "logged_prompts.csv",
            [],
            [
                "model_spec",
                "child_id",
                "clip_id",
                "backend_name",
                "model_name",
                "split",
                "premise_text",
                "full_prompt_text",
                "candidate_labels",
                "predicted_label",
                "confidence",
                "evidence",
                "limitations",
                "timestamp",
                "prompt_version",
                "feature_summary",
            ],
        )
        write_csv(
            output_dir / "merged_predictions.csv",
            [],
            [
                "child_id",
                "clip_id",
                "split",
                "feature_summary",
                "final_label",
                "models_agree",
                "disagreement_flag",
                "num_correct_models",
                "all_models_correct",
                "predictions_json",
                "confidences_json",
                "prompt_texts_json",
            ],
        )
        write_json(output_dir / "merged_predictions.json", [])
        write_csv(
            output_dir / "disagreement_report.csv",
            [],
            ["child_id", "clip_id", "split", "feature_summary", "final_label", "predictions_json", "confidences_json"],
        )
        write_csv(
            output_dir / "confidence_summary.csv",
            [],
            ["model_name", "backend_family", "num_samples", "average_confidence"],
        )
        write_csv(
            output_dir / "per_class_summary.csv",
            [],
            ["model_name", "backend_family", "label", "predicted_count", "per_class_accuracy"],
        )
        generate_paper_artifacts(
            output_dir=output_dir,
            merged_rows=[],
            model_summaries=[],
            pairwise_rows=[],
            model_metadata=[],
        )

    reproducibility_notes = build_reproducibility_notes(
        command=command or " ".join(shlex.quote(item) for item in os.sys.argv),
        dataset_root=dataset_root,
        selected_split=selected_split,
        execution_mode=execution_mode,
        model_specs=model_specs,
    )
    write_text(output_dir / "reproducibility_notes.txt", reproducibility_notes)

    run_summary = {
        "timestamp_utc": iso_timestamp(),
        "dataset_root": str(dataset_root),
        "execution_mode": execution_mode,
        "selected_split": selected_split,
        "requested_model_specs": model_specs,
        "successful_model_specs": [run.spec for run in successful_runs],
        "failed_models": issues,
        "num_sequences": len(sequences),
        "num_successful_models": len(successful_runs),
        "supported_hf_models": SUPPORTED_HF_MODELS,
        "output_dir": str(output_dir),
        "artifacts": {
            "merged_predictions_csv": str(output_dir / "merged_predictions.csv"),
            "paper_artifacts_dir": str(output_dir),
            "setup_issues": str(output_dir / "setup_issues_and_fixes.txt") if issues else "",
        },
    }
    write_json(output_dir / "run_summary.json", run_summary)
    return run_summary
