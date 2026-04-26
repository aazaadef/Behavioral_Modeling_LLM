"""Benchmark orchestration and model comparison utilities.

Runs the pipeline for each model spec, computes pairwise agreement,
builds ranked summaries, and supports ensemble creation via weighted
tiebreak (55% primary / 45% secondary).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from project_llm.features import BehavioralFeatures
from project_llm.llm import InterpretationResult
from project_llm.pipeline import run_pipeline, save_artifacts
from project_llm.reports import write_comparison_reports


@dataclass(frozen=True)
class BenchmarkRun:
    """Results of running one model spec through the pipeline."""
    spec: str
    features: list[BehavioralFeatures]
    interpretations: list[InterpretationResult]


def parse_model_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        if spec == "rule-based":
            return ("rule-based", "rule-based")
        raise ValueError(f"Invalid model spec: {spec}")
    backend, model = spec.split(":", 1)
    return backend, model


def pairwise_agreement(runs: list[BenchmarkRun]) -> dict[str, float]:
    agreements: dict[str, float] = {}
    for i, left in enumerate(runs):
        left_map = {(item.child_id, item.clip_id): item.interaction_type for item in left.interpretations}
        for right in runs[i + 1 :]:
            right_map = {(item.child_id, item.clip_id): item.interaction_type for item in right.interpretations}
            shared_keys = sorted(set(left_map) & set(right_map))
            if not shared_keys:
                agreements[f"{left.spec}__vs__{right.spec}"] = 0.0
                continue
            matches = sum(1 for key in shared_keys if left_map[key] == right_map[key])
            agreements[f"{left.spec}__vs__{right.spec}"] = round(matches / len(shared_keys), 4)
    return agreements


def _load_interpretations(path: Path) -> list[InterpretationResult]:
    results: list[InterpretationResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            results.append(InterpretationResult(**payload))
    return results


def load_saved_benchmark_runs(spec_to_path: dict[str, Path]) -> list[BenchmarkRun]:
    runs: list[BenchmarkRun] = []
    for spec, path in spec_to_path.items():
        interpretations = _load_interpretations(path)
        runs.append(BenchmarkRun(spec=spec, features=[], interpretations=interpretations))
    return runs


def summarize_saved_agreement(spec_to_path: dict[str, Path]) -> dict[str, object]:
    runs = load_saved_benchmark_runs(spec_to_path)
    return {
        "models": [
            {
                "spec": run.spec,
                "num_sequences": len(run.interpretations),
                "interaction_type_counts": dict(Counter(item.interaction_type for item in run.interpretations)),
            }
            for run in runs
        ],
        "pairwise_agreement": pairwise_agreement(runs),
    }


def build_combined_summary(
    evaluated_split: str,
    completed_model_summaries: list[dict[str, object]],
    pairwise: dict[str, float],
    failed_models: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    failed_models = failed_models or []
    ranking = [
        item["spec"]
        for item in sorted(completed_model_summaries, key=lambda row: float(row["avg_confidence"]), reverse=True)
    ]
    zero_shot_models = [item for item in completed_model_summaries if str(item["spec"]).startswith("zero-shot:")]
    best_zero_shot = max(zero_shot_models, key=lambda row: float(row["avg_confidence"]))["spec"] if zero_shot_models else None
    return {
        "evaluated_split": evaluated_split,
        "num_sequences": int(completed_model_summaries[0]["num_sequences"]) if completed_model_summaries else 0,
        "completed_models": completed_model_summaries,
        "failed_models": failed_models,
        "ranking_by_avg_confidence": ranking,
        "recommendation": {
            "best_current_model": ranking[0] if ranking else None,
            "best_zero_shot_baseline": best_zero_shot,
            "note": "This ranking reflects confidence and output stability on an unlabeled high-level behavior task, not accuracy against ground truth.",
        },
        "pairwise_agreement": pairwise,
    }


def write_combined_benchmark_report(output_dir: Path, summary: dict[str, object]) -> None:
    write_comparison_reports(output_dir, summary)


def _keyed_results(results: list[InterpretationResult]) -> dict[tuple[str, str], InterpretationResult]:
    return {(item.child_id, item.clip_id): item for item in results}


def build_ensemble_from_saved_runs(
    primary_spec: str,
    primary_path: Path,
    secondary_spec: str,
    secondary_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = _load_interpretations(primary_path)
    secondary = _load_interpretations(secondary_path)
    primary_map = _keyed_results(primary)
    secondary_map = _keyed_results(secondary)
    shared_keys = sorted(set(primary_map) & set(secondary_map))
    ensemble_results: list[InterpretationResult] = []

    for key in shared_keys:
        left = primary_map[key]
        right = secondary_map[key]
        if left.interaction_type == right.interaction_type:
            chosen = InterpretationResult(
                child_id=left.child_id,
                clip_id=left.clip_id,
                backend_name="ensemble",
                model_name=f"{primary_spec}+{secondary_spec}",
                interaction_type=left.interaction_type,
                interpretation=left.interpretation,
                confidence=round((left.confidence + right.confidence) / 2.0, 4),
                evidence=left.evidence[:2] + right.evidence[:2],
                limitations=[
                    "Agreement between rule-based and zero-shot baseline; confidence averaged across both models."
                ],
                prompt_version=left.prompt_version,
            )
        else:
            left_score = left.confidence * 0.55
            right_score = right.confidence * 0.45
            winner = left if left_score >= right_score else right
            loser = right if winner is left else left
            chosen = InterpretationResult(
                child_id=winner.child_id,
                clip_id=winner.clip_id,
                backend_name="ensemble",
                model_name=f"{primary_spec}+{secondary_spec}",
                interaction_type=winner.interaction_type,
                interpretation=winner.interpretation,
                confidence=round(max(left_score, right_score), 4),
                evidence=winner.evidence[:3],
                limitations=[
                    "Models disagreed; weighted tie-break favored the more reliable baseline.",
                    f"alternate_label={loser.interaction_type}",
                ],
                prompt_version=winner.prompt_version,
            )
        ensemble_results.append(chosen)

    interpretations_path = output_dir / "interpretations.jsonl"
    with interpretations_path.open("w", encoding="utf-8") as handle:
        for item in ensemble_results:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=True) + "\n")

    runs = [
        BenchmarkRun(spec=primary_spec, features=[], interpretations=primary),
        BenchmarkRun(spec=secondary_spec, features=[], interpretations=secondary),
        BenchmarkRun(spec=f"ensemble:{primary_spec}+{secondary_spec}", features=[], interpretations=ensemble_results),
    ]
    summary = {
        "primary_spec": primary_spec,
        "secondary_spec": secondary_spec,
        "ensemble_spec": f"ensemble:{primary_spec}+{secondary_spec}",
        "num_sequences": len(ensemble_results),
        "pairwise_agreement": pairwise_agreement(runs),
        "interaction_type_counts": dict(Counter(item.interaction_type for item in ensemble_results)),
        "avg_confidence": round(
            sum(item.confidence for item in ensemble_results) / len(ensemble_results), 4
        ) if ensemble_results else 0.0,
    }
    write_comparison_reports(
        output_dir,
        {
            "evaluated_split": "test",
            "num_sequences": summary["num_sequences"],
            "completed_models": [
                {
                    "spec": summary["ensemble_spec"],
                    "avg_confidence": summary["avg_confidence"],
                    "num_sequences": summary["num_sequences"],
                    "interaction_type_counts": summary["interaction_type_counts"],
                }
            ],
            "failed_models": [],
            "ranking_by_avg_confidence": [summary["ensemble_spec"]],
            "recommendation": {
                "best_current_model": summary["ensemble_spec"],
                "best_zero_shot_baseline": secondary_spec,
                "note": "This ensemble combines the rule-based baseline with the strongest zero-shot baseline using weighted tie-breaking.",
            },
            "pairwise_agreement": summary["pairwise_agreement"],
        },
    )
    return summary


def summarize_run(run: BenchmarkRun) -> dict[str, object]:
    label_counts = Counter(item.interaction_type for item in run.interpretations)
    avg_confidence = round(
        sum(item.confidence for item in run.interpretations) / len(run.interpretations), 4
    ) if run.interpretations else 0.0
    by_split: dict[str, list[float]] = defaultdict(list)
    for feature, interpretation in zip(run.features, run.interpretations):
        by_split[feature.split].append(interpretation.confidence)
    split_confidence = {
        split: round(sum(values) / len(values), 4) for split, values in by_split.items()
    }
    return {
        "spec": run.spec,
        "num_sequences": len(run.interpretations),
        "avg_confidence": avg_confidence,
        "interaction_type_counts": dict(label_counts),
        "mean_confidence_by_split": split_confidence,
    }


def run_benchmark(
    dataset_root: Path,
    output_dir: Path,
    model_specs: list[str],
    split: str | None = None,
    max_sequences: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[BenchmarkRun] = []
    for spec in model_specs:
        backend_name, model_name = parse_model_spec(spec)
        run_dir = output_dir / spec.replace("/", "_").replace(":", "__")
        artifacts = run_pipeline(
            dataset_root=dataset_root,
            split=split,
            backend_name=backend_name,
            model=model_name,
            max_sequences=max_sequences,
        )
        save_artifacts(artifacts, run_dir)
        runs.append(BenchmarkRun(spec=spec, features=artifacts.features, interpretations=artifacts.interpretations))

    summary = {
        "models": [summarize_run(run) for run in runs],
        "pairwise_agreement": pairwise_agreement(runs),
        "ranking_note": "Without high-level ground-truth labels, these comparisons reflect confidence and agreement, not accuracy.",
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    return summary
