"""V1 pipeline orchestrator.

Loads ChildPlay data, runs a single backend, and saves all intermediate
and final artifacts (sequences, segments, interactions, features,
interpretations, and summary reports) to the output directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from project_llm.dataset import ChildClipSequence, load_child_sequences
from project_llm.features import BehavioralFeatures, extract_behavioral_features
from project_llm.interactions import InteractionEvent, infer_interactions
from project_llm.llm import (
    InterpretationResult,
    LLMBackend,
    OpenAIBackend,
    RuleBasedBackend,
    ZeroShotTransformerBackend,
)
from project_llm.reports import write_reports
from project_llm.temporal import TemporalSegment, build_temporal_segments


@dataclass(frozen=True)
class PipelineArtifacts:
    """All intermediate and final outputs from a single pipeline run."""

    sequences: list[ChildClipSequence]
    temporal_sequences: list[list[TemporalSegment]]
    interactions: list[list[InteractionEvent]]
    features: list[BehavioralFeatures]
    interpretations: list[InterpretationResult]


def build_backend(name: str, model: str) -> LLMBackend:
    """Factory: dispatch a backend name to its v1 class."""
    if name == "rule-based":
        return RuleBasedBackend()
    if name == "openai":
        return OpenAIBackend(model=model)
    if name == "zero-shot":
        return ZeroShotTransformerBackend(model=model)
    if name == "hf-llm":
        from project_llm.llm import HuggingFaceLLMBackend
        return HuggingFaceLLMBackend(model=model)
    raise ValueError(f"Unsupported backend: {name}")


def run_pipeline(
    dataset_root: Path,
    split: str | None = None,
    backend_name: str = "rule-based",
    model: str = "gpt-4.1-mini",
    max_sequences: int | None = None,
) -> PipelineArtifacts:
    """Run the full pipeline: load data -> extract features -> build segments -> interpret."""
    sequences = load_child_sequences(dataset_root, split=split)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    backend = build_backend(backend_name, model=model)
    features = [extract_behavioral_features(sequence) for sequence in sequences]
    temporal_sequences = [build_temporal_segments(sequence) for sequence in sequences]
    interactions = [infer_interactions(segments) for segments in temporal_sequences]
    interpretations = [
        backend.interpret(feature, sequence_segments, sequence_interactions)
        for feature, sequence_segments, sequence_interactions in zip(features, temporal_sequences, interactions)
    ]
    return PipelineArtifacts(
        sequences=sequences,
        temporal_sequences=temporal_sequences,
        interactions=interactions,
        features=features,
        interpretations=interpretations,
    )


def save_artifacts(artifacts: PipelineArtifacts, output_dir: Path) -> None:
    """Write all pipeline artifacts as JSONL files plus summary reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences_path = output_dir / "child_sequences.jsonl"
    temporal_path = output_dir / "temporal_sequences.jsonl"
    interactions_path = output_dir / "interaction_events.jsonl"
    features_path = output_dir / "behavioral_features.jsonl"
    interpretations_path = output_dir / "interpretations.jsonl"

    with sequences_path.open("w", encoding="utf-8") as handle:
        for sequence in artifacts.sequences:
            handle.write(json.dumps(sequence.to_dict(), ensure_ascii=True) + "\n")

    with temporal_path.open("w", encoding="utf-8") as handle:
        for sequence_segments in artifacts.temporal_sequences:
            for segment in sequence_segments:
                handle.write(json.dumps(segment.to_dict(), ensure_ascii=True) + "\n")

    with interactions_path.open("w", encoding="utf-8") as handle:
        for sequence_interactions in artifacts.interactions:
            for event in sequence_interactions:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")

    with features_path.open("w", encoding="utf-8") as handle:
        for feature in artifacts.features:
            handle.write(json.dumps(feature.to_dict(), ensure_ascii=True) + "\n")

    with interpretations_path.open("w", encoding="utf-8") as handle:
        for result in artifacts.interpretations:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=True) + "\n")

    flat_interactions = [event for sequence_interactions in artifacts.interactions for event in sequence_interactions]
    write_reports(output_dir, artifacts.features, flat_interactions, artifacts.interpretations)
