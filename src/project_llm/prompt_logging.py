"""Prompt audit logging utilities.

Provides helpers to construct reproducible audit records for every
prompt-response pair: compact premise text for NLI models, feature
summary strings, and the :class:`PromptLogEntry` dataclass that
captures the full audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from project_llm.features import BehavioralFeatures
from project_llm.interactions import InteractionEvent
from project_llm.temporal import TemporalSegment


def build_zero_shot_premise(
    features: BehavioralFeatures,
    segments: list[TemporalSegment],
    interactions: list[InteractionEvent],
) -> str:
    # Keep the premise compact but still grounded in the same derived signals used elsewhere.
    event_types = [event.interaction_type for event in interactions[:8]]
    segment_types = [segment.segment_type for segment in segments[:8]]
    return (
        "Child gaze behavior summary. "
        f"split={features.split}, "
        f"observed_frames={features.observed_frames}, "
        f"visible_ratio={features.visible_ratio:.3f}, "
        f"on_screen_ratio={features.on_screen_ratio:.3f}, "
        f"gaze_shift_ratio={features.gaze_shift_ratio:.3f}, "
        f"occlusion_ratio={features.occlusion_ratio:.3f}, "
        f"eyes_closed_ratio={features.eyes_closed_ratio:.3f}, "
        f"mean_head_motion={features.mean_head_motion:.2f}, "
        f"mean_gaze_motion={features.mean_gaze_motion:.2f}, "
        f"attention_stability_score={features.attention_stability_score:.3f}, "
        f"dominant_gaze_class={features.dominant_gaze_class}, "
        f"num_segments={len(segments)}, "
        f"num_interactions={len(interactions)}, "
        f"segment_types={segment_types}, "
        f"interaction_events={event_types}."
    )


def build_feature_summary(features: BehavioralFeatures) -> str:
    # This string is reused in merged exports and qualitative analysis tables.
    return (
        f"visible_ratio={features.visible_ratio:.3f}; "
        f"gaze_shift_ratio={features.gaze_shift_ratio:.3f}; "
        f"occlusion_ratio={features.occlusion_ratio:.3f}; "
        f"eyes_closed_ratio={features.eyes_closed_ratio:.3f}; "
        f"mean_gaze_motion={features.mean_gaze_motion:.2f}; "
        f"attention_stability_score={features.attention_stability_score:.3f}; "
        f"dominant_gaze_class={features.dominant_gaze_class}; "
        f"observed_frames={features.observed_frames}"
    )


@dataclass(frozen=True)
class PromptLogEntry:
    """Full audit record for one sample-model evaluation."""

    # One row per evaluated sample-model pair for reproducible prompt auditing.
    child_id: str
    clip_id: str
    backend_name: str
    model_name: str
    split: str
    premise_text: str
    full_prompt_text: str
    candidate_labels: list[str]
    predicted_label: str
    confidence: float
    evidence: list[str]
    limitations: list[str]
    timestamp: str
    prompt_version: str
    feature_summary: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
