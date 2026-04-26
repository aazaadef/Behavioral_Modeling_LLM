"""Prompt construction for LLM backends.

Builds a structured text prompt containing the system role, output JSON
schema, allowed labels, behavioral features, temporal segments, and
interaction events.  The first 12 segments/events are included to keep
prompt length bounded.
"""

from __future__ import annotations

import json

from project_llm.features import BehavioralFeatures
from project_llm.interactions import InteractionEvent
from project_llm.temporal import TemporalSegment


PROMPT_VERSION = "childplay-behavior-v1"


def build_behavior_prompt(
    features: BehavioralFeatures,
    segments: list[TemporalSegment],
    interactions: list[InteractionEvent],
) -> str:
    """Build the full text prompt sent to generative LLM backends."""
    schema = {
        "interaction_type": "string",
        "interpretation": "string",
        "confidence": "float in [0,1]",
        "evidence": ["short strings grounded in numeric features"],
        "limitations": ["short strings about ambiguity or missing evidence"],
    }
    guidance = {
        "allowed_interaction_types": [
            "focused_attention",
            "exploratory_attention",
            "occluded_attention",
            "reduced_visual_availability",
            "mixed_attention",
        ],
        "constraints": [
            "Use only the provided frame-level annotation features.",
            "Do not mention diagnosis, emotions, or causes not evidenced by gaze annotations.",
            "Prefer conservative interpretations when visibility is limited.",
            "Ground evidence in the numeric features.",
        ],
    }
    temporal_description = [
        {
            "type": segment.segment_type,
            "start": segment.start_time,
            "end": segment.end_time,
            "duration": segment.duration_seconds,
        }
        for segment in segments[:12]
    ]
    interaction_description = [
        {
            "type": event.interaction_type,
            "start": event.start_time,
            "end": event.end_time,
            "duration": event.duration_seconds,
        }
        for event in interactions[:12]
    ]
    return (
        "You are a research assistant analyzing ChildPlay gaze annotations.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Output schema: {json.dumps(schema, ensure_ascii=True)}\n"
        f"Guidance: {json.dumps(guidance, ensure_ascii=True)}\n"
        f"Features: {json.dumps(features.to_dict(), sort_keys=True, ensure_ascii=True)}\n"
        f"Temporal sequences: {json.dumps(temporal_description, ensure_ascii=True)}\n"
        f"Interaction events: {json.dumps(interaction_description, ensure_ascii=True)}"
    )
