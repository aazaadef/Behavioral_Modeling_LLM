"""Interaction inference from temporal segments.

Maps each :class:`TemporalSegment` to a higher-level
:class:`InteractionEvent` (focused_attention, attention_shift,
reduced_visual_availability, or monitoring).  When 3+ shift segments
exist, an additional ``exploratory_behavior`` event spanning the full
clip is appended.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from project_llm.temporal import TemporalSegment


@dataclass(frozen=True)
class InteractionEvent:
    """A higher-level behavioral event inferred from one or more segments."""

    child_id: str
    clip_id: str
    split: str
    interaction_index: int
    interaction_type: str
    start_time: float
    end_time: float
    duration_seconds: float
    supporting_segments: list[int]

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def infer_interactions(segments: list[TemporalSegment]) -> list[InteractionEvent]:
    """Convert temporal segments into interaction events.

    Each segment produces one event.  If the clip contains 3 or more
    ``shift`` segments, a clip-spanning ``exploratory_behavior`` event
    is appended as well.
    """
    if not segments:
        return []

    events: list[InteractionEvent] = []
    segment_types = Counter(segment.segment_type for segment in segments)

    for segment in segments:
        if segment.segment_type == "stable" and segment.duration_seconds >= 1.0:
            interaction_type = "focused_attention"
        elif segment.segment_type == "shift":
            interaction_type = "attention_shift"
        elif segment.segment_type in {"occluded", "outside", "eyes_closed"}:
            interaction_type = "reduced_visual_availability"
        else:
            interaction_type = "monitoring"

        events.append(
            InteractionEvent(
                child_id=segment.child_id,
                clip_id=segment.clip_id,
                split=segment.split,
                interaction_index=len(events),
                interaction_type=interaction_type,
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration_seconds=segment.duration_seconds,
                supporting_segments=[segment.segment_index],
            )
        )

    if segment_types["shift"] >= 3:
        first = segments[0]
        last = segments[-1]
        events.append(
            InteractionEvent(
                child_id=first.child_id,
                clip_id=first.clip_id,
                split=first.split,
                interaction_index=len(events),
                interaction_type="exploratory_behavior",
                start_time=first.start_time,
                end_time=last.end_time,
                duration_seconds=round(last.end_time - first.start_time, 4),
                supporting_segments=[
                    segment.segment_index for segment in segments if segment.segment_type == "shift"
                ],
            )
        )

    return events
