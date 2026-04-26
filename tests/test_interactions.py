"""Unit tests for the interaction inference layer.

Builds synthetic temporal-segment lists and verifies that
``infer_interactions`` produces the right ``InteractionEvent`` types and
that the clip-spanning ``exploratory_behavior`` event is appended only
when there are 3 or more shift segments.
"""

from __future__ import annotations

from project_llm.interactions import InteractionEvent, infer_interactions
from project_llm.temporal import TemporalSegment


def _segment(idx: int, segment_type: str, *, duration: float = 1.5) -> TemporalSegment:
    return TemporalSegment(
        child_id="clip_x:person_1",
        clip_id="clip_x",
        split="test",
        segment_index=idx,
        start_frame=idx * 30 + 1,
        end_frame=(idx + 1) * 30,
        start_time=idx * duration,
        end_time=(idx + 1) * duration,
        duration_seconds=duration,
        segment_type=segment_type,
        gaze_class="inside_visible" if segment_type == "stable" else "inside_occluded",
        avg_gaze_x=100.0,
        avg_gaze_y=100.0,
        frame_count=30,
    )


class TestInferInteractions:
    def test_empty_segments_yield_empty_events(self) -> None:
        assert infer_interactions([]) == []

    def test_long_stable_segment_becomes_focused_attention(self) -> None:
        events = infer_interactions([_segment(0, "stable", duration=2.0)])
        assert len(events) == 1
        assert events[0].interaction_type == "focused_attention"

    def test_short_stable_segment_is_monitoring(self) -> None:
        # Stable but < 1 second — not enough for "focused_attention".
        events = infer_interactions([_segment(0, "stable", duration=0.5)])
        assert len(events) == 1
        assert events[0].interaction_type == "monitoring"

    def test_shift_segment_becomes_attention_shift(self) -> None:
        events = infer_interactions([_segment(0, "shift")])
        assert events[0].interaction_type == "attention_shift"

    def test_occluded_outside_eyes_closed_collapse_to_reduced_visibility(self) -> None:
        for stype in ("occluded", "outside", "eyes_closed"):
            events = infer_interactions([_segment(0, stype)])
            assert events[0].interaction_type == "reduced_visual_availability"

    def test_three_shifts_appends_exploratory_behavior(self) -> None:
        segments = [_segment(i, "shift") for i in range(3)]
        events = infer_interactions(segments)
        # 3 shift events + 1 clip-spanning exploratory_behavior
        assert len(events) == 4
        assert events[-1].interaction_type == "exploratory_behavior"
        # Spans the entire clip.
        assert events[-1].start_time == segments[0].start_time
        assert events[-1].end_time == segments[-1].end_time

    def test_two_shifts_does_not_append_exploratory(self) -> None:
        segments = [_segment(i, "shift") for i in range(2)]
        events = infer_interactions(segments)
        assert all(e.interaction_type != "exploratory_behavior" for e in events)


class TestInteractionEventDataclass:
    def test_interaction_event_to_dict_round_trip(self) -> None:
        event = InteractionEvent(
            child_id="clip_x:person_1",
            clip_id="clip_x",
            split="test",
            interaction_index=0,
            interaction_type="focused_attention",
            start_time=0.0,
            end_time=2.0,
            duration_seconds=2.0,
            supporting_segments=[0],
        )
        data = event.to_dict()
        assert data["interaction_type"] == "focused_attention"
        assert data["supporting_segments"] == [0]
