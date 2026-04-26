"""Unit tests for temporal segmentation.

Verifies that ``build_temporal_segments`` correctly groups consecutive
frames sharing the same behavior class and that boundary timestamps,
durations, and segment indices are computed correctly.
"""

from __future__ import annotations

from project_llm.temporal import build_temporal_segments


class TestBuildTemporalSegments:
    def test_uniform_visible_collapses_to_single_stable_segment(self, make_seq) -> None:
        seq = make_seq(["inside_visible"] * 10)
        segments = build_temporal_segments(seq)
        assert len(segments) == 1
        assert segments[0].segment_type == "stable"
        assert segments[0].frame_count == 10
        assert segments[0].start_frame == 1
        assert segments[0].end_frame == 10

    def test_alternating_classes_split_into_separate_segments(self, make_seq) -> None:
        seq = make_seq(["inside_visible", "inside_visible", "inside_occluded", "inside_occluded"])
        segments = build_temporal_segments(seq)
        assert len(segments) == 2
        assert segments[0].segment_type == "stable"
        assert segments[1].segment_type == "occluded"

    def test_segment_indices_are_sequential(self, make_seq) -> None:
        seq = make_seq(
            [
                "inside_visible",
                "inside_occluded",
                "inside_visible",
                "outside_frame",
                "eyes_closed",
            ]
        )
        segments = build_temporal_segments(seq)
        assert [seg.segment_index for seg in segments] == list(range(len(segments)))

    def test_outside_and_eyes_closed_are_distinct_types(self, make_seq) -> None:
        seq = make_seq(["outside_frame"] * 3 + ["eyes_closed"] * 3)
        segments = build_temporal_segments(seq)
        assert len(segments) == 2
        assert {s.segment_type for s in segments} == {"outside", "eyes_closed"}

    def test_durations_match_fps(self, make_seq) -> None:
        seq = make_seq(["inside_visible"] * 30, fps=30.0)
        segments = build_temporal_segments(seq)
        # 30 frames @ 30 fps == 1.0 second
        assert segments[0].duration_seconds == 1.0

    def test_empty_sequence_yields_empty_segments(self, make_seq) -> None:
        seq = make_seq([])
        segments = build_temporal_segments(seq)
        assert segments == []
