"""Unit tests for behavioral feature extraction.

Builds in-memory ``ChildClipSequence`` objects with controlled gaze_class
patterns and verifies that ``extract_behavioral_features`` produces the
expected numeric ratios, streaks, and the composite stability score.
"""

from __future__ import annotations

from project_llm.features import BehavioralFeatures, extract_behavioral_features
from project_llm.labels_3class import rule_based_3class


class TestExtractBehavioralFeatures:
    def test_all_visible_yields_full_visibility(self, make_seq) -> None:
        seq = make_seq(["inside_visible"] * 30)
        feat = extract_behavioral_features(seq)
        assert feat.observed_frames == 30
        assert feat.visible_ratio == 1.0
        assert feat.on_screen_ratio == 1.0
        assert feat.occlusion_ratio == 0.0
        assert feat.eyes_closed_ratio == 0.0
        assert feat.gaze_shift_ratio == 0.0
        assert feat.dominant_gaze_class == "inside_visible"

    def test_all_occluded_collapses_to_zero_visibility(self, make_seq) -> None:
        seq = make_seq(["inside_occluded"] * 20)
        feat = extract_behavioral_features(seq)
        assert feat.visible_ratio == 0.0
        assert feat.occlusion_ratio == 1.0
        assert feat.max_visible_streak == 0
        assert feat.max_occlusion_streak == 20
        assert feat.dominant_gaze_class == "inside_occluded"

    def test_mixed_classes_compute_correct_ratios(self, make_seq) -> None:
        # 6 visible + 2 shift + 2 occluded → ratios 0.6 / 0.2 / 0.2
        classes = (["inside_visible"] * 6) + (["gaze_shift"] * 2) + (["inside_occluded"] * 2)
        seq = make_seq(classes)
        feat = extract_behavioral_features(seq)
        assert feat.visible_ratio == 0.6
        assert feat.gaze_shift_ratio == 0.2
        assert feat.occlusion_ratio == 0.2

    def test_max_streak_picks_longest_run(self, make_seq) -> None:
        # visible(3), occluded(1), visible(5), occluded(2)
        classes = (
            ["inside_visible"] * 3
            + ["inside_occluded"]
            + ["inside_visible"] * 5
            + ["inside_occluded"] * 2
        )
        seq = make_seq(classes)
        feat = extract_behavioral_features(seq)
        assert feat.max_visible_streak == 5
        assert feat.max_occlusion_streak == 2

    def test_stability_score_within_unit_interval(self, make_seq) -> None:
        seq = make_seq(["inside_visible"] * 30)
        feat = extract_behavioral_features(seq)
        assert 0.0 <= feat.attention_stability_score <= 1.0

    def test_uncertain_counts_as_occlusion(self, make_seq) -> None:
        seq = make_seq(["inside_uncertain"] * 10)
        feat = extract_behavioral_features(seq)
        assert feat.occlusion_ratio == 1.0
        assert feat.visible_ratio == 0.0

    def test_eyes_closed_ratio_isolated(self, make_seq) -> None:
        seq = make_seq(["eyes_closed"] * 5 + ["inside_visible"] * 5)
        feat = extract_behavioral_features(seq)
        assert feat.eyes_closed_ratio == 0.5
        assert feat.visible_ratio == 0.5
        # Eyes-closed frames are not "on-screen" — only visible+occluded count.
        assert feat.on_screen_ratio == 0.5


class TestBehavioralFeaturesDataclass:
    def test_to_dict_contains_all_fields(self, make_features) -> None:
        feat = make_features()
        data = feat.to_dict()
        # to_dict should be a flat dict of all dataclass fields.
        for field in BehavioralFeatures.__dataclass_fields__:
            assert field in data

    def test_is_immutable(self, make_features) -> None:
        feat = make_features()
        try:
            feat.visible_ratio = 0.0  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("BehavioralFeatures should be frozen but assignment succeeded")


class TestFeaturesIntegrateWithRuleBased:
    """Sanity check that extracted features can drive the 3-class rule."""

    def test_full_visibility_yields_focused(self, make_seq) -> None:
        seq = make_seq(["inside_visible"] * 50)
        feat = extract_behavioral_features(seq)
        label, _ = rule_based_3class(feat)
        # Stable, visible, no shifts → focused.
        assert label == "focused"

    def test_full_occlusion_yields_others(self, make_seq) -> None:
        seq = make_seq(["inside_occluded"] * 50)
        feat = extract_behavioral_features(seq)
        label, _ = rule_based_3class(feat)
        assert label == "others"
