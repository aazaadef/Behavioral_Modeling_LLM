"""Unit tests for the canonical 3-class label schema and rule-based classifier.

Covers the 5→3 mapping, the rule-based classifier branches, and the
NLI / LLM helpers. All tests are dataset-free.
"""

from __future__ import annotations

import json

import pytest

from project_llm.labels_3class import (
    LABELS_3CLASS,
    LABEL_DESCRIPTIONS_3CLASS,
    MAP_5_TO_3,
    llm_prompt_3class,
    map_5_to_3,
    nli_candidate_labels_3class,
    nli_hypothesis_template_3class,
    rule_based_3class,
)


class TestLabelsContract:
    def test_three_canonical_labels(self) -> None:
        assert LABELS_3CLASS == ["focused", "mix", "others"]

    def test_descriptions_cover_every_label(self) -> None:
        assert set(LABEL_DESCRIPTIONS_3CLASS.keys()) == set(LABELS_3CLASS)

    def test_5_to_3_covers_all_five_legacy_labels(self) -> None:
        expected = {
            "focused_attention",
            "mixed_attention",
            "exploratory_attention",
            "occluded_attention",
            "reduced_visual_availability",
        }
        assert set(MAP_5_TO_3.keys()) == expected

    def test_5_to_3_only_targets_canonical_labels(self) -> None:
        assert set(MAP_5_TO_3.values()) <= set(LABELS_3CLASS)


class TestMap5To3:
    @pytest.mark.parametrize(
        "five,three",
        [
            ("focused_attention", "focused"),
            ("mixed_attention", "mix"),
            ("exploratory_attention", "mix"),
            ("occluded_attention", "others"),
            ("reduced_visual_availability", "others"),
        ],
    )
    def test_known_mapping(self, five: str, three: str) -> None:
        assert map_5_to_3(five) == three

    def test_idempotent_on_canonical_labels(self) -> None:
        for label in LABELS_3CLASS:
            assert map_5_to_3(label) == label

    def test_strips_whitespace(self) -> None:
        assert map_5_to_3("  focused_attention  ") == "focused"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_none_and_empty_pass_through(self, empty) -> None:
        assert map_5_to_3(empty) is None

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown label"):
            map_5_to_3("not_a_real_label")


class TestRuleBased3Class:
    def test_eyes_closed_dominates_returns_others(self, make_features) -> None:
        feat = make_features(eyes_closed_ratio=0.5, occlusion_ratio=0.0)
        label, conf = rule_based_3class(feat)
        assert label == "others"
        assert 0.0 <= conf <= 1.0

    def test_high_occlusion_returns_others(self, make_features) -> None:
        feat = make_features(eyes_closed_ratio=0.0, occlusion_ratio=0.5)
        label, _ = rule_based_3class(feat)
        assert label == "others"

    def test_clear_focused_pattern(self, make_features) -> None:
        feat = make_features(
            eyes_closed_ratio=0.0,
            occlusion_ratio=0.0,
            visible_ratio=0.95,
            gaze_shift_ratio=0.05,
            mean_gaze_motion=20.0,
            attention_stability_score=0.85,
        )
        label, _ = rule_based_3class(feat)
        assert label == "focused"

    def test_high_shift_returns_mix(self, make_features) -> None:
        feat = make_features(
            eyes_closed_ratio=0.0,
            occlusion_ratio=0.0,
            visible_ratio=0.6,
            gaze_shift_ratio=0.3,
            mean_gaze_motion=50.0,
            attention_stability_score=0.4,
        )
        label, _ = rule_based_3class(feat)
        assert label == "mix"

    def test_fallback_returns_mix(self, make_features) -> None:
        # Doesn't match focused, doesn't trigger high-shift branch — fallback.
        feat = make_features(
            eyes_closed_ratio=0.0,
            occlusion_ratio=0.0,
            visible_ratio=0.7,
            gaze_shift_ratio=0.15,
            mean_gaze_motion=100.0,
            attention_stability_score=0.5,
        )
        label, _ = rule_based_3class(feat)
        assert label == "mix"

    def test_returns_canonical_label(self, make_features) -> None:
        feat = make_features()
        label, _ = rule_based_3class(feat)
        assert label in LABELS_3CLASS


class TestLLMPrompt3Class:
    def test_prompt_contains_all_three_labels(self) -> None:
        prompt = llm_prompt_3class({"visible_ratio": 0.8})
        for label in LABELS_3CLASS:
            assert label in prompt

    def test_prompt_includes_features_payload_as_json(self) -> None:
        payload = {"visible_ratio": 0.8, "occlusion_ratio": 0.1}
        prompt = llm_prompt_3class(payload)
        # The features block is appended as serialized JSON; round-tripping
        # the trailing JSON object should reproduce the payload.
        json_start = prompt.index("{", prompt.rfind("Features:"))
        decoded = json.loads(prompt[json_start:])
        assert decoded == payload


class TestNLIHelpers:
    def test_candidate_labels_match_canonical(self) -> None:
        assert nli_candidate_labels_3class() == LABELS_3CLASS

    def test_candidate_labels_returns_fresh_list(self) -> None:
        labels = nli_candidate_labels_3class()
        labels.append("tampered")
        # Mutating the returned list must not affect the canonical list.
        assert LABELS_3CLASS == ["focused", "mix", "others"]

    def test_hypothesis_template_has_format_placeholder(self) -> None:
        template = nli_hypothesis_template_3class()
        assert "{}" in template
        # Should produce a grammatical hypothesis when filled.
        assert template.format("focused").endswith(".")
