"""Unit tests for LLM prompt construction.

Verifies that ``build_behavior_prompt`` produces a prompt containing the
expected schema, allowed labels, and serialized features. The prompt is
text and we don't pin its exact wording — only its structural contract.
"""

from __future__ import annotations

import json

from project_llm.prompts import PROMPT_VERSION, build_behavior_prompt


class TestBuildBehaviorPrompt:
    def test_prompt_includes_version(self, make_features) -> None:
        prompt = build_behavior_prompt(make_features(), [], [])
        assert PROMPT_VERSION in prompt

    def test_prompt_lists_allowed_labels(self, make_features) -> None:
        prompt = build_behavior_prompt(make_features(), [], [])
        for label in (
            "focused_attention",
            "exploratory_attention",
            "occluded_attention",
            "reduced_visual_availability",
            "mixed_attention",
        ):
            assert label in prompt

    def test_features_serialized_as_json(self, make_features) -> None:
        feat = make_features(visible_ratio=0.9, occlusion_ratio=0.05)
        prompt = build_behavior_prompt(feat, [], [])
        # Find the "Features:" block and parse the JSON object that follows.
        idx = prompt.index("Features:")
        json_start = prompt.index("{", idx)
        # Match the matching closing brace by scanning depth.
        depth = 0
        end = json_start
        for i, ch in enumerate(prompt[json_start:], start=json_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        decoded = json.loads(prompt[json_start:end])
        assert decoded["visible_ratio"] == 0.9
        assert decoded["occlusion_ratio"] == 0.05

    def test_segment_window_is_capped_at_twelve(self, make_features) -> None:
        # Build 20 fake segment-like dicts is unsafe (TemporalSegment is
        # frozen), so we just confirm the prompt builder accepts long
        # lists without raising.
        from project_llm.temporal import TemporalSegment

        segments = [
            TemporalSegment(
                child_id="x",
                clip_id="x",
                split="test",
                segment_index=i,
                start_frame=i * 30,
                end_frame=(i + 1) * 30,
                start_time=float(i),
                end_time=float(i + 1),
                duration_seconds=1.0,
                segment_type="stable",
                gaze_class="inside_visible",
                avg_gaze_x=100.0,
                avg_gaze_y=100.0,
                frame_count=30,
            )
            for i in range(20)
        ]
        prompt = build_behavior_prompt(make_features(), segments, [])
        # The "Temporal sequences:" payload should serialize at most 12 entries.
        idx = prompt.index("Temporal sequences:")
        json_start = prompt.index("[", idx)
        depth = 0
        end = json_start
        for i, ch in enumerate(prompt[json_start:], start=json_start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        decoded = json.loads(prompt[json_start:end])
        assert len(decoded) == 12
