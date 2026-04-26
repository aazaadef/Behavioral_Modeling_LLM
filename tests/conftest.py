"""Shared pytest fixtures for the project_llm test suite.

Builds in-memory ``ChildClipSequence`` and ``BehavioralFeatures`` objects
so unit tests can run without the ChildPlay-gaze dataset on disk. The
fixtures here are deliberately small and fully synthetic — anything that
needs the real dataset belongs in a ``@pytest.mark.slow`` test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from project_llm.dataset import ChildClipSequence, FrameRecord  # noqa: E402
from project_llm.features import BehavioralFeatures  # noqa: E402


def _frame(
    frame: int,
    gaze_class: str,
    *,
    gaze_x: float = 100.0,
    gaze_y: float = 100.0,
    bbox_x: float = 0.0,
    bbox_y: float = 0.0,
    bbox_width: float = 50.0,
    bbox_height: float = 50.0,
) -> FrameRecord:
    return FrameRecord(
        frame=frame,
        person_id=1,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        gaze_class=gaze_class,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        is_child=1,
    )


def make_sequence(
    classes: list[str],
    *,
    child_id: str = "clip_x:person_1",
    clip_id: str = "clip_x",
    split: str = "test",
    fps: float = 30.0,
    base_x: float = 100.0,
    base_y: float = 100.0,
) -> ChildClipSequence:
    """Build a minimal ChildClipSequence from a list of gaze_class values."""
    frames = [
        _frame(
            i + 1,
            cls,
            gaze_x=base_x + i * 0.5,
            gaze_y=base_y + i * 0.5,
        )
        for i, cls in enumerate(classes)
    ]
    return ChildClipSequence(
        child_id=child_id,
        clip_id=clip_id,
        person_id=1,
        split=split,
        video_id="video_x",
        channel_id="channel_x",
        fps=fps,
        frame_count=len(frames),
        resolution="1280x720",
        frames=frames,
    )


@pytest.fixture
def make_seq():
    """Factory fixture: pass a list of gaze_class strings."""
    return make_sequence


@pytest.fixture
def make_features():
    """Factory fixture: build a BehavioralFeatures with sensible defaults."""

    def _build(**overrides) -> BehavioralFeatures:
        defaults = dict(
            child_id="clip_x:person_1",
            clip_id="clip_x",
            split="test",
            observed_frames=100,
            visible_ratio=0.85,
            on_screen_ratio=0.9,
            gaze_shift_ratio=0.05,
            occlusion_ratio=0.05,
            eyes_closed_ratio=0.0,
            mean_head_area=2500.0,
            mean_head_motion=10.0,
            mean_gaze_motion=20.0,
            visible_gaze_fraction=0.85,
            gaze_shift_events=5,
            max_visible_streak=80,
            max_occlusion_streak=2,
            attention_stability_score=0.75,
            dominant_gaze_class="inside_visible",
        )
        defaults.update(overrides)
        return BehavioralFeatures(**defaults)

    return _build
