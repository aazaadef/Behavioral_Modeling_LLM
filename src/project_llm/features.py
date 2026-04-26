"""Behavioral feature extraction from gaze sequences.

Computes 14+ numeric features from a :class:`ChildClipSequence`, producing
a :class:`BehavioralFeatures` dataclass that serves as the primary input
to all classification backends.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from math import hypot

from project_llm.dataset import ChildClipSequence


@dataclass(frozen=True)
class BehavioralFeatures:
    """Numeric behavioral metrics derived from one child's gaze sequence."""
    child_id: str
    clip_id: str
    split: str
    observed_frames: int
    visible_ratio: float
    on_screen_ratio: float
    gaze_shift_ratio: float
    occlusion_ratio: float
    eyes_closed_ratio: float
    mean_head_area: float
    mean_head_motion: float
    mean_gaze_motion: float
    visible_gaze_fraction: float
    gaze_shift_events: int
    max_visible_streak: int
    max_occlusion_streak: int
    attention_stability_score: float
    dominant_gaze_class: str

    def to_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def extract_behavioral_features(sequence: ChildClipSequence) -> BehavioralFeatures:
    """Compute all behavioral features for a single child-clip sequence.

    The ``attention_stability_score`` is a weighted composite:
    0.45 * visibility + 0.25 * streak_ratio + 0.15 * (1 - motion) + 0.15 * (1 - shift).
    """
    frames = sequence.frames
    counts = Counter(frame.gaze_class for frame in frames)
    total = len(frames)
    visible = counts["inside_visible"]
    gaze_shift = counts["gaze_shift"]
    occluded = counts["inside_occluded"] + counts["inside_uncertain"]
    eyes_closed = counts["eyes_closed"]

    head_centers = [
        (frame.bbox_x + frame.bbox_width / 2.0, frame.bbox_y + frame.bbox_height / 2.0)
        for frame in frames
    ]
    gaze_points = [(frame.gaze_x, frame.gaze_y) for frame in frames if frame.gaze_class == "inside_visible"]

    visibility_mask = [frame.gaze_class == "inside_visible" for frame in frames]
    occlusion_mask = [frame.gaze_class in {"inside_occluded", "inside_uncertain"} for frame in frames]

    def average_motion(points: list[tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        distances = [hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:])]
        return sum(distances) / len(distances)

    def max_streak(mask: list[bool]) -> int:
        streaks = [sum(1 for _ in group) for value, group in groupby(mask) if value]
        return max(streaks, default=0)

    mean_head_area = sum(frame.bbox_width * frame.bbox_height for frame in frames) / total
    mean_head_motion = average_motion(head_centers)
    mean_gaze_motion = average_motion(gaze_points)
    visible_ratio = visible / total
    gaze_shift_ratio = gaze_shift / total
    occlusion_ratio = occluded / total
    eyes_closed_ratio = eyes_closed / total
    attention_stability_score = max(
        0.0,
        min(
            1.0,
            (0.45 * visible_ratio)
            + (0.25 * (max_streak(visibility_mask) / total))
            + (0.15 * (1.0 - min(mean_gaze_motion / 300.0, 1.0)))
            + (0.15 * (1.0 - gaze_shift_ratio)),
        ),
    )

    return BehavioralFeatures(
        child_id=sequence.child_id,
        clip_id=sequence.clip_id,
        split=sequence.split,
        observed_frames=total,
        visible_ratio=visible_ratio,
        on_screen_ratio=(visible + counts["inside_occluded"] + counts["inside_uncertain"]) / total,
        gaze_shift_ratio=gaze_shift_ratio,
        occlusion_ratio=occlusion_ratio,
        eyes_closed_ratio=eyes_closed_ratio,
        mean_head_area=mean_head_area,
        mean_head_motion=mean_head_motion,
        mean_gaze_motion=mean_gaze_motion,
        visible_gaze_fraction=visible_ratio,
        gaze_shift_events=gaze_shift,
        max_visible_streak=max_streak(visibility_mask),
        max_occlusion_streak=max_streak(occlusion_mask),
        attention_stability_score=attention_stability_score,
        dominant_gaze_class=max(counts, key=counts.get),
    )
