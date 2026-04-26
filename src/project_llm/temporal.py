"""Temporal segmentation of gaze sequences.

Converts a :class:`ChildClipSequence` into contiguous temporal segments
(``stable``, ``shift``, ``occluded``, ``outside``, ``eyes_closed``,
``uncertain``).  Each segment groups consecutive frames that share the
same gaze behavior classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from project_llm.dataset import ChildClipSequence, FrameRecord


@dataclass(frozen=True)
class TemporalSegment:
    """One contiguous block of same-type gaze behavior within a clip."""

    child_id: str
    clip_id: str
    split: str
    segment_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    segment_type: str
    gaze_class: str
    avg_gaze_x: float | None
    avg_gaze_y: float | None
    frame_count: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _segment_type(current: FrameRecord, previous: FrameRecord | None) -> str:
    """Classify a single frame into one of six segment types.

    A ``shift`` is triggered by an explicit ``gaze_shift`` class OR by
    visible gaze movement >= 60 px between consecutive frames.
    """
    if current.gaze_class in {"inside_occluded", "inside_uncertain"}:
        return "occluded"
    if current.gaze_class == "outside_frame":
        return "outside"
    if current.gaze_class == "eyes_closed":
        return "eyes_closed"
    if current.gaze_class == "gaze_shift":
        return "shift"
    if current.gaze_class == "inside_visible":
        if previous and previous.gaze_class == "inside_visible":
            dx = current.gaze_x - previous.gaze_x
            dy = current.gaze_y - previous.gaze_y
            if (dx * dx + dy * dy) ** 0.5 >= 60.0:
                return "shift"
        return "stable"
    return "uncertain"


def build_temporal_segments(sequence: ChildClipSequence) -> list[TemporalSegment]:
    """Group consecutive same-type frames into :class:`TemporalSegment` objects."""
    segments: list[TemporalSegment] = []
    current_frames: list[FrameRecord] = []
    current_type: str | None = None
    previous_frame: FrameRecord | None = None

    def flush() -> None:
        if not current_frames or current_type is None:
            return
        start_frame = current_frames[0].frame
        end_frame = current_frames[-1].frame
        visible_points = [frame for frame in current_frames if frame.gaze_class == "inside_visible"]
        avg_gaze_x = mean(frame.gaze_x for frame in visible_points) if visible_points else None
        avg_gaze_y = mean(frame.gaze_y for frame in visible_points) if visible_points else None
        segments.append(
            TemporalSegment(
                child_id=sequence.child_id,
                clip_id=sequence.clip_id,
                split=sequence.split,
                segment_index=len(segments),
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=round((start_frame - 1) / sequence.fps, 4),
                end_time=round(end_frame / sequence.fps, 4),
                duration_seconds=round((end_frame - start_frame + 1) / sequence.fps, 4),
                segment_type=current_type,
                gaze_class=current_frames[0].gaze_class,
                avg_gaze_x=round(avg_gaze_x, 4) if avg_gaze_x is not None else None,
                avg_gaze_y=round(avg_gaze_y, 4) if avg_gaze_y is not None else None,
                frame_count=len(current_frames),
            )
        )

    for frame in sequence.frames:
        next_type = _segment_type(frame, previous_frame)
        if current_type is None or next_type == current_type:
            current_type = next_type
            current_frames.append(frame)
        else:
            flush()
            current_frames = [frame]
            current_type = next_type
        previous_frame = frame

    flush()
    return segments
