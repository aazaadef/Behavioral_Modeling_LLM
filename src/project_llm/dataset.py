"""Data loading layer for the ChildPlay-gaze dataset.

Reads CSV annotation files, validates their schema, filters to child-only
rows (``is_child == 1``), and groups frame-level records into
:class:`ChildClipSequence` objects -- one per child per clip.  This module
is the single entry point for all raw data consumed by the pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Required columns that every annotation CSV must contain.
ANNOTATION_COLUMNS = {
    "clip",
    "frame",
    "person_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "gaze_class",
    "gaze_x",
    "gaze_y",
    "is_child",
}


@dataclass(frozen=True)
class FrameRecord:
    """A single annotated frame for one tracked person in one clip."""

    frame: int
    person_id: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    gaze_class: str
    gaze_x: float
    gaze_y: float
    is_child: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChildClipSequence:
    """All annotated frames for one child in one clip, plus clip metadata.

    ``child_id`` has the format ``<clip_id>:person_<N>``.
    """

    child_id: str
    clip_id: str
    person_id: int
    split: str
    video_id: str
    channel_id: str
    fps: float
    frame_count: int
    resolution: str
    frames: list[FrameRecord]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [frame.to_dict() for frame in self.frames]
        return payload


@dataclass(frozen=True)
class DatasetIndex:
    """Lookup tables built from clips.csv, splits.csv, and videos.csv."""

    clips: dict[str, dict[str, Any]]
    splits: dict[str, str]
    videos: dict[str, dict[str, Any]]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metadata(dataset_root: Path) -> DatasetIndex:
    """Load and cross-validate clips.csv, splits.csv, and videos.csv."""
    clips_path = dataset_root / "clips.csv"
    splits_path = dataset_root / "splits.csv"
    videos_path = dataset_root / "videos.csv"
    clips_rows = _read_csv_rows(clips_path)
    splits_rows = _read_csv_rows(splits_path)
    videos_rows = _read_csv_rows(videos_path)

    clips = {row["clip"]: row for row in clips_rows}
    splits = {row["clip"]: row["split"] for row in splits_rows}
    videos = {row["video_id"]: row for row in videos_rows}

    missing_split = sorted(set(clips) - set(splits))
    if missing_split:
        raise ValueError(f"Missing split rows for clips: {missing_split[:5]}")

    for clip, row in clips.items():
        if row["split"] != splits[clip]:
            raise ValueError(f"Split mismatch for clip {clip}: {row['split']} != {splits[clip]}")
        if row["video_id"] not in videos:
            raise ValueError(f"Clip {clip} references missing video_id {row['video_id']}")

    return DatasetIndex(clips=clips, splits=splits, videos=videos)


def discover_annotation_files(dataset_root: Path, split: str | None = None) -> list[Path]:
    """Find annotation CSVs under ``annotations/<split>/``, sorted by name."""
    annotations_root = dataset_root / "annotations"
    splits = [split] if split else ["train", "val", "test"]
    files: list[Path] = []
    for split_name in splits:
        files.extend(sorted((annotations_root / split_name).glob("*.csv")))
    return files


def _parse_frame(row: dict[str, str]) -> FrameRecord:
    return FrameRecord(
        frame=int(row["frame"]),
        person_id=int(row["person_id"]),
        bbox_x=float(row["bbox_x"]),
        bbox_y=float(row["bbox_y"]),
        bbox_width=float(row["bbox_width"]),
        bbox_height=float(row["bbox_height"]),
        gaze_class=row["gaze_class"],
        gaze_x=float(row["gaze_x"]),
        gaze_y=float(row["gaze_y"]),
        is_child=int(row["is_child"]),
    )


def _validate_annotation_schema(path: Path, fieldnames: list[str]) -> None:
    missing = ANNOTATION_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def load_child_sequences(dataset_root: Path, split: str | None = None) -> list[ChildClipSequence]:
    """Main data loader: read annotations, keep child rows, return sorted sequences."""
    metadata = load_metadata(dataset_root)
    sequences: list[ChildClipSequence] = []

    for annotation_path in discover_annotation_files(dataset_root, split=split):
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_annotation_schema(annotation_path, reader.fieldnames or [])
            grouped_frames: dict[int, list[FrameRecord]] = {}
            clip_id: str | None = None
            for row in reader:
                clip_id = row["clip"]
                if int(row["is_child"]) != 1:
                    continue
                frame = _parse_frame(row)
                grouped_frames.setdefault(frame.person_id, []).append(frame)

        if not clip_id:
            continue

        clip_meta = metadata.clips[clip_id]
        for person_id, frames in grouped_frames.items():
            sorted_frames = sorted(frames, key=lambda item: item.frame)
            sequences.append(
                ChildClipSequence(
                    child_id=f"{clip_id}:person_{person_id}",
                    clip_id=clip_id,
                    person_id=person_id,
                    split=clip_meta["split"],
                    video_id=clip_meta["video_id"],
                    channel_id=clip_meta["channel_id"],
                    fps=float(clip_meta["fps"]),
                    frame_count=int(clip_meta["frame_count"]),
                    resolution=clip_meta["resolution"],
                    frames=sorted_frames,
                )
            )

    return sorted(sequences, key=lambda item: (item.split, item.clip_id, item.person_id))
