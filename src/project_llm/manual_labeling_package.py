"""Manual labeling package builder.

For each evaluation item, extracts a representative video frame,
draws a bounding box around the target child using OpenCV, and saves
the annotated image as a PNG.  Source videos are resolved locally or
downloaded via yt-dlp when not available on disk.
"""

from __future__ import annotations

import csv
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from project_llm.dataset import ChildClipSequence, load_metadata, load_child_sequences


DEFAULT_DATASET_ROOT = Path("data set/ChildPlay-gaze/ChildPlay-gaze")
DEFAULT_INPUT_CSV = Path("output/paper_eval/manual_eval_annotations.csv")
DEFAULT_OUTPUT_DIR = Path("output/paper_eval/manual_labeling_package")


@dataclass(frozen=True)
class ResolvedRow:
    sample_id: str
    child_id: str
    clip_id: str
    target_person_id: str
    target_person_number: int
    source_video_url: str
    source_video_id: str
    source_video_path: Path
    sequence: ChildClipSequence
    expected_video_width: int
    expected_video_height: int
    clip_frame_number: int
    source_frame_number: int
    timestamp_seconds: float
    timestamp_mmss: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    notes: str


def _normalize_columns(columns: list[str]) -> dict[str, str]:
    return {
        column.strip().lower().replace(" ", "_"): column
        for column in columns
    }


def _find_column(columns: list[str], aliases: list[str], required: bool = True) -> str | None:
    normalized = _normalize_columns(columns)
    for alias in aliases:
        key = alias.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Could not resolve any of the required columns: {aliases}")
    return None


def _parse_person_id(value: str) -> tuple[str, int]:
    token = value.strip()
    if token.startswith("person_"):
        return token, int(token.split("_", 1)[1])
    return f"person_{int(token)}", int(token)


def _parse_person_from_row(row: dict[str, Any], child_id_col: str, person_col: str | None) -> tuple[str, int]:
    if person_col and row.get(person_col):
        return _parse_person_id(str(row[person_col]))

    child_id = str(row[child_id_col])
    if ":person_" in child_id:
        person_text = child_id.split(":person_", 1)[1]
        return f"person_{int(person_text)}", int(person_text)
    raise ValueError(f"Could not infer target person from row: {child_id}")


def _parse_source_video_url(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("source_video_url is empty")
    return text


def _parse_video_id_from_url(url: str) -> str:
    if "v=" in url:
        return url.split("v=", 1)[1].split("&", 1)[0]
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    if tail:
        return tail
    raise ValueError(f"Could not parse video_id from url: {url}")


def _parse_clip_range(clip_id: str) -> tuple[int, int]:
    clip_suffix = clip_id.rsplit("_", 1)[1]
    frame_range = clip_suffix.replace("-downsampled", "")
    start_text, end_text = frame_range.split("-", 1)
    return int(start_text), int(end_text)


def _format_mmss(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _choose_representative_frame(
    row: dict[str, Any],
    sequence: ChildClipSequence,
    frame_col: str | None,
    time_col: str | None,
) -> tuple[int, str]:
    sequence_frames = sorted(sequence.frames, key=lambda item: item.frame)
    available = {frame.frame for frame in sequence_frames}

    if frame_col and row.get(frame_col) not in ("", None):
        frame_number = int(float(row[frame_col]))
        if frame_number in available:
            return frame_number, f"Used explicit frame column `{frame_col}`."
        raise ValueError(f"Explicit frame number {frame_number} is not present for {sequence.child_id}")

    if time_col and row.get(time_col) not in ("", None):
        timestamp_seconds = float(row[time_col])
        clip_frame = int(round(timestamp_seconds * sequence.fps)) + 1
        clip_frame = min(max(clip_frame, sequence_frames[0].frame), sequence_frames[-1].frame)
        if clip_frame in available:
            return clip_frame, f"Derived clip frame from `{time_col}`."

    middle_index = len(sequence_frames) // 2
    return sequence_frames[middle_index].frame, "Fell back to the middle annotated clip frame."


def _map_clip_frame_to_source_frame(clip_id: str, clip_frame_number: int, clip_frame_count: int) -> int:
    source_start_frame, source_end_frame = _parse_clip_range(clip_id)
    source_frame_count = source_end_frame - source_start_frame + 1
    if clip_frame_count <= 1:
        return source_start_frame
    scale = (source_frame_count - 1) / (clip_frame_count - 1)
    source_frame = source_start_frame + round((clip_frame_number - 1) * scale)
    return min(max(source_frame, source_start_frame), source_end_frame)


def _build_video_search_paths(dataset_root: Path, video_id: str, clip_id: str) -> list[Path]:
    return [
        dataset_root / "videos" / f"{video_id}.mp4",
        dataset_root / "videos" / f"{video_id}.webm",
        dataset_root / "clips" / f"{clip_id}.mp4",
        dataset_root / "clips" / f"{clip_id}.webm",
    ]


def _resolve_local_video_path(dataset_root: Path, video_id: str, clip_id: str) -> Path | None:
    for candidate in _build_video_search_paths(dataset_root, video_id, clip_id):
        if candidate.exists():
            return candidate
    return None


def _download_video_with_ytdlp(video_url: str, video_path: Path, logger: logging.Logger) -> Path:
    if video_path.exists():
        logger.info("Reusing cached source video: %s", video_path)
        return video_path

    video_path.parent.mkdir(parents=True, exist_ok=True)
    output_template = str(video_path)
    command = [
        str(Path(".venv/bin/yt-dlp")),
        "--no-progress",
        "--no-warnings",
        "--extractor-args",
        "youtube:player_client=android",
        "--format",
        "best[ext=mp4]/best",
        "--output",
        output_template,
        video_url,
    ]
    logger.info("Downloading source video with yt-dlp: %s", video_url)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        parts = [part.strip() for part in [result.stderr, result.stdout] if part and part.strip()]
        message = "\n".join(parts) if parts else "yt-dlp failed without output"
        raise RuntimeError(message)
    if not video_path.exists():
        matches = sorted(video_path.parent.glob(f"{video_path.stem}*"))
        media_matches = [path for path in matches if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}]
        if len(media_matches) == 1:
            media_matches[0].rename(video_path)
        elif not video_path.exists():
            raise RuntimeError(f"yt-dlp completed but expected video file was not found for {video_url}")
    return video_path


def _draw_target_box(
    image: Any,
    bbox: tuple[float, float, float, float],
    label: str,
) -> Any:
    height, width = image.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(x1 + 1, min(width - 1, int(round(x + w))))
    y2 = max(y1 + 1, min(height - 1, int(round(y + h))))

    color = (0, 255, 255)
    thickness = max(3, round(min(width, height) / 250))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(width, height) / 1200)
    text = label
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
    text_x = x1
    text_y = y1 - 12 if y1 - 12 > text_height else min(height - 10, y2 + text_height + 12)
    box_start = (text_x, max(0, text_y - text_height - baseline - 6))
    box_end = (min(width - 1, text_x + text_width + 10), min(height - 1, text_y + baseline + 4))
    cv2.rectangle(image, box_start, box_end, color, thickness=-1)
    cv2.putText(image, text, (text_x + 5, text_y), font, font_scale, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return image


def _open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def _extract_frame(video_path: Path, source_frame_number: int, timestamp_seconds: float) -> tuple[Any, int, int]:
    capture = _open_video_capture(video_path)
    zero_based_index = max(0, source_frame_number - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, zero_based_index)
    success, frame = capture.read()
    if (not success or frame is None) and timestamp_seconds >= 0:
        capture.release()
        capture = _open_video_capture(video_path)
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        success, frame = capture.read()
    if not success or frame is None:
        capture.release()
        raise RuntimeError(f"Could not read frame {source_frame_number} from {video_path}")
    height, width = frame.shape[:2]
    capture.release()
    return frame, width, height


def _scale_bbox(
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    expected_width: int,
    expected_height: int,
    actual_width: int,
    actual_height: int,
) -> tuple[float, float, float, float]:
    if expected_width <= 0 or expected_height <= 0:
        return bbox_x, bbox_y, bbox_width, bbox_height
    scale_x = actual_width / expected_width
    scale_y = actual_height / expected_height
    return (
        bbox_x * scale_x,
        bbox_y * scale_y,
        bbox_width * scale_x,
        bbox_height * scale_y,
    )


def _build_readme(output_dir: Path) -> None:
    lines = [
        "# Manual Labeling Package",
        "",
        "This package contains one representative annotated frame for each of the 69 manual-evaluation items.",
        "",
        "## Contents",
        "- `images/`: one PNG per sample, named `sample_001.png` through `sample_069.png`.",
        "- `index.csv`: master table linking each output image to its source video file, source YouTube URL, clip, frame, timestamp, and bounding box.",
        "- `processing_log.txt`: detailed processing log with resolution and fallback notes.",
        "- `unresolved_items.csv`: only present if any rows could not be resolved automatically.",
        "",
        "## How Images Were Generated",
        "- Rows were read from `output/paper_eval/manual_eval_annotations.csv`.",
        "- Clip metadata and per-frame boxes were resolved from the ChildPlay dataset files already used by the project.",
        "- For each row, the target person was matched to the corresponding child sequence and one representative frame was selected deterministically.",
        "- If a row did not contain an explicit frame or timestamp, the middle annotated clip frame was used.",
        "- The selected clip frame was mapped back to the source video frame, extracted, and annotated with a visible bounding box and short target label.",
        "",
        "## How To Inspect A Row",
        "- Open `index.csv` and look up the `sample_id`, such as `sample_014`.",
        "- Open the matching image path in `output_image_path` to inspect the annotated frame.",
        "- Use `source_video_url`, `source_video_path`, `frame_number`, `timestamp_seconds`, or `timestamp_mmss` to trace the frame back to the original source video.",
        "",
        "## Traceback To Source",
        "- `timestamp_mmss` is meant for quick human inspection.",
        "- `frame_number` is the exact source-video frame used when extracting the image.",
        "- `bbox_x`, `bbox_y`, `bbox_width`, and `bbox_height` are the box coordinates used on the exported image.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("project_llm.manual_labeling_package")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(output_dir / "processing_log.txt", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _infer_schema(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)
    return {
        "child_id": _find_column(columns, ["child_id", "sample_child_id"]),
        "clip_id": _find_column(columns, ["clip_id", "source_clip_id"]),
        "source_video_url": _find_column(columns, ["source_video_url", "video_url", "youtube_url"]),
        "target_person": _find_column(columns, ["target_person_id", "person_id", "target_person"], required=False),
        "frame_number": _find_column(columns, ["frame_number", "source_frame", "frame"], required=False),
        "timestamp_seconds": _find_column(columns, ["timestamp_seconds", "source_timestamp_seconds", "time_seconds"], required=False),
    }


def _index_sequences(sequences: list[ChildClipSequence]) -> dict[tuple[str, str], ChildClipSequence]:
    return {(sequence.child_id, sequence.clip_id): sequence for sequence in sequences}


def _resolve_row(
    row: dict[str, Any],
    sample_id: str,
    schema: dict[str, str | None],
    sequence_index: dict[tuple[str, str], ChildClipSequence],
    videos_meta: dict[str, dict[str, Any]],
    dataset_root: Path,
    downloaded_videos_dir: Path,
) -> ResolvedRow:
    child_id = str(row[schema["child_id"]]).strip()
    clip_id = str(row[schema["clip_id"]]).strip()
    target_person_id, target_person_number = _parse_person_from_row(row, schema["child_id"], schema["target_person"])
    sequence_key = (child_id, clip_id)
    if sequence_key not in sequence_index:
        raise ValueError(f"Missing child sequence for {child_id}")
    sequence = sequence_index[sequence_key]
    video_url = _parse_source_video_url(str(row[schema["source_video_url"]]))
    video_id = _parse_video_id_from_url(video_url)
    if video_id not in videos_meta:
        raise ValueError(f"Video metadata missing for {video_id}")
    clip_frame_number, frame_note = _choose_representative_frame(row, sequence, schema["frame_number"], schema["timestamp_seconds"])
    clip_frame_count = len(sequence.frames)
    source_frame_number = _map_clip_frame_to_source_frame(clip_id, clip_frame_number, clip_frame_count)
    fps = float(sequence.fps)
    timestamp_seconds = round((source_frame_number - 1) / fps, 3)
    timestamp_mmss = _format_mmss(timestamp_seconds)

    frame_lookup = {frame.frame: frame for frame in sequence.frames}
    if clip_frame_number not in frame_lookup:
        raise ValueError(f"Representative clip frame {clip_frame_number} is missing for {child_id}")
    chosen_frame = frame_lookup[clip_frame_number]

    local_video = _resolve_local_video_path(dataset_root, video_id, clip_id)
    if local_video is None:
        local_video = downloaded_videos_dir / f"{video_id}.mp4"

    return ResolvedRow(
        sample_id=sample_id,
        child_id=child_id,
        clip_id=clip_id,
        target_person_id=target_person_id,
        target_person_number=target_person_number,
        source_video_url=video_url,
        source_video_id=video_id,
        source_video_path=local_video,
        sequence=sequence,
        expected_video_width=int(videos_meta[video_id]["width"]),
        expected_video_height=int(videos_meta[video_id]["height"]),
        clip_frame_number=clip_frame_number,
        source_frame_number=source_frame_number,
        timestamp_seconds=timestamp_seconds,
        timestamp_mmss=timestamp_mmss,
        bbox_x=float(chosen_frame.bbox_x),
        bbox_y=float(chosen_frame.bbox_y),
        bbox_width=float(chosen_frame.bbox_width),
        bbox_height=float(chosen_frame.bbox_height),
        notes=frame_note,
    )


def build_manual_labeling_package(
    input_csv: Path = DEFAULT_INPUT_CSV,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    logger = _setup_logger(output_dir)
    images_dir = output_dir / "images"
    videos_dir = output_dir / "source_videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    schema = _infer_schema(df)
    rows = df.to_dict(orient="records")
    logger.info("Loaded %s rows from %s", len(rows), input_csv)

    metadata = load_metadata(dataset_root)
    sequences = load_child_sequences(dataset_root, split="test")
    sequence_index = _index_sequences(sequences)

    output_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    successful_images = 0

    for row_number, row in enumerate(rows, start=1):
        sample_id = f"sample_{row_number:03d}"
        output_image_path = images_dir / f"{sample_id}.png"
        try:
            resolved = _resolve_row(
                row=row,
                sample_id=sample_id,
                schema=schema,
                sequence_index=sequence_index,
                videos_meta=metadata.videos,
                dataset_root=dataset_root,
                downloaded_videos_dir=videos_dir,
            )
            source_video_path = resolved.source_video_path
            if not source_video_path.exists():
                source_video_path = _download_video_with_ytdlp(resolved.source_video_url, source_video_path, logger)

            frame, actual_width, actual_height = _extract_frame(
                source_video_path,
                resolved.source_frame_number,
                resolved.timestamp_seconds,
            )
            scaled_bbox = _scale_bbox(
                resolved.bbox_x,
                resolved.bbox_y,
                resolved.bbox_width,
                resolved.bbox_height,
                resolved.expected_video_width,
                resolved.expected_video_height,
                actual_width,
                actual_height,
            )
            annotated = _draw_target_box(frame, scaled_bbox, f"target: {resolved.target_person_id}")
            written = cv2.imwrite(str(output_image_path), annotated)
            if not written:
                raise RuntimeError(f"Could not write output image to {output_image_path}")
            if output_image_path.stat().st_size <= 0:
                raise RuntimeError(f"Output image is empty: {output_image_path}")

            bbox_x, bbox_y, bbox_width, bbox_height = scaled_bbox
            output_rows.append(
                {
                    "sample_id": sample_id,
                    "output_image_path": str(output_image_path.resolve()),
                    "source_video_path": str(source_video_path.resolve()),
                    "source_video_url": resolved.source_video_url,
                    "source_clip_id": resolved.clip_id,
                    "target_person_id": resolved.target_person_id,
                    "frame_number": resolved.source_frame_number,
                    "timestamp_seconds": resolved.timestamp_seconds,
                    "timestamp_mmss": resolved.timestamp_mmss,
                    "bbox_x": round(bbox_x, 2),
                    "bbox_y": round(bbox_y, 2),
                    "bbox_width": round(bbox_width, 2),
                    "bbox_height": round(bbox_height, 2),
                    "resolution_width": actual_width,
                    "resolution_height": actual_height,
                    "status": "success",
                    "notes": resolved.notes,
                }
            )
            successful_images += 1
            logger.info(
                "Generated %s for %s using video=%s frame=%s clip_frame=%s",
                sample_id,
                resolved.child_id,
                source_video_path,
                resolved.source_frame_number,
                resolved.clip_frame_number,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            logger.exception("Failed to process row %s (%s)", row_number, sample_id)
            output_rows.append(
                {
                    "sample_id": sample_id,
                    "output_image_path": "",
                    "source_video_path": "",
                    "source_video_url": str(row.get(schema["source_video_url"], "")),
                    "source_clip_id": str(row.get(schema["clip_id"], "")),
                    "target_person_id": "",
                    "frame_number": "",
                    "timestamp_seconds": "",
                    "timestamp_mmss": "",
                    "bbox_x": "",
                    "bbox_y": "",
                    "bbox_width": "",
                    "bbox_height": "",
                    "resolution_width": "",
                    "resolution_height": "",
                    "status": "failed",
                    "notes": message,
                }
            )
            unresolved_rows.append(
                {
                    "sample_id": sample_id,
                    "child_id": row.get(schema["child_id"], ""),
                    "clip_id": row.get(schema["clip_id"], ""),
                    "reason": message,
                }
            )

    index_columns = [
        "sample_id",
        "output_image_path",
        "source_video_path",
        "source_video_url",
        "source_clip_id",
        "target_person_id",
        "frame_number",
        "timestamp_seconds",
        "timestamp_mmss",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "resolution_width",
        "resolution_height",
        "status",
        "notes",
    ]
    index_path = output_dir / "index.csv"
    pd.DataFrame(output_rows, columns=index_columns).to_csv(index_path, index=False)

    if unresolved_rows:
        pd.DataFrame(unresolved_rows).to_csv(output_dir / "unresolved_items.csv", index=False)
    elif (output_dir / "unresolved_items.csv").exists():
        (output_dir / "unresolved_items.csv").unlink()

    _build_readme(output_dir)

    index_df = pd.read_csv(index_path)
    if len(index_df) != len(rows):
        raise RuntimeError(f"index.csv row count mismatch: expected {len(rows)}, found {len(index_df)}")

    success_df = index_df[index_df["status"] == "success"]
    for image_path in success_df["output_image_path"]:
        image_file = Path(image_path)
        if not image_file.exists():
            raise RuntimeError(f"Missing generated image referenced by index.csv: {image_file}")
        image = cv2.imread(str(image_file))
        if image is None or image.size == 0:
            raise RuntimeError(f"Generated image is unreadable: {image_file}")

    summary = {
        "total_rows_read": len(rows),
        "total_images_generated": successful_images,
        "total_unresolved": len(unresolved_rows),
        "output_directory": str(output_dir.resolve()),
    }
    logger.info("Summary: %s", summary)
    return summary
