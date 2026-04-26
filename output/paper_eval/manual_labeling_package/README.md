# Manual Labeling Package

This package contains one representative annotated frame for each of the 69 manual-evaluation items.

## Contents
- `images/`: one PNG per sample, named `sample_001.png` through `sample_069.png`.
- `index.csv`: master table linking each output image to its source video file, source YouTube URL, clip, frame, timestamp, and bounding box.
- `processing_log.txt`: detailed processing log with resolution and fallback notes.
- `unresolved_items.csv`: only present if any rows could not be resolved automatically.

## How Images Were Generated
- Rows were read from `output/paper_eval/manual_eval_annotations.csv`.
- Clip metadata and per-frame boxes were resolved from the ChildPlay dataset files already used by the project.
- For each row, the target person was matched to the corresponding child sequence and one representative frame was selected deterministically.
- If a row did not contain an explicit frame or timestamp, the middle annotated clip frame was used.
- The selected clip frame was mapped back to the source video frame, extracted, and annotated with a visible bounding box and short target label.

## How To Inspect A Row
- Open `index.csv` and look up the `sample_id`, such as `sample_014`.
- Open the matching image path in `output_image_path` to inspect the annotated frame.
- Use `source_video_url`, `source_video_path`, `frame_number`, `timestamp_seconds`, or `timestamp_mmss` to trace the frame back to the original source video.

## Traceback To Source
- `timestamp_mmss` is meant for quick human inspection.
- `frame_number` is the exact source-video frame used when extracting the image.
- `bbox_x`, `bbox_y`, `bbox_width`, and `bbox_height` are the box coordinates used on the exported image.
