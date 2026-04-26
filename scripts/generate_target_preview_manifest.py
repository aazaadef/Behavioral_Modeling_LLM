"""Create a preview-request manifest for selected target child IDs.

Checks local asset availability (clips, extracted images) and writes
a CSV + JSON manifest for downstream preview generation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze"
MANUAL_ANNOTATIONS = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"
OUTPUT_CSV = ROOT / "output" / "paper_eval" / "target_preview_requests.csv"
OUTPUT_JSON = ROOT / "output" / "paper_eval" / "target_preview_requests.json"

TARGET_CHILD_IDS = [
    "6mA6UAoT3M0_6165-6361:person_1",
    "9DNwRwt5kI4_14988-15270:person_2",
    "9DNwRwt5kI4_14988-15270:person_3",
    "9DNwRwt5kI4_2442-3179:person_3",
    "LHT2zVYvObg_1790-1917:person_1",
    "ND7pXuhs3VM_1800-1925:person_2",
    "ND7pXuhs3VM_3066-3159:person_1",
    "NIk1-ck4c6Q_13651-13719:person_3",
    "aWV7UUMddCU_5934-6205:person_2",
    "aWV7UUMddCU_6517-7514:person_1",
    "f6wqlpG9rd0_9404-9541:person_1",
    "f6wqlpG9rd0_9704-9949:person_1",
    "zmZpa1p5zaE_1028-1129:person_1",
]


def _detect_available_assets(dataset_root: Path, clip_id: str) -> dict[str, str]:
    clip_mp4 = dataset_root / "clips" / f"{clip_id}.mp4"
    image_dir = dataset_root / "images" / clip_id
    return {
        "clip_mp4_path": str(clip_mp4.relative_to(ROOT)) if clip_mp4.exists() else "",
        "image_dir_path": str(image_dir.relative_to(ROOT)) if image_dir.exists() else "",
        "asset_status": "ready" if clip_mp4.exists() or image_dir.exists() else "missing_clips_and_images",
    }


def main() -> int:
    with MANUAL_ANNOTATIONS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_child_id = {row["child_id"]: row for row in rows}
    missing = [child_id for child_id in TARGET_CHILD_IDS if child_id not in by_child_id]
    if missing:
        raise ValueError(f"Missing requested child_ids in manual_eval_annotations.csv: {missing}")

    output_rows = []
    for child_id in TARGET_CHILD_IDS:
        row = by_child_id[child_id]
        asset_info = _detect_available_assets(DATASET_ROOT, row["clip_id"])
        output_rows.append(
            {
                "child_id": row["child_id"],
                "clip_id": row["clip_id"],
                "source_video_url": row["source_video_url"],
                "target_person_and_time": row["target_person_and_time"],
                "target_person_bbox_hint": row["target_person_bbox_hint"],
                "rule_based_label": row["rule_based_label"],
                "deberta_label": row["deberta_label"],
                "distilbert_label": row["distilbert_label"],
                "bart_label": row["bart_label"],
                "preview_image_path": "",
                **asset_info,
                "note": "Real preview image cannot be generated until the ChildPlay clips or extracted images are available locally.",
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    OUTPUT_JSON.write_text(json.dumps(output_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"rows={len(output_rows)}")
    print(f"csv={OUTPUT_CSV.relative_to(ROOT)}")
    print(f"json={OUTPUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
