"""Convert an external gaze-annotation dataset into the
ChildPlay-compatible CSV format consumed by this project's pipeline.

This is a deliberately thin scaffold. The intent is:

  * Lock down the *output* schema to match exactly what the rest
    of the pipeline expects (ChildPlay annotation CSVs).
  * Provide explicit ``extract_*`` hooks for every column. Each
    hook takes one row of the external dataset and returns the
    ChildPlay-compatible value, or NaN if no value can be derived.
  * Fail loudly when a derivation is impossible (e.g., source
    dataset has no occlusion class) rather than silently filling
    with zeros.

Target output format — one CSV per (clip × person) under
``data set/<external_name>/annotations/<split>/``:

    clip,frame,person_id,bbox_x,bbox_y,bbox_width,bbox_height,
    gaze_class,gaze_x,gaze_y,is_child

Allowed gaze_class values:
    inside_visible, inside_occluded, inside_uncertain,
    outside_frame, eyes_closed, gaze_shift

Three usage paths:

  1. **VideoAttentionTarget (Chong 2020).** Use ``adapt_vat()`` —
     the function is a stub today; fill in the path-resolution
     logic once the VAT release is downloaded.

  2. **GazeFollow (Recasens 2015).** Single-frame data; use
     ``adapt_gazefollow()`` only for the supplementary
     "single-frame ablation" use case.

  3. **Custom dataset.** Subclass ``ExternalDatasetAdapter`` and
     implement the ``extract_*`` hooks for your schema.

The script is intentionally not invoked from the test suite — it
will not run until the external dataset is on disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


# Allowed gaze_class values (must match ChildPlay-gaze schema).
ALLOWED_GAZE_CLASSES = {
    "inside_visible",
    "inside_occluded",
    "inside_uncertain",
    "outside_frame",
    "eyes_closed",
    "gaze_shift",
}

# The exact column order ChildPlay annotation CSVs use.
CHILDPLAY_COLUMNS = [
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
]


@dataclass
class ExternalRow:
    """One row from the external dataset, after light normalisation.

    The adapter's extract_* hooks consume this and emit one row of
    the ChildPlay schema. Field names mirror the most common
    annotation columns across VideoAttentionTarget / GazeFollow.
    """

    clip_id: str
    frame: int
    person_id: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    gaze_x: float | None  # None if outside frame / not annotated
    gaze_y: float | None
    is_outside_frame: bool
    is_child: int  # 1 or 0; pass through if known, else 1 by default


class ExternalDatasetAdapter:
    """Base adapter — subclass and implement extract_* hooks per dataset."""

    name: str = "external"

    def iter_rows(self) -> Iterable[ExternalRow]:
        """Yield one ExternalRow per annotated person-frame in the source."""
        raise NotImplementedError

    # ---- Per-column extract hooks ------------------------------------

    def extract_gaze_class(self, row: ExternalRow) -> str:
        """Map external annotation to ChildPlay gaze_class.

        Default policy:
          outside frame → "outside_frame"
          gaze coordinates present → "inside_visible"
          gaze coordinates missing AND head visible → "inside_occluded"

        Subclasses should override when the external dataset has
        richer annotations (e.g. an explicit eyes_closed flag).
        """
        if row.is_outside_frame:
            return "outside_frame"
        if row.gaze_x is None or row.gaze_y is None:
            return "inside_occluded"
        return "inside_visible"

    def extract_is_child(self, row: ExternalRow) -> int:
        return int(row.is_child)

    # ---- Entry point --------------------------------------------------

    def to_childplay_dataframe(self) -> pd.DataFrame:
        records = []
        for row in self.iter_rows():
            gaze_class = self.extract_gaze_class(row)
            if gaze_class not in ALLOWED_GAZE_CLASSES:
                raise ValueError(
                    f"{self.name}: extract_gaze_class returned "
                    f"{gaze_class!r}, which is not in {ALLOWED_GAZE_CLASSES}"
                )
            gx = -1.0 if row.gaze_x is None else float(row.gaze_x)
            gy = -1.0 if row.gaze_y is None else float(row.gaze_y)
            records.append(
                {
                    "clip": row.clip_id,
                    "frame": int(row.frame),
                    "person_id": int(row.person_id),
                    "bbox_x": float(row.bbox_x),
                    "bbox_y": float(row.bbox_y),
                    "bbox_width": float(row.bbox_width),
                    "bbox_height": float(row.bbox_height),
                    "gaze_class": gaze_class,
                    "gaze_x": gx,
                    "gaze_y": gy,
                    "is_child": self.extract_is_child(row),
                }
            )
        return pd.DataFrame.from_records(records, columns=CHILDPLAY_COLUMNS)

    def write_childplay_split(self, output_dir: Path, split: str) -> None:
        """Write per-clip CSVs under output_dir/<split>/ following
        ChildPlay's one-CSV-per-clip convention."""
        df = self.to_childplay_dataframe()
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for clip, sub in df.groupby("clip"):
            out_csv = split_dir / f"{clip}.csv"
            sub.sort_values(["frame", "person_id"]).to_csv(out_csv, index=False)
        print(f"[{self.name}] wrote {df['clip'].nunique()} clips to {split_dir}/")


# ----------------------------------------------------------------------
# VideoAttentionTarget adapter — recommended primary target.
#
# Stub: fill in iter_rows() once the VAT release is on disk. The
# function below documents the expected file layout. Returns of
# inside_uncertain, gaze_shift, and eyes_closed are intentionally
# *not* synthesised — those flags are absent in VAT and we keep them
# absent to make the comparison fair.
# ----------------------------------------------------------------------
class VATAdapter(ExternalDatasetAdapter):
    name = "video_attention_target"

    def __init__(self, vat_root: Path, split: str) -> None:
        self.vat_root = vat_root
        self.split = split

    def iter_rows(self) -> Iterable[ExternalRow]:
        # Expected VAT layout:
        #   vat_root/annotations/<split>/<show>/<clip_id>.txt
        # Each row of each .txt: frame, x_min, y_min, x_max, y_max, gaze_x, gaze_y
        # gaze_x = -1 means "outside frame" in VAT's convention.
        # See the official VAT release for column details:
        #   http://chong.in/research/social-gaze
        raise NotImplementedError(
            "VATAdapter.iter_rows is a stub. Fill in once the "
            "VideoAttentionTarget release is downloaded; consult the "
            "VAT README for the exact annotation layout."
        )

    def extract_gaze_class(self, row: ExternalRow) -> str:
        # VAT only has {inside, outside}. Map to the ChildPlay subset.
        if row.is_outside_frame:
            return "outside_frame"
        return "inside_visible"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
ADAPTER_REGISTRY: dict[str, Callable[..., ExternalDatasetAdapter]] = {
    "vat": VATAdapter,
    # Add custom adapters here.
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter",
        required=True,
        choices=sorted(ADAPTER_REGISTRY),
        help="Which external-dataset adapter to use.",
    )
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the external dataset on disk.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Where to write the ChildPlay-compatible output.",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Split name to convert (e.g. test, val).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cls = ADAPTER_REGISTRY[args.adapter]
    adapter = cls(args.source_root, args.split)
    adapter.write_childplay_split(args.output_root, args.split)


if __name__ == "__main__":
    main()
