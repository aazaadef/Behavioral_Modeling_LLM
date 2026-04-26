"""Regenerate paper artifacts from saved merged outputs without re-running inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.paper_artifacts_v2 import generate_paper_artifacts


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    output_dir = Path(argv[0]) if argv else Path("output_experiments_v2/run_default")
    merged_path = output_dir / "merged_predictions.json"
    metadata_path = output_dir / "model_metadata.json"
    agreement_path = output_dir / "agreement_table.csv"

    if not merged_path.exists():
        raise FileNotFoundError(f"Missing merged predictions: {merged_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")

    merged_rows = json.loads(merged_path.read_text(encoding="utf-8"))
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pairwise_rows: list[dict[str, object]] = []
    if agreement_path.exists():
        import csv

        with agreement_path.open("r", encoding="utf-8", newline="") as handle:
            pairwise_rows = list(csv.DictReader(handle))

    # Rebuild the paper tables from the saved merged outputs without rerunning inference.
    model_summaries = [
        {
            **row,
            "prediction_column": f"{row['spec'].replace('/', '_').replace(':', '__')}__label",
            "per_class_accuracy": {},
            "notable_failure_modes": row.get("notable_failure_modes", []),
        }
        for row in model_metadata
    ]
    generate_paper_artifacts(
        output_dir=output_dir,
        merged_rows=merged_rows,
        model_summaries=model_summaries,
        pairwise_rows=pairwise_rows,
        model_metadata=model_metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
