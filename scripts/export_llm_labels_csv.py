"""Export per-model LLM predictions as a CSV mirroring manual_eval_annotations.csv.

For every run directory under ``output/llm_runs/`` this script reads the
``parsed_outputs.jsonl`` file and writes a CSV with one row per child_id:
    child_id, clip_id, split, predicted_label, confidence, reasoning

A combined CSV (``all_models_predictions.csv``) is also written, with one
column per model holding the predicted label — analogous to how
``manual_eval_annotations.csv`` holds the final human label per row.

Usage:
    python scripts/export_llm_labels_csv.py
    python scripts/export_llm_labels_csv.py --output-dir output/llm_labels_csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_parsed(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rows.append({
                "child_id": row["child_id"],
                "clip_id": row.get("clip_id", ""),
                "split": row.get("split", ""),
                "predicted_label": row.get("label", ""),
                "confidence": f"{float(row.get('confidence', 0.0)):.4f}",
                "reasoning": (row.get("reasoning", "") or "").replace("\n", " "),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-runs", type=Path, default=ROOT / "output" / "llm_runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "llm_labels_csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_model_labels: dict[str, dict[str, str]] = {}
    id_meta: dict[str, tuple[str, str]] = {}  # child_id -> (clip_id, split)

    for run_dir in sorted(args.llm_runs.iterdir()):
        parsed_path = run_dir / "parsed_outputs.jsonl"
        if not parsed_path.exists():
            continue
        rows = _load_parsed(parsed_path)
        model_name = run_dir.name

        out_csv = args.output_dir / f"{model_name}_predictions.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "child_id", "clip_id", "split",
                "predicted_label", "confidence", "reasoning",
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        per_model_labels[model_name] = {r["child_id"]: r["predicted_label"] for r in rows}
        for r in rows:
            id_meta.setdefault(r["child_id"], (r["clip_id"], r["split"]))

        print(f"Wrote {out_csv} ({len(rows)} rows)")

    if per_model_labels:
        combined_path = args.output_dir / "all_models_predictions.csv"
        model_names = sorted(per_model_labels.keys())
        child_ids = sorted(id_meta.keys())
        with combined_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["child_id", "clip_id", "split", *model_names])
            for cid in child_ids:
                clip_id, split = id_meta[cid]
                writer.writerow([
                    cid, clip_id, split,
                    *[per_model_labels[m].get(cid, "") for m in model_names],
                ])
        print(f"Wrote combined {combined_path} ({len(child_ids)} rows x {len(model_names)} models)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
