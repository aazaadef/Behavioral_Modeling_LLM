"""One-time script: copy final_manual_label from all_predictions_for_labeling.csv
into the final_label column of manual_eval_annotations.csv."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

src = ROOT / "output" / "all_predictions_for_labeling.csv"
dst = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"

# Read manual labels.
manual: dict[str, str] = {}
with src.open("r", encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        label = (row.get("final_manual_label") or "").strip()
        if label:
            manual[row["child_id"]] = label

print(f"Read {len(manual)} manual labels from {src.name}")

# Update annotations CSV.
rows = []
with dst.open("r", encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    for row in reader:
        cid = row["child_id"]
        if cid in manual:
            row["final_label"] = manual[cid]
        rows.append(row)

updated = sum(1 for r in rows if r.get("final_label", "").strip())
with dst.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {updated}/{len(rows)} rows in {dst.name}")
