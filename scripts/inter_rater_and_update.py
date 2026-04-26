"""Compute inter-rater reliability between the two human annotators and
update the ground-truth file with the new consensus labels.

Source CSV: output/all_predictions_with_two_annotators.csv
  columns rater1, rater2, final_manual_label

Writes:
  output/phase_a_analysis/inter_rater_agreement.json
  output/phase_a_analysis/inter_rater_report.md
  updates output/paper_eval/manual_eval_annotations.csv
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "output" / "all_predictions_with_two_annotators.csv"
ANN = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"
OUT_DIR = ROOT / "output" / "phase_a_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    with SRC.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Loaded {len(rows)} samples from {SRC.name}")

    rater1 = [r["rater1"].strip() for r in rows]
    rater2 = [r["rater2"].strip() for r in rows]
    final = [r["final_manual_label"].strip() for r in rows]
    cids = [r["child_id"] for r in rows]

    # ── Inter-rater agreement ─────────────────────────────────────────
    label_set = sorted(set(rater1) | set(rater2))
    kappa = cohen_kappa_score(rater1, rater2)
    agree_raw = accuracy_score(rater1, rater2)
    n_disagree = sum(1 for a, b in zip(rater1, rater2) if a != b)

    # Disagreement list with consensus.
    disagreements = [
        {
            "child_id": cids[i],
            "rater1": rater1[i],
            "rater2": rater2[i],
            "consensus": final[i],
        }
        for i in range(len(rows)) if rater1[i] != rater2[i]
    ]

    print(f"\n== Inter-rater (rater1 vs rater2) ==")
    print(f"  Raw agreement: {agree_raw:.4f}  ({len(rows) - n_disagree}/{len(rows)})")
    print(f"  Cohen kappa  : {kappa:.4f}")
    print(f"  Disagreements: {n_disagree}")

    cm = confusion_matrix(rater1, rater2, labels=label_set)

    # Distribution comparison.
    dist_r1 = Counter(rater1)
    dist_r2 = Counter(rater2)
    dist_final = Counter(final)

    # Consensus behavior on disagreements.
    sided = Counter()
    for d in disagreements:
        if d["consensus"] == d["rater1"]:
            sided["sided_with_rater1"] += 1
        elif d["consensus"] == d["rater2"]:
            sided["sided_with_rater2"] += 1
        else:
            sided["third_option"] += 1

    print(f"\n== Consensus resolution of {n_disagree} disagreements ==")
    for k, v in sided.items():
        print(f"  {k}: {v}")

    # ── Diff vs previous stored final_label ───────────────────────────
    prev_final: dict[str, str] = {}
    with ANN.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        ann_fields = reader.fieldnames
        ann_rows = list(reader)
    for row in ann_rows:
        lbl = (row.get("final_label") or "").strip()
        if lbl:
            prev_final[row["child_id"]] = lbl

    new_final = {cid: lbl for cid, lbl in zip(cids, final) if lbl}

    changed: list[dict[str, str]] = []
    for cid, new_lbl in new_final.items():
        old_lbl = prev_final.get(cid, "")
        if old_lbl != new_lbl:
            changed.append({"child_id": cid, "old": old_lbl, "new": new_lbl})

    print(f"\n== Ground truth changes vs previous final_label ==")
    print(f"  Changed: {len(changed)}/{len(new_final)}")
    for c in changed:
        print(f"    {c['child_id']}  {c['old']} -> {c['new']}")

    # ── Update annotations CSV with new final_label ───────────────────
    updated = 0
    for row in ann_rows:
        cid = row["child_id"]
        if cid in new_final:
            if row.get("final_label", "") != new_final[cid]:
                row["final_label"] = new_final[cid]
            updated += 1
    with ANN.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ann_fields)
        writer.writeheader()
        writer.writerows(ann_rows)
    print(f"\nUpdated {ANN} ({updated} rows)")

    # ── Save JSON summary ─────────────────────────────────────────────
    summary = {
        "n_samples": len(rows),
        "raw_agreement": round(float(agree_raw), 4),
        "cohen_kappa": round(float(kappa), 4),
        "n_disagreements": n_disagree,
        "label_set": label_set,
        "confusion_matrix_rater1_vs_rater2": cm.tolist(),
        "distribution_rater1": dict(dist_r1),
        "distribution_rater2": dict(dist_r2),
        "distribution_final_consensus": dict(dist_final),
        "consensus_bias": dict(sided),
        "disagreements": disagreements,
        "ground_truth_changes_vs_previous_final_label": changed,
    }
    json_path = OUT_DIR / "inter_rater_agreement.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")

    # ── Save markdown inter-rater report ──────────────────────────────
    lines: list[str] = []
    lines.append("# Inter-rater agreement — rater1 vs rater2\n")
    lines.append(f"- **n** = {len(rows)}")
    lines.append(f"- **Raw agreement** = {agree_raw:.4f} "
                 f"({len(rows) - n_disagree}/{len(rows)})")
    lines.append(f"- **Cohen's κ** = {kappa:.4f}  "
                 "(Landis–Koch: 0.6–0.8 = substantial, >0.8 = almost perfect)")
    lines.append(f"- **Disagreements** = {n_disagree}\n")

    lines.append("## Per-rater label distribution\n")
    lines.append("| Label | Rater 1 | Rater 2 | Final consensus |")
    lines.append("|---|---:|---:|---:|")
    all_labels = sorted(set(dist_r1) | set(dist_r2) | set(dist_final))
    for lbl in all_labels:
        lines.append(
            f"| {lbl} | {dist_r1.get(lbl, 0)} | "
            f"{dist_r2.get(lbl, 0)} | {dist_final.get(lbl, 0)} |"
        )
    lines.append("")

    lines.append("## Confusion matrix (rater1 rows vs rater2 cols)\n")
    lines.append("| rater1\\rater2 | " + " | ".join(label_set) + " |")
    lines.append("|---" + "|---" * len(label_set) + "|")
    for i, lbl in enumerate(label_set):
        lines.append(f"| {lbl} | " + " | ".join(str(x) for x in cm[i]) + " |")
    lines.append("")

    lines.append("## Consensus bias on disagreements\n")
    for k, v in sided.items():
        lines.append(f"- **{k.replace('_', ' ')}**: {v}")
    lines.append("")

    lines.append(f"## Disagreements resolved by discussion ({n_disagree})\n")
    lines.append("| child_id | rater1 | rater2 | consensus |")
    lines.append("|---|---|---|---|")
    for d in disagreements:
        lines.append(
            f"| {d['child_id']} | {d['rater1']} | {d['rater2']} | "
            f"**{d['consensus']}** |"
        )
    lines.append("")

    if changed:
        lines.append(f"## Ground-truth changes vs previous file ({len(changed)})\n")
        lines.append("| child_id | old | new |")
        lines.append("|---|---|---|")
        for c in changed:
            lines.append(f"| {c['child_id']} | {c['old']} | {c['new']} |")
    else:
        lines.append("## Ground-truth unchanged vs previous final_label\n")

    md_path = OUT_DIR / "inter_rater_report.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
