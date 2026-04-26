"""Build a single CSV aggregating predictions from every available model
for all 69 test samples, designed as a companion to manual labeling.

Columns (in order):
    child_id, clip_id, split, final_manual_label,
    <model>_label, <model>_confidence (when available)
    ...

Models included (if their outputs exist):
    - rule_based
    - zero-shot NLI: distilbert, bart, deberta
    - supervised baselines: logreg, linear_svm, random_forest
    - LLM runs discovered under output/llm_runs/
    - legacy qwen7b (only if its label space matches)

Confidence columns are included for LLM runs (parsed_outputs.jsonl) and
any interpretation file exposing a confidence field.  The
``final_manual_label`` column is left empty for the user to fill in while
reviewing peer-model predictions side by side.

Usage:
    python scripts/build_all_predictions_csv.py
    python scripts/build_all_predictions_csv.py --output output/all_predictions_for_labeling.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LABELS = {
    "focused_attention",
    "exploratory_attention",
    "occluded_attention",
    "reduced_visual_availability",
    "mixed_attention",
}


def _load_interpretations(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = row["child_id"]
            out[cid] = {
                "label": row.get("interaction_type", ""),
                "confidence": (
                    f"{float(row['confidence']):.4f}"
                    if "confidence" in row and row["confidence"] is not None
                    else ""
                ),
                "clip_id": row.get("clip_id", ""),
                "split": row.get("split", ""),
            }
    return out


def _load_parsed(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = row["child_id"]
            out[cid] = {
                "label": row.get("label", ""),
                "confidence": f"{float(row.get('confidence', 0.0)):.4f}",
                "clip_id": row.get("clip_id", ""),
                "split": row.get("split", ""),
            }
    return out


def _load_supervised(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_clf: dict[str, dict[str, dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            clf = row["classifier"]
            cid = row["child_id"]
            by_clf.setdefault(clf, {})[cid] = {
                "label": row["predicted_label"],
                "confidence": row.get("confidence", "") or "",
                "clip_id": row.get("clip_id", ""),
                "split": row.get("split", ""),
            }
    return by_clf


def _load_manual(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = row["child_id"]
            out[cid] = {
                "clip_id": row.get("clip_id", ""),
                "split": row.get("split", ""),
                "final_label": (row.get("final_label") or "").strip(),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / "all_predictions_for_labeling.csv",
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv",
    )
    args = parser.parse_args()

    models: dict[str, dict[str, dict[str, str]]] = {}

    # Rule-based.
    rb = ROOT / "output" / "benchmark_rule_based" / "rule-based" / "interpretations.jsonl"
    if rb.exists():
        models["rule_based"] = _load_interpretations(rb)

    # Zero-shot NLI.
    zs = {
        "distilbert": ROOT / "output" / "benchmark_zero_shot" / "zero-shot__typeform_distilbert-base-uncased-mnli" / "interpretations.jsonl",
        "bart": ROOT / "output" / "benchmark_bart_large_mnli" / "zero-shot__facebook_bart-large-mnli" / "interpretations.jsonl",
        "deberta": ROOT / "output" / "benchmark_deberta_zeroshot" / "zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0" / "interpretations.jsonl",
    }
    for name, path in zs.items():
        if path.exists():
            models[name] = _load_interpretations(path)

    # LLM runs.
    llm_dir = ROOT / "output" / "llm_runs"
    if llm_dir.exists():
        for run in sorted(llm_dir.iterdir()):
            p = run / "parsed_outputs.jsonl"
            if p.exists():
                models[f"llm_{run.name}"] = _load_parsed(p)

    # Supervised — prefer all-69 predictions if available, else CV predictions.
    sup_all = ROOT / "output" / "supervised_baselines" / "supervised_all69_predictions.csv"
    sup_cv = ROOT / "output" / "supervised_baselines" / "supervised_predictions.csv"
    sup = sup_all if sup_all.exists() else sup_cv
    if sup is not None and sup.exists():
        for clf, data in _load_supervised(sup).items():
            models[f"supervised_{clf}"] = data

    # Manual annotations — provides canonical id/clip/split order and any existing labels.
    manual = _load_manual(args.annotations)

    # Collect every child_id we have seen.
    all_ids: set[str] = set(manual.keys())
    meta: dict[str, tuple[str, str]] = {
        cid: (m["clip_id"], m["split"]) for cid, m in manual.items()
    }
    for preds in models.values():
        for cid, rec in preds.items():
            all_ids.add(cid)
            meta.setdefault(cid, (rec.get("clip_id", ""), rec.get("split", "")))

    child_ids = sorted(all_ids)

    # Build column order: metadata, manual label, then per-model label + confidence.
    model_names = sorted(models.keys())
    header: list[str] = ["child_id", "clip_id", "split", "final_manual_label"]
    for m in model_names:
        header.append(f"{m}_label")
        header.append(f"{m}_confidence")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cid in child_ids:
            clip_id, split = meta.get(cid, ("", ""))
            existing = manual.get(cid, {}).get("final_label", "")
            row = [cid, clip_id, split, existing]
            for m in model_names:
                rec = models[m].get(cid, {})
                row.append(rec.get("label", ""))
                row.append(rec.get("confidence", ""))
            writer.writerow(row)

    print(f"Wrote {args.output}")
    print(f"  {len(child_ids)} rows × {len(model_names)} models")
    print(f"  Models: {', '.join(model_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
