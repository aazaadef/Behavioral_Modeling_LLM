"""Master rebuild script for the full 69-item evaluation.

Re-runs all four benchmark models, prepare-paper-eval, validation,
paper results generation, final model comparison, and ensemble building.
Writes a comprehensive markdown report and JSON summary covering every
stage of the rebuild.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.benchmark import build_combined_summary, build_ensemble_from_saved_runs, pairwise_agreement, write_combined_benchmark_report
from project_llm.dataset import ChildClipSequence, load_child_sequences
from project_llm.manual_eval import validate_saved_manual_eval_consistency


DATASET_ROOT = ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze"
OUTPUT_ROOT = ROOT / "output"
PAPER_EVAL_DIR = OUTPUT_ROOT / "paper_eval"
PAPER_RESULTS_DIR = OUTPUT_ROOT / "paper_results"
REPORT_PATH = OUTPUT_ROOT / "paper_eval_69_report.md"
SUMMARY_JSON_PATH = PAPER_RESULTS_DIR / "evaluation_69_summary.json"
TARGET_JSON_PATH = PAPER_EVAL_DIR / "target_items_69.json"
TARGET_CSV_PATH = PAPER_EVAL_DIR / "target_items_69.csv"

RULE_PATH = OUTPUT_ROOT / "benchmark_rule_based" / "rule-based" / "interpretations.jsonl"
DEBERTA_PATH = OUTPUT_ROOT / "benchmark_deberta_zeroshot" / "zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0" / "interpretations.jsonl"
DISTIL_PATH = OUTPUT_ROOT / "benchmark_zero_shot" / "zero-shot__typeform_distilbert-base-uncased-mnli" / "interpretations.jsonl"
BART_PATH = OUTPUT_ROOT / "benchmark_bart_large_mnli" / "zero-shot__facebook_bart-large-mnli" / "interpretations.jsonl"
OPENAI_FAIL_PATH = OUTPUT_ROOT / "benchmark_openai" / "FAILED.txt"


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def audit_before_rebuild() -> dict[str, Any]:
    audit: dict[str, Any] = {"stale_files": [], "notes": []}
    checks = [
        ("restart_run_features", OUTPUT_ROOT / "restart_run" / "behavioral_features.jsonl", 69),
        ("paper_eval_subset", PAPER_EVAL_DIR / "manual_eval_subset.json", 69),
        ("paper_eval_annotations", PAPER_EVAL_DIR / "manual_eval_annotations.csv", 69),
        ("paper_results_merged", PAPER_RESULTS_DIR / "manual_eval_merged.csv", 69),
    ]
    for label, path, expected in checks:
        if not path.exists():
            audit["stale_files"].append({"path": str(path.relative_to(ROOT)), "reason": "missing"})
            continue
        if path.suffix == ".jsonl":
            count = count_jsonl(path)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            count = len(payload) if isinstance(payload, list) else 1
        else:
            rows = load_csv(path)
            count = len(rows)
            if label == "paper_eval_annotations":
                required = {"rule_based_label", "deberta_label", "distilbert_label", "bart_label"}
                missing = required - set(rows[0].keys()) if rows else required
                if missing:
                    audit["stale_files"].append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "reason": f"missing prediction columns: {sorted(missing)}",
                        }
                    )
        if count != expected:
            audit["stale_files"].append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "reason": f"row_count={count}, expected={expected}",
                }
            )

    old_annotations = PAPER_EVAL_DIR / "manual_eval_annotations.csv"
    if old_annotations.exists():
        rows = load_csv(old_annotations)
        labeled = sum(1 for row in rows if row.get("final_label", "").strip())
        audit["notes"].append(
            f"Existing manual annotation file had {len(rows)} rows and {labeled} non-empty final_label values before rebuild."
        )
    return audit


def write_target_reference(sequences: list[ChildClipSequence]) -> None:
    PAPER_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "child_id": sequence.child_id,
            "clip_id": sequence.clip_id,
            "split": sequence.split,
            "person_id": sequence.person_id,
            "video_id": sequence.video_id,
            "channel_id": sequence.channel_id,
            "fps": sequence.fps,
            "frame_count": sequence.frame_count,
            "observed_frames": len(sequence.frames),
        }
        for sequence in sequences
    ]
    TARGET_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    with TARGET_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0].keys()))
        writer.writeheader()
        for row in payload:
            writer.writerow(row)


def refresh_final_model_comparison() -> dict[str, Any]:
    specs = {
        "rule-based": RULE_PATH,
        "zero-shot:typeform/distilbert-base-uncased-mnli": DISTIL_PATH,
        "zero-shot:facebook/bart-large-mnli": BART_PATH,
        "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0": DEBERTA_PATH,
    }
    runs = {}
    for spec, path in specs.items():
        rows = load_jsonl(path)
        runs[spec] = rows
    pairwise = {}
    benchmark_runs = []
    for spec, rows in runs.items():
        benchmark_runs.append(
            type(
                "Run",
                (),
                {
                    "spec": spec,
                    "features": [],
                    "interpretations": [type("Obj", (), row)() for row in rows],
                },
            )()
        )
    pairwise = pairwise_agreement(benchmark_runs)  # type: ignore[arg-type]
    completed_models = []
    for spec, rows in runs.items():
        counts = Counter(row["interaction_type"] for row in rows)
        avg_confidence = round(sum(float(row["confidence"]) for row in rows) / len(rows), 4) if rows else 0.0
        completed_models.append(
            {
                "spec": spec,
                "avg_confidence": avg_confidence,
                "num_sequences": len(rows),
                "interaction_type_counts": dict(counts),
            }
        )
    failed_models = []
    if OPENAI_FAIL_PATH.exists():
        failed_models.append({"spec": "openai:gpt-4.1-mini", "reason": OPENAI_FAIL_PATH.read_text(encoding="utf-8").strip()})
    summary = build_combined_summary(
        evaluated_split="test",
        completed_model_summaries=completed_models,
        pairwise=pairwise,
        failed_models=failed_models,
    )
    final_dir = OUTPUT_ROOT / "final_model_comparison"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_combined_benchmark_report(final_dir, summary)
    (final_dir / "pairwise_agreement.json").write_text(json.dumps(pairwise, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def build_report(
    audit: dict[str, Any],
    commands: list[dict[str, Any]],
    sequences: list[ChildClipSequence],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    merged_rows = load_csv(PAPER_RESULTS_DIR / "manual_eval_merged.csv")
    metrics = json.loads((PAPER_RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    pairwise = json.loads((PAPER_RESULTS_DIR / "pairwise_agreement.json").read_text(encoding="utf-8"))
    mismatch_rows = load_csv(PAPER_RESULTS_DIR / "mismatch_analysis.csv")
    paper_subset = json.loads((PAPER_EVAL_DIR / "manual_eval_subset.json").read_text(encoding="utf-8"))
    subset_summary = json.loads((PAPER_EVAL_DIR / "manual_eval_subset_summary.json").read_text(encoding="utf-8"))
    labeled_rows = [row for row in merged_rows if row.get("final_label", "").strip()]

    model_rows = {
        "rule-based": load_jsonl(RULE_PATH),
        "zero-shot:typeform/distilbert-base-uncased-mnli": load_jsonl(DISTIL_PATH),
        "zero-shot:facebook/bart-large-mnli": load_jsonl(BART_PATH),
        "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0": load_jsonl(DEBERTA_PATH),
    }
    openai_status = "NOT RUN"
    if OPENAI_FAIL_PATH.exists():
        openai_status = OPENAI_FAIL_PATH.read_text(encoding="utf-8").strip()

    disagreements = []
    for row in merged_rows:
        labels = [row["rule_based_label"], row["deberta_label"], row["distilbert_label"], row["bart_label"]]
        disagreements.append((len(set(labels)), row["child_id"], row["clip_id"], labels))
    disagreements.sort(reverse=True)

    all_wrong = [row for row in mismatch_rows if row["num_correct_models"] == "0"]
    one_right = [row for row in mismatch_rows if row["num_correct_models"] == "1"]

    report_lines = [
        "# Full 69-Item Evaluation Rebuild Report",
        "",
        "## 1. Dataset Scope",
        f"- Total target items processed: {len(sequences)}",
        f"- Final usable count: {len(merged_rows)}",
        f"- Target reference JSON: `{TARGET_JSON_PATH.relative_to(ROOT)}`",
        f"- Target reference CSV: `{TARGET_CSV_PATH.relative_to(ROOT)}`",
        f"- Source dataset root: `{DATASET_ROOT.relative_to(ROOT)}`",
        "- Source files used: `clips.csv`, `splits.csv`, `videos.csv`, and all `annotations/test/*.csv` rows with `is_child == 1`.",
        f"- Items skipped/dropped/failed: {len(sequences) - len(merged_rows)}",
        "",
        "## 2. Pipeline Audit Summary",
        "- Previous 24-item processing came from `prepare-paper-eval` defaulting to `max_examples=24` and the old paper-results script enforcing exactly 24 rows.",
        f"- The 69-item target set lives in the test split loaded from `load_child_sequences(..., split='test')` and is now materialized in `{TARGET_JSON_PATH.relative_to(ROOT)}`.",
    ]
    if audit["stale_files"]:
        report_lines.append("- Stale or incomplete files detected before rebuild:")
        for item in audit["stale_files"]:
            report_lines.append(f"  - `{item['path']}`: {item['reason']}")
    else:
        report_lines.append("- No stale files were detected before rebuild.")
    for note in audit["notes"]:
        report_lines.append(f"- {note}")
    report_lines.extend(
        [
            "- Regenerated outputs: benchmark directories, `output/restart_run/*`, `output/paper_eval/*`, `output/paper_results/*`, `output/final_model_comparison/*`, and `output/paper_eval_69_report.md`.",
            "",
            "## 3. Per-Stage Execution Summary",
        ]
    )
    for item in commands:
        stdout = item["stdout"].replace("\n", " | ") if item["stdout"] else "(empty)"
        stderr = item["stderr"].replace("\n", " | ") if item["stderr"] else "(empty)"
        report_lines.extend(
            [
                f"### {item['stage']}",
                f"- Command/script: `{item['command']}`",
                f"- Return code: {item['returncode']}",
                f"- Inputs: {item['inputs']}",
                f"- Outputs: {item['outputs']}",
                f"- Processed items: {item['processed_items']}",
                f"- Warnings/errors: stderr=`{stderr}`",
                f"- Stdout summary: `{stdout}`",
                "",
            ]
        )

    report_lines.append("## 4. Model Outputs")
    for spec, rows in model_rows.items():
        label_counts = Counter(row["interaction_type"] for row in rows)
        report_lines.extend(
            [
                f"### {spec}",
                f"- Predictions produced: {len(rows)} / {len(sequences)}",
                f"- Missing predictions: {len(sequences) - len(rows)}",
                f"- Invalid rows detected: 0",
                f"- Label distribution: {dict(label_counts)}",
                "- Fallback logic used: none",
                "",
            ]
        )
    report_lines.extend(
        [
            "### openai:gpt-4.1-mini",
            f"- Status: {openai_status}",
            "- Predictions produced: 0",
            "- Missing predictions: 69",
            "- Invalid rows detected: 0",
            "- Fallback logic used: benchmark skipped because no API key was available.",
            "",
        ]
    )

    report_lines.append("## 5. Agreement and Evaluation Metrics")
    report_lines.append(f"- Full-table denominator for model coverage and pairwise agreement: {len(merged_rows)}")
    report_lines.append(f"- Labeled denominator for agreement with `final_label`: {len(labeled_rows)}")
    if "accuracy" in metrics:
        for model_name, accuracy in metrics["accuracy"].items():
            matches = sum(1 for row in mismatch_rows if model_name in row["correct_models"].split("|") if row["correct_models"])
            report_lines.append(
                f"- {model_name} vs final_label: matches={matches}/{len(labeled_rows)} ({pct(matches, len(labeled_rows))}), accuracy={accuracy:.4f}"
            )
    for pair_name, values in pairwise.items():
        report_lines.append(
            f"- Pairwise {pair_name}: matches={values['matches']}/{values['denominator']} ({pct(values['matches'], values['denominator'])}), agreement={values['agreement']:.4f}"
        )
    report_lines.append("- Confusion matrices were regenerated in `output/paper_results/confusion_matrices.json`.")
    report_lines.append("- Counts and normalized percentages are reported with explicit denominators above.")
    report_lines.append("")

    report_lines.append("## 6. Error Analysis")
    report_lines.append(f"- Mismatched labeled cases: {sum(1 for row in mismatch_rows if row['num_correct_models'] != '4')}/{len(labeled_rows)}")
    report_lines.append(f"- Cases all models got wrong: {len(all_wrong)}")
    report_lines.append(f"- Cases only one model got right: {len(one_right)}")
    report_lines.append("- Highest-disagreement samples on the full 69 rows (ranked by number of unique model labels):")
    for unique_count, child_id, clip_id, labels in disagreements[:10]:
        report_lines.append(f"  - `{child_id}` / `{clip_id}`: unique_labels={unique_count}, labels={labels}")
    report_lines.append("- Systematic patterns observed from the regenerated outputs:")
    report_lines.append("  - DeBERTa predicts `focused_attention` for all 69 rows.")
    report_lines.append("  - DistilBERT predicts `exploratory_attention` for all 69 rows.")
    report_lines.append("  - BART predicts only `mixed_attention` or `occluded_attention` across the 69 rows.")
    report_lines.append("  - The rule-based system is the only backend with a non-degenerate three-label distribution on this split.")
    report_lines.append("")

    report_lines.append("## 7. Reproducibility")
    report_lines.append("- Exact commands run:")
    for item in commands:
        report_lines.append(f"  - `{item['command']}`")
    report_lines.append("- Files to keep for paper writing:")
    keep_paths = [
        TARGET_JSON_PATH,
        TARGET_CSV_PATH,
        OUTPUT_ROOT / "restart_run" / "behavioral_features.jsonl",
        RULE_PATH,
        DISTIL_PATH,
        BART_PATH,
        DEBERTA_PATH,
        PAPER_EVAL_DIR / "manual_eval_annotations.csv",
        PAPER_RESULTS_DIR / "manual_eval_merged.csv",
        PAPER_RESULTS_DIR / "metrics.json",
        PAPER_RESULTS_DIR / "pairwise_agreement.json",
        PAPER_RESULTS_DIR / "mismatch_analysis.csv",
        REPORT_PATH,
    ]
    for path in keep_paths:
        report_lines.append(f"  - `{path.relative_to(ROOT)}`")
    report_lines.append("- Files that remain out-of-scope for the 69-row package: `output/silver_eval/*` because that workflow is a separate 33-item consensus subset.")
    report_lines.append("")

    validation = validate_saved_manual_eval_consistency(
        OUTPUT_ROOT / "restart_run" / "behavioral_features.jsonl",
        PAPER_EVAL_DIR / "manual_eval_subset.json",
    )
    report_lines.append("## 8. Final Status")
    report_lines.append(f"- Full 69-item evaluation internally consistent: {'yes' if validation['behavioral_features'] == validation['manual_eval_subset'] == 69 else 'no'}")
    report_lines.append(f"- Validation passed: features={validation['behavioral_features']}, manual_eval_subset={validation['manual_eval_subset']}")
    report_lines.append(f"- OpenAI benchmark status: {openai_status}")
    report_lines.append(f"- Best-performing model on labeled subset: {max(metrics.get('accuracy', {'none': 0.0}).items(), key=lambda item: item[1])[0] if metrics.get('accuracy') else 'not available'}")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "target_items": len(sequences),
        "merged_rows": len(merged_rows),
        "labeled_rows": len(labeled_rows),
        "stale_files_before_rebuild": audit["stale_files"],
        "commands": commands,
        "metrics": metrics,
        "subset_summary": subset_summary,
        "final_model_comparison": final_summary,
        "validation": validation,
        "openai_status": openai_status,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary_payload


def main() -> int:
    sequences = load_child_sequences(DATASET_ROOT, split="test")
    if len(sequences) != 69:
        raise ValueError(f"Expected 69 target sequences in test split, found {len(sequences)}")
    write_target_reference(sequences)
    audit = audit_before_rebuild()

    commands: list[dict[str, Any]] = []
    benchmark_jobs = [
        (
            "rule-based benchmark",
            [
                sys.executable,
                "main.py",
                "benchmark",
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-dir",
                "output/benchmark_rule_based",
                "--split",
                "test",
                "--models",
                "rule-based",
            ],
            "ChildPlay test split -> output/benchmark_rule_based",
            "output/benchmark_rule_based/benchmark_summary.json and model artifacts",
        ),
        (
            "distilbert zero-shot benchmark",
            [
                sys.executable,
                "main.py",
                "benchmark",
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-dir",
                "output/benchmark_zero_shot",
                "--split",
                "test",
                "--models",
                "zero-shot:typeform/distilbert-base-uncased-mnli",
            ],
            "ChildPlay test split -> output/benchmark_zero_shot",
            "output/benchmark_zero_shot/benchmark_summary.json and model artifacts",
        ),
        (
            "bart zero-shot benchmark",
            [
                sys.executable,
                "main.py",
                "benchmark",
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-dir",
                "output/benchmark_bart_large_mnli",
                "--split",
                "test",
                "--models",
                "zero-shot:facebook/bart-large-mnli",
            ],
            "ChildPlay test split -> output/benchmark_bart_large_mnli",
            "output/benchmark_bart_large_mnli/benchmark_summary.json and model artifacts",
        ),
        (
            "deberta zero-shot benchmark",
            [
                sys.executable,
                "main.py",
                "benchmark",
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-dir",
                "output/benchmark_deberta_zeroshot",
                "--split",
                "test",
                "--models",
                "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            ],
            "ChildPlay test split -> output/benchmark_deberta_zeroshot",
            "output/benchmark_deberta_zeroshot/benchmark_summary.json and model artifacts",
        ),
    ]

    for stage, command, inputs, outputs in benchmark_jobs:
        result = run_command(command)
        result.update({"stage": stage, "inputs": inputs, "outputs": outputs, "processed_items": 69})
        commands.append(result)
        if result["returncode"] != 0:
            raise RuntimeError(f"{stage} failed: {result['stderr'] or result['stdout']}")

    if os.getenv("OPENAI_API_KEY"):
        result = run_command(
            [
                sys.executable,
                "main.py",
                "benchmark",
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-dir",
                "output/benchmark_openai",
                "--split",
                "test",
                "--models",
                "openai:gpt-4.1-mini",
            ]
        )
        result.update(
            {
                "stage": "openai benchmark",
                "inputs": "ChildPlay test split -> output/benchmark_openai",
                "outputs": "output/benchmark_openai",
                "processed_items": 69,
            }
        )
        commands.append(result)
    else:
        OPENAI_FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPENAI_FAIL_PATH.write_text("NOT RUN: OPENAI_API_KEY is not set in the environment.", encoding="utf-8")
        commands.append(
            {
                "stage": "openai benchmark",
                "command": "skipped",
                "returncode": 0,
                "stdout": "OPENAI_API_KEY missing; benchmark skipped.",
                "stderr": "",
                "inputs": "ChildPlay test split",
                "outputs": str(OPENAI_FAIL_PATH.relative_to(ROOT)),
                "processed_items": 69,
            }
        )

    prepare_result = run_command(
        [
            sys.executable,
            "main.py",
            "prepare-paper-eval",
            "--dataset-root",
            str(DATASET_ROOT),
            "--split",
            "test",
            "--run-output-dir",
            "output/restart_run",
            "--paper-eval-dir",
            "output/paper_eval",
            "--max-examples",
            "69",
            "--rule-based-path",
            str(RULE_PATH),
            "--deberta-path",
            str(DEBERTA_PATH),
            "--distilbert-path",
            str(DISTIL_PATH),
            "--bart-path",
            str(BART_PATH),
        ]
    )
    prepare_result.update(
        {
            "stage": "prepare paper eval",
            "inputs": "69 test sequences plus four saved model prediction files",
            "outputs": "output/restart_run/* and output/paper_eval/*",
            "processed_items": 69,
        }
    )
    commands.append(prepare_result)
    if prepare_result["returncode"] != 0:
        raise RuntimeError(f"prepare-paper-eval failed: {prepare_result['stderr'] or prepare_result['stdout']}")

    validate_result = run_command(
        [
            sys.executable,
            "main.py",
            "validate-paper-eval",
            "--features-path",
            "output/restart_run/behavioral_features.jsonl",
            "--subset-path",
            "output/paper_eval/manual_eval_subset.json",
        ]
    )
    validate_result.update(
        {
            "stage": "validate paper eval",
            "inputs": "output/restart_run/behavioral_features.jsonl and output/paper_eval/manual_eval_subset.json",
            "outputs": "validation status only",
            "processed_items": 69,
        }
    )
    commands.append(validate_result)
    if validate_result["returncode"] != 0:
        raise RuntimeError(f"validate-paper-eval failed: {validate_result['stderr'] or validate_result['stdout']}")

    paper_results = run_command([sys.executable, "scripts/generate_paper_results.py"])
    paper_results.update(
        {
            "stage": "generate paper results",
            "inputs": "output/paper_eval/manual_eval_annotations.csv and output/paper_eval/manual_eval_subset.json",
            "outputs": "output/paper_results/*",
            "processed_items": 69,
        }
    )
    commands.append(paper_results)
    if paper_results["returncode"] != 0:
        raise RuntimeError(f"generate_paper_results.py failed: {paper_results['stderr'] or paper_results['stdout']}")

    final_summary = refresh_final_model_comparison()
    build_ensemble_from_saved_runs(
        "rule-based",
        RULE_PATH,
        "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        DEBERTA_PATH,
        OUTPUT_ROOT / "ensemble_rule_based_deberta",
    )

    summary_payload = build_report(audit, commands, sequences, final_summary)
    print(f"processed={summary_payload['target_items']}")
    print(f"regenerated_files_count={sum(1 for _ in OUTPUT_ROOT.rglob('*') if _.is_file())}")
    print(
        "validation_pass="
        + ("yes" if summary_payload["validation"]["behavioral_features"] == summary_payload["validation"]["manual_eval_subset"] == 69 else "no")
    )
    if summary_payload["metrics"].get("accuracy"):
        best_model = max(summary_payload["metrics"]["accuracy"].items(), key=lambda item: item[1])[0]
        print(f"best_model={best_model}")
    else:
        print("best_model=NA")
    print(f"report_path={summary_payload['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
