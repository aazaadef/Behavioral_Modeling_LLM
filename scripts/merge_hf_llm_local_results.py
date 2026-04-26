"""Merge local HuggingFace LLM (Qwen-7B) results with existing baseline runs.

Loads saved predictions from four baseline models (rule-based, DistilBERT,
BART, DeBERTa) and the local Qwen-7B HF-LLM run, then produces a unified
comparison directory containing merged predictions, pairwise agreement tables,
per-class / confidence summaries, disagreement reports, and paper-ready
artifacts.  This script does NOT re-run any inference — it only reads
previously saved outputs and combines them for cross-model analysis.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.benchmark import _load_interpretations
from project_llm.features import BehavioralFeatures
from project_llm.io_utils_v2 import ensure_new_output_dir, write_csv, write_json, write_text
from project_llm.paper_artifacts_v2 import generate_paper_artifacts
from project_llm.prompt_logging import build_feature_summary, build_zero_shot_premise
from project_llm.temporal import TemporalSegment
from project_llm.interactions import InteractionEvent


HF_RUN_DIR = ROOT / "output_experiments_v2" / "hf_llm_local_runs_v5"
OUTPUT_DIR = ROOT / "output_experiments_v2" / "hf_llm_local_comparison_v5"


def main() -> int:
    ensure_new_output_dir(OUTPUT_DIR)

    spec_to_path = {
        "rule-based": ROOT / "output/benchmark_rule_based/rule-based/interpretations.jsonl",
        "zero-shot:typeform/distilbert-base-uncased-mnli": ROOT / "output/benchmark_zero_shot/zero-shot__typeform_distilbert-base-uncased-mnli/interpretations.jsonl",
        "zero-shot:facebook/bart-large-mnli": ROOT / "output/benchmark_bart_large_mnli/zero-shot__facebook_bart-large-mnli/interpretations.jsonl",
        "zero-shot:MoritzLaurer/deberta-v3-large-zeroshot-v2.0": ROOT / "output/benchmark_deberta_zeroshot/zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0/interpretations.jsonl",
    }

    features_path = ROOT / "output/benchmark_rule_based/rule-based/behavioral_features.jsonl"
    temporal_path = ROOT / "output/benchmark_rule_based/rule-based/temporal_sequences.jsonl"
    interaction_path = ROOT / "output/benchmark_rule_based/rule-based/interaction_events.jsonl"
    manual_eval_path = ROOT / "output/paper_eval/manual_eval_annotations.csv"
    hf_outputs_path = HF_RUN_DIR / "parsed_outputs.jsonl"

    features: list[BehavioralFeatures] = []
    with features_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            features.append(BehavioralFeatures(**json.loads(line)))

    segments_by_key: dict[tuple[str, str], list[TemporalSegment]] = {}
    with temporal_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = TemporalSegment(**json.loads(line))
            segments_by_key.setdefault((item.child_id, item.clip_id), []).append(item)

    interactions_by_key: dict[tuple[str, str], list[InteractionEvent]] = {}
    with interaction_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = InteractionEvent(**json.loads(line))
            interactions_by_key.setdefault((item.child_id, item.clip_id), []).append(item)

    manual_labels: dict[tuple[str, str], str] = {}
    with manual_eval_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            final_label = row.get("final_label", "").strip()
            if final_label:
                manual_labels[(row["child_id"], row["clip_id"])] = final_label

    runs = {spec: {(item.child_id, item.clip_id): item for item in _load_interpretations(path)} for spec, path in spec_to_path.items()}
    hf_rows = []
    with hf_outputs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            hf_rows.append(json.loads(line))
    hf_run = {(row["child_id"], row["clip_id"]): row for row in hf_rows}

    merged_rows = []
    pairwise_rows = []
    per_class_rows = []
    confidence_rows = []
    model_metadata = []
    model_specs = list(spec_to_path.keys())
    hf_spec = "hf-llm:./models/qwen-7b"

    for feature in features:
        key = (feature.child_id, feature.clip_id)
        segments = segments_by_key.get(key, [])
        interactions = interactions_by_key.get(key, [])
        row = {
            "child_id": feature.child_id,
            "clip_id": feature.clip_id,
            "split": feature.split,
            "feature_summary": build_feature_summary(feature),
            "final_label": manual_labels.get(key, ""),
        }
        predictions_json = {}
        confidences_json = {}
        prompt_texts_json = {}
        labels = []

        for spec in model_specs:
            result = runs[spec][key]
            col_base = spec.replace("/", "_").replace(":", "__")
            row[f"{col_base}__label"] = result.interaction_type
            row[f"{col_base}__confidence"] = result.confidence
            predictions_json[spec] = result.interaction_type
            confidences_json[spec] = result.confidence
            prompt_texts_json[spec] = {
                "premise_text": build_zero_shot_premise(feature, segments, interactions),
                "full_prompt_text": "",
            }
            labels.append(result.interaction_type)

        hf_result = hf_run[key]
        hf_col_base = hf_spec.replace("/", "_").replace(":", "__")
        row[f"{hf_col_base}__label"] = hf_result["label"]
        row[f"{hf_col_base}__confidence"] = hf_result["confidence"]
        predictions_json[hf_spec] = hf_result["label"]
        confidences_json[hf_spec] = hf_result["confidence"]
        prompt_texts_json[hf_spec] = {"premise_text": "", "full_prompt_text": "stored_in_hf_llm_local_runs_v3/prompt_logs.jsonl"}
        labels.append(hf_result["label"])

        row["models_agree"] = "yes" if len(set(labels)) == 1 else "no"
        row["disagreement_flag"] = "yes" if row["models_agree"] == "no" else "no"
        row["predictions_json"] = json.dumps(predictions_json, ensure_ascii=True, sort_keys=True)
        row["confidences_json"] = json.dumps(confidences_json, ensure_ascii=True, sort_keys=True)
        row["prompt_texts_json"] = json.dumps(prompt_texts_json, ensure_ascii=True, sort_keys=True)
        row["num_correct_models"] = ""
        row["all_models_correct"] = ""
        merged_rows.append(row)

    pairwise_specs = model_specs + [hf_spec]
    for index, left in enumerate(pairwise_specs):
        left_col = f"{left.replace('/', '_').replace(':', '__')}__label"
        for right in pairwise_specs[index + 1 :]:
            right_col = f"{right.replace('/', '_').replace(':', '__')}__label"
            shared = [(row[left_col], row[right_col]) for row in merged_rows]
            matches = sum(1 for a, b in shared if a == b)
            pairwise_rows.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "agreement": round(matches / len(shared), 4) if shared else 0.0,
                    "matches": matches,
                    "denominator": len(shared),
                }
            )

    agreement_index: dict[str, list[dict[str, object]]] = {}
    for row in pairwise_rows:
        agreement_index.setdefault(row["left_model"], []).append({"other_model": row["right_model"], "agreement": row["agreement"]})
        agreement_index.setdefault(row["right_model"], []).append({"other_model": row["left_model"], "agreement": row["agreement"]})

    for spec in model_specs:
        col_base = spec.replace("/", "_").replace(":", "__")
        label_col = f"{col_base}__label"
        conf_col = f"{col_base}__confidence"
        labels = [row[label_col] for row in merged_rows]
        counts = Counter(labels)
        avg_conf = round(sum(float(row[conf_col]) for row in merged_rows) / len(merged_rows), 4)
        model_name = spec.split(":", 1)[-1] if ":" in spec else spec
        backend_family = spec.split(":", 1)[0] if ":" in spec else spec
        confidence_rows.append({"model_name": model_name, "backend_family": backend_family, "num_samples": len(merged_rows), "average_confidence": avg_conf})
        for label, count in sorted(counts.items()):
            per_class_rows.append({"model_name": model_name, "backend_family": backend_family, "label": label, "predicted_count": count, "per_class_accuracy": ""})
        model_metadata.append(
            {
                "spec": spec,
                "backend_name": backend_family,
                "backend_family": backend_family,
                "model_name": model_name,
                "run_dir": str(spec_to_path[spec].parent),
                "prompt_version": "childplay-behavior-v1-reused",
                "num_samples": len(merged_rows),
                "label_distribution": dict(counts),
                "average_confidence": avg_conf,
                "overall_accuracy": None,
                "notable_failure_modes": ["legacy label space reused as-is"],
                "agreement_with_other_models": agreement_index.get(spec, []),
            }
        )

    hf_label_col = f"{hf_spec.replace('/', '_').replace(':', '__')}__label"
    hf_conf_col = f"{hf_spec.replace('/', '_').replace(':', '__')}__confidence"
    hf_counts = Counter(row[hf_label_col] for row in merged_rows)
    hf_avg_conf = round(sum(float(row[hf_conf_col]) for row in merged_rows) / len(merged_rows), 4)
    confidence_rows.append({"model_name": "./models/qwen-7b", "backend_family": "hf-llm", "num_samples": len(merged_rows), "average_confidence": hf_avg_conf})
    for label, count in sorted(hf_counts.items()):
        per_class_rows.append({"model_name": "./models/qwen-7b", "backend_family": "hf-llm", "label": label, "predicted_count": count, "per_class_accuracy": ""})
    model_metadata.append(
        {
            "spec": hf_spec,
            "backend_name": "hf-llm",
            "backend_family": "hf-llm",
            "model_name": "./models/qwen-7b",
            "run_dir": str(HF_RUN_DIR),
            "prompt_version": "hf-llm-local-json-v1",
            "num_samples": len(merged_rows),
            "label_distribution": dict(hf_counts),
            "average_confidence": hf_avg_conf,
            "overall_accuracy": None,
            "notable_failure_modes": ["label space differs from legacy baselines; accuracy against legacy final_label omitted"],
            "agreement_with_other_models": agreement_index.get(hf_spec, []),
        }
    )

    root_fields = [
        "child_id",
        "clip_id",
        "split",
        "feature_summary",
        "final_label",
        "models_agree",
        "disagreement_flag",
        "num_correct_models",
        "all_models_correct",
        "predictions_json",
        "confidences_json",
        "prompt_texts_json",
    ]
    dynamic_fields = []
    for spec in pairwise_specs:
        base = spec.replace("/", "_").replace(":", "__")
        dynamic_fields.extend([f"{base}__label", f"{base}__confidence"])

    write_csv(OUTPUT_DIR / "merged_predictions.csv", merged_rows, root_fields + dynamic_fields)
    write_json(OUTPUT_DIR / "merged_predictions.json", merged_rows)
    write_csv(
        OUTPUT_DIR / "disagreement_report.csv",
        [
            {
                "child_id": row["child_id"],
                "clip_id": row["clip_id"],
                "split": row["split"],
                "feature_summary": row["feature_summary"],
                "final_label": row["final_label"],
                "predictions_json": row["predictions_json"],
                "confidences_json": row["confidences_json"],
            }
            for row in merged_rows
            if row["disagreement_flag"] == "yes"
        ],
        ["child_id", "clip_id", "split", "feature_summary", "final_label", "predictions_json", "confidences_json"],
    )
    write_csv(OUTPUT_DIR / "confidence_summary.csv", confidence_rows, ["model_name", "backend_family", "num_samples", "average_confidence"])
    write_csv(OUTPUT_DIR / "per_class_summary.csv", per_class_rows, ["model_name", "backend_family", "label", "predicted_count", "per_class_accuracy"])

    comparison_model_summaries = [
        {
            **metadata,
            "prediction_column": f"{metadata['spec'].replace('/', '_').replace(':', '__')}__label",
            "per_class_accuracy": {},
        }
        for metadata in model_metadata
    ]
    generate_paper_artifacts(OUTPUT_DIR, merged_rows, comparison_model_summaries, pairwise_rows, model_metadata)
    write_json(
        OUTPUT_DIR / "run_summary.json",
        {
            "reused_model_specs": model_specs,
            "new_model_specs": [hf_spec],
            "num_sequences": len(merged_rows),
            "hf_run_dir": str(HF_RUN_DIR),
            "notes": "Zero-shot baselines were reused exactly as-is; local hf-llm predictions were added from ./models/qwen-7b.",
        },
    )
    write_text(
        OUTPUT_DIR / "reproducibility_notes.txt",
        "\n".join(
            [
                "Comparison built from existing saved baselines plus local hf-llm predictions.",
                f"HF run directory: {HF_RUN_DIR}",
                f"Comparison directory: {OUTPUT_DIR}",
            ]
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
