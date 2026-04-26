"""Unit tests for the V1 pipeline and supporting modules.

Covers dataset loading, feature extraction, pipeline execution, report
generation, benchmark orchestration, ensemble building, manual-eval
artifact creation, and silver-label evaluation workflows.  All tests
use the real ChildPlay-gaze dataset (test / val / train splits) with
small sequence caps so they finish quickly.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.dataset import load_child_sequences, load_metadata
from project_llm.features import extract_behavioral_features
from project_llm.benchmark import (
    build_combined_summary,
    build_ensemble_from_saved_runs,
    parse_model_spec,
    run_benchmark,
)
from project_llm.manual_eval import (
    LABEL_SCHEMA,
    build_manual_eval_subset,
    build_silver_label_subset,
    evaluate_against_manual_labels,
    evaluate_against_silver_labels,
    prepare_manual_eval_workflow,
    validate_saved_manual_eval_consistency,
    write_label_schema,
    write_manual_eval_subset,
    write_silver_label_subset,
)
from project_llm.pipeline import run_pipeline, save_artifacts


DATASET_ROOT = ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze"


class DatasetTests(unittest.TestCase):
    def test_metadata_consistency(self) -> None:
        metadata = load_metadata(DATASET_ROOT)
        self.assertEqual(len(metadata.clips), 401)
        self.assertEqual(len(metadata.splits), 401)
        self.assertEqual(len(metadata.videos), 95)

    def test_child_sequences_are_loaded(self) -> None:
        sequences = load_child_sequences(DATASET_ROOT, split="test")
        self.assertTrue(sequences)
        sample = sequences[0]
        self.assertTrue(sample.child_id.endswith(f"person_{sample.person_id}"))
        self.assertEqual(sample.split, "test")
        self.assertTrue(sample.frames)
        self.assertTrue(all(frame.is_child == 1 for frame in sample.frames))

    def test_feature_extraction(self) -> None:
        sequence = load_child_sequences(DATASET_ROOT, split="val")[0]
        features = extract_behavioral_features(sequence)
        self.assertEqual(features.child_id, sequence.child_id)
        self.assertGreater(features.observed_frames, 0)
        self.assertGreaterEqual(features.visible_ratio, 0.0)
        self.assertLessEqual(features.visible_ratio, 1.0)
        self.assertGreaterEqual(features.attention_stability_score, 0.0)
        self.assertLessEqual(features.attention_stability_score, 1.0)

    def test_pipeline_emits_interpretations(self) -> None:
        artifacts = run_pipeline(DATASET_ROOT, split="train", max_sequences=3)
        self.assertEqual(len(artifacts.sequences), 3)
        self.assertEqual(len(artifacts.temporal_sequences), 3)
        self.assertEqual(len(artifacts.interactions), 3)
        self.assertEqual(len(artifacts.features), 3)
        self.assertEqual(len(artifacts.interpretations), 3)
        payload = artifacts.interpretations[0].to_dict()
        json.dumps(payload)
        self.assertIn("interaction_type", payload)
        self.assertIn("limitations", payload)
        self.assertIn("prompt_version", payload)

    def test_reports_are_written(self) -> None:
        artifacts = run_pipeline(DATASET_ROOT, split="test", max_sequences=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_artifacts(artifacts, Path(temp_dir))
            self.assertTrue((Path(temp_dir) / "summary.json").exists())
            self.assertTrue((Path(temp_dir) / "report.md").exists())
            self.assertTrue((Path(temp_dir) / "temporal_sequences.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "interaction_events.jsonl").exists())

    def test_benchmark_summary_is_written(self) -> None:
        self.assertEqual(parse_model_spec("rule-based"), ("rule-based", "rule-based"))
        self.assertEqual(parse_model_spec("openai:gpt-4.1-mini"), ("openai", "gpt-4.1-mini"))
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_benchmark(
                DATASET_ROOT,
                Path(temp_dir),
                model_specs=["rule-based"],
                split="test",
                max_sequences=2,
            )
            self.assertIn("models", summary)
            self.assertTrue((Path(temp_dir) / "benchmark_summary.json").exists())

    def test_combined_summary_prefers_best_zero_shot(self) -> None:
        summary = build_combined_summary(
            evaluated_split="test",
            completed_model_summaries=[
                {"spec": "rule-based", "avg_confidence": 0.8, "num_sequences": 2},
                {"spec": "zero-shot:model-a", "avg_confidence": 0.4, "num_sequences": 2},
                {"spec": "zero-shot:model-b", "avg_confidence": 0.6, "num_sequences": 2},
            ],
            pairwise={"a__vs__b": 0.5},
        )
        self.assertEqual(summary["recommendation"]["best_zero_shot_baseline"], "zero-shot:model-b")

    def test_saved_run_ensemble_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            left = Path(temp_dir) / "left.jsonl"
            right = Path(temp_dir) / "right.jsonl"
            payloads = [
                {
                    "child_id": "c1",
                    "clip_id": "clip1",
                    "backend_name": "rule-based",
                    "model_name": "rule-based",
                    "interaction_type": "focused_attention",
                    "interpretation": "x",
                    "confidence": 0.84,
                    "evidence": ["a"],
                    "limitations": ["b"],
                    "prompt_version": "v1",
                },
                {
                    "child_id": "c2",
                    "clip_id": "clip2",
                    "backend_name": "rule-based",
                    "model_name": "rule-based",
                    "interaction_type": "mixed_attention",
                    "interpretation": "x",
                    "confidence": 0.68,
                    "evidence": ["a"],
                    "limitations": ["b"],
                    "prompt_version": "v1",
                },
            ]
            alt_payloads = [
                {
                    "child_id": "c1",
                    "clip_id": "clip1",
                    "backend_name": "zero-shot",
                    "model_name": "deberta",
                    "interaction_type": "focused_attention",
                    "interpretation": "y",
                    "confidence": 0.59,
                    "evidence": ["c"],
                    "limitations": ["d"],
                    "prompt_version": "v1",
                },
                {
                    "child_id": "c2",
                    "clip_id": "clip2",
                    "backend_name": "zero-shot",
                    "model_name": "deberta",
                    "interaction_type": "focused_attention",
                    "interpretation": "y",
                    "confidence": 0.91,
                    "evidence": ["c"],
                    "limitations": ["d"],
                    "prompt_version": "v1",
                },
            ]
            left.write_text("".join(json.dumps(item) + "\n" for item in payloads), encoding="utf-8")
            right.write_text("".join(json.dumps(item) + "\n" for item in alt_payloads), encoding="utf-8")
            summary = build_ensemble_from_saved_runs(
                "rule-based",
                left,
                "zero-shot:deberta",
                right,
                Path(temp_dir) / "ensemble",
            )
            self.assertIn("pairwise_agreement", summary)
            self.assertTrue((Path(temp_dir) / "ensemble" / "interpretations.jsonl").exists())

    def test_manual_eval_artifacts_and_empty_eval(self) -> None:
        sequences = load_child_sequences(DATASET_ROOT, split="test")[:4]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            rule = temp / "rule.jsonl"
            deberta = temp / "deberta.jsonl"
            distil = temp / "distil.jsonl"
            bart = temp / "bart.jsonl"
            rows = []
            for index, sequence in enumerate(sequences):
                rows.append(
                    {
                        "child_id": sequence.child_id,
                        "clip_id": sequence.clip_id,
                        "backend_name": "rule-based",
                        "model_name": "rule-based",
                        "interaction_type": "focused_attention" if index % 2 == 0 else "mixed_attention",
                        "interpretation": "x",
                        "confidence": 0.8,
                        "evidence": ["a"],
                        "limitations": ["b"],
                        "prompt_version": "v1",
                    }
                )
            rule.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            deberta.write_text(
                "".join(json.dumps({**item, "backend_name": "zero-shot", "model_name": "deberta"}) + "\n" for item in rows),
                encoding="utf-8",
            )
            distil.write_text(
                "".join(json.dumps({**item, "backend_name": "zero-shot", "model_name": "distil", "interaction_type": "exploratory_attention"}) + "\n" for item in rows),
                encoding="utf-8",
            )
            bart.write_text(
                "".join(json.dumps({**item, "backend_name": "zero-shot", "model_name": "bart", "interaction_type": "occluded_attention"}) + "\n" for item in rows),
                encoding="utf-8",
            )
            subset = build_manual_eval_subset(sequences, rule, deberta, distil, bart, max_examples=4)
            self.assertTrue(subset)
            write_label_schema(temp / "paper_eval")
            write_manual_eval_subset(temp / "paper_eval", subset)
            self.assertEqual(set(LABEL_SCHEMA), set(json.loads((temp / "paper_eval" / "label_schema.json").read_text())))
            summary = evaluate_against_manual_labels(
                temp / "paper_eval" / "manual_eval_annotations.csv",
                {"rule-based": rule},
                temp / "paper_eval",
            )
            self.assertEqual(summary["num_labeled_examples"], 0)

    def test_prepare_manual_eval_workflow_validates_feature_alignment(self) -> None:
        sequences = load_child_sequences(DATASET_ROOT, split="test")[:4]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            rule = temp / "rule.jsonl"
            deberta = temp / "deberta.jsonl"
            distil = temp / "distil.jsonl"
            bart = temp / "bart.jsonl"

            def write_rows(path: Path, backend: str, model: str, label_fn) -> None:
                rows = []
                for index, sequence in enumerate(sequences):
                    rows.append(
                        {
                            "child_id": sequence.child_id,
                            "clip_id": sequence.clip_id,
                            "backend_name": backend,
                            "model_name": model,
                            "interaction_type": label_fn(index),
                            "interpretation": "x",
                            "confidence": 0.8,
                            "evidence": ["a"],
                            "limitations": ["b"],
                            "prompt_version": "v1",
                        }
                    )
                path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

            write_rows(rule, "rule-based", "rule-based", lambda index: "focused_attention" if index % 2 == 0 else "mixed_attention")
            write_rows(deberta, "zero-shot", "deberta", lambda index: "focused_attention")
            write_rows(distil, "zero-shot", "distil", lambda index: "exploratory_attention")
            write_rows(bart, "zero-shot", "bart", lambda index: "occluded_attention")

            counts = prepare_manual_eval_workflow(
                dataset_root=DATASET_ROOT,
                run_output_dir=temp / "run",
                paper_eval_output_dir=temp / "paper_eval",
                rule_based_path=rule,
                deberta_path=deberta,
                distilbert_path=distil,
                bart_path=bart,
                split="test",
                max_examples=4,
                max_sequences=4,
            )

            self.assertEqual(counts["behavioral_features"], 4)
            self.assertEqual(counts["manual_eval_subset"], 4)
            validation = validate_saved_manual_eval_consistency(
                temp / "run" / "behavioral_features.jsonl",
                temp / "paper_eval" / "manual_eval_subset.json",
            )
            self.assertEqual(validation["behavioral_features"], 4)
            self.assertEqual(validation["manual_eval_subset"], 4)

    def test_silver_eval_workflow(self) -> None:
        sequences = load_child_sequences(DATASET_ROOT, split="test")[:4]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            rule = temp / "rule.jsonl"
            deberta = temp / "deberta.jsonl"
            distil = temp / "distil.jsonl"
            rows_rule = []
            rows_deberta = []
            rows_distil = []
            for sequence in sequences:
                common = {
                    "child_id": sequence.child_id,
                    "clip_id": sequence.clip_id,
                    "prompt_version": "v1",
                    "evidence": ["a"],
                    "limitations": ["b"],
                    "interpretation": "x",
                }
                rows_rule.append({**common, "backend_name": "rule-based", "model_name": "rule-based", "interaction_type": "focused_attention", "confidence": 0.84})
                rows_deberta.append({**common, "backend_name": "zero-shot", "model_name": "deberta", "interaction_type": "focused_attention", "confidence": 0.61})
                rows_distil.append({**common, "backend_name": "zero-shot", "model_name": "distil", "interaction_type": "exploratory_attention", "confidence": 0.52})
            rule.write_text("".join(json.dumps(item) + "\n" for item in rows_rule), encoding="utf-8")
            deberta.write_text("".join(json.dumps(item) + "\n" for item in rows_deberta), encoding="utf-8")
            distil.write_text("".join(json.dumps(item) + "\n" for item in rows_distil), encoding="utf-8")
            subset = build_silver_label_subset(sequences, rule, deberta)
            self.assertEqual(len(subset), 4)
            write_silver_label_subset(temp / "silver_eval", subset)
            summary = evaluate_against_silver_labels(
                temp / "silver_eval" / "silver_labels.json",
                {"rule-based": rule, "zero-shot:distil": distil},
                temp / "silver_eval",
            )
            self.assertEqual(summary["num_silver_examples"], 4)


if __name__ == "__main__":
    unittest.main()
