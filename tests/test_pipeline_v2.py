"""Unit tests for the V2 extended-experiments pipeline.

Verifies that ``run_extended_experiments()`` produces the full set of
required output artifacts (merged predictions, agreement tables, paper
artifacts, etc.) and that it correctly refuses to overwrite an existing
output directory.
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

from project_llm.pipeline_v2 import run_extended_experiments


DATASET_ROOT = ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze"


class ExtendedPipelineTests(unittest.TestCase):
    def test_rule_based_extended_pipeline_writes_new_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output_experiments_v2" / "rule_based_smoke"
            summary = run_extended_experiments(
                dataset_root=DATASET_ROOT,
                output_dir=output_dir,
                model_specs=["rule-based"],
                execution_mode="test-only",
                max_sequences=2,
                command="pytest smoke",
            )
            self.assertEqual(summary["num_successful_models"], 1)
            required = [
                "child_sequences.jsonl",
                "temporal_sequences.jsonl",
                "interaction_events.jsonl",
                "behavioral_features.jsonl",
                "interpretations.jsonl",
                "logged_prompts.jsonl",
                "logged_prompts.csv",
                "merged_predictions.csv",
                "merged_predictions.json",
                "disagreement_report.csv",
                "confidence_summary.csv",
                "per_class_summary.csv",
                "run_summary.json",
                "reproducibility_notes.txt",
                "model_comparison_table.csv",
                "model_comparison_table.md",
                "confusion_matrices.json",
                "confusion_matrices.csv",
                "agreement_table.csv",
                "examples_for_error_analysis.csv",
                "examples_for_case_study.csv",
                "prompt_examples.csv",
                "model_metadata.json",
            ]
            for name in required:
                self.assertTrue((output_dir / name).exists(), name)

            with (output_dir / "merged_predictions.json").open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            self.assertEqual(len(rows), 2)
            self.assertIn("prompt_texts_json", rows[0])

    def test_extended_pipeline_refuses_to_overwrite_existing_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output_experiments_v2" / "rule_based_smoke"
            output_dir.mkdir(parents=True, exist_ok=False)
            with self.assertRaises(FileExistsError):
                run_extended_experiments(
                    dataset_root=DATASET_ROOT,
                    output_dir=output_dir,
                    model_specs=["rule-based"],
                    execution_mode="test-only",
                    max_sequences=1,
                )


if __name__ == "__main__":
    unittest.main()
