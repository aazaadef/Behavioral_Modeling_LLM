"""CLI entry point for the ChildPlay gaze-to-behavior pipeline.

Subcommands:
  run                        -- single backend pipeline execution
  benchmark                  -- multi-model comparison
  prepare-paper-eval         -- build manual evaluation package
  validate-paper-eval        -- validate paper eval artifact consistency
  build-manual-labeling-package -- generate annotated frame images
  run-extended-experiments   -- v2 pipeline with prompt logging
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_llm.benchmark import run_benchmark
from project_llm.manual_eval import (
    prepare_manual_eval_workflow,
    validate_saved_manual_eval_consistency,
)
from project_llm.manual_labeling_package import build_manual_labeling_package
from project_llm.pipeline import run_pipeline, save_artifacts
from project_llm.pipeline_v2 import run_extended_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChildPlay gaze-to-behavior pipeline")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data set/ChildPlay-gaze/ChildPlay-gaze"),
        help="Path to the ChildPlay dataset root containing annotations/, clips.csv, splits.csv, and videos.csv.",
    )
    run_parser.add_argument("--output-dir", type=Path, default=Path("output/results"))
    run_parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    run_parser.add_argument(
        "--backend", choices=["rule-based", "openai", "zero-shot", "hf-llm"], default="rule-based"
    )
    run_parser.add_argument("--model", default="gpt-4.1-mini")
    run_parser.add_argument("--max-sequences", type=int, default=None)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data set/ChildPlay-gaze/ChildPlay-gaze"),
    )
    benchmark_parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark"))
    benchmark_parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    benchmark_parser.add_argument("--max-sequences", type=int, default=None)
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model specs like rule-based, openai:gpt-4.1-mini, zero-shot:typeform/distilbert-base-uncased-mnli, hf-llm:Qwen/Qwen1.5-0.5B-Chat",
    )

    paper_eval_parser = subparsers.add_parser("prepare-paper-eval")
    paper_eval_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data set/ChildPlay-gaze/ChildPlay-gaze"),
    )
    paper_eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    paper_eval_parser.add_argument(
        "--run-output-dir", type=Path, default=Path("output/restart_run")
    )
    paper_eval_parser.add_argument("--paper-eval-dir", type=Path, default=Path("output/paper_eval"))
    paper_eval_parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of manual-eval examples to emit. Defaults to all eligible sequences in the split.",
    )
    paper_eval_parser.add_argument("--max-sequences", type=int, default=None)
    paper_eval_parser.add_argument(
        "--rule-based-path",
        type=Path,
        default=Path("output/benchmark_rule_based/rule-based/interpretations.jsonl"),
    )
    paper_eval_parser.add_argument(
        "--deberta-path",
        type=Path,
        default=Path(
            "output/benchmark_deberta_zeroshot/zero-shot__MoritzLaurer_deberta-v3-large-zeroshot-v2.0/interpretations.jsonl"
        ),
    )
    paper_eval_parser.add_argument(
        "--distilbert-path",
        type=Path,
        default=Path(
            "output/benchmark_zero_shot/zero-shot__typeform_distilbert-base-uncased-mnli/interpretations.jsonl"
        ),
    )
    paper_eval_parser.add_argument(
        "--bart-path",
        type=Path,
        default=Path(
            "output/benchmark_bart_large_mnli/zero-shot__facebook_bart-large-mnli/interpretations.jsonl"
        ),
    )

    validate_parser = subparsers.add_parser("validate-paper-eval")
    validate_parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("output/restart_run/behavioral_features.jsonl"),
    )
    validate_parser.add_argument(
        "--subset-path",
        type=Path,
        default=Path("output/paper_eval/manual_eval_subset.json"),
    )

    manual_package_parser = subparsers.add_parser("build-manual-labeling-package")
    manual_package_parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/paper_eval/manual_eval_annotations.csv"),
    )
    manual_package_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data set/ChildPlay-gaze/ChildPlay-gaze"),
    )
    manual_package_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/paper_eval/manual_labeling_package"),
    )

    extended_parser = subparsers.add_parser("run-extended-experiments")
    extended_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data set/ChildPlay-gaze/ChildPlay-gaze"),
    )
    extended_parser.add_argument(
        "--output-dir", type=Path, default=Path("output_experiments_v2/run_default")
    )
    extended_parser.add_argument(
        "--execution-mode",
        choices=["test-only", "full-analysis"],
        default="test-only",
        help="Use test-only for unbiased evaluation or full-analysis to aggregate train+val+test for exploratory analysis.",
    )
    extended_parser.add_argument("--max-sequences", type=int, default=None)
    extended_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Model specs like rule-based, "
            "zero-shot:typeform/distilbert-base-uncased-mnli, "
            "openai:gpt-4.1-mini, deepseek:deepseek-chat, hf-llm:Qwen/Qwen1.5-0.5B-Chat"
        ),
    )
    extended_parser.add_argument(
        "--prompt-version",
        default="childplay-behavior-v2",
        help="Version label stored in v2 reproducibility metadata.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "run"
    if command == "run":
        artifacts = run_pipeline(
            dataset_root=args.dataset_root,
            split=args.split,
            backend_name=args.backend,
            model=args.model,
            max_sequences=args.max_sequences,
        )
        save_artifacts(artifacts, args.output_dir)
        return 0

    if command == "benchmark":
        run_benchmark(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            model_specs=args.models,
            split=args.split,
            max_sequences=args.max_sequences,
        )
        return 0

    if command == "prepare-paper-eval":
        prepare_manual_eval_workflow(
            dataset_root=args.dataset_root,
            run_output_dir=args.run_output_dir,
            paper_eval_output_dir=args.paper_eval_dir,
            rule_based_path=args.rule_based_path,
            deberta_path=args.deberta_path,
            distilbert_path=args.distilbert_path,
            bart_path=args.bart_path,
            split=args.split,
            max_examples=args.max_examples,
            max_sequences=args.max_sequences,
        )
        return 0

    if command == "validate-paper-eval":
        validate_saved_manual_eval_consistency(
            features_path=args.features_path,
            subset_path=args.subset_path,
        )
        return 0

    if command == "build-manual-labeling-package":
        summary = build_manual_labeling_package(
            input_csv=args.input_csv,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
        )
        print(f"total_rows_read={summary['total_rows_read']}")
        print(f"total_images_generated={summary['total_images_generated']}")
        print(f"total_unresolved={summary['total_unresolved']}")
        print(f"output_directory={summary['output_directory']}")
        return 0

    if command == "run-extended-experiments":
        run_extended_experiments(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            model_specs=args.models,
            execution_mode=args.execution_mode,
            max_sequences=args.max_sequences,
            prompt_version=args.prompt_version,
        )
        return 0

    parser.error(f"Unsupported command: {command}")
    return 0
