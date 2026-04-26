"""Run supervised classification baselines on extracted behavioral features.

Trains Logistic Regression, Linear SVM, and Random Forest classifiers
using stratified k-fold cross-validation on the labeled subset of the
69 test-split samples.  Outputs per-classifier metrics, per-sample
predictions, confusion matrices, and classification reports to the
designated output directory.

Usage:
    python scripts/run_supervised_baselines.py [--folds N]

Requires:
    - output/restart_run/behavioral_features.jsonl  (extracted features)
    - output/paper_eval/manual_eval_annotations.csv (manual labels)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.supervised_baselines import run_supervised_baselines


# Default paths matching the project's existing output layout.
DEFAULT_FEATURES = ROOT / "output_experiments_v2" / "hf_comparison_v1" / "behavioral_features.jsonl"
DEFAULT_ANNOTATIONS = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"
DEFAULT_OUTPUT = ROOT / "output" / "supervised_baselines"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run supervised baselines on behavioral features")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Path to behavioral_features.jsonl")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS, help="Path to manual_eval_annotations.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory for results")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    if not args.features.exists():
        logging.error("Features file not found: %s", args.features)
        return 1
    if not args.annotations.exists():
        logging.error("Annotations file not found: %s", args.annotations)
        return 1

    results = run_supervised_baselines(
        features_path=args.features,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        n_folds=args.folds,
    )

    # Print summary table to stdout.
    print("\n" + "=" * 72)
    print("SUPERVISED BASELINE RESULTS")
    print("=" * 72)
    print(f"{'Classifier':<22} {'Accuracy':>12} {'Macro-F1':>12} {'Kappa':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r.classifier_name:<22} "
            f"{r.mean_accuracy:>8.4f}±{r.std_accuracy:<4.4f}"
            f"{r.mean_macro_f1:>8.4f}±{r.std_macro_f1:<4.4f}"
            f"{r.mean_kappa:>10.4f}"
        )
    print("-" * 72)
    print(f"Output saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
