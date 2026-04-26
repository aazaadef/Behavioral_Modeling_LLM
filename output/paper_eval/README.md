# Paper Evaluation Assets

This folder contains the manual-evaluation workflow for turning the current exploratory benchmark into a paper-grade evaluation.

Files expected here:

- `label_schema.json`
- `annotation_guidelines.md`
- `manual_eval_subset.json`
- `manual_eval_annotations.csv`
- `manual_eval_subset_summary.json`
- `paper_eval_summary.json` after manual labeling is completed

Workflow:

1. Review `label_schema.json` and `annotation_guidelines.md`.
2. Fill `manual_eval_annotations.csv` with at least `final_label`.
3. Run the evaluation utility against saved model predictions.
