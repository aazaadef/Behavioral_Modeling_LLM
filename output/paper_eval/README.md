# Paper Evaluation Assets

This folder contains the manual-evaluation assets used to build the
3-class consensus reference labels for the 69-clip test subset.

## Files

| File | Purpose |
|---|---|
| `label_schema.json` | Canonical 3-class label schema (`focused` / `mix` / `others`) with definitions. |
| `annotation_guidelines.md` | Rater guidelines used during the manual labeling pass. |
| `manual_eval_annotations_3class.csv` | Two-rater 3-class annotations for the 69-clip subset, plus consensus column. |
| `target_items_69.csv` / `.json` | Metadata for the 69 evaluation clips (clip ID, fps, frame count, etc.). |
| `target_person_tracks.json` | Per-clip target person ID / time-window metadata. |

## Workflow

The labels were collected as follows:

1. Two raters read `annotation_guidelines.md` and the schema in
   `label_schema.json`.
2. Each rater independently assigned one label per clip in
   `target_items_69.csv`.
3. Cohen's κ was computed over the two label columns; consensus was
   produced by tie-breaking with a third reviewer for the
   inter-rater disagreement zone (12 of 69 samples).
4. The final consensus labels live in
   `manual_eval_annotations_3class.csv` (column
   `final_label_3class`) and in
   `output/3class_eval/ground_truth_3class.csv`.

The downstream Phase A evaluation uses
`output/3class_eval/ground_truth_3class.csv` as the reference;
this folder is kept for reproducibility of the labeling protocol
itself.
