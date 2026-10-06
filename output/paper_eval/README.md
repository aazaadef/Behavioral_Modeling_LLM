# Paper Evaluation Assets

This folder contains the assets behind the human reference labels for the 69 child–clip sequences of the ChildPlay-gaze test split.

## Files

| File | Purpose |
|---|---|
| `annotation_guidelines.md` | The guideline the raters used (original five-category protocol), with the merge into three classes. |
| `label_schema.json` | Definitions of the three merged classes (`focused` / `mix` / `others`), as used in the model prompts. |
| `manual_eval_annotations_3class.csv` | Analysis sheet: both raters' labels and the consensus (merged to three classes), with the rule and model predictions added after labelling. |
| `target_items_69.csv` / `.json` | Metadata for the 69 evaluation sequences (clip ID, fps, frame count, etc.). |
| `target_person_tracks.json` | Per-clip target person ID / time-window metadata. |

## Workflow

1. Two authors (A.F. and N.P.) labelled every sequence independently from the video clip, with the target child marked by a bounding box on a reference frame. They used `annotation_guidelines.md` (five categories) and had no feature values, summaries or model outputs.
2. The raters disagreed on 15 sequences in the five categories. All disagreements were resolved by joint discussion between the two raters.
3. While reviewing the clips together, the raters merged the five categories into three classes, because some categories had few or no members and could not be reliably distinguished: exploratory and mixed attention became `mix`, and occluded attention and reduced visual availability became `others`. No sequence was relabelled; after merging, 12 disagreements remain.
4. All systems were then run from scratch under the three-class schema ("Phase A" in file names).
5. The rule and model predictions were joined to the labels in `manual_eval_annotations_3class.csv` only for analysis, after labelling. The column `suggested_reference_label_3class` is a rule-derived column of that analysis sheet.

The consensus labels used as the reference are in `output/3class_eval/ground_truth_3class.csv`.
