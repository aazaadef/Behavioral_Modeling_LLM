# Labelling Guideline Used by the Raters

> This is the guideline the two raters used, from the original five-category protocol. The raters labelled each child–clip sequence from the video clip (the target child was marked by a bounding box on a reference frame). They were not given feature values, summaries or model outputs; the first instruction below refers to the evidence visible in the clip.
>
> After labelling, the five categories were merged into the three classes used in the paper; no sequence was relabelled:
>
> | Five-category label | Three-class label |
> |---|---|
> | `focused_attention` | `focused` |
> | `exploratory_attention`, `mixed_attention` | `mix` |
> | `occluded_attention`, `reduced_visual_availability` | `others` |
>
> The three-class definitions used in the model prompts are in `label_schema.json`.

---

# Manual Evaluation Guidelines

Use one label per child-sequence.
Base decisions only on the structured gaze evidence and clip-level summaries.
Do not infer diagnosis, emotion, or social intent beyond the visible evidence.

## Labels
- `focused_attention`: Predominantly stable, sustained visual attention on a target within the clip.
- `exploratory_attention`: Frequent shifts or scanning across multiple targets without one stable focus.
- `occluded_attention`: Attention cannot be characterized confidently because visible gaze is often unavailable or uncertain.
- `reduced_visual_availability`: Interpretation is limited by eyes closed, outside-frame gaze, or similar loss of visual evidence.
- `mixed_attention`: The clip contains both stable attention and notable shifts without one dominant pattern.

## Annotation Notes
- Prefer `occluded_attention` when visual evidence is frequently unavailable.
- Prefer `reduced_visual_availability` when eyes are closed or gaze is persistently outside the frame.
- Use `mixed_attention` only when no single pattern clearly dominates.
