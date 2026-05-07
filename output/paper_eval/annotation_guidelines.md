# Manual Evaluation Guidelines

Use one label per child-sequence (per child × clip).
Base decisions only on the structured gaze evidence and clip-level
summaries.
Do not infer diagnosis, emotion, or social intent beyond the visible
evidence.

## Labels (3-class schema)

- **`focused`** — The child shows stable, sustained visual attention
  on a target with minimal gaze shifts.
- **`mix`** — The child actively switches gaze or scans across
  multiple targets, including mixed stable-plus-shifting and purely
  exploratory patterns.
- **`others`** — Gaze is not reliably usable because of occlusion,
  extended eye closure, or the child being off-frame.

## Annotation Notes

- Use `others` when more than ~half the frames have unusable gaze
  signal (occluded, eyes closed, or outside-frame), regardless of
  what the visible portion suggests.
- Use `focused` when the dominant pattern across the clip is a
  single stable target with very few shifts.
- Use `mix` for everything in between — clips that contain
  meaningful visible gaze but with active switching, scanning, or
  no single dominant target.
- When in doubt between `focused` and `mix`, lean `mix`.
- When in doubt between `mix` and `others`, lean toward whichever
  the *majority of the clip* exhibits.
