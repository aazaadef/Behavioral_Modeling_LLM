"""ChildPlay LLM-based behavioral analysis pipeline.

This package converts frame-level gaze annotations from the ChildPlay-gaze
dataset into high-level behavioral interpretations.  The processing stages are:

1. **dataset** -- load and filter CSV annotations into per-child clip sequences.
2. **temporal** -- segment each sequence into contiguous gaze-behavior blocks.
3. **features** -- extract numeric behavioral metrics from each sequence.
4. **interactions** -- infer higher-level interaction events from segments.
5. **llm / backends_v2** -- classify behavior via rule-based, zero-shot NLI,
   or generative LLM backends.
6. **pipeline / pipeline_v2** -- orchestrate the above stages end-to-end.
7. **benchmark / manual_eval** -- compare models, evaluate against human labels.
8. **paper_artifacts_v2 / reports** -- generate tables and exports for the paper.
"""

__all__ = [
    "benchmark",
    "cli",
    "dataset",
    "features",
    "llm",
    "pipeline",
    "prompts",
    "reports",
    "interactions",
    "manual_eval",
    "manual_labeling_package",
    "temporal",
    "backends_v2",
    "io_utils_v2",
    "paper_artifacts_v2",
    "pipeline_v2",
    "prompt_logging",
]
