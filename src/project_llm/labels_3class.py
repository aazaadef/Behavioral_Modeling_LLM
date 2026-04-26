"""Canonical 3-class label schema for ChildPlay attention analysis.

The original 5-class schema was collapsed by semantic grouping (not by
rarity) into a 3-class schema after two-annotator consensus analysis
showed that:

  - mixed_attention and exploratory_attention describe the same
    underlying behavior (active gaze switching / scanning) at different
    intensities. Inter-rater disagreements also frequently landed on
    this boundary.

  - occluded_attention and reduced_visual_availability both describe
    *visibility failure* — the gaze signal is not usable, regardless of
    the underlying behavior.

  - focused_attention remains its own class (stable sustained gaze).

This module is the single source of truth for the 3-class schema and
provides utilities used by the rule-based, NLI, supervised, and LLM
backends plus Phase A analysis.
"""

from __future__ import annotations

from project_llm.features import BehavioralFeatures


# Canonical 3-class label set.
LABELS_3CLASS: list[str] = ["focused", "mix", "others"]

# Deterministic 5→3 mapping. Every 5-class label maps to exactly one
# 3-class label; inverse is one-to-many.
MAP_5_TO_3: dict[str, str] = {
    "focused_attention": "focused",
    "mixed_attention": "mix",
    "exploratory_attention": "mix",
    "occluded_attention": "others",
    "reduced_visual_availability": "others",
}

# Human-readable descriptions used in NLI hypothesis templates and LLM
# prompts.
LABEL_DESCRIPTIONS_3CLASS: dict[str, str] = {
    "focused": (
        "The child shows stable, sustained visual attention on a target "
        "with minimal gaze shifts."
    ),
    "mix": (
        "The child actively switches gaze or scans across multiple "
        "targets, including mixed stable-plus-shifting and purely "
        "exploratory patterns."
    ),
    "others": (
        "Gaze is not reliably usable because of occlusion, extended eye "
        "closure, or the child being off-frame."
    ),
}


def map_5_to_3(label: str | None) -> str | None:
    """Map a 5-class label to its 3-class equivalent.

    Returns None for None/empty input so callers can distinguish
    unlabeled rows from valid labels. Unknown labels raise ValueError —
    silent coercion would hide data issues.
    """
    if label is None or label == "":
        return None
    label = label.strip()
    if label in LABELS_3CLASS:
        return label
    if label not in MAP_5_TO_3:
        raise ValueError(f"Unknown label: {label!r}")
    return MAP_5_TO_3[label]


def rule_based_3class(features: BehavioralFeatures) -> tuple[str, float]:
    """3-class rule-based classifier using the same thresholds as the
    5-class rule-based backend, then semantically mapped.

    Decision order mirrors the 5-class rule and returns a confidence
    calibrated to the decision branch.

    Returns (label, confidence).
    """
    if features.eyes_closed_ratio >= 0.3:
        return ("others", 0.72)  # was reduced_visual_availability
    if features.occlusion_ratio >= 0.35:
        return ("others", 0.74)  # was occluded_attention
    if (
        features.visible_ratio >= 0.75
        and features.gaze_shift_ratio <= 0.1
        and features.mean_gaze_motion <= 120
        and features.attention_stability_score >= 0.65
    ):
        return ("focused", 0.84)
    # Everything else — both the old exploratory branch and the old
    # mixed fallback — collapses into "mix".
    if features.gaze_shift_ratio >= 0.2 or features.mean_gaze_motion >= 180:
        return ("mix", 0.80)  # was exploratory_attention
    return ("mix", 0.68)  # was mixed_attention


def llm_prompt_3class(features_payload: dict[str, object]) -> str:
    """Build a 3-class LLM prompt with explicit label definitions."""
    import json as _json

    label_block = "\n".join(
        f"  - {name}: {desc}" for name, desc in LABEL_DESCRIPTIONS_3CLASS.items()
    )
    return (
        "You are an expert in child visual attention analysis.\n\n"
        "Given the following structured behavioral features extracted from "
        "a child's gaze tracking data in a video clip, classify the child's "
        "overall attention behavior into exactly ONE of these three categories:\n\n"
        f"{label_block}\n\n"
        "Return ONLY a valid JSON object with these three fields:\n"
        '  - "label": one of the three category names above (exactly as written)\n'
        '  - "confidence": a float between 0 and 1\n'
        '  - "reasoning": a short explanation (1-2 sentences)\n\n'
        "Do NOT include any text before or after the JSON object.\n\n"
        f"Features:\n{_json.dumps(features_payload, ensure_ascii=True, sort_keys=True)}"
    )


def nli_candidate_labels_3class() -> list[str]:
    """Candidate labels to feed into a zero-shot NLI pipeline."""
    return list(LABELS_3CLASS)


def nli_hypothesis_template_3class() -> str:
    """Hypothesis template for zero-shot NLI classification."""
    return "This clip is best described as {}."
