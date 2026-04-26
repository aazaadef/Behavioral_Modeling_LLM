"""V1 backend implementations and label schema.

Defines the five valid behavioral labels, human-readable interpretation
templates, and all v1 backend classes: rule-based heuristic, OpenAI API,
HuggingFace zero-shot NLI, and HuggingFace generative LLM.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from typing import Protocol

from project_llm.features import BehavioralFeatures
from project_llm.interactions import InteractionEvent
from project_llm.prompts import PROMPT_VERSION, build_behavior_prompt
from project_llm.temporal import TemporalSegment


# The five valid behavioral labels used across all backends.
INTERACTION_TYPES = [
    "focused_attention",
    "exploratory_attention",
    "occluded_attention",
    "reduced_visual_availability",
    "mixed_attention",
]

# Human-readable description for each label, used in zero-shot outputs.
INTERPRETATION_MAP = {
    "focused_attention": "The child shows relatively stable and sustained visual attention within the clip.",
    "exploratory_attention": "The child appears to scan across multiple targets rather than maintaining one stable focus.",
    "occluded_attention": "The child's attention is difficult to characterize because gaze visibility is frequently limited.",
    "reduced_visual_availability": "Visual evidence is reduced due to extended eye closure or similar constraints.",
    "mixed_attention": "The clip contains a mixture of stable attention and shifting gaze behavior.",
}


@dataclass(frozen=True)
class InterpretationResult:
    """Output of any backend: predicted label, interpretation text, confidence, and evidence."""

    child_id: str
    clip_id: str
    backend_name: str
    model_name: str
    interaction_type: str
    interpretation: str
    confidence: float
    evidence: list[str]
    limitations: list[str]
    prompt_version: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class LLMBackend(Protocol):
    """Protocol that all v1 backends must implement."""

    def interpret(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> InterpretationResult: ...


class RuleBasedBackend:
    """Deterministic heuristic classifier using feature thresholds.

    Decision order: eyes_closed >= 0.3 -> reduced_visual_availability,
    occlusion >= 0.35 -> occluded_attention, high visibility + low shift
    + low motion + high stability -> focused_attention, high shift or
    high motion -> exploratory_attention, otherwise -> mixed_attention.
    """

    backend_name = "rule-based"
    model_name = "rule-based"

    def interpret(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> InterpretationResult:
        evidence = [
            f"visible_ratio={features.visible_ratio:.3f}",
            f"gaze_shift_ratio={features.gaze_shift_ratio:.3f}",
            f"occlusion_ratio={features.occlusion_ratio:.3f}",
            f"eyes_closed_ratio={features.eyes_closed_ratio:.3f}",
            f"mean_gaze_motion={features.mean_gaze_motion:.2f}",
            f"attention_stability_score={features.attention_stability_score:.3f}",
            f"max_visible_streak={features.max_visible_streak}",
            f"segments={len(segments)}",
            f"interactions={len(interactions)}",
        ]
        limitations: list[str] = []

        if features.eyes_closed_ratio >= 0.3:
            interaction_type = "reduced_visual_availability"
            interpretation = "Extended periods of closed eyes suggest reduced access to visual cues during the clip."
            confidence = 0.72
        elif features.occlusion_ratio >= 0.35:
            interaction_type = "occluded_attention"
            interpretation = "Frequent occlusion or uncertainty limits direct observation of the child's gaze target."
            confidence = 0.74
        elif (
            features.visible_ratio >= 0.75
            and features.gaze_shift_ratio <= 0.1
            and features.mean_gaze_motion <= 120
            and features.attention_stability_score >= 0.65
        ):
            interaction_type = "focused_attention"
            interpretation = "The child maintains stable visible gaze for most of the clip, consistent with sustained attention."
            confidence = 0.84
        elif features.gaze_shift_ratio >= 0.2 or features.mean_gaze_motion >= 180:
            interaction_type = "exploratory_attention"
            interpretation = (
                "Rapid gaze changes indicate active scanning or exploration of multiple targets."
            )
            confidence = 0.8
        else:
            interaction_type = "mixed_attention"
            interpretation = "The clip shows a mixture of stable attention and gaze changes without a single dominant pattern."
            confidence = 0.68

        if features.occlusion_ratio > 0.2:
            limitations.append(
                "Interpretation is partially limited by occluded or uncertain gaze frames."
            )
        if features.observed_frames < 60:
            limitations.append("Short clips provide less temporal evidence than longer sequences.")
        if not limitations:
            limitations.append(
                "Interpretation is limited to annotation-derived behavior and not video context."
            )

        return InterpretationResult(
            child_id=features.child_id,
            clip_id=features.clip_id,
            backend_name=self.backend_name,
            model_name=self.model_name,
            interaction_type=interaction_type,
            interpretation=interpretation,
            confidence=confidence,
            evidence=evidence,
            limitations=limitations,
            prompt_version=PROMPT_VERSION,
        )


class OpenAIBackend:
    """Sends the structured prompt to OpenAI API and parses JSON response."""

    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.backend_name = "openai"
        self.model_name = model

    def interpret(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> InterpretationResult:
        prompt = build_behavior_prompt(features, segments, interactions)
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            temperature=0,
        )
        text = response.output_text
        payload = json.loads(text)
        return InterpretationResult(
            child_id=features.child_id,
            clip_id=features.clip_id,
            backend_name=self.backend_name,
            model_name=self.model_name,
            interaction_type=payload["interaction_type"],
            interpretation=payload["interpretation"],
            confidence=float(payload["confidence"]),
            evidence=list(payload["evidence"]),
            limitations=list(payload["limitations"]),
            prompt_version=PROMPT_VERSION,
        )


class ZeroShotTransformerBackend:
    """HuggingFace zero-shot NLI classifier using hypothesis template matching."""

    def __init__(self, model: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers package is not installed") from exc

        self.backend_name = "zero-shot"
        self.model_name = model
        self._classifier = pipeline("zero-shot-classification", model=model)

    def interpret(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> InterpretationResult:
        premise = (
            "Child gaze behavior summary. "
            f"visible_ratio={features.visible_ratio:.3f}, "
            f"gaze_shift_ratio={features.gaze_shift_ratio:.3f}, "
            f"occlusion_ratio={features.occlusion_ratio:.3f}, "
            f"eyes_closed_ratio={features.eyes_closed_ratio:.3f}, "
            f"mean_gaze_motion={features.mean_gaze_motion:.2f}, "
            f"attention_stability_score={features.attention_stability_score:.3f}, "
            f"max_visible_streak={features.max_visible_streak}, "
            f"num_segments={len(segments)}, num_interactions={len(interactions)}."
        )
        result: Any = self._classifier(
            premise,
            candidate_labels=INTERACTION_TYPES,
            hypothesis_template="This clip is best described as {}.",
            multi_label=False,
        )
        labels: list[str] = list(result["labels"])
        scores: list[float] = [float(score) for score in result["scores"]]
        top_label = labels[0]
        evidence = [f"{label}={score:.3f}" for label, score in zip(labels[:3], scores[:3])]
        limitations = [
            "This backend maps numeric features to labels through a generic zero-shot model rather than task-specific fine-tuning.",
            "Behavioral quality should be compared cautiously because the dataset lacks high-level ground-truth labels.",
        ]
        return InterpretationResult(
            child_id=features.child_id,
            clip_id=features.clip_id,
            backend_name=self.backend_name,
            model_name=self.model_name,
            interaction_type=top_label,
            interpretation=INTERPRETATION_MAP[top_label],
            confidence=scores[0],
            evidence=evidence,
            limitations=limitations,
            prompt_version=PROMPT_VERSION,
        )


class HuggingFaceLLMBackend:
    """Local HuggingFace text-generation LLM with JSON output parsing.

    Falls back to ``mixed_attention`` with confidence 0.0 when the model
    output cannot be parsed as valid JSON.
    """

    def __init__(self, model: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers package is not installed") from exc

        self.backend_name = "hf-llm"
        self.model_name = model
        self._generator = pipeline("text-generation", model=model, device_map="auto")

    def interpret(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> InterpretationResult:
        prompt = build_behavior_prompt(features, segments, interactions)
        messages = [{"role": "user", "content": prompt}]

        response = self._generator(
            messages,
            max_new_tokens=400,
            temperature=0.0,
            do_sample=False,
            return_full_text=False,
        )
        text = response[0]["generated_text"]

        payload = {}
        try:
            match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
            if match:
                payload = json.loads(match.group(1))
            else:
                payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {
                "interaction_type": "mixed_attention",
                "interpretation": f"Error parsing LLM response: {text[:100]}...",
                "confidence": 0.0,
                "evidence": ["Failed to extract valid JSON format."],
                "limitations": [f"Raw text generated: {text[:200]}"],
            }

        return InterpretationResult(
            child_id=features.child_id,
            clip_id=features.clip_id,
            backend_name=self.backend_name,
            model_name=self.model_name,
            interaction_type=payload.get("interaction_type", "mixed_attention"),
            interpretation=payload.get("interpretation", ""),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=list(payload.get("evidence", [])),
            limitations=list(payload.get("limitations", [])),
            prompt_version=PROMPT_VERSION,
        )
