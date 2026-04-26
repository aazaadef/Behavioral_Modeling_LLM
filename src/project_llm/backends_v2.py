"""V2 experiment backends with integrated prompt logging.

Each backend wraps the core inference logic and returns a
:class:`BackendResponse` that bundles the interpretation result together
with a :class:`PromptLogEntry` for reproducible auditing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from project_llm.features import BehavioralFeatures
from project_llm.interactions import InteractionEvent
from project_llm.io_utils_v2 import iso_timestamp
from project_llm.llm import (
    INTERACTION_TYPES,
    INTERPRETATION_MAP,
    InterpretationResult,
    RuleBasedBackend,
)
from project_llm.prompt_logging import (
    PromptLogEntry,
    build_feature_summary,
    build_zero_shot_premise,
)
from project_llm.prompts import PROMPT_VERSION, build_behavior_prompt
from project_llm.temporal import TemporalSegment


@dataclass(frozen=True)
class BackendResponse:
    """Bundles an interpretation result with its prompt audit log entry."""

    interpretation: InterpretationResult
    prompt_log: PromptLogEntry


class ExperimentBackend(Protocol):
    """Protocol that all v2 backends must implement."""

    backend_name: str
    model_name: str
    backend_family: str

    def interpret_with_logging(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> BackendResponse: ...


def _build_prompt_log(
    *,
    features: BehavioralFeatures,
    backend_name: str,
    model_name: str,
    premise_text: str,
    full_prompt_text: str,
    interpretation: InterpretationResult,
) -> PromptLogEntry:
    # Centralize prompt logging so every backend emits the same audit schema.
    return PromptLogEntry(
        child_id=features.child_id,
        clip_id=features.clip_id,
        backend_name=backend_name,
        model_name=model_name,
        split=features.split,
        premise_text=premise_text,
        full_prompt_text=full_prompt_text,
        candidate_labels=list(INTERACTION_TYPES),
        predicted_label=interpretation.interaction_type,
        confidence=interpretation.confidence,
        evidence=list(interpretation.evidence),
        limitations=list(interpretation.limitations),
        timestamp=iso_timestamp(),
        prompt_version=interpretation.prompt_version,
        feature_summary=build_feature_summary(features),
    )


class RuleBasedBackendV2:
    """V2 wrapper around the v1 rule-based backend, adding prompt logging."""

    backend_name = "rule-based"
    model_name = "rule-based"
    backend_family = "rule-based"

    def __init__(self) -> None:
        # Reuse the existing baseline implementation to stay backward compatible.
        self._backend = RuleBasedBackend()

    def interpret_with_logging(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> BackendResponse:
        interpretation = self._backend.interpret(features, segments, interactions)
        premise = build_zero_shot_premise(features, segments, interactions)
        return BackendResponse(
            interpretation=interpretation,
            prompt_log=_build_prompt_log(
                features=features,
                backend_name=self.backend_name,
                model_name=self.model_name,
                premise_text=premise,
                full_prompt_text="",
                interpretation=interpretation,
            ),
        )


class HuggingFaceZeroShotBackendV2:
    """HuggingFace NLI zero-shot classifier with prompt logging."""

    backend_name = "zero-shot"
    backend_family = "zero-shot"

    def __init__(self, model: str) -> None:
        # This stays generic so additional NLI models can be added via model spec alone.
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers package is not installed in the active environment"
            ) from exc

        self.model_name = model
        self._classifier = pipeline("zero-shot-classification", model=model)

    def interpret_with_logging(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> BackendResponse:
        premise = build_zero_shot_premise(features, segments, interactions)
        result: Any = self._classifier(
            premise,
            candidate_labels=INTERACTION_TYPES,
            hypothesis_template="This clip is best described as {}.",
            multi_label=False,
        )
        labels = [str(item) for item in result["labels"]]
        scores = [float(item) for item in result["scores"]]
        top_label = labels[0]
        interpretation = InterpretationResult(
            child_id=features.child_id,
            clip_id=features.clip_id,
            backend_name=self.backend_name,
            model_name=self.model_name,
            interaction_type=top_label,
            interpretation=INTERPRETATION_MAP[top_label],
            confidence=scores[0],
            evidence=[f"{label}={score:.3f}" for label, score in zip(labels[:5], scores[:5])],
            limitations=[
                "Zero-shot/NLI prediction over annotation-derived features instead of task-specific fine-tuning.",
                "Outputs should be validated against manual labels before paper claims are finalized.",
            ],
            prompt_version=PROMPT_VERSION,
        )
        return BackendResponse(
            interpretation=interpretation,
            prompt_log=_build_prompt_log(
                features=features,
                backend_name=self.backend_name,
                model_name=self.model_name,
                premise_text=premise,
                full_prompt_text="",
                interpretation=interpretation,
            ),
        )


class OpenAIBackendV2:
    """OpenAI API backend with prompt logging."""

    backend_name = "openai"
    backend_family = "openai"

    def __init__(self, model: str) -> None:
        # Match the existing repo behavior while adding prompt logging around it.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed in the active environment") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._client = OpenAI(api_key=api_key)
        self.model_name = model

    def interpret_with_logging(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> BackendResponse:
        prompt = build_behavior_prompt(features, segments, interactions)
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0,
        )
        payload = json.loads(response.output_text)
        interpretation = InterpretationResult(
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
        return BackendResponse(
            interpretation=interpretation,
            prompt_log=_build_prompt_log(
                features=features,
                backend_name=self.backend_name,
                model_name=self.model_name,
                premise_text="",
                full_prompt_text=prompt,
                interpretation=interpretation,
            ),
        )


class DeepSeekBackendV2:
    """DeepSeek backend via OpenAI-compatible client, with prompt logging."""

    backend_name = "deepseek"
    backend_family = "deepseek"

    def __init__(self, model: str) -> None:
        # DeepSeek is wired through an OpenAI-compatible client for later enablement.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for the DeepSeek-compatible client"
            ) from exc

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model

    def interpret_with_logging(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> BackendResponse:
        prompt = build_behavior_prompt(features, segments, interactions)
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0,
        )
        payload = json.loads(response.output_text)
        interpretation = InterpretationResult(
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
        return BackendResponse(
            interpretation=interpretation,
            prompt_log=_build_prompt_log(
                features=features,
                backend_name=self.backend_name,
                model_name=self.model_name,
                premise_text="",
                full_prompt_text=prompt,
                interpretation=interpretation,
            ),
        )


def build_backend_v2(name: str, model: str) -> ExperimentBackend:
    """Factory: dispatch a backend name to its v2 class."""
    if name == "rule-based":
        return RuleBasedBackendV2()
    if name == "zero-shot":
        return HuggingFaceZeroShotBackendV2(model=model)
    if name == "openai":
        return OpenAIBackendV2(model=model)
    if name == "deepseek":
        return DeepSeekBackendV2(model=model)
    if name == "hf-llm":
        return HuggingFaceLLMBackendV2(model=model)
    raise ValueError(f"Unsupported backend for v2 experiments: {name}")


class HuggingFaceLLMBackendV2:
    """Local HuggingFace text-generation LLM with JSON parsing and prompt logging.

    Falls back to ``mixed_attention`` with confidence 0.0 on JSON parse failure.
    """

    backend_name = "hf-llm"
    backend_family = "hf-llm"

    def __init__(self, model: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers package is not installed in the active environment"
            ) from exc

        self.model_name = model
        self._generator = pipeline("text-generation", model=model, device_map="auto")

    def interpret_with_logging(
        self,
        features: BehavioralFeatures,
        segments: list[TemporalSegment],
        interactions: list[InteractionEvent],
    ) -> BackendResponse:
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

        interpretation = InterpretationResult(
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

        return BackendResponse(
            interpretation=interpretation,
            prompt_log=_build_prompt_log(
                features=features,
                backend_name=self.backend_name,
                model_name=self.model_name,
                premise_text="",
                full_prompt_text=prompt,
                interpretation=interpretation,
            ),
        )
