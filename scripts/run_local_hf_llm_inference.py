"""Standalone local LLM inference script for Qwen-7B.

Loads the model with ``device_map="auto"``, runs on 69 test sequences
using a custom prompt (label space: inside_visible / outside / unsure),
and saves prompt logs, raw outputs, parsed outputs, and errors.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.dataset import load_child_sequences
from project_llm.features import extract_behavioral_features
from project_llm.interactions import infer_interactions
from project_llm.io_utils_v2 import ensure_new_output_dir, iso_timestamp, write_json, write_jsonl
from project_llm.temporal import build_temporal_segments


LOCAL_MODEL_PATH = ROOT / "models" / "qwen-7b"
OUTPUT_DIR = ROOT / "output_experiments_v2" / "hf_llm_local_runs_v5"
ALLOWED_LABELS = {"inside_visible", "outside", "unsure"}


def build_prompt(features_payload: dict[str, object]) -> str:
    # Keep the prompt deterministic and force a single JSON object as output.
    return (
        "You are an expert in child behavior analysis.\n"
        "Given the following structured features, classify gaze behavior.\n\n"
        "Return ONLY valid JSON with:\n"
        "- label (one of: inside_visible, outside, unsure)\n"
        "- confidence (0 to 1)\n"
        "- reasoning (short explanation)\n\n"
        f"Features:\n{json.dumps(features_payload, ensure_ascii=True, sort_keys=True)}"
    )


def build_model_inputs(tokenizer, prompt: str):
    # Qwen instruct checkpoints respond more reliably when we use the chat template.
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt")["input_ids"]


def extract_json_payload(text: str) -> dict[str, object]:
    # Accept raw JSON, fenced JSON, or truncated JSON that can be repaired conservatively.
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in fenced if item.strip())
    inline = re.findall(r"({[\s\S]*})", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in inline if item.strip())

    if "{" in text:
        partial = text[text.find("{") :].strip()
        candidates.append(partial)

    for candidate in candidates:
        repaired = candidate
        if repaired and not repaired.endswith("}"):
            repaired = repaired.rstrip()
            if '"reasoning"' in repaired and repaired.count("{") > repaired.count("}"):
                repaired = repaired + '"}'
            while repaired.count("{") > repaired.count("}"):
                repaired += "}"
        try:
            payload = json.loads(repaired)
        except Exception:
            continue
        label = str(payload.get("label", "unsure")).strip()
        confidence = float(payload.get("confidence", 0.0))
        reasoning = str(payload.get("reasoning", "")).strip()
        if label not in ALLOWED_LABELS:
            label = "unsure"
        confidence = max(0.0, min(1.0, confidence))
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    raise ValueError("Could not extract valid JSON payload from model output")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_new_output_dir(OUTPUT_DIR)
    sequences = load_child_sequences(ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze", split="test")
    sequences = sequences[:69]

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )

    prompt_logs: list[dict[str, object]] = []
    raw_outputs: list[dict[str, object]] = []
    parsed_outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for sequence in sequences:
        features = extract_behavioral_features(sequence)
        segments = build_temporal_segments(sequence)
        interactions = infer_interactions(segments)
        features_payload = {
            **features.to_dict(),
            "num_segments": len(segments),
            "num_interactions": len(interactions),
        }
        prompt = build_prompt(features_payload)
        prompt_logs.append(
            {
                "child_id": sequence.child_id,
                "clip_id": sequence.clip_id,
                "split": sequence.split,
                "prompt": prompt,
                "timestamp": iso_timestamp(),
            }
        )

        try:
            input_ids = build_model_inputs(tokenizer, prompt)
            if hasattr(input_ids, "to") and hasattr(model, "device") and str(model.device) != "cpu":
                input_ids = input_ids.to(model.device)
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=192,
                min_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(outputs[0][input_ids.shape[1] :], skip_special_tokens=True).strip()
            raw_outputs.append(
                {
                    "child_id": sequence.child_id,
                    "clip_id": sequence.clip_id,
                    "raw_output": generated,
                }
            )
            parsed = extract_json_payload(generated)
            parsed_outputs.append(
                {
                    "child_id": sequence.child_id,
                    "clip_id": sequence.clip_id,
                    "split": sequence.split,
                    "backend_name": "hf-llm",
                    "model_name": str(LOCAL_MODEL_PATH),
                    "label": parsed["label"],
                    "confidence": parsed["confidence"],
                    "reasoning": parsed["reasoning"],
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "child_id": sequence.child_id,
                    "clip_id": sequence.clip_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            parsed_outputs.append(
                {
                    "child_id": sequence.child_id,
                    "clip_id": sequence.clip_id,
                    "split": sequence.split,
                    "backend_name": "hf-llm",
                    "model_name": str(LOCAL_MODEL_PATH),
                    "label": "unsure",
                    "confidence": 0.0,
                    "reasoning": f"fallback_due_to_error={type(exc).__name__}",
                }
            )

    write_jsonl(OUTPUT_DIR / "prompt_logs.jsonl", prompt_logs)
    write_jsonl(OUTPUT_DIR / "raw_outputs.jsonl", raw_outputs)
    write_jsonl(OUTPUT_DIR / "parsed_outputs.jsonl", parsed_outputs)
    write_jsonl(OUTPUT_DIR / "errors.jsonl", errors)
    write_json(
        OUTPUT_DIR / "run_summary.json",
        {
            "model_path": str(LOCAL_MODEL_PATH),
            "gpu_used": bool(getattr(model, "hf_device_map", {}).get("") != "cpu"),
            "device_map": getattr(model, "hf_device_map", None),
            "num_sequences": len(sequences),
            "num_successful_predictions": len(sequences) - len(errors),
            "num_errors": len(errors),
            "timestamp": iso_timestamp(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
