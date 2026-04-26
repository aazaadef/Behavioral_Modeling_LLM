"""Unified local LLM inference script with correct 5-label behavioral schema.

Runs any HuggingFace causal-LM checkpoint (Qwen, Llama, Mistral, etc.)
on the 69 test-split sequences using the project's canonical five-label
attention schema.  Supports both local model paths and HuggingFace Hub
model IDs.

Label space (aligned with all other backends):
    focused_attention, exploratory_attention, occluded_attention,
    reduced_visual_availability, mixed_attention

Outputs per model run:
    - prompt_logs.jsonl      — full prompts sent to the model
    - raw_outputs.jsonl      — raw decoded text from the model
    - parsed_outputs.jsonl   — structured label / confidence / reasoning
    - errors.jsonl           — any sequences that failed
    - run_summary.json       — metadata and aggregate stats

Usage:
    python scripts/run_local_llm_unified.py --model ./models/qwen-7b
    python scripts/run_local_llm_unified.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_local_llm_unified.py --model Qwen/Qwen2.5-72B-Instruct --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import logging
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
from project_llm.io_utils_v2 import iso_timestamp, write_json, write_jsonl
from project_llm.temporal import build_temporal_segments

logger = logging.getLogger(__name__)

# ── Canonical label space (must match all other backends) ────────────
ALLOWED_LABELS = {
    "focused_attention",
    "exploratory_attention",
    "occluded_attention",
    "reduced_visual_availability",
    "mixed_attention",
}

LABEL_DESCRIPTIONS = {
    "focused_attention": "Predominantly stable, sustained visual attention on a target.",
    "exploratory_attention": "Frequent shifts or scanning across multiple targets.",
    "occluded_attention": "Attention cannot be characterized due to unavailable gaze.",
    "reduced_visual_availability": "Interpretation limited by closed eyes or off-frame gaze.",
    "mixed_attention": "Both stable attention and notable shifts without one dominant pattern.",
}


# ── Prompt construction ──────────────────────────────────────────────

def build_prompt(features_payload: dict[str, object]) -> str:
    """Build a structured prompt that forces the model to respond with
    one of the five canonical behavioral-attention labels.

    The prompt includes label definitions so the model understands
    exactly what each label means — this prevents the label-space
    mismatch that occurred with the earlier inside_visible/outside
    prompt design.
    """
    label_block = "\n".join(
        f"  - {name}: {desc}" for name, desc in LABEL_DESCRIPTIONS.items()
    )
    return (
        "You are an expert in child visual attention analysis.\n\n"
        "Given the following structured behavioral features extracted from "
        "a child's gaze tracking data in a video clip, classify the child's "
        "overall attention behavior into exactly ONE of these five categories:\n\n"
        f"{label_block}\n\n"
        "Return ONLY a valid JSON object with these three fields:\n"
        '  - "label": one of the five category names above (exactly as written)\n'
        '  - "confidence": a float between 0 and 1\n'
        '  - "reasoning": a short explanation (1-2 sentences)\n\n'
        "Do NOT include any text before or after the JSON object.\n\n"
        f"Features:\n{json.dumps(features_payload, ensure_ascii=True, sort_keys=True)}"
    )


# ── Model input preparation ─────────────────────────────────────────

def build_model_inputs(tokenizer, prompt: str):
    """Prepare tokenized input IDs using the chat template if available.

    Instruct-tuned models respond more reliably when the chat template
    is used.  Falls back to raw tokenization for base models.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt")["input_ids"]


# ── JSON extraction from model output ────────────────────────────────

def extract_json_payload(text: str) -> dict[str, object]:
    """Extract and validate a JSON classification payload from raw text.

    Tries multiple strategies: raw parse, fenced code blocks, inline
    braces, and conservative truncation repair.  Normalizes the label
    to one of the five allowed values (falls back to 'mixed_attention'
    if unrecognized).
    """
    candidates = [text.strip()]

    # Try fenced JSON blocks (```json ... ```).
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in fenced if item.strip())

    # Try inline JSON objects ({ ... }).
    inline = re.findall(r"(\{[\s\S]*\})", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in inline if item.strip())

    # Try from first brace to end (for truncated outputs).
    if "{" in text:
        partial = text[text.find("{"):].strip()
        candidates.append(partial)

    for candidate in candidates:
        repaired = candidate
        # Attempt to close truncated JSON.
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

        label = str(payload.get("label", "mixed_attention")).strip().lower()
        confidence = float(payload.get("confidence", 0.0))
        reasoning = str(payload.get("reasoning", "")).strip()

        # Normalize label to the allowed set.
        if label not in ALLOWED_LABELS:
            # Try partial matching (e.g. "focused" → "focused_attention").
            matched = [l for l in ALLOWED_LABELS if label in l or l in label]
            label = matched[0] if matched else "mixed_attention"

        confidence = max(0.0, min(1.0, confidence))
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    raise ValueError("Could not extract valid JSON payload from model output")


# ── Main inference loop ──────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run local LLM inference with correct 5-label schema")
    parser.add_argument("--model", type=str, required=True, help="Local path or HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-generated if omitted)")
    parser.add_argument("--max-sequences", type=int, default=69, help="Maximum number of test sequences to process")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization (for large models)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization")
    args = parser.parse_args()

    # Import heavy dependencies only when actually running.
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    # Resolve model path and output directory.
    model_path = Path(args.model) if Path(args.model).exists() else args.model
    model_short_name = str(model_path).replace("/", "_").replace(".", "_")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / "output" / "llm_runs" / model_short_name

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model: %s", model_path)
    logger.info("Output: %s", output_dir)

    # Load dataset sequences.
    dataset_root = ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze"
    sequences = load_child_sequences(dataset_root, split="test")
    sequences = sequences[: args.max_sequences]
    logger.info("Loaded %d test sequences", len(sequences))

    # Load model and tokenizer.
    load_kwargs: dict[str, object] = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if isinstance(model_path, Path):
        load_kwargs["local_files_only"] = True

    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization")
    elif args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=isinstance(model_path, Path),
    )
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)

    # Run inference on each sequence.
    prompt_logs: list[dict[str, object]] = []
    raw_outputs: list[dict[str, object]] = []
    parsed_outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for idx, sequence in enumerate(sequences):
        logger.info("Processing %d/%d: %s", idx + 1, len(sequences), sequence.child_id)

        features = extract_behavioral_features(sequence)
        segments = build_temporal_segments(sequence)
        interactions = infer_interactions(segments)
        features_payload = {
            **features.to_dict(),
            "num_segments": len(segments),
            "num_interactions": len(interactions),
        }
        # Remove metadata fields from the feature payload sent to the LLM.
        for key in ("child_id", "clip_id", "split"):
            features_payload.pop(key, None)

        prompt = build_prompt(features_payload)
        prompt_logs.append({
            "child_id": sequence.child_id,
            "clip_id": sequence.clip_id,
            "split": sequence.split,
            "prompt": prompt,
            "timestamp": iso_timestamp(),
        })

        try:
            input_ids = build_model_inputs(tokenizer, prompt)
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                outputs[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            raw_outputs.append({
                "child_id": sequence.child_id,
                "clip_id": sequence.clip_id,
                "raw_output": generated,
            })

            parsed = extract_json_payload(generated)
            parsed_outputs.append({
                "child_id": sequence.child_id,
                "clip_id": sequence.clip_id,
                "split": sequence.split,
                "backend_name": "hf-llm",
                "model_name": str(model_path),
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
            })
            logger.info("  → %s (conf=%.2f)", parsed["label"], parsed["confidence"])

        except Exception as exc:
            logger.error("  → ERROR: %s: %s", type(exc).__name__, exc)
            errors.append({
                "child_id": sequence.child_id,
                "clip_id": sequence.clip_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            parsed_outputs.append({
                "child_id": sequence.child_id,
                "clip_id": sequence.clip_id,
                "split": sequence.split,
                "backend_name": "hf-llm",
                "model_name": str(model_path),
                "label": "mixed_attention",
                "confidence": 0.0,
                "reasoning": f"fallback_due_to_error={type(exc).__name__}",
            })

    # Save all outputs.
    write_jsonl(output_dir / "prompt_logs.jsonl", prompt_logs)
    write_jsonl(output_dir / "raw_outputs.jsonl", raw_outputs)
    write_jsonl(output_dir / "parsed_outputs.jsonl", parsed_outputs)
    write_jsonl(output_dir / "errors.jsonl", errors)

    # Build label distribution summary.
    from collections import Counter
    label_dist = dict(Counter(r["label"] for r in parsed_outputs))

    write_json(output_dir / "run_summary.json", {
        "model_path": str(model_path),
        "label_space": sorted(ALLOWED_LABELS),
        "quantization": "4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "none"),
        "gpu_used": True,
        "num_sequences": len(sequences),
        "num_successful": len(sequences) - len(errors),
        "num_errors": len(errors),
        "label_distribution": label_dist,
        "average_confidence": round(
            sum(r["confidence"] for r in parsed_outputs) / len(parsed_outputs), 4
        ) if parsed_outputs else 0.0,
        "timestamp": iso_timestamp(),
    })

    # Print summary.
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Sequences: {len(sequences)}, Errors: {len(errors)}")
    print(f"Label distribution: {label_dist}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
