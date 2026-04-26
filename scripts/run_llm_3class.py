"""Re-run LLM inference with the 3-class prompt on the 69 test sequences.

Supports running one or many local checkpoints. For each model a fresh
prompt is built with 3 labels only; raw output is logged, the JSON is
parsed, and the predicted 3-class label is written out alongside the
two-rater consensus 3-class ground truth.

Usage:
    python scripts/run_llm_3class.py --model ./models/Qwen--Qwen2.5-7B-Instruct
    python scripts/run_llm_3class.py --model Qwen/Qwen2.5-72B-Instruct --load-in-4bit
    python scripts/run_llm_3class.py --all  # run all downloaded models
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.dataset import load_child_sequences
from project_llm.features import extract_behavioral_features
from project_llm.interactions import infer_interactions
from project_llm.labels_3class import (
    LABELS_3CLASS,
    llm_prompt_3class,
)
from project_llm.temporal import build_temporal_segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output" / "3class_eval"
GT_CSV = OUTPUT_DIR / "ground_truth_3class.csv"
ALLOWED = set(LABELS_3CLASS)


def extract_json_payload_3class(text: str) -> dict[str, object]:
    """Extract and validate a 3-class JSON payload from model output."""
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in fenced if item.strip())
    inline = re.findall(r"(\{[\s\S]*\})", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in inline if item.strip())
    if "{" in text:
        candidates.append(text[text.find("{"):].strip())

    for cand in candidates:
        repaired = cand
        if repaired and not repaired.endswith("}"):
            repaired = repaired.rstrip()
            while repaired.count("{") > repaired.count("}"):
                repaired += "}"
        try:
            payload = json.loads(repaired)
        except Exception:
            continue
        label = str(payload.get("label", "mix")).strip().lower()
        confidence = float(payload.get("confidence", 0.0))
        reasoning = str(payload.get("reasoning", "")).strip()
        if label not in ALLOWED:
            matched = [l for l in ALLOWED if label in l or l in label]
            label = matched[0] if matched else "mix"
        confidence = max(0.0, min(1.0, confidence))
        return {"label": label, "confidence": confidence, "reasoning": reasoning}
    raise ValueError("Could not extract valid JSON payload")


def run_one_model(
    model_path: str | Path,
    gt: pd.DataFrame,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    max_new_tokens: int = 256,
) -> pd.DataFrame:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_path = Path(model_path) if Path(model_path).exists() else model_path
    short = str(model_path).replace("/", "_").replace(".", "_")
    logger.info("Model: %s", model_path)

    dataset_root = ROOT / "data set" / "ChildPlay-gaze" / "ChildPlay-gaze"
    sequences = {s.child_id: s for s in load_child_sequences(dataset_root, split="test")}

    load_kwargs: dict[str, object] = {"device_map": "auto", "torch_dtype": "auto"}
    if isinstance(model_path, Path):
        load_kwargs["local_files_only"] = True
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=isinstance(model_path, Path)
    )
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)

    rows, raw_rows, prompt_rows, errors = [], [], [], []
    for i, (_, r) in enumerate(gt.iterrows(), 1):
        cid = r["child_id"]
        seq = sequences[cid]
        features = extract_behavioral_features(seq)
        segments = build_temporal_segments(seq)
        interactions = infer_interactions(segments)
        payload = {
            **features.to_dict(),
            "num_segments": len(segments),
            "num_interactions": len(interactions),
        }
        for key in ("child_id", "clip_id", "split"):
            payload.pop(key, None)
        prompt = llm_prompt_3class(payload)
        prompt_rows.append({"child_id": cid, "prompt": prompt})

        logger.info("  [%d/%d] %s", i, len(gt), cid)
        try:
            messages = [{"role": "user", "content": prompt}]
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
            except Exception:
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            raw_rows.append({"child_id": cid, "raw": text})
            parsed = extract_json_payload_3class(text)
            pred = parsed["label"]
            conf = parsed["confidence"]
            reasoning = parsed["reasoning"]
            logger.info("    → %s (%.2f)", pred, conf)
        except Exception as exc:
            logger.error("    ERROR: %s", exc)
            errors.append({"child_id": cid, "error": str(exc)})
            pred, conf, reasoning = "mix", 0.0, f"error:{type(exc).__name__}"

        rows.append({
            "child_id": cid,
            "true_3class": r["final_3class"],
            "pred_3class": pred,
            "confidence": round(conf, 4),
            "reasoning": reasoning,
            "correct": pred == r["final_3class"],
        })

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / f"llm_{short}_3class.csv"
    df.to_csv(out_csv, index=False)
    # Save raw outputs and prompts for reproducibility.
    with (OUTPUT_DIR / f"llm_{short}_raw.jsonl").open("w", encoding="utf-8") as fh:
        for rr in raw_rows:
            fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / f"llm_{short}_prompts.jsonl").open("w", encoding="utf-8") as fh:
        for pr in prompt_rows:
            fh.write(json.dumps(pr, ensure_ascii=False) + "\n")
    if errors:
        with (OUTPUT_DIR / f"llm_{short}_errors.jsonl").open("w", encoding="utf-8") as fh:
            for e in errors:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    acc = df["correct"].mean()
    dist = dict(df["pred_3class"].value_counts())
    logger.info("  %s: acc=%.4f  dist=%s", short, acc, dist)

    # Free GPU.
    del model, tokenizer
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Local path or HF model ID")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all models in ./models/")
    args = parser.parse_args()

    if not GT_CSV.exists():
        logger.error("Ground truth not found at %s — run run_3class_pipeline.py first", GT_CSV)
        return 1
    gt = pd.read_csv(GT_CSV)

    if args.all:
        models_root = ROOT / "models"
        candidates = [
            (models_root / "qwen-7b", False),
            (models_root / "Qwen--Qwen2.5-7B-Instruct", False),
            (models_root / "Qwen--Qwen2.5-72B-Instruct", True),  # 72B uses 4-bit
        ]
        for path, use_4bit in candidates:
            if path.exists():
                run_one_model(path, gt, load_in_4bit=use_4bit)
            else:
                logger.warning("Skipping missing model at %s", path)
    elif args.model:
        run_one_model(
            args.model, gt,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
    else:
        parser.error("Either --model or --all is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
