"""Prompt ablations requested by Reviewer 4 (major comment 7) and Reviewer 1.

Runs six open-weight LLMs (Qwen2.5-72B excluded: it does not fit in 8 GB of
VRAM) on the 69 evaluation sequences under five prompt conditions:

  C0 original     the exact prompt used in the paper (re-run locally)
  C1 thresholds   C0 plus the rule thresholds as explicit decision guidance
  C2 categorical  numeric features replaced by low / medium / high
  C3 masked       C0 without the `dominant_gaze_class` feature
  C4 perturbed    C0 with every numeric feature scaled by U(0.95, 1.05)

All models run in 4-bit NF4 on the local GPU, so C0 is re-run under the same
setup and every comparison is made within this setup. Decoding matches the
paper: chat template with a single user message, greedy, 256 new tokens.

Resumable: each (model, condition) writes one JSONL file and finished
sequences are skipped on restart.

Usage (from the project root, with .venv-llm):
  .venv-llm/Scripts/python.exe revision/prompt_ablation/run_prompt_ablation.py [--models NAME ...] [--conditions C0 ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "code" if (ROOT / "code" / "output").is_dir() else ROOT  # project layout or repository layout
EVAL = CODE / "output" / "3class_eval"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(CODE / "models")))  # local Hugging Face snapshots
OUT = HERE / "outputs"

sys.path.insert(0, str(CODE / "src"))
from project_llm.labels_3class import llm_prompt_3class  # noqa: E402

MODELS = {
    "Yi-1.5-9B-Chat": "01-ai--Yi-1_5-9B-Chat",
    "Qwen2.5-7B-Instruct": "Qwen--Qwen2.5-7B-Instruct",
    "qwen-7b": "qwen-7b",
    "Llama-3.1-8B-Instruct": "meta-llama--Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai--Mistral-7B-Instruct-v0_3",
    "Phi-4-mini-instruct": "microsoft--Phi-4-mini-instruct",
}
CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
MAX_NEW_TOKENS = 256
PERTURB_SEED = 20261005

THRESHOLD_GUIDANCE = (
    "Use these reference thresholds when interpreting the features:\n"
    "  - others: eyes_closed_ratio >= 0.30, or occlusion_ratio >= 0.35.\n"
    "  - focused: visible_ratio >= 0.75 and gaze_shift_ratio <= 0.10 and "
    "mean_gaze_motion <= 120 and attention_stability_score >= 0.65.\n"
    "  - mix: every other case.\n\n"
)
RATIO_FEATURES = [
    "attention_stability_score", "eyes_closed_ratio", "gaze_shift_ratio",
    "occlusion_ratio", "on_screen_ratio", "visible_gaze_fraction", "visible_ratio",
]
STREAK_FEATURES = ["max_occlusion_streak", "max_visible_streak"]
INT_FEATURES = {"gaze_shift_events", "max_occlusion_streak", "max_visible_streak",
                "num_interactions", "num_segments", "observed_frames"}


def load_original() -> tuple[dict[str, str], dict[str, dict]]:
    """Exact prompts and feature payloads used in the paper."""
    pattern = re.compile(r"Features:\s*(\{.+\})", re.DOTALL)
    prompts, payloads = {}, {}
    with (EVAL / "llm_models_Qwen--Qwen2_5-72B-Instruct_prompts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            prompts[rec["child_id"]] = rec["prompt"]
            payloads[rec["child_id"]] = json.loads(pattern.search(rec["prompt"]).group(1))
    return prompts, payloads


def tertile(x: float) -> str:
    return "low" if x < 1 / 3 else ("medium" if x < 2 / 3 else "high")


def categorical(p: dict) -> dict:
    """A-priori bins: ratios and streak fractions in thirds of [0, 1];
    mean_gaze_motion in thirds of the 300-pixel scale used by the stability
    score. Features without a natural scale stay numeric."""
    q = dict(p)
    for k in RATIO_FEATURES:
        q[k] = tertile(p[k])
    for k in STREAK_FEATURES:
        q[k] = tertile(p[k] / p["observed_frames"] if p["observed_frames"] else 0.0)
    m = p["mean_gaze_motion"]
    q["mean_gaze_motion"] = "low" if m < 100 else ("medium" if m < 200 else "high")
    return q


def perturbed(p: dict, rng: random.Random) -> dict:
    q = dict(p)
    for k, v in p.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        new = v * rng.uniform(0.95, 1.05)
        if k in RATIO_FEATURES:
            new = min(max(new, 0.0), 1.0)
        q[k] = int(round(new)) if k in INT_FEATURES else new
    return q


def build_prompts() -> dict[str, dict[str, str]]:
    original, payloads = load_original()
    rebuilt = {cid: llm_prompt_3class(p) for cid, p in payloads.items()}
    assert rebuilt == original, "prompt builder no longer reproduces the paper's prompts"
    rng = random.Random(PERTURB_SEED)
    out = {c: {} for c in CONDITIONS}
    for cid in sorted(payloads):
        p = payloads[cid]
        out["C0"][cid] = original[cid]
        base = llm_prompt_3class(p)
        out["C1"][cid] = base.replace("Return ONLY a valid JSON object",
                                      THRESHOLD_GUIDANCE + "Return ONLY a valid JSON object", 1)
        out["C2"][cid] = llm_prompt_3class(categorical(p))
        out["C3"][cid] = llm_prompt_3class({k: v for k, v in p.items() if k != "dominant_gaze_class"})
        out["C4"][cid] = llm_prompt_3class(perturbed(p, rng))
    assert all(out["C1"][c] != out["C0"][c] for c in out["C0"]), "threshold guidance not inserted"
    (OUT / "prompts_by_condition.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_model(path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    # dtype pinned to bf16 so the non-quantised layers (embeddings, lm_head) do not
    # default to fp32 (Llama-3.1-8B otherwise needs 8.5 GB and spills out of the 8 GB GPU).
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, dtype=torch.bfloat16,
                                                 quantization_config=quant, device_map={"": 0})
    model.eval()
    return tok, model


def generate(tok, model, prompt: str) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"]
    except Exception:
        input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)
    # Stop at any of the model's own end tokens (tokenizer + generation config).
    # With greedy decoding this only removes tokens generated after the answer
    # has ended; the text before the stop, and hence the parsed label, is the
    # same as with the paper's single eos_token_id.
    eos = {tok.eos_token_id}
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    eos.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos = sorted(e for e in eos if e is not None)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=eos)
    return tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None, help="first N sequences only (smoke test)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts()
    ids = sorted(prompts["C0"])[: args.limit] if args.limit else sorted(prompts["C0"])

    import torch
    import transformers
    meta = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "quantization": "bitsandbytes 4-bit NF4, double quant, bf16 compute",
            "decoding": f"greedy, max_new_tokens={MAX_NEW_TOKENS}, min_new_tokens=8, "
                        "stop at tokenizer + generation-config eos tokens", "perturb_seed": PERTURB_SEED}
    (OUT / "run_environment.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)

    for name in args.models:
        todo = {}
        for cond in args.conditions:
            path = OUT / f"{name}__{cond}.jsonl"
            done = set()
            if path.exists():
                done = {json.loads(l)["child_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
            todo[cond] = [cid for cid in ids if cid not in done]
        if not any(todo.values()):
            print(f"[{name}] already complete", flush=True)
            continue
        t0 = time.time()
        tok, model = load_model(MODELS_DIR / MODELS[name])
        print(f"[{name}] loaded in {time.time() - t0:.0f}s; VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
        for cond in args.conditions:
            with (OUT / f"{name}__{cond}.jsonl").open("a", encoding="utf-8") as fh:
                for i, cid in enumerate(todo[cond], 1):
                    t1 = time.time()
                    raw = generate(tok, model, prompts[cond][cid])
                    fh.write(json.dumps({"child_id": cid, "raw": raw}, ensure_ascii=False) + "\n")
                    fh.flush()
                    # Release cached blocks so VRAM does not creep towards the 8 GB limit,
                    # where Windows starts paging GPU memory and generation slows sharply.
                    torch.cuda.empty_cache()
                    print(f"[{name}] {cond} {i}/{len(todo[cond])} {time.time() - t1:.1f}s "
                          f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB", flush=True)
        del model
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
