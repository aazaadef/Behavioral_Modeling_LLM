"""Prompt ablation on the HULTIG GPU server, in the paper's original precision.

Same prompts as the local 4-bit run (prompts_by_condition.json, built and
checked by revision/prompt_ablation/run_prompt_ablation.py). Precision follows
the paper's original runs (code/scripts/run_llm_3class.py):
  * 7-9B models: torch_dtype="auto" (bf16), device_map="auto", no quantisation
  * Qwen2.5-72B: 4-bit nf4, fp16 compute, device_map="auto"
Decoding as the paper: chat template with one user message, greedy,
max_new_tokens=256, min_new_tokens=8; generation stops at the model's own
end tokens (this only removes tokens after the answer has ended).

Resumable: one JSONL per (model, condition); finished sequences are skipped.
Usage (on the server):
  CUDA_VISIBLE_DEVICES=2 python run_ablation_server.py --models Yi-1.5-9B-Chat qwen-7b
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path.home() / "project" / "project.LLM" / "models"
OUT = HERE / "outputs_full_precision"
MODELS = {
    "Qwen2.5-72B-Instruct": ("Qwen--Qwen2.5-72B-Instruct", True),
    "Yi-1.5-9B-Chat": ("01-ai--Yi-1_5-9B-Chat", False),
    "Qwen2.5-7B-Instruct": ("Qwen--Qwen2.5-7B-Instruct", False),
    "qwen-7b": ("qwen-7b", False),
    "Llama-3.1-8B-Instruct": ("meta-llama--Llama-3.1-8B-Instruct", False),
    "Mistral-7B-Instruct-v0.3": ("mistralai--Mistral-7B-Instruct-v0_3", False),
    "Phi-4-mini-instruct": ("microsoft--Phi-4-mini-instruct", False),
}
CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
MAX_NEW_TOKENS = 256


def load(path: Path, four_bit: bool):
    kw = {"device_map": "auto", "torch_dtype": "auto", "local_files_only": True}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), **kw)
    model.eval()
    return tok, model


def generate(tok, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(input_ids):
            input_ids = input_ids["input_ids"]
    except Exception:
        input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)
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
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", default="prompts_by_condition.json")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((HERE / args.prompts).read_text(encoding="utf-8"))
    conditions = args.conditions
    ids = sorted(prompts[conditions[0]])
    meta = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "precision": "7-9B models: bf16 (torch_dtype=auto); Qwen2.5-72B: 4-bit nf4, fp16 compute",
            "decoding": f"greedy, max_new_tokens={MAX_NEW_TOKENS}, min_new_tokens=8, stop at model eos tokens"}
    (OUT / f"run_environment_{'_'.join(args.models)}_{conditions[0]}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)

    for name in args.models:
        rel, four_bit = MODELS[name]
        todo = {}
        for cond in conditions:
            path = OUT / f"{name}__{cond}.jsonl"
            done = set()
            if path.exists():
                done = {json.loads(l)["child_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
            todo[cond] = [c for c in ids if c not in done]
        if not any(todo.values()):
            print(f"[{name}] already complete", flush=True)
            continue
        t0 = time.time()
        tok, model = load(MODELS_DIR / rel, four_bit)
        print(f"[{name}] loaded in {time.time() - t0:.0f}s", flush=True)
        for cond in conditions:
            with (OUT / f"{name}__{cond}.jsonl").open("a", encoding="utf-8") as fh:
                for i, cid in enumerate(todo[cond], 1):
                    t1 = time.time()
                    raw = generate(tok, model, prompts[cond][cid])
                    fh.write(json.dumps({"child_id": cid, "raw": raw}, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"[{name}] {cond} {i}/{len(todo[cond])} {time.time() - t1:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
