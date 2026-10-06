"""Score the prompt-ablation outputs written by run_prompt_ablation.py.

For every model x condition: parse failures, label distribution, accuracy,
Cohen's kappa against consensus and each rater, F1 for `others`, and
agreement with the rule's own prediction. For C4 (perturbed features) it also
reports the prediction flip rate against C0 and, for comparison, the rule's
flip rate under the same perturbation, and checks C0 against the paper's
original predictions (same label / identical raw text).

  python evaluate_ablation.py                 local 4-bit run   -> ABLATION_RESULTS.md, ablation_metrics.csv
  python evaluate_ablation.py --setup server  server run (paper's precision; outputs_full_precision/)
                                              -> ABLATION_RESULTS_FULL_PRECISION.md, ablation_metrics_full_precision.csv
numpy and pandas only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "code" if (ROOT / "code" / "output").is_dir() else ROOT  # project layout or repository layout
EVAL = CODE / "output" / "3class_eval"
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))
from run_prompt_ablation import CONDITIONS, MODELS  # noqa: E402

LABELS = ["focused", "mix", "others"]
ORIGINAL = {  # paper's original prediction files in code/output/3class_eval
    "Qwen2.5-72B-Instruct": "Qwen--Qwen2_5-72B-Instruct", "Yi-1.5-9B-Chat": "01-ai--Yi-1_5-9B-Chat",
    "Qwen2.5-7B-Instruct": "Qwen--Qwen2_5-7B-Instruct", "qwen-7b": "qwen-7b",
    "Llama-3.1-8B-Instruct": "meta-llama--Llama-3_1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai--Mistral-7B-Instruct-v0_3",
    "Phi-4-mini-instruct": "microsoft--Phi-4-mini-instruct",
}
SETUPS = {
    "local": {"dir": OUT, "suffix": "", "models": list(MODELS),
              "intro": "Six models, 4-bit NF4 on the local GPU, greedy decoding. Compare conditions **within** this "
                       "table only; the paper's original numbers come from a different (full-precision) setup."},
    "server": {"dir": HERE / "outputs_full_precision", "suffix": "_FULL_PRECISION", "models": list(ORIGINAL),
               "intro": "Seven models on the HULTIG GPU server in the paper's original precision (7-9B models bf16; "
                        "Qwen2.5-72B 4-bit nf4 with fp16 compute, as in the original runs), greedy decoding, "
                        "same prompts as the local run."},
}
COND_NAMES = {"C0": "original (local re-run)", "C1": "+ rule thresholds", "C2": "categorical values",
              "C3": "dominant_gaze_class masked", "C4": "perturbed values (±5%)"}

_src = (CODE / "scripts" / "run_llm_3class.py").read_text(encoding="utf-8")
_ns = {"re": re, "json": json, "ALLOWED": set(LABELS)}
_end = 'raise ValueError("Could not extract valid JSON payload")'
exec(_src[_src.index("def extract_json_payload_3class"):_src.index(_end) + len(_end)], _ns)
parse_payload = _ns["extract_json_payload_3class"]  # the paper's own parser


def parse(raw: str) -> tuple[str, str]:
    """Return (label, status); status is ok / no_json / invalid_label."""
    try:
        label = parse_payload(raw)["label"]
    except ValueError:
        return "mix", "no_json"
    m = re.search(r'"label"\s*:\s*"([^"]*)"', raw)
    return label, ("ok" if m and m.group(1).strip().lower() in LABELS else "invalid_label")


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in LABELS)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def f1(pred: np.ndarray, y: np.ndarray, c: str) -> float:
    tp = np.sum((pred == c) & (y == c))
    fp = np.sum((pred == c) & (y != c))
    fn = np.sum((pred != c) & (y == c))
    return 0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn))


def rule(p: dict) -> str:
    if p["eyes_closed_ratio"] >= 0.3 or p["occlusion_ratio"] >= 0.35:
        return "others"
    if (p["visible_ratio"] >= 0.75 and p["gaze_shift_ratio"] <= 0.1
            and p["mean_gaze_motion"] <= 120 and p["attention_stability_score"] >= 0.65):
        return "focused"
    return "mix"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=list(SETUPS), default="local")
    setup = SETUPS[ap.parse_args().setup]
    run_dir, models = setup["dir"], setup["models"]
    gt = pd.read_csv(EVAL / "ground_truth_3class.csv").set_index("child_id")
    prompts = json.loads((OUT / "prompts_by_condition.json").read_text(encoding="utf-8"))
    feats = lambda txt: json.loads(re.search(r"Features:\s*(\{.+\})", txt, re.S).group(1))  # noqa: E731
    rule_c0 = pd.Series({cid: rule(feats(t)) for cid, t in prompts["C0"].items()})
    rule_c4 = pd.Series({cid: rule(feats(t)) for cid, t in prompts["C4"].items()})

    rows, preds, raws = [], {}, {}
    for name in models:
        for cond in CONDITIONS:
            path = run_dir / f"{name}__{cond}.jsonl"
            if not path.exists():
                continue
            recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            parsed = {r["child_id"]: parse(r["raw"]) for r in recs}
            raws[(name, cond)] = {r["child_id"]: r["raw"] for r in recs}
            ids = [cid for cid in gt.index if cid in parsed]
            p = np.array([parsed[c][0] for c in ids])
            preds[(name, cond)] = pd.Series(p, index=ids)
            y = gt.loc[ids]
            rows.append({
                "model": name, "condition": cond, "n": len(ids),
                "no_json": sum(parsed[c][1] == "no_json" for c in ids),
                "invalid_label": sum(parsed[c][1] == "invalid_label" for c in ids),
                "pred_focused": int(np.sum(p == "focused")), "pred_mix": int(np.sum(p == "mix")),
                "pred_others": int(np.sum(p == "others")),
                "accuracy": float(np.mean(p == y["final_3class"].to_numpy())),
                "kappa_consensus": kappa(p, y["final_3class"].to_numpy()),
                "kappa_rater1": kappa(p, y["rater1_3"].to_numpy()),
                "kappa_rater2": kappa(p, y["rater2_3"].to_numpy()),
                "f1_others": f1(p, y["final_3class"].to_numpy(), "others"),
                "agree_with_rule": float(np.mean(p == rule_c0.loc[ids].to_numpy())),
            })
    df = pd.DataFrame(rows)
    flips = []
    for name in models:
        if (name, "C0") in preds and (name, "C4") in preds:
            a, b = preds[(name, "C0")], preds[(name, "C4")]
            common = a.index.intersection(b.index)
            flips.append({"model": name, "n": len(common), "flip_rate": float(np.mean(a[common] != b[common]))})
    rule_flip = float(np.mean(rule_c0 != rule_c4.loc[rule_c0.index]))
    repro = []
    for name in models:
        if (name, "C0") not in preds:
            continue
        orig = pd.read_csv(EVAL / f"llm_models_{ORIGINAL[name]}_3class.csv").set_index("child_id")
        orig_raw = {json.loads(l)["child_id"]: json.loads(l)["raw"] for l in
                    (EVAL / f"llm_models_{ORIGINAL[name]}_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        a = preds[(name, "C0")]
        common = a.index.intersection(orig.index)
        po = orig.loc[common, "pred_3class"].to_numpy()
        y = gt.loc[common, "final_3class"].to_numpy()
        repro.append({"model": name, "n": len(common), "same_label": int(np.sum(a[common].to_numpy() == po)),
                      "identical_raw": sum(raws[(name, "C0")][c].strip() == orig_raw.get(c, "").strip() for c in common),
                      "kappa_original": kappa(po, y), "kappa_rerun": kappa(a[common].to_numpy(), y)})
    df.to_csv(HERE / f"ablation_metrics{setup['suffix'].lower()}.csv", index=False)

    lines = ["# Prompt ablation results", "", setup["intro"], ""]
    if not df.empty:
        piv = df.pivot(index="model", columns="condition", values="kappa_consensus").reindex(columns=CONDITIONS)
        lines += ["## Cohen's kappa vs consensus", "", "| Model | " + " | ".join(f"{c} {COND_NAMES[c]}" for c in piv.columns) + " |",
                  "|---" * (len(piv.columns) + 1) + "|"]
        for m, r in piv.iterrows():
            lines.append(f"| {m} | " + " | ".join("—" if pd.isna(v) else f"{v:.3f}" for v in r) + " |")
        lines += ["", f"Rule-based classifier for reference: kappa {kappa(rule_c0.loc[gt.index].to_numpy(), gt['final_3class'].to_numpy()):.3f}.", "",
                  "## All metrics", "", "| " + " | ".join(df.columns) + " |", "|---" * len(df.columns) + "|"]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    if flips:
        lines += ["", "## Stability under ±5% perturbation (C4 vs C0)", "",
                  f"Rule-based classifier flip rate under the same perturbation: **{rule_flip:.3f}**.", "",
                  "| Model | n | Flip rate |", "|---|---|---|"]
        lines += [f"| {f['model']} | {f['n']} | {f['flip_rate']:.3f} |" for f in flips]
    if repro:
        lines += ["", "## C0 against the paper's original predictions", "",
                  "Same prompt as the paper; `same_label` and `identical_raw` count sequences out of n.", "",
                  "| Model | n | Same label | Identical raw text | kappa original | kappa re-run |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['n']} | {r['same_label']} | {r['identical_raw']} | {r['kappa_original']:.3f} | "
                  f"{r['kappa_rerun']:.3f} |" for r in repro]
    (HERE / f"ABLATION_RESULTS{setup['suffix']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
