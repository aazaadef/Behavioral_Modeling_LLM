"""NLI formulation sensitivity (Reviewer 1: "hypotheses, verbalisation of
structured numerical features, label descriptions, and entailment mapping").

Re-runs the paper's three zero-shot NLI models under a small grid of
formulations, all fixed in advance (none chosen on the results):

  premise   P1 paper's key=value summary | P2 natural-language summary
  labels    L1 bare names (paper)        | L2 short phrases | L3 full definitions
  template  T1 "This clip is best described as {}." (paper)
            T2 "The child's attention in this clip is {}."
            T3 "{}"  (used only with L3, whose definitions are full sentences)

Entailment mapping (unchanged from the paper): Hugging Face
`zero-shot-classification`, multi_label=False, i.e. a softmax over the
entailment logits of the three hypotheses; the top hypothesis maps back to
its label. Exploratory: every formulation is scored on the same 69 labels.
Run with .venv-llm after the prompt ablation has finished (uses the GPU).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "code" if (ROOT / "code" / "output").is_dir() else ROOT  # project layout or repository layout
EVAL = CODE / "output" / "3class_eval"
LABELS = ["focused", "mix", "others"]
MODELS = {
    "NLI DeBERTa": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "NLI BART": "facebook/bart-large-mnli",
    "NLI DistilBERT": "typeform/distilbert-base-uncased-mnli",
}
LABEL_SETS = {
    "L1 bare names": {"focused": "focused", "mix": "mix", "others": "others"},
    "L2 short phrases": {"focused": "focused attention", "mix": "mixed or exploratory attention",
                         "others": "unreliable gaze evidence"},
    "L3 full definitions": {  # code/src/project_llm/labels_3class.py LABEL_DESCRIPTIONS_3CLASS
        "focused": "The child shows stable, sustained visual attention on a target with minimal gaze shifts.",
        "mix": "The child actively switches gaze or scans across multiple targets, including mixed "
               "stable-plus-shifting and purely exploratory patterns.",
        "others": "Gaze is not reliably usable because of occlusion, extended eye closure, or the child being off-frame.",
    },
}
TEMPLATES = {"T1 paper": "This clip is best described as {}.",
             "T2 attention": "The child's attention in this clip is {}.",
             "T3 definition": "{}"}
COMBOS = [("L1 bare names", "T1 paper"), ("L1 bare names", "T2 attention"), ("L2 short phrases", "T1 paper"),
          ("L2 short phrases", "T2 attention"), ("L3 full definitions", "T3 definition")]


def premise_p1(f: dict) -> str:  # code/scripts/run_3class_pipeline.py, step 3
    return ("Child gaze behavior summary. "
            f"visible_ratio={f['visible_ratio']:.3f}, gaze_shift_ratio={f['gaze_shift_ratio']:.3f}, "
            f"occlusion_ratio={f['occlusion_ratio']:.3f}, eyes_closed_ratio={f['eyes_closed_ratio']:.3f}, "
            f"mean_gaze_motion={f['mean_gaze_motion']:.2f}, attention_stability_score={f['attention_stability_score']:.3f}, "
            f"max_visible_streak={f['max_visible_streak']}.")


def premise_p2(f: dict) -> str:
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    return (f"In this clip, the child's gaze is visible in {pct(f['visible_ratio'])} of the frames, shifts between "
            f"targets in {pct(f['gaze_shift_ratio'])} of the frames, is occluded or uncertain in "
            f"{pct(f['occlusion_ratio'])} of the frames, and the eyes are closed in {pct(f['eyes_closed_ratio'])} "
            f"of the frames. The longest uninterrupted visible stretch lasts {f['max_visible_streak']} of "
            f"{f['observed_frames']} frames. The gaze point moves on average {f['mean_gaze_motion']:.0f} pixels "
            f"between frames, and the attention stability score is {f['attention_stability_score']:.2f} out of 1.")


def kappa(a, b):
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in LABELS)
    return float("nan") if pe >= 1 else float((po - pe) / (1 - pe))


def main() -> None:
    import torch
    from transformers import pipeline

    pattern = re.compile(r"Features:\s*(\{.+\})", re.DOTALL)
    feats = {}
    with (EVAL / "llm_models_Qwen--Qwen2_5-72B-Instruct_prompts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            feats[rec["child_id"]] = json.loads(pattern.search(rec["prompt"]).group(1))
    gt = pd.read_csv(EVAL / "ground_truth_3class.csv").set_index("child_id")
    ids = list(gt.index)
    y = gt["final_3class"].to_numpy()
    device = 0 if torch.cuda.is_available() else -1

    rows, preds = [], {}
    for mname, hf_id in MODELS.items():
        clf = pipeline("zero-shot-classification", model=hf_id, device=device)
        for pname, pfun in (("P1 key=value (paper)", premise_p1), ("P2 natural language", premise_p2)):
            for lset, tname in COMBOS:
                verb = LABEL_SETS[lset]
                inv = {v: k for k, v in verb.items()}
                out = []
                for cid in ids:
                    r = clf(pfun(feats[cid]), candidate_labels=list(verb.values()),
                            hypothesis_template=TEMPLATES[tname], multi_label=False)
                    out.append(inv[r["labels"][0]])
                p = np.array(out)
                key = f"{mname} | {pname} | {lset} | {tname}"
                preds[key] = p
                rows.append({"model": mname, "premise": pname, "labels": lset, "template": tname,
                             "is_paper_formulation": pname.startswith("P1") and lset.startswith("L1") and tname.startswith("T1"),
                             "accuracy": float(np.mean(p == y)), "kappa": kappa(p, y),
                             "pred_counts": "/".join(str(int(np.sum(p == c))) for c in LABELS)})
                print(f"{key}: acc={rows[-1]['accuracy']:.3f} kappa={rows[-1]['kappa']:.3f} pred={rows[-1]['pred_counts']}", flush=True)
        del clf
        torch.cuda.empty_cache()

    R = pd.DataFrame(rows)
    R.to_csv(HERE / "nli_sensitivity_metrics.csv", index=False)
    pd.DataFrame(preds, index=ids).rename_axis("child_id").to_csv(HERE / "nli_sensitivity_predictions.csv")
    L = ["# NLI formulation sensitivity", "",
         "Generated by `run_nli_sensitivity.py`. Exploratory: all formulations fixed in advance and scored on the "
         "same 69 consensus labels. Entailment mapping: softmax over the three hypotheses' entailment logits "
         "(Hugging Face zero-shot pipeline, multi_label=False). Majority baseline accuracy 0.696, κ 0.", "",
         "| Model | Premise | Labels | Template | Accuracy | κ | Predicted f/m/o |", "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        mark = " **(paper)**" if r["is_paper_formulation"] else ""
        L.append(f"| {r['model']}{mark} | {r['premise']} | {r['labels']} | {r['template']} | {r['accuracy']:.3f} | "
                 f"{r['kappa']:.3f} | {r['pred_counts']} |")
    best = R.loc[R.groupby("model")["kappa"].idxmax()]
    L += ["", "Best κ per model across the 10 formulations (exploratory; selected on the evaluation labels):", ""]
    L += [f"- {r['model']}: κ {r['kappa']:.3f} ({r['premise']}, {r['labels']}, {r['template']})" for _, r in best.iterrows()]
    (HERE / "NLI_SENSITIVITY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
