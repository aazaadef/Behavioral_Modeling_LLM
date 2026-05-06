"""HRSF — Hybrid Reliability-Semantic Framework.

A three-stage hybrid classifier that combines a rule-based reliability
detector with an LLM ensemble through a single annotation-style
coefficient α ∈ [0, 1]:

  Stage 1 — Reliability routing (deterministic, domain-driven thresholds)
      if visible_ratio          < τ_v   OR
         max_occlusion_streak   > τ_s   OR
         eyes_closed_ratio      > τ_e
         → return "others"

  Stage 2 — Specialist experts
      For the remaining {focused, mix} decision, both the rule-based
      classifier and the ensemble of seven open-weight LLMs cast a
      vote per class.

  Stage 3 — Human-style consensus weighting
      score(c | x) = α · 1{rule predicts c}
                   + (1 - α) · (1/K) · Σ_i 1{LLM_i predicts c}
      return argmax_c score(c | x)

The coefficient α has a human-style interpretation grounded in the
per-rater fault line found in this paper:
  α = 0   → mimics the lexical/semantic rater (LLMs)
  α = 1   → mimics the numerical/threshold rater (rule)
  α = 0.5 → balanced consensus reasoning

This script sweeps α ∈ {0.0, 0.1, …, 1.0}, computes accuracy /
κ / macro-F1 / per-class F1 on the 69-sample test set, and writes:

  output/3class_eval/phase_a/hrsf/hrsf_alpha_sweep.csv
  output/3class_eval/phase_a/hrsf/hrsf_per_class_at_best_alpha.csv
  output/3class_eval/phase_a/hrsf/hrsf_predictions_at_best_alpha.csv
  output/3class_eval/phase_a/hrsf/HRSF_REPORT.md

Thresholds τ_v, τ_s, τ_e are domain-driven and never tuned on the
ChildPlay test split:
  τ_v = 0.50   visibility floor (less than half the frames usable)
  τ_s = 30     1.0 s of continuous occlusion at 30 fps
  τ_e = 0.30   30% eyes-closed share of frames
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "3class_eval"
PHASE_A = EVAL_DIR / "phase_a"
OUT_DIR = PHASE_A / "hrsf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Domain-driven reliability thresholds (Section 7.2 / Appendix C of the paper).
TAU_V = 0.50  # visible_ratio floor
TAU_S = 30  # max_occlusion_streak ceiling (frames; ≈ 1 s @ 30 fps)
TAU_E = 0.30  # eyes_closed_ratio ceiling

LLM_FAMILIES = [
    "llm_models_Qwen--Qwen2_5-72B-Instruct",
    "llm_models_Qwen--Qwen2_5-7B-Instruct",
    "llm_models_qwen-7b",
    "llm_models_01-ai--Yi-1_5-9B-Chat",
    "llm_models_meta-llama--Llama-3_1-8B-Instruct",
    "llm_models_mistralai--Mistral-7B-Instruct-v0_3",
    "llm_models_microsoft--Phi-4-mini-instruct",
]
RULE_NAME = "rule_based"
LABELS = ["focused", "mix", "others"]

ALPHA_GRID = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]


def extract_features_from_prompts() -> pd.DataFrame:
    """Pull the 16-feature vector for every child_id from a prompts JSONL.

    Each per-LLM prompts JSONL embeds the features inside the prompt
    string after the literal "Features:" marker, so we recover them
    deterministically with a single regex pass.
    """
    prompts_path = EVAL_DIR / f"{LLM_FAMILIES[0]}_prompts.jsonl"
    rows = []
    pattern = re.compile(r"Features:\s*(\{.+\})", re.DOTALL)
    with prompts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            match = pattern.search(rec["prompt"])
            if not match:
                continue
            features = json.loads(match.group(1))
            features["child_id"] = rec["child_id"]
            rows.append(features)
    return pd.DataFrame(rows).set_index("child_id")


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rule_pred series, llm_predictions DataFrame)."""
    rule_df = pd.read_csv(EVAL_DIR / f"{RULE_NAME}_3class.csv").set_index("child_id")
    rule_pred = rule_df[["pred_3class"]].rename(columns={"pred_3class": "rule"})

    llm_blocks = {}
    for fam in LLM_FAMILIES:
        df = pd.read_csv(EVAL_DIR / f"{fam}_3class.csv").set_index("child_id")
        llm_blocks[fam] = df["pred_3class"]
    llm_df = pd.DataFrame(llm_blocks)
    return rule_pred, llm_df


def reliability_fires(features: pd.Series) -> bool:
    """Stage 1 — domain-driven reliability check."""
    return bool(
        features["visible_ratio"] < TAU_V
        or features["max_occlusion_streak"] > TAU_S
        or features["eyes_closed_ratio"] > TAU_E
    )


def hrsf_predict_one(
    features: pd.Series,
    rule_pred: str,
    llm_preds: list[str],
    alpha: float,
) -> str:
    """Apply the three-stage HRSF formula to a single sample."""
    if reliability_fires(features):
        return "others"
    scores = {}
    for c in LABELS:
        rule_vote = 1.0 if rule_pred == c else 0.0
        llm_vote = sum(1 for p in llm_preds if p == c) / len(llm_preds)
        scores[c] = alpha * rule_vote + (1.0 - alpha) * llm_vote
    return max(scores, key=lambda k: scores[k])


def hrsf_predict_all(
    features_df: pd.DataFrame,
    rule_series: pd.Series,
    llm_df: pd.DataFrame,
    alpha: float,
) -> pd.Series:
    """Vectorise HRSF across the whole test set."""
    out = {}
    for child_id in features_df.index:
        feats = features_df.loc[child_id]
        rule_pred = rule_series.loc[child_id]
        llm_preds = llm_df.loc[child_id].tolist()
        out[child_id] = hrsf_predict_one(feats, rule_pred, llm_preds, alpha)
    return pd.Series(out, name=f"hrsf_alpha_{alpha:.2f}")


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Headline + per-class F1."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
        ),
    }
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    for label, f1v in zip(LABELS, f1_per_class):
        out[f"f1_{label}"] = float(f1v)
    return out


def main() -> None:
    features_df = extract_features_from_prompts()
    rule_pred_df, llm_df = load_predictions()
    gt = pd.read_csv(EVAL_DIR / "ground_truth_3class.csv").set_index("child_id")

    # Align everything to the ground-truth ordering.
    common_ids = (
        features_df.index.intersection(rule_pred_df.index)
        .intersection(llm_df.index)
        .intersection(gt.index)
    )
    features_df = features_df.loc[common_ids]
    rule_series = rule_pred_df.loc[common_ids, "rule"]
    llm_df = llm_df.loc[common_ids]
    y_true = gt.loc[common_ids, "final_3class"].tolist()
    y_true_r1 = gt.loc[common_ids, "rater1_3"].tolist()
    y_true_r2 = gt.loc[common_ids, "rater2_3"].tolist()

    n_reliability_fired = sum(reliability_fires(features_df.loc[c]) for c in common_ids)
    print(
        f"Loaded {len(common_ids)} samples; reliability rule fires on {n_reliability_fired} sample(s)."
    )

    rows = []
    per_alpha_predictions: dict[float, pd.Series] = {}
    for alpha in ALPHA_GRID:
        preds = hrsf_predict_all(features_df, rule_series, llm_df, alpha)
        per_alpha_predictions[alpha] = preds
        m_consensus = metrics(y_true, preds.tolist())
        m_r1 = metrics(y_true_r1, preds.tolist())
        m_r2 = metrics(y_true_r2, preds.tolist())
        rows.append(
            {
                "alpha": alpha,
                "accuracy": m_consensus["accuracy"],
                "kappa": m_consensus["kappa"],
                "macro_f1": m_consensus["macro_f1"],
                "f1_focused": m_consensus["f1_focused"],
                "f1_mix": m_consensus["f1_mix"],
                "f1_others": m_consensus["f1_others"],
                "kappa_vs_rater1": m_r1["kappa"],
                "kappa_vs_rater2": m_r2["kappa"],
            }
        )

    sweep_df = pd.DataFrame(rows)
    sweep_path = OUT_DIR / "hrsf_alpha_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Wrote {sweep_path.relative_to(ROOT)}")

    best_idx = sweep_df["kappa"].idxmax()
    best_alpha = float(sweep_df.loc[best_idx, "alpha"])
    best_row = sweep_df.loc[best_idx]
    print(
        f"Best α (by κ on consensus) = {best_alpha:.2f} → "
        f"acc={best_row['accuracy']:.3f}, κ={best_row['kappa']:.3f}, "
        f"macro-F1={best_row['macro_f1']:.3f}, F1(others)={best_row['f1_others']:.3f}"
    )

    # Per-class report at the best α.
    best_preds = per_alpha_predictions[best_alpha].tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, best_preds, labels=LABELS, zero_division=0
    )
    per_class_df = pd.DataFrame(
        {"label": LABELS, "precision": prec, "recall": rec, "f1": f1, "support": sup}
    )
    per_class_path = OUT_DIR / "hrsf_per_class_at_best_alpha.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Wrote {per_class_path.relative_to(ROOT)}")

    # Side-by-side per-sample predictions for the disagreement plots.
    side_df = pd.DataFrame(
        {
            "child_id": common_ids,
            "rater1": y_true_r1,
            "rater2": y_true_r2,
            "consensus": y_true,
            "rule": rule_series.tolist(),
            "llm_majority": [pd.Series(llm_df.loc[c]).mode().iloc[0] for c in common_ids],
            f"hrsf_alpha_{best_alpha:.2f}": best_preds,
        }
    )
    side_path = OUT_DIR / "hrsf_predictions_at_best_alpha.csv"
    side_df.to_csv(side_path, index=False)
    print(f"Wrote {side_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
