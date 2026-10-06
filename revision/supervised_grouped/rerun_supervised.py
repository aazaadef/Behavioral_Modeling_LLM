"""Supervised baselines: reproduce the paper, then re-run with grouped CV.

Reviewer 1 (pipeline, participant independence) and Reviewer 4 (major
comment 6: "clips from the same video must stay together").

The only attention labels that exist are the 69 consensus labels, so every
supervised result is out-of-fold cross-validation within those 69 sequences.
The paper used StratifiedKFold(3, shuffle, random_state=42) with sequences
from the same video and channel split across folds. Here:

  1. reproduce: the paper's exact CV, to check the code and data still match
     the stored predictions;
  2. grouped: leave-one-channel-out (10 folds; primary, the closest available
     proxy for the same child or family) and leave-one-video-out (21 folds).

Models and hyperparameters are the paper's fixed settings (no search).
Run with .venv-llm (scikit-learn, xgboost, lightgbm).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "code" if (ROOT / "code" / "output").is_dir() else ROOT  # project layout or repository layout
EVAL = CODE / "output" / "3class_eval"
LABELS = ["focused", "mix", "others"]
FEATURE_COLUMNS = [  # code/src/project_llm/supervised_baselines.py
    "observed_frames", "visible_ratio", "on_screen_ratio", "gaze_shift_ratio", "occlusion_ratio",
    "eyes_closed_ratio", "mean_head_area", "mean_head_motion", "mean_gaze_motion", "visible_gaze_fraction",
    "gaze_shift_events", "max_visible_streak", "max_occlusion_streak", "attention_stability_score",
]


def classifiers() -> dict[str, tuple[object, bool]]:
    """The paper's classifier zoo (code/scripts/run_3class_pipeline.py, step 4)."""
    return {
        "LogisticRegression": (Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))]), False),
        "LinearSVM": (Pipeline([("scaler", StandardScaler()),
                                ("clf", LinearSVC(max_iter=5000, dual="auto", random_state=42))]), False),
        "RandomForest": (Pipeline([("scaler", StandardScaler()),
                                   ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))]), False),
        "Dummy_majority": (DummyClassifier(strategy="most_frequent", random_state=42), False),
        "Dummy_stratified": (DummyClassifier(strategy="stratified", random_state=42), False),
        "XGBoost": (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="mlogloss",
                                  random_state=42, verbosity=0), True),
        "LightGBM": (LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbosity=-1), False),
    }


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pattern = re.compile(r"Features:\s*(\{.+\})", re.DOTALL)
    feats = {}
    with (EVAL / "llm_models_Qwen--Qwen2_5-72B-Instruct_prompts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            feats[rec["child_id"]] = json.loads(pattern.search(rec["prompt"]).group(1))
    gt = pd.read_csv(EVAL / "ground_truth_3class.csv")  # same order as the paper's pipeline
    man = pd.read_csv(ROOT / "revision" / "label_audit" / "sample_manifest.csv").set_index("sequence_id")
    gt = gt.join(man[["video_id", "channel_id"]], on="child_id")
    X = np.array([[float(feats[c][k]) for k in FEATURE_COLUMNS] for c in gt["child_id"]])
    return gt, X, gt["final_3class"].to_numpy()


def oof_predict(model, needs_le, X, y, splits) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        clf = clone(model)
        if needs_le:
            le = LabelEncoder()
            clf.fit(X[tr], le.fit_transform(y[tr]))
            pred[te] = le.inverse_transform(clf.predict(X[te]))
        else:
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
    return pred


def scores(y, p) -> dict:
    f = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
    return {"accuracy": accuracy_score(y, p), "kappa": cohen_kappa_score(y, p),
            "macro_f1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
            "f1_focused": f[0], "f1_mix": f[1], "f1_others": f[2],
            "pred_counts": "/".join(str(int(np.sum(p == c))) for c in LABELS)}


def main() -> None:
    gt, X, y = load()
    n_folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    schemes = {
        "paper (StratifiedKFold 3, ungrouped)": list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y)),
        "leave-one-channel-out (10 folds)": list(LeaveOneGroupOut().split(X, y, gt["channel_id"])),
        "leave-one-video-out (21 folds)": list(LeaveOneGroupOut().split(X, y, gt["video_id"])),
    }
    rows, repro = [], []
    preds_out = gt[["child_id", "video_id", "channel_id", "final_3class"]].copy()
    for name, (model, needs_le) in classifiers().items():
        for scheme, splits in schemes.items():
            p = oof_predict(model, needs_le, X, y, splits)
            rows.append({"model": name, "scheme": scheme, **scores(y, p)})
            preds_out[f"{name}__{scheme.split(' (')[0]}"] = p
            if scheme.startswith("paper"):
                stored = pd.read_csv(EVAL / f"supervised_{name}_3class.csv").set_index("child_id")["pred_3class"]
                repro.append((name, int((stored.reindex(gt["child_id"]).to_numpy() == p).sum())))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "supervised_grouped_metrics.csv", index=False)
    preds_out.to_csv(HERE / "supervised_grouped_predictions.csv", index=False)

    L = ["# Supervised baselines: paper CV vs grouped CV", "",
         "Generated by `rerun_supervised.py`. All results are out-of-fold predictions within the 69 labelled "
         "sequences (no other attention labels exist). Fixed hyperparameters from the paper; no search. "
         "Within each fold, scaling and model fitting use only the training sequences.", "",
         f"Training folds per scheme: paper = StratifiedKFold({n_folds}); channel = 10 folds; video = 21 folds. "
         f"Folds missing a class in training cannot predict that class.", "",
         "## Reproduction of the paper's predictions (same CV)", "",
         "| Model | Identical predictions (of 69) |", "|---|---|"]
    L += [f"| {n} | {k} |" for n, k in repro]
    L += ["", "## Agreement with the consensus", "",
          "| Model | CV scheme | Accuracy | κ | Macro-F1 | F1 others | Predicted f/m/o |", "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r['model']} | {r['scheme']} | {r['accuracy']:.3f} | {r['kappa']:.3f} | {r['macro_f1']:.3f} | "
                 f"{r['f1_others']:.3f} | {r['pred_counts']} |")
    piv = R.pivot(index="model", columns="scheme", values="kappa")
    paper_col = [c for c in piv.columns if c.startswith("paper")][0]
    ch_col = [c for c in piv.columns if "channel" in c][0]
    L += ["", "## Change in κ when sequences from the same channel are kept together", "",
          "| Model | κ paper CV | κ leave-one-channel-out | Change |", "|---|---|---|---|"]
    for m, r in piv.iterrows():
        L.append(f"| {m} | {r[paper_col]:.3f} | {r[ch_col]:.3f} | {r[ch_col] - r[paper_col]:+.3f} |")
    (HERE / "SUPERVISED_GROUPED.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
