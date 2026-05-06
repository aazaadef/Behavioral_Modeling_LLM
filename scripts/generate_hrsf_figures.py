"""HRSF-specific figures.

Reads:
  output/3class_eval/phase_a/hrsf/hrsf_alpha_sweep.csv
  output/3class_eval/phase_a/hrsf/hrsf_predictions_at_best_alpha.csv

Produces three figures, all 300 DPI:

  fig6_hrsf_alpha_sensitivity.png
      Two-panel sensitivity plot — left: accuracy/κ/macro-F1 vs α;
      right: per-class F1 vs α with the inter-rater κ ceiling shown
      as a dashed reference.

  fig7_hrsf_per_class.png
      Grouped bar chart comparing rule_based, LLM ensemble (majority
      vote), best single LLM (Qwen2.5-72B), and HRSF at the best α
      across the three classes (focused / mix / others).

  fig8_hrsf_disagreement_heatmap.png
      Per-sample agreement heatmap on the 12 inter-rater
      disagreement samples — columns: rater 1, rater 2, rule_based,
      LLM majority, HRSF; rows: child IDs; cells coloured by class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "3class_eval"
PHASE_A = EVAL_DIR / "phase_a"
HRSF_DIR = PHASE_A / "hrsf"
FIG_DIR = PHASE_A / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INTER_RATER_KAPPA = 0.6129
LABELS = ["focused", "mix", "others"]
LABEL_COLOURS = {"focused": "#2E86AB", "mix": "#F4A261", "others": "#E63946"}


def fig6_alpha_sensitivity() -> None:
    sweep = pd.read_csv(HRSF_DIR / "hrsf_alpha_sweep.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel — headline metrics vs α.
    ax = axes[0]
    ax.plot(
        sweep["alpha"],
        sweep["accuracy"],
        marker="o",
        linewidth=2,
        label="Accuracy",
        color="#2E86AB",
    )
    ax.plot(
        sweep["alpha"], sweep["kappa"], marker="s", linewidth=2, label="Cohen's κ", color="#E63946"
    )
    ax.plot(
        sweep["alpha"],
        sweep["macro_f1"],
        marker="^",
        linewidth=2,
        label="Macro-F1",
        color="#06A77D",
    )
    ax.axhline(
        INTER_RATER_KAPPA,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Inter-rater κ = {INTER_RATER_KAPPA:.3f}",
    )
    ax.set_xlabel("α  (annotation-style coefficient)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("HRSF headline metrics vs α", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.4, 0.95)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # α-axis annotations.
    ax.annotate(
        "← rater 1 style\n(pure LLM)",
        xy=(0.02, 0.43),
        fontsize=9,
        color="#666666",
    )
    ax.annotate(
        "rater 2 style →\n(pure rule + gate)",
        xy=(0.78, 0.43),
        fontsize=9,
        color="#666666",
        ha="left",
    )

    # Right panel — per-class F1 vs α.
    ax = axes[1]
    ax.plot(
        sweep["alpha"],
        sweep["f1_focused"],
        marker="o",
        linewidth=2,
        label="F1(focused)",
        color=LABEL_COLOURS["focused"],
    )
    ax.plot(
        sweep["alpha"],
        sweep["f1_mix"],
        marker="s",
        linewidth=2,
        label="F1(mix)",
        color=LABEL_COLOURS["mix"],
    )
    ax.plot(
        sweep["alpha"],
        sweep["f1_others"],
        marker="^",
        linewidth=2,
        label="F1(others)",
        color=LABEL_COLOURS["others"],
    )
    # Reference line: F1(others) = 0 for all 18 baseline systems.
    ax.axhline(
        0.0,
        color="grey",
        linestyle=":",
        linewidth=1,
        label="Baseline F1(others) = 0 (all 18 systems)",
    )
    ax.set_xlabel("α  (annotation-style coefficient)", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("HRSF per-class F1 vs α", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Hybrid Reliability-Semantic Framework (HRSF) — sensitivity to α",
        fontsize=13,
        y=1.00,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig6_hrsf_alpha_sensitivity.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def _load_baseline_predictions() -> dict[str, list[str]]:
    """Get rule_based, LLM majority vote, best single LLM predictions on all 69 samples."""
    rule = pd.read_csv(EVAL_DIR / "rule_based_3class.csv").set_index("child_id")["pred_3class"]
    qwen72 = pd.read_csv(EVAL_DIR / "llm_models_Qwen--Qwen2_5-72B-Instruct_3class.csv").set_index(
        "child_id"
    )["pred_3class"]

    llm_families = [
        "llm_models_Qwen--Qwen2_5-72B-Instruct",
        "llm_models_Qwen--Qwen2_5-7B-Instruct",
        "llm_models_qwen-7b",
        "llm_models_01-ai--Yi-1_5-9B-Chat",
        "llm_models_meta-llama--Llama-3_1-8B-Instruct",
        "llm_models_mistralai--Mistral-7B-Instruct-v0_3",
        "llm_models_microsoft--Phi-4-mini-instruct",
    ]
    llm_block = pd.DataFrame(
        {
            f: pd.read_csv(EVAL_DIR / f"{f}_3class.csv").set_index("child_id")["pred_3class"]
            for f in llm_families
        }
    )
    llm_majority = llm_block.mode(axis=1).iloc[:, 0]

    return {
        "rule_based": rule.tolist(),
        "qwen2.5-72b": qwen72.tolist(),
        "llm_majority": llm_majority.tolist(),
        "ids": rule.index.tolist(),
    }


def fig7_per_class_comparison() -> None:
    gt = pd.read_csv(EVAL_DIR / "ground_truth_3class.csv").set_index("child_id")
    baselines = _load_baseline_predictions()
    hrsf_pred = pd.read_csv(HRSF_DIR / "hrsf_predictions_at_best_alpha.csv").set_index("child_id")
    best_alpha_col = [c for c in hrsf_pred.columns if c.startswith("hrsf_alpha_")][0]
    best_alpha = float(best_alpha_col.replace("hrsf_alpha_", ""))

    ids = baselines["ids"]
    y_true = gt.loc[ids, "final_3class"].tolist()

    systems = {
        "Rule-based": baselines["rule_based"],
        "LLM majority\nvote (7 LLMs)": baselines["llm_majority"],
        "Qwen2.5-72B\n(best single LLM)": baselines["qwen2.5-72b"],
        f"HRSF (α={best_alpha:.2f})\n[ours]": hrsf_pred.loc[ids, best_alpha_col].tolist(),
    }

    rows = []
    for name, preds in systems.items():
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, preds, labels=LABELS, zero_division=0
        )
        for i, lbl in enumerate(LABELS):
            rows.append({"system": name, "class": lbl, "f1": f1[i]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.22
    x = np.arange(len(systems))
    for i, lbl in enumerate(LABELS):
        sub = df[df["class"] == lbl]
        ax.bar(
            x + (i - 1) * width,
            sub["f1"].values,
            width,
            label=lbl,
            color=LABEL_COLOURS[lbl],
            edgecolor="black",
            linewidth=0.4,
        )
        # Annotate each bar with its F1 value.
        for xi, val in zip(x + (i - 1) * width, sub["f1"].values):
            ax.text(xi, val + 0.015, f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(systems.keys()), fontsize=10)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title(
        "Per-class F1 — HRSF is the first system to detect the `others` class (F1 > 0)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(title="Class", loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig7_hrsf_per_class.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def fig8_disagreement_heatmap() -> None:
    """Heatmap on the 12 inter-rater disagreement samples."""
    hrsf_pred = pd.read_csv(HRSF_DIR / "hrsf_predictions_at_best_alpha.csv")
    best_alpha_col = [c for c in hrsf_pred.columns if c.startswith("hrsf_alpha_")][0]

    # Restrict to the 12 disagreement samples.
    dz = hrsf_pred[hrsf_pred["rater1"] != hrsf_pred["rater2"]].copy()

    cols = ["rater1", "rater2", "rule", "llm_majority", best_alpha_col]
    col_labels = [
        "Rater 1",
        "Rater 2",
        "Rule-based",
        "LLM majority",
        f"HRSF (α={best_alpha_col.replace('hrsf_alpha_', '')})",
    ]
    label_to_int = {lbl: i for i, lbl in enumerate(LABELS)}

    matrix = np.array([[label_to_int[dz.iloc[r][c]] for c in cols] for r in range(len(dz))])
    short_ids = [cid.split(":")[0][-12:] for cid in dz["child_id"].tolist()]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(dz) + 1)))
    cmap = plt.cm.colors.ListedColormap([LABEL_COLOURS[lbl] for lbl in LABELS])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=2.5)

    # Cell text — class name in each cell.
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(
                c,
                r,
                LABELS[matrix[r, c]],
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
            )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(dz)))
    ax.set_yticklabels(short_ids, fontsize=8)
    ax.set_title(
        "Disagreement-zone behaviour — 12 samples where rater 1 ≠ rater 2",
        fontsize=12,
    )

    # Legend showing class colours.
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=LABEL_COLOURS[lbl], edgecolor="black", label=lbl) for lbl in LABELS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)

    plt.tight_layout()
    out = FIG_DIR / "fig8_hrsf_disagreement_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def main() -> None:
    fig6_alpha_sensitivity()
    fig7_per_class_comparison()
    fig8_disagreement_heatmap()
    print(f"\nHRSF figures written to {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
