"""Generate publication-quality figures for the paper.

Produces five figures from existing CSVs (no model re-runs needed):

  Figure 1 — system_ranking_kappa.png
      Horizontal bar chart of all 18 systems' Cohen's kappa with
      bootstrap 95% CI, family-coloured, with the inter-rater ceiling
      drawn as a vertical reference line.

  Figure 2 — per_rater_kappa_heatmap.png
      18 systems x {rater1, rater2, consensus} kappa heatmap, sorted
      by consensus kappa.

  Figure 3 — alignment_fault_line.png
      Diverging horizontal bar chart of Δ = κ(vs r1) − κ(vs r2),
      colour-coded by family, illustrating the LLM/rule-tree split.

  Figure 4 — confusion_matrices_top4.png
      2x2 grid of confusion matrices for rule_based, Qwen2.5-72B,
      Yi-1.5-9B, and RandomForest.

  Figure 5 — schema_5_vs_3_class.png
      Paired bar chart of accuracy and kappa for the 10 systems that
      have both 5-class and 3-class runs.

All figures are rendered at 300 DPI and saved to
output/3class_eval/phase_a/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
PHASE_A = ROOT / "output" / "3class_eval" / "phase_a"
FIG_DIR = PHASE_A / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INTER_RATER_KAPPA = 0.6129  # rater1 vs rater2 — "human ceiling"

# Family palette — colour-blind safe, consistent across figures.
FAMILY_COLOURS = {
    "rule": "#2E86AB",  # deep blue
    "llm": "#E63946",  # red
    "supervised": "#06A77D",  # green
    "nli": "#9C6ADE",  # purple
    "dummy": "#9A9A9A",  # grey
}


def family_of(system: str) -> str:
    """Bucket a system name into one of five families for colouring."""
    s = system.lower()
    if s.startswith("rule_based"):
        return "rule"
    if s.startswith("llm_models_") or "llm_" in s:
        return "llm"
    if "dummy" in s:
        return "dummy"
    if s.startswith("supervised_"):
        return "supervised"
    if s in {"bart", "deberta", "distilbert"} or s.startswith("nli_"):
        return "nli"
    return "supervised"


def short_name(system: str) -> str:
    """Compress verbose HuggingFace-style identifiers for axis labels."""
    s = system
    if s.endswith("_3class"):
        s = s[: -len("_3class")]
    s = s.replace("llm_models_", "")
    s = s.replace("supervised_", "")
    s = s.replace("Qwen--Qwen2_5", "Qwen2.5")
    s = s.replace("01-ai--Yi-1_5", "Yi-1.5")
    s = s.replace("meta-llama--Llama-3_1", "Llama-3.1")
    s = s.replace("microsoft--Phi-4-mini", "Phi-4-mini")
    s = s.replace("mistralai--Mistral-7B-Instruct-v0_3", "Mistral-7B-v0.3")
    s = s.replace("-Instruct", "")
    s = s.replace("--", "/")
    s = s.replace("_", " ")
    return s


# ----------------------------------------------------------------------
# Figure 1 — System ranking with bootstrap CI
# ----------------------------------------------------------------------
def figure_1_ranking() -> None:
    df = pd.read_csv(PHASE_A / "bootstrap_ci_3class.csv").sort_values("kappa", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(df))

    families = [family_of(m) for m in df["model"]]
    colours = [FAMILY_COLOURS[f] for f in families]

    err_low = df["kappa"] - df["kappa_ci_low"]
    err_high = df["kappa_ci_high"] - df["kappa"]
    ax.barh(
        y_pos,
        df["kappa"],
        xerr=[err_low, err_high],
        color=colours,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )

    ax.axvline(
        INTER_RATER_KAPPA,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Inter-rater κ = {INTER_RATER_KAPPA:.3f} (human ceiling)",
    )
    ax.axvline(0.0, color="grey", linewidth=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_name(m) for m in df["model"]], fontsize=9)
    ax.set_xlabel("Cohen's κ vs consensus (3-class)", fontsize=11)
    ax.set_title(
        "System ranking on 69 ChildPlay-gaze samples\n" "(bootstrap 95% CI, 1000 iterations)",
        fontsize=12,
    )
    ax.set_xlim(-0.25, 0.95)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend.
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=FAMILY_COLOURS["rule"], edgecolor="black", label="Rule-based"),
        Patch(facecolor=FAMILY_COLOURS["llm"], edgecolor="black", label="LLM (open-weight)"),
        Patch(facecolor=FAMILY_COLOURS["supervised"], edgecolor="black", label="Supervised"),
        Patch(facecolor=FAMILY_COLOURS["nli"], edgecolor="black", label="Zero-shot NLI"),
        Patch(facecolor=FAMILY_COLOURS["dummy"], edgecolor="black", label="Dummy baseline"),
    ]
    ax.legend(
        handles=legend_elems
        + [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label=f"Human ceiling (κ={INTER_RATER_KAPPA:.3f})",
            )
        ],
        loc="lower right",
        fontsize=9,
        framealpha=0.95,
    )

    plt.tight_layout()
    out = FIG_DIR / "fig1_system_ranking_kappa.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------
# Figure 2 — Per-rater kappa heatmap
# ----------------------------------------------------------------------
def figure_2_per_rater_heatmap() -> None:
    df = pd.read_csv(PHASE_A / "per_rater_kappa.csv")
    pivot = (
        df.pivot(index="system", columns="reference", values="kappa")
        .loc[:, ["rater1", "rater2", "consensus"]]
        .sort_values("consensus", ascending=False)
    )
    pivot.index = [short_name(s) for s in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=INTER_RATER_KAPPA,
        vmin=-0.2,
        vmax=0.8,
        cbar_kws={"label": "Cohen's κ"},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Reference annotation", fontsize=11)
    ax.set_ylabel("")
    ax.set_xticklabels(["Rater 1", "Rater 2", "Consensus"], fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_title(
        f"Per-rater κ matrix — green ≥ inter-rater ceiling ({INTER_RATER_KAPPA:.3f})",
        fontsize=12,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig2_per_rater_kappa_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------
# Figure 3 — Alignment fault-line (Δ = κ_r1 − κ_r2)
# ----------------------------------------------------------------------
def figure_3_alignment_fault_line() -> None:
    df = pd.read_csv(PHASE_A / "per_rater_kappa.csv")
    pivot = df.pivot(index="system", columns="reference", values="kappa")
    pivot["delta"] = pivot["rater1"] - pivot["rater2"]
    # Drop the degenerate κ = 0 systems — Δ for them is uninformative.
    keep = pivot[(pivot["rater1"].abs() > 1e-6) | (pivot["rater2"].abs() > 1e-6)]
    keep = keep.sort_values("delta", ascending=True)

    families = [family_of(s) for s in keep.index]
    colours = [FAMILY_COLOURS[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(keep))
    ax.barh(y, keep["delta"], color=colours, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([short_name(s) for s in keep.index], fontsize=9)
    ax.set_xlabel("Δ = κ(vs rater 1) − κ(vs rater 2)", fontsize=11)
    ax.set_title(
        "Alignment fault line: which annotator does each system match more?\n"
        "Negative ⇒ closer to rater 2 (numerical/threshold style); "
        "positive ⇒ closer to rater 1 (lexical/semantic style)",
        fontsize=11,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # In-axis annotations marking the two clusters.
    ax.text(
        -0.07,
        0.5,
        "rule-based\n+ tree",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        color=FAMILY_COLOURS["rule"],
        weight="bold",
    )
    ax.text(
        1.04,
        0.5,
        "LLMs",
        transform=ax.transAxes,
        rotation=270,
        ha="center",
        va="center",
        fontsize=10,
        color=FAMILY_COLOURS["llm"],
        weight="bold",
    )

    plt.tight_layout()
    out = FIG_DIR / "fig3_alignment_fault_line.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------
# Figure 4 — Confusion matrices for the top 4 systems
# ----------------------------------------------------------------------
def figure_4_confusion_matrices() -> None:
    cm_dir = PHASE_A / "confusion_matrices"
    selected = [
        ("rule_based", "rule_based.csv"),
        (
            "Qwen2.5-72B-Instruct (LLM)",
            "llm_models_Qwen--Qwen2_5-72B-Instruct_3class.csv",
        ),
        ("Yi-1.5-9B-Chat (LLM)", "llm_models_01-ai--Yi-1_5-9B-Chat_3class.csv"),
        ("RandomForest (supervised)", "supervised_RandomForest.csv"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, fname) in zip(axes.flat, selected):
        cm = pd.read_csv(cm_dir / fname, index_col=0)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            annot_kws={"size": 12},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
    fig.suptitle(
        "Confusion matrices — three classes (focused / mix / others)",
        fontsize=13,
        y=1.00,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig4_confusion_matrices_top4.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------
# Figure 5 — 5-class vs 3-class paired comparison
# ----------------------------------------------------------------------
def figure_5_schema_comparison() -> None:
    # Static table (the same numbers reported in THREECLASS_REPORT §3).
    rows = [
        # name, acc_5, acc_3, kappa_5, kappa_3
        ("rule_based", 0.854, 0.870, 0.679, 0.706),
        ("Qwen2.5-72B (LLM)", 0.783, 0.812, 0.515, 0.581),
        ("RandomForest", 0.768, 0.797, 0.444, 0.491),
        ("qwen-7b (LLM)", 0.725, 0.768, 0.441, 0.453),
        ("Qwen2.5-7B (LLM)", 0.709, 0.754, 0.151, 0.392),
        ("LogisticRegression", 0.696, 0.725, 0.268, 0.331),
        ("LinearSVM", 0.710, 0.667, 0.314, 0.190),
        ("distilbert (NLI)", 0.014, 0.696, 0.000, 0.000),
        ("deberta (NLI)", 0.696, 0.681, 0.000, -0.026),
        ("bart (NLI)", 0.144, 0.261, -0.021, 0.000),
    ]
    df = pd.DataFrame(rows, columns=["system", "acc_5", "acc_3", "kappa_5", "kappa_3"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    y = np.arange(len(df))
    width = 0.4

    # Left panel: Accuracy.
    axes[0].barh(
        y - width / 2,
        df["acc_5"],
        width,
        label="5-class",
        color="#7BB3D6",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].barh(
        y + width / 2,
        df["acc_3"],
        width,
        label="3-class",
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_xlabel("Accuracy", fontsize=11)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["system"], fontsize=10)
    axes[0].set_title("Accuracy: 5-class vs 3-class", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(axis="x", linestyle=":", alpha=0.4)
    axes[0].set_xlim(0, 1)

    # Right panel: Cohen's κ.
    axes[1].barh(
        y - width / 2,
        df["kappa_5"],
        width,
        label="5-class",
        color="#F4A261",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].barh(
        y + width / 2,
        df["kappa_3"],
        width,
        label="3-class",
        color="#E76F51",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].axvline(0, color="grey", linewidth=0.6)
    axes[1].axvline(
        INTER_RATER_KAPPA,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Inter-rater (3-class) κ = {INTER_RATER_KAPPA:.3f}",
    )
    axes[1].set_xlabel("Cohen's κ", fontsize=11)
    axes[1].set_title("Cohen's κ: 5-class vs 3-class", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(axis="x", linestyle=":", alpha=0.4)
    axes[1].set_xlim(-0.2, 0.85)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Schema refinement effect — same 10 systems, both schemas",
        fontsize=13,
        y=1.00,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig5_schema_5_vs_3_class.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    figure_1_ranking()
    figure_2_per_rater_heatmap()
    figure_3_alignment_fault_line()
    figure_4_confusion_matrices()
    figure_5_schema_comparison()
    print(f"\nAll figures written to {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
