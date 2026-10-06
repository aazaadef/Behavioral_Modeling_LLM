"""Redrawn figures for the revision (Reviewer 4 minor comments 2-3, Reviewer 1/4 on "human ceiling").

  fig2_kappa_consensus   Cohen's kappa vs consensus with channel-cluster CIs; the
                         inter-rater kappa is a labelled reference band, not a "ceiling".
  fig3_kappa_per_rater   kappa against rater 1, rater 2 and the consensus (sequential blue).
  fig4_alignment_delta   delta-kappa per varied system with channel CI and the
                         permutation null of the system's own label frequencies.

Inputs: revision/statistics/*.csv. Outputs: PNG (300 dpi) and PDF next to this file.
Colours: reference palette slots 1-3 (blue, orange, aqua; documented all-pairs CVD-safe
in light mode) plus a neutral gray for baselines. Print figures: light mode only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
STATS = HERE.parent / "statistics"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAMILY_COLOR = {"Rule": "#2a78d6", "LLM": "#eb6834", "Supervised": "#1baf7a", "NLI": MUTED, "Dummy": MUTED}
FAMILY_LABEL = {"Rule": "Rule-based", "LLM": "LLM (zero-shot)", "Supervised": "Supervised ML",
                "NLI": "NLI / dummy baselines", "Dummy": "NLI / dummy baselines"}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INTER_RATER = (0.613, 0.470, 0.842)  # kappa, 95% CI (channel-cluster bootstrap), label audit

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.titlecolor": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.linewidth": 0.8, "savefig.dpi": 300,
})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend(ax, families, loc="lower right", **kw):
    seen, handles = set(), []
    order = ["Rule", "LLM", "Supervised", "NLI", "Dummy"]
    for f in sorted(set(families), key=order.index):
        lab = FAMILY_LABEL[f]
        if lab in seen:
            continue
        seen.add(lab)
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=FAMILY_COLOR[f], label=lab))
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8, labelcolor=INK2, **kw)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig2(T):
    d = T.sort_values("kappa__consensus")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    k, lo, hi = INTER_RATER
    ax.axvspan(lo, hi, color="#f0efec", zorder=0)
    ax.axvline(k, color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
    label = f"Rater 1 vs rater 2: κ = {k:.3f}" + "\n" + "(shaded: 95% channel CI)"
    ax.text(k, len(d) - 0.2, label, ha="center", va="bottom", fontsize=7.5, color=INK2,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        c = FAMILY_COLOR[r["family"]]
        ax.plot([r["kappa__channel_id_lo"], r["kappa__channel_id_hi"]], [yi, yi], color=c, linewidth=1.6,
                solid_capstyle="round", zorder=2)
        ax.plot(r["kappa__consensus"], yi, "o", markersize=6, color=c, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
    ax.set_yticks(y, d["system"])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Cohen's κ vs consensus (95% CI, paired cluster bootstrap over channels)")
    ax.set_xlim(-0.35, 1.0)
    ax.set_ylim(-0.7, len(d) + 0.9)
    style(ax)
    legend(ax, d["family"], loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, columnspacing=1.2,
           handletextpad=0.3)
    save(fig, "fig2_kappa_consensus")


def fig3(T):
    d = T.sort_values("kappa__consensus", ascending=False)
    M = d[["kappa__rater1", "kappa__rater2", "kappa__consensus"]].to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(np.clip(M, 0, None), cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.45 else INK)
    ax.set_xticks(range(3), ["Rater 1", "Rater 2", "Consensus"])
    ax.set_yticks(range(len(d)), d["system"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Cohen's κ (negative values shown as 0)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig3_kappa_per_rater")


def fig4(A):
    d = A[A["varied"]].sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.add_patch(plt.Rectangle((r["null_lo"], yi - 0.32), r["null_hi"] - r["null_lo"], 0.64,
                                   color="#f0efec", zorder=0, linewidth=0))
        c = FAMILY_COLOR[r["family"]]
        ax.plot([r["delta_lo_channel"], r["delta_hi_channel"]], [yi, yi], color=c, linewidth=1.6,
                solid_capstyle="round", zorder=2)
        ax.plot(r["delta"], yi, "o", markersize=6, color=c, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_yticks(y, d["system"])
    ax.set_xlabel("Δκ = κ(rater 1) − κ(rater 2)   (line: 95% channel CI; gray box: permutation null 95%)")
    ax.text(0.99, 0.02, "← closer to rater 2      closer to rater 1 →", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7.5, color=MUTED)
    style(ax)
    legend(ax, d["family"], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, columnspacing=1.2,
           handletextpad=0.3)
    save(fig, "fig4_alignment_delta")


def box(ax, x, y, w, h, title, lines, color, fill, title_size=6.6, size=6.0, dashed=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", facecolor=fill,
                                edgecolor=color, linewidth=0.9, linestyle=(0, (4, 2)) if dashed else "-"))
    band = 3.2
    ax.add_patch(FancyBboxPatch((x, y + h - band), w, band, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=color, edgecolor=color, linewidth=0.9))
    ax.text(x + w / 2, y + h - band / 2, title, ha="center", va="center", fontsize=title_size, color="white",
            fontweight="bold")
    ax.text(x + w / 2, y + (h - band) / 2, "\n".join(lines), ha="center", va="center", fontsize=size, color=INK,
            linespacing=1.4)


def arrow(ax, p, q):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=INK2, linewidth=0.8,
                                 shrinkA=0, shrinkB=0))


def line(ax, xs, ys):
    ax.plot(xs, ys, color=INK2, linewidth=0.8, solid_capstyle="butt")


def fig1():
    """Study design (replaces the hand-drawn Fig. 1; Reviewer 4 minor comment 2)."""
    fig, ax = plt.subplots(figsize=(7.16, 4.4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(-0.5, 62)
    ax.axis("off")
    NEUTRAL, NFILL = "#52514e", "#f4f3ef"
    top, th = 43, 18.5
    box(ax, 0, top, 14, th, "Source data", ["ChildPlay-gaze", "test split", "", "Frame-level", "human gaze",
                                           "annotations"], NEUTRAL, NFILL)
    box(ax, 16.5, top, 13, th, "Unit", ["Child–clip", "sequence", "", "n = 69", "45 clips", "21 videos",
                                       "10 channels"], NEUTRAL, NFILL)
    box(ax, 32, top, 45.5, th, "Gaze-derived features (same input for all systems)", [], NEUTRAL, NFILL)
    chips = [("Visibility", "visible ratio, on-screen ratio,\nmax visible streak", 33, 51.2),
             ("Gaze dynamics and stability", "gaze-shift ratio, mean gaze\nmotion, stability score", 55, 51.2),
             ("Evidence quality", "occlusion ratio, eyes-closed\nratio, max occlusion streak", 33, 44),
             ("Categorical cue", "dominant gaze class", 55, 44)]
    for t, body, cx, cy in chips:
        ax.add_patch(plt.Rectangle((cx, cy), 21.5, 6.6, facecolor="white", edgecolor=AXIS, linewidth=0.6))
        ax.text(cx + 10.75, cy + 5.15, t, ha="center", va="center", fontsize=6.0, fontweight="bold", color=INK)
        ax.text(cx + 10.75, cy + 2.35, body, ha="center", va="center", fontsize=5.7, color=INK2, linespacing=1.3)
    box(ax, 80, top, 20, th, "Human reference", ["Two raters viewed", "the clips only", "(no features, no",
                                                 "model outputs)", "Consensus by discussion",
                                                 "48 focused · 18 mix · 3 others"], NEUTRAL, NFILL)
    arrow(ax, (14, 52.25), (16.5, 52.25))
    arrow(ax, (29.5, 52.25), (32, 52.25))
    xs, w = [0, 19.5, 39, 58.5], 17.5
    mid = 23
    paradigms = [("A · Rule-based", ["Thresholds fixed", "a priori", "No training"], "#2a78d6", "#e6f0fb"),
                 ("B · Supervised ML", ["LR, SVM, RF,", "XGBoost, LightGBM,", "2 dummy baselines",
                                        "Out-of-fold CV on the 69"], "#1baf7a", "#e3f5ee"),
                 ("C · Zero-shot NLI", ["BART, DeBERTa,", "DistilBERT", "Entailment over", "label hypotheses"],
                  MUTED, "#efeeea"),
                 ("D · Zero-shot LLMs", ["7 open-weight models", "Fixed JSON prompt", "Greedy decoding"],
                  "#eb6834", "#fdebe3")]
    for x, (t, lines, c, f) in zip(xs, paradigms):
        box(ax, x, mid, w, 15.5, t, lines, c, f)
    line(ax, [54.75, 54.75], [top, 40.5])
    line(ax, [xs[0] + w / 2, xs[-1] + w / 2], [40.5, 40.5])
    for x in xs:
        arrow(ax, (x + w / 2, 40.5), (x + w / 2, mid + 15.5))
    bot, bh = 0, 15.5
    box(ax, 0, bot, 45, bh, "Exploratory analyses (same 69 sequences)",
        ["Prompt ablations (D) and NLI formulations (C)",
         "HRSF: reliability gate, then a vote with rule weight α",
         "",
         r"$\mathrm{score}(c\,|\,x)=\alpha\,\mathbb{I}[r(x)=c]+(1-\alpha)\,\frac{1}{K}\,\Sigma_{i}\,\mathbb{I}[\ell_i(x)=c]$"],
        NEUTRAL, "white", dashed=True)
    box(ax, 48, bot, 52, bh, "Evaluation on the 69 sequences",
        ["Cohen's κ and Gwet's AC1 against the consensus", "and against each rater; per-class F1",
         "Paired cluster bootstrap over channels (videos: sensitivity)",
         "Exact McNemar tests with Holm correction"], NEUTRAL, NFILL)
    line(ax, [xs[0] + w / 2, xs[-1] + w / 2], [20, 20])
    for x in xs:
        line(ax, [x + w / 2, x + w / 2], [mid, 20])
    arrow(ax, (67.25, 20), (67.25, bh))
    line(ax, [90, 90], [top, 20])
    arrow(ax, (90, 20), (90, bh))
    save(fig, "fig1_study_design")


def fig_hrsf():
    """HRSF alpha sweep and per-class F1 in one larger figure (old Figs. 7 and 8; Reviewer 4 minor 3)."""
    S = pd.read_csv((HERE.parent.parent / "code" if (HERE.parent.parent / "code" / "output").is_dir() else HERE.parent.parent) / "output" / "3class_eval" / "phase_a" / "hrsf" / "hrsf_alpha_sweep.csv")
    rule_k, rule_f = 0.706, (0.926, 0.800, 0.000)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.8), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.28})
    a.axvspan(0.55, 1.05, color="#f0efec", zorder=0, linewidth=0)
    a.text(0.8, 0.42, "α > 0.5: LLM votes\ncannot change\nthe rule's class", ha="center", va="center", fontsize=6.8,
           color=INK2)
    a.axhline(rule_k, color="#2a78d6", linewidth=1.1, linestyle=(0, (4, 2)))
    a.text(0.0, rule_k + 0.012, "Rule only (κ = 0.706)", fontsize=6.8, color="#2a78d6", va="bottom")
    a.plot(S["alpha"], S["kappa"], "-o", color=INK, markersize=3.8, linewidth=1.1, markeredgecolor="white")
    a.set_xlim(-0.05, 1.05)
    a.set_ylim(0.35, 0.78)
    a.set_xlabel("Rule weight α", fontsize=7.5)
    a.set_ylabel("Cohen's κ vs consensus", fontsize=7.5)
    a.set_title("(a) Gate and vote, by rule weight", fontsize=8, loc="left")
    b.set_title("(b) Per-class F1", fontsize=8, loc="left")
    pick = lambda al: tuple(S.loc[np.isclose(S["alpha"], al)].iloc[0][["f1_focused", "f1_mix", "f1_others"]])  # noqa: E731
    configs = [("Rule only", rule_f, "#2a78d6"), ("Rule + gate (α ≥ 0.6)", pick(0.6), "#104281"),
               ("Gate + vote (α = 0.4–0.5)", pick(0.5), "#86b6ef"), ("LLM ensemble + gate (α = 0)", pick(0.0), "#eb6834")]
    x = np.arange(3)
    wd = 0.2
    for i, (lab, vals, c) in enumerate(configs):
        b.bar(x + (i - 1.5) * wd, vals, wd * 0.92, color=c, label=lab)
    b.set_xticks(x, ["focused\n(n = 48)", "mix\n(n = 18)", "others\n(n = 3)"])
    b.set_ylim(0, 1.0)
    b.set_ylabel("F1 vs consensus", fontsize=7.5)
    b.text(2, 0.27, "gate: 7 flags,\n1 correct", ha="center", fontsize=6.6, color=INK2)
    b.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, labelcolor=INK2,
             handlelength=1.0, columnspacing=1.0)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    save(fig, "fig7_hrsf_ablation")


def main():
    T = pd.read_csv(STATS / "system_metrics_with_cis.csv")
    A = pd.read_csv(STATS / "alignment_delta.csv")
    fig1()
    fig2(T)
    fig3(T)
    fig4(A)
    fig_hrsf()
    print("written:", sorted(p.name for p in HERE.glob("fig*.png")))


if __name__ == "__main__":
    main()
