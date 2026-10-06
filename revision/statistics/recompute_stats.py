"""Recomputed statistics for all 18 systems (Reviewer 1, Reviewer 2, Reviewer 4 major comments 3, 5, 8).

  1. Accuracy, Cohen's kappa, Gwet's AC1, macro-F1 and per-class F1 against the
     consensus, with 95% percentile CIs from a paired cluster bootstrap.
     Primary cluster unit: YouTube channel (10 clusters); sensitivity: source
     video (21 clusters). Every system is evaluated on the same resampled
     clusters, so comparisons stay paired.
  2. Kappa and AC1 against each rater separately, with CIs.
  3. Exact two-sided McNemar tests, rule-based vs each other system (family of
     17) and all 153 pairs, with Holm correction, discordant counts, and the
     cluster-bootstrap CI of the paired accuracy difference.
  4. Alignment pattern (Reviewer 4 major comment 5): delta = kappa(rater 1) -
     kappa(rater 2) per system with cluster CI, a permutation null that keeps
     each system's own label frequencies, and the prespecified restriction to
     systems with varied predictions.

Metric conventions: kappa is undefined only if expected agreement is 1.
Macro-F1 averages the three labels, and a class absent from both reference and
prediction contributes F1 = 0 (scikit-learn zero_division=0); the share of
replicates without any reference `others` is reported, and a sensitivity
macro-F1 over the classes present in the reference is given.

numpy and pandas only. Writes STATISTICS.md and CSV tables next to this file.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "code" if (ROOT / "code" / "output").is_dir() else ROOT  # project layout or repository layout
LABELS = ["focused", "mix", "others"]
N_BOOT = 5000
SEED = 20261005
VARIED_MAX_SHARE = 0.95  # prespecified: a system is "varied" if no label covers >= 95% of predictions

SYSTEMS = {
    "rule_based_label": "Rule-based",
    "llm_models_Qwen--Qwen2_5-72B-Instruct_label": "Qwen2.5-72B",
    "llm_models_01-ai--Yi-1_5-9B-Chat_label": "Yi-1.5-9B",
    "llm_models_mistralai--Mistral-7B-Instruct-v0_3_label": "Mistral-7B",
    "llm_models_microsoft--Phi-4-mini-instruct_label": "Phi-4-mini",
    "llm_models_meta-llama--Llama-3_1-8B-Instruct_label": "Llama-3.1-8B",
    "llm_models_qwen-7b_label": "qwen-7b",
    "llm_models_Qwen--Qwen2_5-7B-Instruct_label": "Qwen2.5-7B",
    "supervised_RandomForest_label": "Random Forest",
    "supervised_XGBoost_label": "XGBoost",
    "supervised_LightGBM_label": "LightGBM",
    "supervised_LogisticRegression_label": "Logistic Regression",
    "supervised_LinearSVM_label": "Linear SVM",
    "nli_bart_label": "NLI BART",
    "nli_deberta_label": "NLI DeBERTa",
    "nli_distilbert_label": "NLI DistilBERT",
    "supervised_Dummy_majority_label": "Dummy (majority)",
    "supervised_Dummy_stratified_label": "Dummy (stratified)",
}
FAMILY = {name: ("LLM" if c.startswith("llm_") else "Supervised" if c.startswith("supervised_") and "Dummy" not in c
                 else "Dummy" if "Dummy" in c else "NLI" if c.startswith("nli_") else "Rule")
          for c, name in SYSTEMS.items()}


def kappa(a, b):
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in range(3))
    return np.nan if pe >= 1 else (po - pe) / (1 - pe)


def ac1(a, b):
    po = np.mean(a == b)
    pi = [(np.mean(a == c) + np.mean(b == c)) / 2 for c in range(3)]
    pe = sum(p * (1 - p) for p in pi) / 2
    return (po - pe) / (1 - pe)


def f1(pred, y, c):
    tp = np.sum((pred == c) & (y == c))
    den = 2 * tp + np.sum((pred == c) & (y != c)) + np.sum((pred != c) & (y == c))
    return 0.0 if den == 0 else 2 * tp / den


def metrics(pred, y):
    f = [f1(pred, y, c) for c in range(3)]
    present = [c for c in range(3) if np.any(y == c)]
    return {"accuracy": np.mean(pred == y), "kappa": kappa(pred, y), "ac1": ac1(pred, y),
            "macro_f1": np.mean(f), "macro_f1_present": np.mean([f[c] for c in present]),
            "f1_focused": f[0], "f1_mix": f[1], "f1_others": f[2]}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def ci(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def fmt(x, lo=None, hi=None):
    return f"{x:.3f}" if lo is None else f"{x:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    pred_df = pd.read_csv(CODE / "output" / "all_predictions_3class.csv").set_index("child_id")
    man = pd.read_csv(ROOT / "revision" / "label_audit" / "sample_manifest.csv").set_index("sequence_id")
    df = pred_df.join(man[["video_id", "channel_id"]], how="inner")
    assert len(df) == 69
    code = {lab: i for i, lab in enumerate(LABELS)}
    y = {"consensus": df["final_manual_label_3class"].map(code).to_numpy(),
         "rater1": df["rater1_3class"].map(code).to_numpy(),
         "rater2": df["rater2_3class"].map(code).to_numpy()}
    P = {name: df[col].map(code).to_numpy() for col, name in SYSTEMS.items()}
    assert all(not np.isnan(v.astype(float)).any() for v in P.values())
    names = list(P)
    rng = np.random.default_rng(SEED)

    def boot_indices(unit):
        groups = [np.flatnonzero(df[unit].to_numpy() == g) for g in df[unit].unique()]
        return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)], len(groups)

    boots = {}
    for unit in ("channel_id", "video_id"):
        idx_list, n_groups = boot_indices(unit)
        boots[unit] = (idx_list, n_groups)

    # ---- 1-2. point estimates and cluster CIs ---------------------------------
    rows = []
    rep_no_others = {u: sum(not np.any(y["consensus"][ix] == 2) for ix in b[0]) for u, b in boots.items()}
    for name in names:
        row = {"system": name, "family": FAMILY[name]}
        for ref in ("consensus", "rater1", "rater2"):
            m = metrics(P[name], y[ref])
            for k, v in m.items():
                row[f"{k}__{ref}"] = v
        for unit, (idx_list, _) in boots.items():
            reps = {"accuracy": [], "kappa": [], "ac1": [], "macro_f1": [], "macro_f1_present": [], "f1_others": [],
                    "kappa_r1": [], "kappa_r2": [], "ac1_r1": [], "ac1_r2": [], "acc_minus_majority": [], "delta": []}
            for ix in idx_list:
                p, yc = P[name][ix], y["consensus"][ix]
                m = metrics(p, yc)
                for k in ("accuracy", "kappa", "ac1", "macro_f1", "macro_f1_present", "f1_others"):
                    reps[k].append(m[k])
                k1, k2 = kappa(p, y["rater1"][ix]), kappa(p, y["rater2"][ix])
                reps["kappa_r1"].append(k1); reps["kappa_r2"].append(k2); reps["delta"].append(k1 - k2)
                reps["ac1_r1"].append(ac1(p, y["rater1"][ix])); reps["ac1_r2"].append(ac1(p, y["rater2"][ix]))
                reps["acc_minus_majority"].append(np.mean(p == yc) - np.mean(yc == 0))
            for k, v in reps.items():
                lo, hi = ci(v)
                row[f"{k}__{unit}_lo"], row[f"{k}__{unit}_hi"] = lo, hi
            row[f"kappa_undefined__{unit}"] = int(np.isnan(np.asarray(reps["kappa"], dtype=float)).sum())
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(HERE / "system_metrics_with_cis.csv", index=False)

    # ---- 3. McNemar -----------------------------------------------------------
    correct = {n: (P[n] == y["consensus"]) for n in names}
    mc = []
    for a, b in combinations(names, 2):
        b_cnt = int(np.sum(correct[a] & ~correct[b]))
        c_cnt = int(np.sum(~correct[a] & correct[b]))
        mc.append({"system_a": a, "system_b": b, "a_right_b_wrong": b_cnt, "a_wrong_b_right": c_cnt,
                   "p_exact": mcnemar_exact(b_cnt, c_cnt)})
    MC = pd.DataFrame(mc)
    MC["p_holm_153"] = holm(MC["p_exact"])
    MC.to_csv(HERE / "mcnemar_all_153_pairs.csv", index=False)

    vs = []
    idx_c = boots["channel_id"][0]
    for other in names[1:]:
        b_cnt = int(np.sum(correct["Rule-based"] & ~correct[other]))
        c_cnt = int(np.sum(~correct["Rule-based"] & correct[other]))
        diffs = [np.mean(correct["Rule-based"][ix]) - np.mean(correct[other][ix]) for ix in idx_c]
        lo, hi = ci(diffs)
        vs.append({"other": other, "rule_right_other_wrong": b_cnt, "rule_wrong_other_right": c_cnt,
                   "p_exact": mcnemar_exact(b_cnt, c_cnt),
                   "acc_diff": float(np.mean(correct["Rule-based"]) - np.mean(correct[other])),
                   "acc_diff_lo_channel": lo, "acc_diff_hi_channel": hi})
    VS = pd.DataFrame(vs)
    VS["p_holm_17"] = holm(VS["p_exact"])
    VS.to_csv(HERE / "mcnemar_rule_vs_17.csv", index=False)

    # ---- 4. alignment pattern with permutation null --------------------------
    perm_rng = np.random.default_rng(SEED + 1)
    al = []
    for name in names:
        p = P[name]
        share = max(np.mean(p == c) for c in range(3))
        delta = kappa(p, y["rater1"]) - kappa(p, y["rater2"])
        null = []
        for _ in range(N_BOOT):
            q = perm_rng.permutation(p)
            null.append(kappa(q, y["rater1"]) - kappa(q, y["rater2"]))
        null = np.asarray(null, dtype=float)
        null = null[~np.isnan(null)]
        lo_n, hi_n = (np.percentile(null, 2.5), np.percentile(null, 97.5)) if len(null) else (np.nan, np.nan)
        r = T.set_index("system").loc[name]
        al.append({"system": name, "family": FAMILY[name], "max_label_share": share,
                   "varied": share < VARIED_MAX_SHARE, "delta": delta,
                   "delta_lo_channel": r["delta__channel_id_lo"], "delta_hi_channel": r["delta__channel_id_hi"],
                   "null_mean": float(np.mean(null)) if len(null) else np.nan, "null_lo": lo_n, "null_hi": hi_n,
                   "p_perm_two_sided": float(np.mean(np.abs(null - np.mean(null)) >= abs(delta - np.mean(null)))) if len(null) else np.nan})
    AL = pd.DataFrame(al)
    AL.to_csv(HERE / "alignment_delta.csv", index=False)

    # family-level delta for varied LLMs vs varied rule/tree systems, cluster bootstrap
    varied = set(AL.loc[AL["varied"], "system"])
    llm = [n for n in names if FAMILY[n] == "LLM" and n in varied]
    rt = [n for n in names if n in varied and n in ("Rule-based", "Random Forest", "XGBoost", "LightGBM")]
    fam_reps = []
    for ix in idx_c:
        d = lambda n: kappa(P[n][ix], y["rater1"][ix]) - kappa(P[n][ix], y["rater2"][ix])  # noqa: E731
        fam_reps.append(np.nanmean([d(n) for n in llm]) - np.nanmean([d(n) for n in rt]))
    fam_point = np.mean([AL.set_index("system").loc[n, "delta"] for n in llm]) - np.mean(
        [AL.set_index("system").loc[n, "delta"] for n in rt])
    fam_lo, fam_hi = ci(fam_reps)

    # ---- report -----------------------------------------------------------------
    t = T.set_index("system")
    L = ["# Recomputed statistics (all 18 systems)", "",
         "Generated by `recompute_stats.py`. Paired cluster bootstrap, "
         f"{N_BOOT} replicates, seed {SEED}. Primary unit: YouTube channel ({boots['channel_id'][1]} clusters); "
         f"sensitivity: source video ({boots['video_id'][1]} clusters). With so few clusters the intervals are "
         "themselves imprecise and should be read as indicative.", "",
         f"- Majority-class baseline accuracy: **{np.mean(y['consensus'] == 0):.3f}** (48/69).",
         f"- Bootstrap replicates with no reference `others`: {rep_no_others['channel_id']} of {N_BOOT} (channel), "
         f"{rep_no_others['video_id']} of {N_BOOT} (video). In these, F1(others) = 0 by the zero_division "
         "convention; `macro_f1_present` (CSV) averages only the classes present in the reference.",
         f"- Replicates with undefined kappa: {int(t[[c for c in t.columns if c.startswith('kappa_undefined')]].to_numpy().sum())}.", "",
         "## Agreement with the consensus (95% CI, channel clusters)", "",
         "| System | Family | Accuracy | Acc − majority | Cohen's κ | Gwet's AC1 | Macro-F1 | F1 others |",
         "|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = t.loc[n]
        L.append(f"| {n} | {FAMILY[n]} | {fmt(r['accuracy__consensus'], r['accuracy__channel_id_lo'], r['accuracy__channel_id_hi'])} | "
                 f"{fmt(r['accuracy__consensus'] - np.mean(y['consensus'] == 0), r['acc_minus_majority__channel_id_lo'], r['acc_minus_majority__channel_id_hi'])} | "
                 f"{fmt(r['kappa__consensus'], r['kappa__channel_id_lo'], r['kappa__channel_id_hi'])} | "
                 f"{fmt(r['ac1__consensus'], r['ac1__channel_id_lo'], r['ac1__channel_id_hi'])} | "
                 f"{fmt(r['macro_f1__consensus'], r['macro_f1__channel_id_lo'], r['macro_f1__channel_id_hi'])} | "
                 f"{r['f1_others__consensus']:.3f} |")
    L += ["", "## Agreement with each rater (κ and AC1, 95% CI, channel clusters)", "",
          "| System | κ rater 1 | κ rater 2 | AC1 rater 1 | AC1 rater 2 |", "|---|---|---|---|---|"]
    for n in names:
        r = t.loc[n]
        L.append(f"| {n} | {fmt(r['kappa__rater1'], r['kappa_r1__channel_id_lo'], r['kappa_r1__channel_id_hi'])} | "
                 f"{fmt(r['kappa__rater2'], r['kappa_r2__channel_id_lo'], r['kappa_r2__channel_id_hi'])} | "
                 f"{fmt(r['ac1__rater1'], r['ac1_r1__channel_id_lo'], r['ac1_r1__channel_id_hi'])} | "
                 f"{fmt(r['ac1__rater2'], r['ac1_r2__channel_id_lo'], r['ac1_r2__channel_id_hi'])} |")
    L += ["", "## Rule-based vs each other system (exact McNemar, Holm over 17 tests)", "",
          "Sequence-level McNemar tests ignore within-channel dependence; the cluster-bootstrap CI of the "
          "paired accuracy difference is the design-respecting complement.", "",
          "| Other system | Rule ✓ other ✗ | Rule ✗ other ✓ | p exact | p Holm (17) | Acc. difference [channel CI] |",
          "|---|---|---|---|---|---|"]
    for _, r in VS.iterrows():
        L.append(f"| {r['other']} | {r['rule_right_other_wrong']} | {r['rule_wrong_other_right']} | {r['p_exact']:.4f} | "
                 f"{r['p_holm_17']:.4f} | {fmt(r['acc_diff'], r['acc_diff_lo_channel'], r['acc_diff_hi_channel'])} |")
    n_sig153 = int((MC["p_holm_153"] < 0.05).sum())
    L += ["", f"All 153 pairs (`mcnemar_all_153_pairs.csv`): {n_sig153} remain significant at 0.05 after Holm correction.", "",
          "## Alignment pattern: Δκ = κ(rater 1) − κ(rater 2)", "",
          f"Varied systems (prespecified: no label covers ≥ {VARIED_MAX_SHARE:.0%} of predictions) are marked. "
          "The permutation null shuffles each system's own predictions across sequences, keeping its label "
          "frequencies, so it shows how much Δκ the raters' different class frequencies produce by themselves.", "",
          "| System | Family | Varied | Δκ [channel CI] | Permutation null mean [95%] | p (perm.) |", "|---|---|---|---|---|---|"]
    for _, r in AL.iterrows():
        L.append(f"| {r['system']} | {r['family']} | {'yes' if r['varied'] else 'no'} | "
                 f"{fmt(r['delta'], r['delta_lo_channel'], r['delta_hi_channel'])} | "
                 f"{fmt(r['null_mean'], r['null_lo'], r['null_hi'])} | {r['p_perm_two_sided']:.3f} |")
    L += ["", f"Family contrast (varied systems only): mean Δκ of LLMs ({len(llm)}) minus mean Δκ of rule/tree systems "
          f"({len(rt)}) = **{fam_point:.3f}** [channel CI {fam_lo:.3f}, {fam_hi:.3f}]."]
    (HERE / "STATISTICS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
