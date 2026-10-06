# Revised analysis

This folder contains the analyses added for the revised article, *Attention Inference from Child Gaze Features: An Exploratory Benchmark of Rules, Machine Learning, and LLMs*. Each script reads the original predictions and labels in `output/` and writes its results next to itself. All numbers in the revised article's tables come from these files.

| Folder | Content | Script |
|---|---|---|
| `statistics/` | Agreement with the consensus and each rater (accuracy, Cohen's κ, Gwet's AC1, macro-F1, per-class F1); paired cluster-bootstrap intervals over channels and videos; exact McNemar tests with Holm correction (17 and 153 tests); Δκ alignment with permutation null | `recompute_stats.py` |
| `supervised_grouped/` | Supervised baselines under the original three-fold, leave-one-channel-out and leave-one-video-out cross-validation | `rerun_supervised.py` |
| `threshold_audit/` | Sensitivity of the rule to the stability-score weights | `stability_weights_sensitivity.py` |
| `prompt_ablation/` | Prompt ablations C0–C4 and out-of-fold few-shot prompting for the seven LLMs: prompts, raw outputs, metrics | `run_prompt_ablation.py` (builds the prompts), `build_fewshot_prompts.py`, `evaluate_ablation.py --setup server`, `ablation_details.py`, `evaluate_fewshot.py` |
| `server/` | Runner used for the LLM re-runs (same models, precision and decoding as the original runs) | `run_ablation_server.py` |
| `nli_sensitivity/` | NLI models under ten premise, label and template formulations | `run_nli_sensitivity.py` |
| `r3_analyses/` | Calibration of the LLMs' stated confidence; case studies with the LLMs' explanations | `r3_analyses.py` |
| `label_audit/` | Sequence metadata (clip, video, channel, frame rate) with both raters' labels and the consensus; the 12 disagreement sequences with their features | data only |
| `figures/` | Figures of the revised article | `make_figures.py` |
| `manuscript/` | Writes the table rows of the revised article and the supplementary files | `build_tables.py` |

## Running

From the repository root, with the dependencies in `requirements.txt` plus `pandas` and `numpy`:

```bash
python revision/statistics/recompute_stats.py
python revision/supervised_grouped/rerun_supervised.py
python revision/threshold_audit/stability_weights_sensitivity.py
python revision/prompt_ablation/evaluate_ablation.py --setup server
python revision/prompt_ablation/ablation_details.py
python revision/prompt_ablation/evaluate_fewshot.py
python revision/r3_analyses/r3_analyses.py
python revision/figures/make_figures.py          # needs matplotlib
python revision/manuscript/build_tables.py
```

Re-running the LLMs (`server/run_ablation_server.py`) or the NLI models (`nli_sensitivity/run_nli_sensitivity.py`) needs GPUs and the model weights (Hugging Face checkpoints listed in Appendix G of the article). The raw outputs of those runs are included in `prompt_ablation/outputs_full_precision/` and `nli_sensitivity/`, so all metrics can be recomputed without a GPU.

Seeds: bootstrap and permutation analyses 20261005; supervised cross-validation `random_state = 42`; prompt perturbation (C4) 20261005; few-shot example draws seeds 1–3.
