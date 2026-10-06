# Prompt ablation results

Seven models on the HULTIG GPU server in the paper's original precision (7-9B models bf16; Qwen2.5-72B 4-bit nf4 with fp16 compute, as in the original runs), greedy decoding, same prompts as the local run.

## Cohen's kappa vs consensus

| Model | C0 original (local re-run) | C1 + rule thresholds | C2 categorical values | C3 dominant_gaze_class masked | C4 perturbed values (±5%) |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.522 | 0.357 | 0.351 | 0.522 | 0.480 |
| Mistral-7B-Instruct-v0.3 | 0.542 | 0.512 | 0.308 | 0.559 | 0.555 |
| Phi-4-mini-instruct | 0.529 | 0.218 | 0.008 | 0.548 | 0.487 |
| Qwen2.5-72B-Instruct | 0.581 | 0.635 | 0.275 | 0.608 | 0.518 |
| Qwen2.5-7B-Instruct | 0.392 | 0.299 | 0.320 | 0.455 | 0.366 |
| Yi-1.5-9B-Chat | 0.549 | 0.514 | 0.494 | 0.668 | 0.480 |
| qwen-7b | 0.453 | 0.318 | 0.103 | 0.523 | 0.495 |

Rule-based classifier for reference: kappa 0.706.

## All metrics

| model | condition | n | no_json | invalid_label | pred_focused | pred_mix | pred_others | accuracy | kappa_consensus | kappa_rater1 | kappa_rater2 | f1_others | agree_with_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | C0 | 69 | 0 | 0 | 47 | 20 | 2 | 0.812 | 0.581 | 0.582 | 0.581 | 0.000 | 0.913 |
| Qwen2.5-72B-Instruct | C1 | 69 | 0 | 0 | 49 | 18 | 2 | 0.841 | 0.635 | 0.572 | 0.602 | 0.000 | 0.942 |
| Qwen2.5-72B-Instruct | C2 | 69 | 0 | 0 | 58 | 9 | 2 | 0.725 | 0.275 | 0.329 | 0.237 | 0.000 | 0.812 |
| Qwen2.5-72B-Instruct | C3 | 69 | 0 | 0 | 41 | 28 | 0 | 0.812 | 0.608 | 0.606 | 0.608 | 0.000 | 0.884 |
| Qwen2.5-72B-Instruct | C4 | 69 | 0 | 0 | 42 | 25 | 2 | 0.768 | 0.518 | 0.516 | 0.578 | 0.000 | 0.870 |
| Yi-1.5-9B-Chat | C0 | 69 | 0 | 0 | 51 | 18 | 0 | 0.812 | 0.549 | 0.690 | 0.549 | 0.000 | 0.913 |
| Yi-1.5-9B-Chat | C1 | 69 | 0 | 0 | 51 | 18 | 0 | 0.797 | 0.514 | 0.552 | 0.584 | 0.000 | 0.913 |
| Yi-1.5-9B-Chat | C2 | 69 | 0 | 0 | 40 | 29 | 0 | 0.754 | 0.494 | 0.520 | 0.494 | 0.000 | 0.812 |
| Yi-1.5-9B-Chat | C3 | 69 | 0 | 0 | 48 | 21 | 0 | 0.855 | 0.668 | 0.669 | 0.668 | 0.000 | 0.957 |
| Yi-1.5-9B-Chat | C4 | 69 | 0 | 0 | 51 | 18 | 0 | 0.783 | 0.480 | 0.552 | 0.480 | 0.000 | 0.884 |
| Qwen2.5-7B-Instruct | C0 | 69 | 0 | 0 | 54 | 13 | 2 | 0.754 | 0.392 | 0.507 | 0.392 | 0.000 | 0.841 |
| Qwen2.5-7B-Instruct | C1 | 69 | 0 | 0 | 57 | 8 | 4 | 0.725 | 0.299 | 0.315 | 0.372 | 0.000 | 0.797 |
| Qwen2.5-7B-Instruct | C2 | 69 | 0 | 0 | 54 | 13 | 2 | 0.725 | 0.320 | 0.366 | 0.285 | 0.000 | 0.812 |
| Qwen2.5-7B-Instruct | C3 | 69 | 0 | 0 | 54 | 15 | 0 | 0.783 | 0.455 | 0.569 | 0.455 | 0.000 | 0.870 |
| Qwen2.5-7B-Instruct | C4 | 69 | 0 | 0 | 53 | 14 | 2 | 0.739 | 0.366 | 0.479 | 0.366 | 0.000 | 0.826 |
| qwen-7b | C0 | 69 | 0 | 0 | 50 | 19 | 0 | 0.768 | 0.453 | 0.592 | 0.453 | 0.000 | 0.870 |
| qwen-7b | C1 | 69 | 0 | 0 | 60 | 9 | 0 | 0.754 | 0.318 | 0.451 | 0.277 | 0.000 | 0.812 |
| qwen-7b | C2 | 69 | 0 | 0 | 66 | 3 | 0 | 0.710 | 0.103 | 0.180 | 0.103 | 0.000 | 0.725 |
| qwen-7b | C3 | 69 | 0 | 0 | 45 | 24 | 0 | 0.783 | 0.523 | 0.586 | 0.586 | 0.000 | 0.913 |
| qwen-7b | C4 | 69 | 0 | 0 | 49 | 20 | 0 | 0.783 | 0.495 | 0.564 | 0.495 | 0.000 | 0.884 |
| Llama-3.1-8B-Instruct | C0 | 69 | 0 | 0 | 50 | 19 | 0 | 0.797 | 0.522 | 0.592 | 0.522 | 0.000 | 0.899 |
| Llama-3.1-8B-Instruct | C1 | 69 | 0 | 0 | 57 | 11 | 1 | 0.754 | 0.357 | 0.481 | 0.319 | 0.000 | 0.841 |
| Llama-3.1-8B-Instruct | C2 | 69 | 0 | 0 | 54 | 14 | 1 | 0.739 | 0.351 | 0.325 | 0.315 | 0.000 | 0.826 |
| Llama-3.1-8B-Instruct | C3 | 69 | 0 | 0 | 50 | 19 | 0 | 0.797 | 0.522 | 0.660 | 0.522 | 0.000 | 0.899 |
| Llama-3.1-8B-Instruct | C4 | 69 | 0 | 0 | 51 | 18 | 0 | 0.783 | 0.480 | 0.621 | 0.480 | 0.000 | 0.884 |
| Mistral-7B-Instruct-v0.3 | C0 | 69 | 0 | 0 | 48 | 19 | 2 | 0.797 | 0.542 | 0.610 | 0.542 | 0.000 | 0.899 |
| Mistral-7B-Instruct-v0.3 | C1 | 69 | 0 | 0 | 56 | 13 | 0 | 0.812 | 0.512 | 0.557 | 0.475 | 0.000 | 0.870 |
| Mistral-7B-Instruct-v0.3 | C2 | 69 | 0 | 0 | 58 | 10 | 1 | 0.739 | 0.308 | 0.360 | 0.269 | 0.000 | 0.826 |
| Mistral-7B-Instruct-v0.3 | C3 | 69 | 0 | 0 | 39 | 30 | 0 | 0.783 | 0.559 | 0.555 | 0.559 | 0.000 | 0.855 |
| Mistral-7B-Instruct-v0.3 | C4 | 69 | 0 | 0 | 46 | 21 | 2 | 0.797 | 0.555 | 0.619 | 0.555 | 0.000 | 0.870 |
| Phi-4-mini-instruct | C0 | 69 | 0 | 0 | 50 | 17 | 2 | 0.797 | 0.529 | 0.599 | 0.529 | 0.000 | 0.899 |
| Phi-4-mini-instruct | C1 | 69 | 0 | 0 | 60 | 6 | 3 | 0.710 | 0.218 | 0.278 | 0.179 | 0.000 | 0.768 |
| Phi-4-mini-instruct | C2 | 69 | 0 | 0 | 65 | 2 | 2 | 0.667 | 0.008 | 0.170 | 0.008 | 0.000 | 0.710 |
| Phi-4-mini-instruct | C3 | 69 | 0 | 0 | 41 | 28 | 0 | 0.783 | 0.548 | 0.545 | 0.548 | 0.000 | 0.884 |
| Phi-4-mini-instruct | C4 | 69 | 0 | 0 | 51 | 16 | 2 | 0.783 | 0.487 | 0.628 | 0.487 | 0.000 | 0.884 |

## Stability under ±5% perturbation (C4 vs C0)

Rule-based classifier flip rate under the same perturbation: **0.014**.

| Model | n | Flip rate |
|---|---|---|
| Qwen2.5-72B-Instruct | 69 | 0.072 |
| Yi-1.5-9B-Chat | 69 | 0.058 |
| Qwen2.5-7B-Instruct | 69 | 0.014 |
| qwen-7b | 69 | 0.014 |
| Llama-3.1-8B-Instruct | 69 | 0.014 |
| Mistral-7B-Instruct-v0.3 | 69 | 0.029 |
| Phi-4-mini-instruct | 69 | 0.043 |

## C0 against the paper's original predictions

Same prompt as the paper; `same_label` and `identical_raw` count sequences out of n.

| Model | n | Same label | Identical raw text | kappa original | kappa re-run |
|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 69 | 69 | 69 | 0.581 | 0.581 |
| Yi-1.5-9B-Chat | 69 | 69 | 69 | 0.549 | 0.549 |
| Qwen2.5-7B-Instruct | 69 | 69 | 69 | 0.392 | 0.392 |
| qwen-7b | 69 | 69 | 69 | 0.453 | 0.453 |
| Llama-3.1-8B-Instruct | 69 | 69 | 69 | 0.522 | 0.522 |
| Mistral-7B-Instruct-v0.3 | 69 | 69 | 69 | 0.542 | 0.542 |
| Phi-4-mini-instruct | 69 | 69 | 0 | 0.529 | 0.529 |
