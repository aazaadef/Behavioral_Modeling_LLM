| model_name | backend_family | num_samples | overall_accuracy | average_confidence | notable_failure_modes |
| --- | --- | --- | --- | --- | --- |
| rule-based | rule-based | 69 | None | 0.7959 | legacy label space reused as-is |
| typeform/distilbert-base-uncased-mnli | zero-shot | 69 | None | 0.5259 | legacy label space reused as-is |
| facebook/bart-large-mnli | zero-shot | 69 | None | 0.2843 | legacy label space reused as-is |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | zero-shot | 69 | None | 0.5889 | legacy label space reused as-is |
| ./models/qwen-7b | hf-llm | 69 | None | 0.9324 | label space differs from legacy baselines; accuracy against legacy final_label omitted |
