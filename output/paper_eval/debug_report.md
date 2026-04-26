# Debug Report

## Bug
`output/paper_eval/manual_eval_subset.json` was generated from the full 69-sequence test benchmark outputs, while `output/restart_run/behavioral_features.jsonl` came from a separate partial run that only processed 3 test sequences.

## Why It Happened
The pipeline had no first-class command that prepared paper-eval assets from a canonical feature run, and it had no validation that the selected manual-eval subset existed in the feature file. That allowed stale artifacts from different runs to coexist silently.

## Files Changed
- `src/project_llm/manual_eval.py`
- `src/project_llm/cli.py`
- `tests/test_pipeline.py`
- `README.md`

## Correct Reproduction Command
```bash
python main.py prepare-paper-eval
python main.py validate-paper-eval
```

## Regenerated Counts
- `output/restart_run/behavioral_features.jsonl`: 69 records
- `output/paper_eval/manual_eval_subset.json`: 24 records

## Validation Result
All 24 manual-eval samples are present in `output/restart_run/behavioral_features.jsonl`.
