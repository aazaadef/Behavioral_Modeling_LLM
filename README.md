# ChildPlay LLM-Based Behavior Analysis

## Overview
This project extends the ChildPlay dataset by transforming low-level gaze data into high-level behavioral interpretations using Large Language Models (LLMs).

## Pipeline
1. Load dataset (annotations only)
2. Convert gaze coordinates to temporal sequences
3. Approximate interaction patterns
4. Interpret behavior using LLM
5. Generate reports

## Requirements
- Python 3.10+
- pandas
- openai (or local LLM)

## Run
```bash
python main.py run
```

### Output Format Example

```json
{
  "child_id": "A",
  "interpretation": "The child shows sustained attention indicating engagement",
  "interaction_type": "focused_attention",
  "confidence": 0.87
}
```

## Paper Evaluation Workflow
Regenerate a full annotation-based feature run and the manual evaluation subset together:

```bash
python main.py prepare-paper-eval
```

Validate that every manual-eval sample exists in the saved behavioral features:

```bash
python main.py validate-paper-eval
```
