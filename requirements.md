# Project Name: ChildPlay LLM-Based Behavioral Analysis

## Objective
Develop a modular system that transforms low-level gaze annotations from the ChildPlay dataset into high-level behavioral interpretations using Large Language Models (LLMs), without requiring video processing.

The system must be designed for research and publication purposes, producing outputs suitable for academic papers.

---

# INPUT

## Dataset
- Source: ChildPlay Gaze Dataset (local)
- Use ONLY:
  - annotations/ (train/val/test CSV files)
  - clips.csv
  - splits.csv

## Data Format
Each annotation file contains:
- frame_id
- person_id
- gaze_x, gaze_y
- is_child

---

# SYSTEM REQUIREMENTS

## 1. Data Loader Module
- Load all annotation CSV files
- Merge with clip metadata
- Filter only children (is_child = 1)
- Organize data per child and per clip

### Output:
```json
{
  "child_id": "...",
  "clip_id": "...",
  "frames": [...]
}
2. Temporal Sequence Builder
Convert frame-level gaze into temporal sequences
Aggregate consecutive frames
Compute:
duration
gaze stability
movement patterns
Output:
{
  "child_id": "...",
  "sequence": [
    {"start": 0.0, "end": 2.3, "avg_x": ..., "avg_y": ..., "type": "stable"},
    {"start": 2.3, "end": 4.1, "type": "shift"}
    ]
}
3. Interaction Approximation Engine
Infer interaction patterns from gaze sequences
Implement rule-based logic:

Rules:

Stable gaze → focused attention
Rapid changes → attention shift
Repeated patterns → exploratory behavior
Output:
{
  "child_id": "...",
  "interactions": [
    {"type": "focused_attention", "duration": 2.3},
    {"type": "attention_shift", "duration": 1.8}
  ]
}
4. LLM Behavior Interpretation Module
IMPORTANT:

This is the core innovation.

Task:

Transform structured gaze/interactions into human-readable behavioral interpretation.

Input to LLM:

Natural language description of sequence:
Example:
"Child A maintains gaze for 2.3 seconds, then shifts attention rapidly."

Prompt Requirements:
Scientific tone
No hallucination
Behavior-focused interpretation
No clinical claims unless supported
Output:
{
  "interpretation": "...",
  "behavior_type": "...",
  "engagement_level": "...",
  "confidence": 0.0
}
5. Report Generator (FOR PAPER)

Generate outputs required for academic publication:

A. Aggregated Statistics
average attention duration
distribution of interaction types
frequency of shifts
B. Behavioral Summaries
per child
per clip
C. Dataset-level Insights
global patterns
variability across children
Output:
{
  "statistics": {...},
  "summaries": [...],
  "insights": [...]
}
