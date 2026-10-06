"""Out-of-fold few-shot prompts (Reviewer 3, point 4; Reviewers 1 and 4 on few-shot).

Design, fixed before any run:
  * 3-shot, one labelled example per class (focused, mix, others), consensus labels.
  * Examples come only from YouTube channels other than the target sequence's
    channel, the same rule as the leave-one-channel-out supervised baselines,
    so no example shares a channel (the proxy for the same child or family)
    with the sequence being classified.
  * The examples are drawn at random and shown in random order; three
    independent draws (seeds 1, 2, 3) show how much the result depends on
    which examples are drawn.
  * Each example is shown as its feature payload (same JSON as the target)
    followed by its category. The rest of the prompt is the original C0
    prompt, unchanged.

Writes outputs/prompts_fewshot.json with conditions FS3_s1, FS3_s2, FS3_s3.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
LABELS = ["focused", "mix", "others"]
SEEDS = [1, 2, 3]
ANCHOR = "Return ONLY a valid JSON object"


def main() -> None:
    prompts = json.loads((HERE / "outputs" / "prompts_by_condition.json").read_text(encoding="utf-8"))["C0"]
    pattern = re.compile(r"Features:\s*(\{.+\})", re.DOTALL)
    payload = {cid: json.loads(pattern.search(t).group(1)) for cid, t in prompts.items()}
    man = pd.read_csv(REV / "label_audit" / "sample_manifest.csv").set_index("sequence_id")
    label = man["consensus"].to_dict()
    channel = man["channel_id"].to_dict()
    ids = sorted(prompts)
    out = {}
    for seed in SEEDS:
        cond = f"FS3_s{seed}"
        out[cond] = {}
        for cid in ids:
            rng = random.Random(f"{seed}|{cid}")
            examples = []
            for lab in LABELS:
                pool = [o for o in ids if o != cid and label[o] == lab and channel[o] != channel[cid]]
                assert pool, (cid, lab)
                examples.append(rng.choice(pool))
            rng.shuffle(examples)
            block = "Here are three labelled example clips from other videos, one per category:\n\n"
            for i, ex in enumerate(examples, 1):
                block += (f"Example {i}:\nFeatures:\n{json.dumps(payload[ex], ensure_ascii=True, sort_keys=True)}\n"
                          f"Category: {label[ex]}\n\n")
            block += "Now classify the following clip.\n\n"
            base = prompts[cid]
            assert base.count(ANCHOR) == 1
            out[cond][cid] = base.replace(ANCHOR, block + ANCHOR, 1)
    meta = {"design": __doc__.strip(), "examples": {}}
    for cond, d in out.items():
        meta["examples"][cond] = {cid: re.findall(r"Category: (\w+)", t) for cid, t in d.items()}
    (HERE / "outputs" / "prompts_fewshot.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "outputs" / "prompts_fewshot_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    n_others_same_channel = sum(1 for cid in ids for o in ids if label[o] == "others" and channel[o] == channel[cid])
    print("conditions:", list(out), "| sequences:", len(ids), "| example labels per prompt:",
          {tuple(sorted(v)) for v in meta["examples"]["FS3_s1"].values()},
          "| target-channel pairs excluded for others:", n_others_same_channel)


if __name__ == "__main__":
    main()
