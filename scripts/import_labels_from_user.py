"""Import all 69 manual labels provided by user into manual_eval_annotations.csv."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# All 69 labels as provided by user.
LABELS = {
    "1Ab4vLMMAbY_2354-2439:person_1": "focused_attention",
    "1Ab4vLMMAbY_412-554:person_1": "focused_attention",
    "1Ab4vLMMAbY_557-690:person_1": "focused_attention",
    "31lG75MDwSA_2857-2928:person_1": "focused_attention",
    "31lG75MDwSA_3180-3306:person_1": "focused_attention",
    "31lG75MDwSA_4686-4764:person_2": "mixed_attention",
    "6mA6UAoT3M0_6165-6361:person_1": "mixed_attention",
    "6mA6UAoT3M0_6165-6361:person_3": "focused_attention",
    "9DNwRwt5kI4_10666-10822:person_2": "focused_attention",
    "9DNwRwt5kI4_10666-10822:person_3": "mixed_attention",
    "9DNwRwt5kI4_14988-15270:person_2": "mixed_attention",
    "9DNwRwt5kI4_14988-15270:person_3": "mixed_attention",
    "9DNwRwt5kI4_2442-3179:person_1": "focused_attention",
    "9DNwRwt5kI4_2442-3179:person_3": "mixed_attention",
    "LHT2zVYvObg_1790-1917:person_1": "focused_attention",
    "LHT2zVYvObg_1790-1917:person_2": "mixed_attention",
    "LHT2zVYvObg_1921-2099:person_1": "focused_attention",
    "LS9Hztyrmmw_1768-1839:person_2": "focused_attention",
    "LS9Hztyrmmw_1768-1839:person_3": "mixed_attention",
    "LS9Hztyrmmw_2667-2845:person_2": "focused_attention",
    "LS9Hztyrmmw_2667-2845:person_3": "focused_attention",
    "Lva4fn4_q88_1270-1473:person_2": "focused_attention",
    "Lva4fn4_q88_1270-1473:person_3": "mixed_attention",
    "Lva4fn4_q88_4391-4523:person_2": "focused_attention",
    "Lva4fn4_q88_4391-4523:person_3": "focused_attention",
    "Lva4fn4_q88_4691-4775:person_2": "focused_attention",
    "Lva4fn4_q88_4691-4775:person_3": "exploratory_attention",
    "Lva4fn4_q88_5291-5420:person_2": "focused_attention",
    "Lva4fn4_q88_5291-5420:person_3": "focused_attention",
    "ND7pXuhs3VM_1800-1925:person_2": "mixed_attention",
    "ND7pXuhs3VM_3066-3159:person_1": "mixed_attention",
    "NIk1-ck4c6Q_13651-13719:person_2": "mixed_attention",
    "NIk1-ck4c6Q_13651-13719:person_3": "mixed_attention",
    "NIk1-ck4c6Q_22766-22806:person_2": "focused_attention",
    "NIk1-ck4c6Q_22766-22806:person_3": "focused_attention",
    "N_9vfhsDz8w_5785-5911:person_2": "focused_attention",
    "N_9vfhsDz8w_5785-5911:person_3": "focused_attention",
    "N_9vfhsDz8w_6085-6298:person_2": "focused_attention",
    "N_9vfhsDz8w_6085-6298:person_3": "focused_attention",
    "Pyb0z_YQjjI_7332-7429:person_2": "exploratory_attention",
    "Pyb0z_YQjjI_7332-7429:person_3": "focused_attention",
    "Pyb0z_YQjjI_7632-7685:person_2": "focused_attention",
    "Pyb0z_YQjjI_7632-7685:person_3": "focused_attention",
    "UUwGLDeYN8c_266-474:person_1": "focused_attention",
    "WpbXt04qWEk_1166-1611:person_1": "focused_attention",
    "WpbXt04qWEk_924-1135:person_1": "focused_attention",
    "aWV7UUMddCU_5934-6205:person_2": "focused_attention",
    "aWV7UUMddCU_6517-7514:person_1": "mixed_attention",
    "dUYIh1U2z-8_3300-3427:person_1": "occluded_attention",
    "dUYIh1U2z-8_3300-3427:person_2": "focused_attention",
    "dUYIh1U2z-8_4007-4220:person_1": "focused_attention",
    "dUYIh1U2z-8_4007-4220:person_2": "focused_attention",
    "dUYIh1U2z-8_4007-4220:person_3": "focused_attention",
    "f6wqlpG9rd0_1402-1688:person_1": "focused_attention",
    "f6wqlpG9rd0_203-262:person_1": "focused_attention",
    "f6wqlpG9rd0_9404-9541:person_1": "mixed_attention",
    "f6wqlpG9rd0_9704-9949:person_1": "mixed_attention",
    "fvJbRuIu4-0_1308-1370:person_1": "focused_attention",
    "fvJbRuIu4-0_1308-1370:person_2": "focused_attention",
    "fvJbRuIu4-0_1907-2018:person_1": "mixed_attention",
    "fvJbRuIu4-0_1907-2018:person_2": "focused_attention",
    "ivX3JXIV1W4_1-1920:person_1": "focused_attention",
    "ivX3JXIV1W4_1-1920:person_2": "focused_attention",
    "odFKscFdEas_14127-14292:person_2": "mixed_attention",
    "odFKscFdEas_15822-16030:person_2": "mixed_attention",
    "whHPATlDw8M_33039-33248-downsampled:person_1": "focused_attention",
    "whHPATlDw8M_5014-5313-downsampled:person_2": "focused_attention",
    "zmZpa1p5zaE_1028-1129:person_1": "focused_attention",
    "zmZpa1p5zaE_1028-1129:person_2": "mixed_attention",
}

print(f"Total labels: {len(LABELS)}")

# Count distribution.
from collections import Counter
dist = Counter(LABELS.values())
for k, v in sorted(dist.items()):
    print(f"  {k}: {v}")

# Update manual_eval_annotations.csv.
dst = ROOT / "output" / "paper_eval" / "manual_eval_annotations.csv"
rows = []
with dst.open("r", encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    for row in reader:
        cid = row["child_id"]
        if cid in LABELS:
            row["final_label"] = LABELS[cid]
        rows.append(row)

updated = sum(1 for r in rows if r.get("final_label", "").strip())
with dst.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nUpdated {updated}/{len(rows)} rows with final_label in {dst.name}")
