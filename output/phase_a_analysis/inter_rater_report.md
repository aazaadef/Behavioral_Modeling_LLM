# Inter-rater agreement — rater1 vs rater2

- **n** = 69
- **Raw agreement** = 0.7826 (54/69)
- **Cohen's κ** = 0.5310  (Landis–Koch: 0.6–0.8 = substantial, >0.8 = almost perfect)
- **Disagreements** = 15

## Per-rater label distribution

| Label | Rater 1 | Rater 2 | Final consensus |
|---|---:|---:|---:|
| exploratory_attention | 3 | 1 | 1 |
| focused_attention | 47 | 48 | 48 |
| mixed_attention | 17 | 17 | 17 |
| occluded_attention | 2 | 3 | 3 |

## Confusion matrix (rater1 rows vs rater2 cols)

| rater1\rater2 | exploratory_attention | focused_attention | mixed_attention | occluded_attention |
|---|---|---|---|---|
| exploratory_attention | 0 | 1 | 2 | 0 |
| focused_attention | 0 | 43 | 4 | 0 |
| mixed_attention | 1 | 4 | 10 | 2 |
| occluded_attention | 0 | 0 | 1 | 1 |

## Consensus bias on disagreements

- **sided with rater1**: 6
- **sided with rater2**: 8
- **third option**: 1

## Disagreements resolved by discussion (15)

| child_id | rater1 | rater2 | consensus |
|---|---|---|---|
| 31lG75MDwSA_4686-4764:person_2 | mixed_attention | focused_attention | **mixed_attention** |
| 6mA6UAoT3M0_6165-6361:person_1 | mixed_attention | occluded_attention | **occluded_attention** |
| 9DNwRwt5kI4_10666-10822:person_2 | mixed_attention | focused_attention | **focused_attention** |
| 9DNwRwt5kI4_10666-10822:person_3 | exploratory_attention | mixed_attention | **mixed_attention** |
| 9DNwRwt5kI4_14988-15270:person_2 | occluded_attention | mixed_attention | **occluded_attention** |
| 9DNwRwt5kI4_14988-15270:person_3 | focused_attention | mixed_attention | **mixed_attention** |
| 9DNwRwt5kI4_2442-3179:person_1 | mixed_attention | occluded_attention | **focused_attention** |
| 9DNwRwt5kI4_2442-3179:person_3 | mixed_attention | exploratory_attention | **mixed_attention** |
| LHT2zVYvObg_1790-1917:person_2 | focused_attention | mixed_attention | **focused_attention** |
| LS9Hztyrmmw_1768-1839:person_3 | mixed_attention | focused_attention | **focused_attention** |
| Lva4fn4_q88_1270-1473:person_3 | exploratory_attention | focused_attention | **focused_attention** |
| Lva4fn4_q88_4391-4523:person_3 | mixed_attention | focused_attention | **mixed_attention** |
| ND7pXuhs3VM_1800-1925:person_2 | focused_attention | mixed_attention | **mixed_attention** |
| Pyb0z_YQjjI_7332-7429:person_2 | exploratory_attention | mixed_attention | **exploratory_attention** |
| aWV7UUMddCU_5934-6205:person_2 | focused_attention | mixed_attention | **mixed_attention** |

## Ground-truth changes vs previous file (8)

| child_id | old | new |
|---|---|---|
| 6mA6UAoT3M0_6165-6361:person_1 | mixed_attention | occluded_attention |
| 9DNwRwt5kI4_14988-15270:person_2 | mixed_attention | occluded_attention |
| LHT2zVYvObg_1790-1917:person_2 | mixed_attention | focused_attention |
| LS9Hztyrmmw_1768-1839:person_3 | mixed_attention | focused_attention |
| Lva4fn4_q88_1270-1473:person_3 | mixed_attention | focused_attention |
| Lva4fn4_q88_4391-4523:person_3 | focused_attention | mixed_attention |
| Lva4fn4_q88_4691-4775:person_3 | exploratory_attention | focused_attention |
| aWV7UUMddCU_5934-6205:person_2 | focused_attention | mixed_attention |