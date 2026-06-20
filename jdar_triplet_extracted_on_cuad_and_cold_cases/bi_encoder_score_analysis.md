# Bi-Encoder Score Distribution Analysis

> Analysis of **5,985 datapoints** from [distributed_threshold_results.json](file:///Users/adrinorosario/Desktop/legal-pragmatic-inference/jdar_triplet_extracted_on_cuad_and_cold_cases/distributed_threshold_results.json), threshold bucket `"0.4"`.  
> Script: [dataset_inspector.py](file:///Users/adrinorosario/Desktop/legal-pragmatic-inference/jdar_triplet_extracted_on_cuad_and_cold_cases/dataset_inspector.py)

---

## Global Summary

| Metric | Value |
|---|---|
| Total datapoints | **5,985** |
| **Mean** | **0.4147** |
| **Median** | **0.4106** |
| Std Dev (σ) | **0.0143** |
| **Min** | **0.4000** |
| **Max** | **0.5319** |
| **Range** | **0.1319** |
| Q1 (25th pctl) | 0.4044 |
| Q3 (75th pctl) | 0.4207 |
| IQR | 0.0163 |
| 5th percentile | 0.4008 |
| 10th percentile | 0.4016 |
| 90th percentile | 0.4331 |
| 95th percentile | 0.4433 |
| Coeff. of Variation | **3.45%** |

---

## Histogram

```
  [0.40, 0.45)   5,804  ( 97.0%)  ████████████████████████████████████████████████████████████
  [0.45, 0.50)     174  (  2.9%)  █
  [0.50, 0.55)       7  (  0.1%)  ▏
```

---

## Cumulative Clustering Diagnostic

| Cutoff | Count Below | % Below |
|---|---|---|
| 0.30 | 0 | 0.0% |
| 0.35 | 0 | 0.0% |
| 0.40 | 0 | 0.0% |
| **0.45** | **5,804** | **97.0%** |
| **0.50** | **5,978** | **99.9%** |
| 0.55 | 5,985 | 100.0% |

---

## Interpretation

> [!WARNING]
> **Yes — scores cluster suspiciously low.** The distribution is extremely compressed into a narrow band just above the 0.40 threshold floor.

### Key findings:

1. **Mean = 0.4147, Median = 0.4106** — Both central-tendency measures sit barely above the 0.40 floor. This is well below the 0.50 mark that would indicate moderate semantic similarity.

2. **97.0% of all scores fall in [0.40, 0.45)** — Nearly the entire dataset is packed into a 0.05-wide band. The histogram is essentially a single spike.

3. **Coefficient of Variation = 3.45%** — This is extremely low, meaning there is almost no meaningful variance between datapoints. The bi-encoder is barely distinguishing between matches.

4. **Range = 0.1319** — While the range looks non-trivial, it's driven by just 7 outlier scores above 0.50 (out of 5,985). The effective spread (IQR = 0.0163) is minuscule.

5. **Floor effect at 0.40** — The minimum score is 0.400002. This isn't coincidence — the pipeline applied a 0.40 threshold, so every score ≥ 0.40 was retained. The distribution is truncated at the left and compressed against the floor.

### What this means:

- The bi-encoder is producing **near-threshold scores for essentially all matches**. These aren't confident semantic matches — they're barely clearing the 0.40 bar.
- The tight clustering with σ = 0.0143 suggests the model sees these CUAD-anchor → case-law-sentence pairs as **roughly equally (dis)similar**, which is a red flag for downstream quality.
- If you raised the threshold to even **0.45**, you'd lose **97%** of your data. If you raised it to **0.50**, you'd lose **99.9%**.
