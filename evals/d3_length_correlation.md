# Set D3 — D3 Semantic Coherence x Candidate Length

**Methodological note:** This analysis mirrors `set_c_d2_length_correlation.py` exactly.
Same population (N = 12,294), same length field (`len(raw_sentence)`), same
correlation functions (`scipy.stats.spearmanr`, `scipy.stats.pointbiserialr`),
same p-value formatting, same extraction logic in `extract_vectors()`.
No methodological drift.

**D3 definition** (from `triplet_quality_scorer.py:score_dimension_3`):
  `d3_semantic_coherence` = 1 when the raw_sentence matches any term
  from the anchor's inferred vagueness category (effort/time/scope/harm/
  necessity/industry_norms/knowledge/confidentiality/financial/survival).
  `d3_matched_category_terms` is the list of matched terms; its length
  is the hit count used for Spearman rho.

---

## Primary Result: Spearman rho  (D3 hit count x candidate chars)

| Statistic | Value |
|-----------|-------|
| N | **12,294** |
| **Spearman rho** | **+0.1498** |
| Spearman p-value | 1.25e-62 |
| Pearson r | +0.1621 |
| Pearson p-value | 3.64e-73 |

---

## Secondary Result: Point-Biserial rpb  (D3 flag x candidate chars)

| Statistic | Value |
|-----------|-------|
| N | **12,294** |
| **Point-biserial rpb** | **+0.1495** |
| p-value | 2.33e-62 |
| Mean chars, D3=0 | 568.4 |
| Mean chars, D3=1 | 641.5 |
| Delta mean chars | +73.0 |

**D3 flag distribution:**
- N (D3=1): 2,223 (18.1%)
- N (D3=0): 10,071 (81.9%)

---

*Script*: `evals/set_d3_length_correlation.py`
*Data*: `datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/concatenated_runs_json_triplets.json`
