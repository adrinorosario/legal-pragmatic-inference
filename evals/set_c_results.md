# Set C — D2 Judicial-Content Density × Candidate Length

## Motivation

D2 scores judicial-content density by counting interpretive and procedural
keyword hits. Longer passages mechanically contain more opportunities for such
hits. If D2 correlates with `len(raw_sentence)`, the threshold gate is partly a
**length gate**, and Claim 4 (threshold-clearing) predicts Claim 3 (length
dominance) — converting a reported oddity into an explained mechanism.

---

## Primary Result: Spearman ρ  (d2 raw hit count × candidate chars)

| Statistic | Value |
|-----------|-------|
| N | **12,294** |
| **Spearman ρ** | **+0.1262** |
| Spearman p-value | 7.6 × 10⁻⁴⁵ |
| Pearson r | +0.1198 |
| Pearson p-value | 1.6 × 10⁻⁴⁰ |

---

## Secondary Result: Point-Biserial r_pb  (D2 flag × candidate chars)

| Statistic | Value |
|-----------|-------|
| N | **12,294** |
| **Point-biserial r_pb** | **+0.0584** |
| p-value | 8.9 × 10⁻¹¹ |
| Mean chars, D2 = 0 | 577 |
| Mean chars, D2 = 1 | 610 |
| Δ mean chars | **+32** |

---

## Interpretation

**Weak but highly significant correlation — the causal-chain hypothesis is
not supported at the population level.**

### What the numbers say

- Spearman ρ = **+0.126** (p = 7.6 × 10⁻⁴⁵, N = 12,294). The correlation is
  unambiguously real (the p-value is machine-floor territory at this N) but the
  effect is tiny. Length accounts for roughly **1.6 %** of the rank-variance in D2
  hit count (ρ² ≈ 0.016).
- Point-biserial r_pb = **+0.058** (p = 8.9 × 10⁻¹¹). Passages flagged D2 = 1
  are on average **32 characters longer** (610 vs 577) — detectable at this
  sample size but negligible in practical magnitude.

### What this resolves

The proposed causal chain (`length → D2 hits → D2 gate → triplet retained`) is
**real but not the dominant story**. The D2 gate is principally a *content* gate,
not a length gate. Claims 3 and 4 remain empirically separable findings; Claim 4
does not subsume Claim 3 — the "oddity" stands independently.

The p-values are extreme because N = 12,294. A ρ of 0.126 at this N is trivially
distinguishable from zero, but "statistically detectable ≠ practically meaningful."
The correct frame is effect size, not significance.

### Mechanism note

The D2 binary flag fires when `(interpretive_hits ≥ 1) OR (procedural_hits ≥ 1)`.
Because even a single-hit short sentence passes, the gate does not substantially
privilege long text. The weak positive signal (~32 chars, ~5.5% of mean length) is
consistent with the trivial expectation that longer text has a marginally higher
chance of containing at least one match — but ρ² = 0.016 means the gate is
94 % explained by *what* is in the passage, not *how much* of it there is.

### Two follow-ups that would fully close the door

1. **Normalised D2** — hits-per-100-chars correlated against length. If ρ
   collapses toward zero, the raw-count correlation is purely a volume effect with
   no rate component.
2. **Conditional on D2 = 0 rows** — does hit count within the failing group rise
   with length? That would confirm a ceiling effect at the gate threshold.

---

*Script*: `evals/set_c_d2_length_correlation.py`
*Data*: `datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/concatenated_runs_json_triplets.json`

