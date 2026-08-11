#!/usr/bin/env python3
"""
Set D3 — D3 Semantic Coherence vs. Candidate Length
==================================================
Mirrors the EXACT methodology of set_c_d2_length_correlation.py:

  Same population: 12,294 scored triplets from concatenated_runs_json_triplets.json
  Same length field: len(raw_sentence) per triplet
  Same correlation functions: scipy.stats.spearmanr and scipy.stats.pointbiserialr
  Same extraction logic: extract_vectors() pulls raw_sentence, flag, hit-count
  Same p-value formatting: fmt_p() identical

D2 measured:
  - Spearman rho(density_raw=candidates, chars=length)
  - Point-biserial r_pb(flag=binary, chars=length)

D3 measures the analog:
  - Spearman rho(d3 hit count, chars)  [d3_matched_category_terms count]
  - Point-biserial r_pb(d3_semantic_coherence flag, chars)

The D3 scorer (triplet_quality_scorer.py score_dimension_3) computes:
  - d3_semantic_coherence: 0/1 binary
  - d3_matched_category_terms: list — use len() as the hit count
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    REPO_ROOT
    / "datasets"
    / "jdar_triplet_extracted_on_cuad_and_cold_cases"
    / "dpo_dataset_construction"
    / "concatenated_runs_json_triplets.json"
)

def load_data(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_vectors(records: list):
    """Mirror the exact extraction logic from set_c_d2_length_correlation.py."""
    chars, hit_count, flag = [], [], []
    skipped = 0
    for r in records:
        rs = r.get("raw_sentence", "")
        if not isinstance(rs, str):
            skipped += 1
            continue

        # D3 hit count: number of category terms matched in the sentence
        matched_terms = r.get("d3_matched_category_terms", [])
        if not isinstance(matched_terms, list):
            matched_terms = []
        hits = len(matched_terms)

        # D3 binary flag
        d3_flag = int(r.get("d3_semantic_coherence", 0) or 0)

        n_chars = len(rs)
        chars.append(n_chars)
        hit_count.append(hits)
        flag.append(d3_flag)

    return (
        np.array(chars, dtype=float),
        np.array(hit_count, dtype=float),
        np.array(flag, dtype=float),
        skipped,
    )


def fmt_p(p):
    """Identical to set_c_d2_length_correlation.py."""
    if p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main():
    print(f"Loading data from:\n  {DATA_PATH}")
    records = load_data(DATA_PATH)
    print(f"Loaded {len(records):,} records")

    chars, hit_count, flag, skipped = extract_vectors(records)
    print(f"Extracted {len(chars):,} rows  (skipped {skipped} with missing raw_sentence)")

    # --- D3 sanity checks against D2 known values ---
    # D2: density_raw = d2_interpretive_hits + d2_procedural_hits
    # D2 flag = d2_judicial_content
    d2_check = sum(1 for r in records if r.get("d2_judicial_content", None) == 1)
    d3_check = int(flag.sum()) if len(flag) else 0
    print(f"D2 flag=1 count (sanity, expected ~3,900+): {d2_check}")
    print(f"D3 flag=1 count: {d3_check}")

    # --- Primary: Spearman rho (D3 hit count x candidate chars) ---
    print("\n-- Spearman rho: D3 hit count x candidate chars --")
    rho, p = stats.spearmanr(hit_count, chars)
    print(f"   rho = {rho:+.6f}   p = {fmt_p(p)}")

    # --- Secondary: Point-biserial r_pb (D3 flag x candidate chars) ---
    print("\n-- Point-biserial r_pb: D3 flag x candidate chars --")
    rpb, ppb = stats.pointbiserialr(flag, chars)
    n = len(chars)
    n1 = int(flag.sum())
    n0 = n - n1
    m0 = chars[flag == 0].mean()
    m1 = chars[flag == 1].mean()
    delta = m1 - m0
    print(f"   rpb = {rpb:+.6f}   p = {fmt_p(ppb)}")
    print(f"   N = {n}")
    print(f"   N (D3=1) = {n1}  ({100*n1/n:.1f}%)")
    print(f"   N (D3=0) = {n0}  ({100*n0/n:.1f}%)")
    print(f"   mean chars | D3=0 = {m0:,.1f}")
    print(f"   mean chars | D3=1 = {m1:,.1f}")
    print(f"   Delta      = {delta:+,.1f} chars")

    # --- Pearson for reporting completeness (same as D2 script) ---
    r_pearson, p_pearson = stats.pearsonr(hit_count, chars)

    # --- Write markdown report ---
    md_path = Path(__file__).resolve().parent / "d3_length_correlation.md"
    md = f"""# Set D3 — D3 Semantic Coherence x Candidate Length

**Methodological note:** This analysis mirrors `set_c_d2_length_correlation.py` exactly.
Same population (N = {n:,}), same length field (`len(raw_sentence)`), same
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
| N | **{n:,}** |
| **Spearman rho** | **{rho:+.4f}** |
| Spearman p-value | {fmt_p(p)} |
| Pearson r | {r_pearson:+.4f} |
| Pearson p-value | {fmt_p(p_pearson)} |

---

## Secondary Result: Point-Biserial rpb  (D3 flag x candidate chars)

| Statistic | Value |
|-----------|-------|
| N | **{n:,}** |
| **Point-biserial rpb** | **{rpb:+.4f}** |
| p-value | {fmt_p(ppb)} |
| Mean chars, D3=0 | {m0:,.1f} |
| Mean chars, D3=1 | {m1:,.1f} |
| Delta mean chars | {delta:+,.1f} |

**D3 flag distribution:**
- N (D3=1): {n1:,} ({100*n1/n:.1f}%)
- N (D3=0): {n0:,} ({100*n0/n:.1f}%)

---

*Script*: `evals/set_d3_length_correlation.py`
*Data*: `datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/concatenated_runs_json_triplets.json`
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"\n[ok] Results written -> {md_path}")


if __name__ == "__main__":
    main()
