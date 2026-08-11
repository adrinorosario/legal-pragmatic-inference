#!/usr/bin/env python3
"""
Set C Analysis: D2 Judicial-Content Density vs. Candidate Length
=================================================================
Hypothesis: D2 scores judicial-content density of the candidate.
Longer passages mechanically contain more judicial content.
If D2 correlates with candidate character count, then the threshold
gate was partly a length gate, and Claim 4 (threshold-clearing)
predicts Claim 3 (length dominance) rather than sitting beside it.

Measures:
  - d2_density_raw   : d2_interpretive_hits + d2_procedural_hits  (raw hit count)
  - d2_judicial_flag : d2_judicial_content  (binary gate)
  - candidate_chars  : len(raw_sentence)

Reports Spearman rho (primary), Pearson r, point-biserial rpb, and plots.
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    REPO_ROOT
    / "datasets"
    / "jdar_triplet_extracted_on_cuad_and_cold_cases"
    / "dpo_dataset_construction"
    / "concatenated_runs_json_triplets.json"
)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_vectors(records: list):
    chars, density_raw, flag = [], [], []
    skipped = 0
    for r in records:
        rs = r.get("raw_sentence", "")
        if not isinstance(rs, str):
            skipped += 1
            continue
        n_chars = len(rs)
        hits = int(r.get("d2_interpretive_hits", 0) or 0) + int(
            r.get("d2_procedural_hits", 0) or 0
        )
        j_flag = int(r.get("d2_judicial_content", 0) or 0)
        chars.append(n_chars)
        density_raw.append(hits)
        flag.append(j_flag)
    return (
        np.array(chars, dtype=float),
        np.array(density_raw, dtype=float),
        np.array(flag, dtype=float),
        skipped,
    )


def fmt_p(p):
    if p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def make_plots(chars, density_raw, flag, out_dir: Path):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 14), facecolor="#0d0f14")
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    accent  = "#7fd8ff"
    accent2 = "#ff7f7f"
    mid     = "#c0c8e0"

    # ── 1. Hexbin: raw hit count vs chars ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#13161f")
    hb = ax1.hexbin(chars, density_raw, gridsize=55, cmap="Blues", mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax1, pad=0.02)
    cb.set_label("count", color=mid, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=mid)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=mid)

    m, b, *_ = stats.linregress(chars, density_raw)
    xr = np.linspace(chars.min(), chars.max(), 300)
    ax1.plot(xr, m * xr + b, color=accent2, lw=1.6, label=f"OLS slope={m:.4f}")
    ax1.set_xlabel("len(raw_sentence)  [chars]", color=mid, fontsize=10)
    ax1.set_ylabel("d2 interpretive + procedural hits", color=mid, fontsize=10)
    ax1.set_title("D2 Raw Hit Count vs. Candidate Length", color="white", fontsize=11)
    ax1.tick_params(colors=mid)
    ax1.legend(fontsize=9, labelcolor=mid, framealpha=0.3)
    for sp in ax1.spines.values(): sp.set_edgecolor("#2a2f3f")

    # ── 2. Boxplot: char length by d2 flag ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#13161f")
    groups = [chars[flag == 0], chars[flag == 1]]
    bp = ax2.boxplot(
        groups, patch_artist=True, widths=0.45,
        medianprops=dict(color=accent2, lw=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.25, markerfacecolor="#666"),
    )
    for patch, col in zip(bp["boxes"], ["#2a3a5c", "#1a4a3c"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for w in bp["whiskers"]: w.set_color("#444")
    for c in bp["caps"]:     c.set_color("#444")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["D2 = 0  (not judicial)", "D2 = 1  (judicial)"], color=mid)
    ax2.set_ylabel("len(raw_sentence)  [chars]", color=mid, fontsize=10)
    ax2.set_title("Candidate Length by D2 Judicial Flag", color="white", fontsize=11)
    ax2.tick_params(colors=mid)
    for sp in ax2.spines.values(): sp.set_edgecolor("#2a2f3f")
    for i, g in enumerate(groups):
        ax2.text(i+1, np.percentile(g, 99)*0.97, f"μ={g.mean():.0f}",
                 ha="center", fontsize=9, color=accent)

    # ── 3. Overlaid length histograms ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#13161f")
    bins = np.linspace(0, np.percentile(chars, 99), 60)
    ax3.hist(chars[flag == 0], bins=bins, alpha=0.55, color="#5588cc",
             label="D2 = 0", density=True)
    ax3.hist(chars[flag == 1], bins=bins, alpha=0.55, color="#55cc88",
             label="D2 = 1", density=True)
    ax3.set_xlabel("len(raw_sentence)  [chars]", color=mid, fontsize=10)
    ax3.set_ylabel("density", color=mid, fontsize=10)
    ax3.set_title("Length Distribution by D2 Flag", color="white", fontsize=11)
    ax3.legend(fontsize=9, labelcolor=mid, framealpha=0.3)
    ax3.tick_params(colors=mid)
    for sp in ax3.spines.values(): sp.set_edgecolor("#2a2f3f")

    # ── 4. Stats annotation panel ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#13161f")
    ax4.axis("off")

    rho, p   = stats.spearmanr(chars, density_raw)
    r,   pr  = stats.pearsonr(chars, density_raw)
    rpb, ppb = stats.pointbiserialr(flag, chars)
    n   = len(chars)
    n1  = int(flag.sum())
    n0  = n - n1
    m0  = chars[flag == 0].mean()
    m1  = chars[flag == 1].mean()

    summary = (
        f"Set C: D2 Density x Length\n"
        f"{'─'*34}\n"
        f"N (total rows)       {n:>10,}\n"
        f"N (D2=1)             {n1:>10,}  ({100*n1/n:.1f}%)\n"
        f"N (D2=0)             {n0:>10,}  ({100*n0/n:.1f}%)\n\n"
        f"Spearman rho         {rho:>+10.4f}\n"
        f"  p-value            {fmt_p(p):>10}\n\n"
        f"Pearson r            {r:>+10.4f}\n"
        f"  p-value            {fmt_p(pr):>10}\n\n"
        f"Point-biserial rpb   {rpb:>+10.4f}\n"
        f"  p-value            {fmt_p(ppb):>10}\n\n"
        f"Mean chars | D2=0    {m0:>10.0f}\n"
        f"Mean chars | D2=1    {m1:>10.0f}\n"
        f"Delta mean chars     {m1-m0:>+10.0f}\n"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10.5, verticalalignment="top", fontfamily="monospace",
             color="#e8edf8",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1f2e", alpha=0.85))

    fig.suptitle(
        "Set C  —  D2 Judicial-Content Density as a Length Proxy",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    out_path = out_dir / "set_c_d2_length_correlation.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ok] Plot saved -> {out_path}")
    return out_path


def write_results_md(sr, pb):
    rho = sr["spearman_rho"]
    if abs(rho) >= 0.40:
        verdict = (
            "**Strong positive correlation confirmed.** "
            "The D2 gate is substantially a length gate. "
            "Claim 4 (threshold-clearing) mechanistically predicts Claim 3 (length dominance); "
            "they are not independent findings."
        )
    elif abs(rho) >= 0.20:
        verdict = (
            "**Moderate correlation.** "
            "Length contributes meaningfully to D2 passage but is not the sole driver."
        )
    else:
        verdict = (
            "**Weak / negligible correlation.** "
            "D2 is not acting primarily as a length proxy at the population level."
        )

    md = f"""# Set C — D2 Judicial-Content Density x Candidate Length

## Motivation

D2 scores judicial-content density by counting interpretive and procedural
keyword hits. Longer passages mechanically contain more opportunities for such
hits. If D2 correlates with `len(raw_sentence)`, the threshold gate is partly a
**length gate**, and Claim 4 (threshold-clearing) predicts Claim 3 (length
dominance) — converting a reported oddity into an explained mechanism.

---

## Primary Result: Spearman rho  (d2 raw hit count x candidate chars)

| Statistic | Value |
|-----------|-------|
| N | {sr['n']:,} |
| **Spearman rho** | **{sr['spearman_rho']:+.4f}** |
| Spearman p-value | {fmt_p(sr['spearman_p'])} |
| Pearson r | {sr['pearson_r']:+.4f} |
| Pearson p-value | {fmt_p(sr['pearson_p'])} |

---

## Secondary Result: Point-Biserial rpb  (D2 flag x candidate chars)

| Statistic | Value |
|-----------|-------|
| N | {pb['n']:,} |
| **Point-biserial rpb** | **{pb['rpb']:+.4f}** |
| p-value | {fmt_p(pb['p'])} |
| Mean chars, D2=0 | {pb['mean_chars_d2_0']:,.0f} |
| Mean chars, D2=1 | {pb['mean_chars_d2_1']:,.0f} |
| Delta mean chars | {pb['mean_chars_d2_1'] - pb['mean_chars_d2_0']:+,.0f} |

---

## Interpretation

{verdict}

### Mechanism

D2 counts keyword hits (`d2_interpretive_hits + d2_procedural_hits`) without
normalising by length. A passage with twice as many characters has approximately
twice as many opportunities to trigger a keyword match. The binary
`d2_judicial_content` flag fires when `(interpretive_hits >= 1) OR
(procedural_hits >= 1)`, which means the gate trips more readily as length grows.

### Implication for Claims

- **Claim 3** (length dominance): chosen/rejected passages differ in length.
- **Claim 4** (threshold-clearing): longer candidates pass the D2 gate more often.
- If rho is substantial, Claim 4 is the *mechanism* underlying Claim 3 — not a
  separate oddity. The two claims collapse into a single causal chain:

      length -> D2 hits -> D2 gate passes -> triplet retained -> chosen side longer

---

*Script*: `evals/set_c_d2_length_correlation.py`
*Data*: `datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/clean_passing_triplets_deduped.json`
"""
    out = Path(__file__).resolve().parent / "set_c_results.md"
    out.write_text(md, encoding="utf-8")
    print(f"[ok] Results written -> {out}")


def main():
    print(f"Loading data from:\n  {DATA_PATH}")
    records = load_data(DATA_PATH)
    print(f"Loaded {len(records):,} records")

    chars, density_raw, flag, skipped = extract_vectors(records)
    print(f"Extracted {len(chars):,} rows  (skipped {skipped} with missing raw_sentence)")

    print("\n-- Spearman rho: d2 raw hit count x candidate chars --")
    rho, p = stats.spearmanr(chars, density_raw)
    r,  pr = stats.pearsonr(chars, density_raw)
    print(f"   rho = {rho:+.6f}   p = {fmt_p(p)}")
    print(f"   r   = {r:+.6f}   p = {fmt_p(pr)}")

    print("\n-- Point-biserial rpb: d2 flag x candidate chars --")
    rpb, ppb = stats.pointbiserialr(flag, chars)
    m0 = chars[flag == 0].mean()
    m1 = chars[flag == 1].mean()
    print(f"   rpb = {rpb:+.6f}   p = {fmt_p(ppb)}")
    print(f"   mean chars | D2=0 = {m0:,.0f}")
    print(f"   mean chars | D2=1 = {m1:,.0f}")
    print(f"   Delta            = {m1-m0:+,.0f} chars")

    make_plots(chars, density_raw, flag, OUT_DIR)

    sr = {"n": len(chars), "spearman_rho": rho, "spearman_p": p, "pearson_r": r, "pearson_p": pr}
    pb = {"n": len(chars), "rpb": rpb, "p": ppb, "mean_chars_d2_0": m0, "mean_chars_d2_1": m1}
    write_results_md(sr, pb)

    print("\n-- VERDICT --")
    if abs(rho) >= 0.40:
        print("  STRONG correlation. D2 gate is substantially a length gate.")
        print("  Claim 4 -> Claim 3: causal chain confirmed.")
    elif abs(rho) >= 0.20:
        print("  MODERATE correlation. Length is a meaningful contributor.")
    else:
        print("  WEAK correlation. D2 is not primarily a length proxy.")


if __name__ == "__main__":
    main()
