#!/usr/bin/env python3
"""
Figure: discrimination accuracy across automatic scorers, a length-only
baseline, and the single-rater blind read.

Marker shape encodes evaluation population, because the three groups are
NOT drawn from the same pool:
    circle   = held-out pool  (500 pairs, 19 anchors)
    square   = 984 training pairs
    diamond  = blind read     (30 presentations, 20 unique clauses)

Shaded band = the length-only prior, the bar any method must clear to
show it is reading anything.

Inputs (working directory):
    held_out_baseline_results.json
    evaluation_results.json
    dpo_dataset_revised.json
    blind_eval_binosh_1785755394293.json

Output:
    figures/fig_discrimination.pdf

Usage:
    python make_discrimination_figure.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ config

CLUSTERED_CI = True     # resample anchors/clauses, not individual pairs
N_BOOT       = 4000
SEED         = 0
FIGSIZE      = (6.9, 2.9)   # figure*; use (3.3, 3.0) for single column
OUTDIR       = "figures"
OUTFILE      = "fig_discrimination.pdf"

C_LEXICAL = "#D85A30"
C_DENSE   = "#888780"
C_LEARNED = "#1D9E75"
C_LENGTH  = "#7F77DD"
C_HUMAN   = "#185FA5"
C_CHANCE  = "#B4B2A9"
C_BAND    = "#EDEBE4"

MARK = {"heldout": "o", "train984": "s", "blind": "D"}

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- helpers

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (c - h) / d, (c + h) / d


def clustered_ci(dec, clusters, n_boot=N_BOOT):
    dec = np.asarray(dec).astype(int)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots[b] = dec[np.concatenate([idx[c] for c in pick])].mean()
    return np.percentile(boots, [2.5, 97.5])


def row(name, dec, clusters, colour, pop, n_clusters=None):
    dec = np.asarray(dec).astype(int)
    p, n = dec.mean(), len(dec)
    if CLUSTERED_CI and clusters is not None:
        lo, hi = clustered_ci(dec, clusters)
    else:
        lo, hi = wilson(int(dec.sum()), n)
    return dict(name=name, acc=p, lo=lo, hi=hi, n=n,
                colour=colour, pop=pop,
                n_clusters=n_clusters if n_clusters
                else (len(np.unique(clusters)) if clusters is not None else n))


# -------------------------------------------------------------------- load

rows = []

with open("held_out_baseline_results.json") as f:
    base = json.load(f)
pp = base["per_pair"]
anch = [p["anchor"] for p in pp]

for label, kc, kr, col in [
    ("TF-IDF cosine",    "tfidf_chosen",    "tfidf_rejected",    C_LEXICAL),
    ("Okapi BM25",       "bm25_chosen",     "bm25_rejected",     C_LEXICAL),
    ("Embedding cosine", "embgemma_chosen", "embgemma_rejected", C_DENSE),
    ("Cross-encoder",    "crossenc_chosen", "crossenc_rejected", C_DENSE),
]:
    rows.append(row(label, [p[kc] > p[kr] for p in pp], anch, col, "heldout"))

# DPO implicit reward — DIFFERENT 500-pair draw. Disclose in caption.
with open("evaluation_results.json") as f:
    ev = json.load(f)
rows.append(row("DPO implicit reward", [p["correct_norm"] for p in ev],
                [p["anchor"] for p in ev], C_LEARNED, "heldout"))

# Length-only baseline, held-out pool
lc = np.array([len(p["chosen"]) for p in pp])
lr = np.array([len(p["rejected"]) for p in pp])
rows.append(row("Length only (held-out)", lc > lr, anch, C_LENGTH, "heldout"))

# Length-only baseline, 984 training pairs
with open("../dpo_training_env/datasets/dpo_dataset_revised.json") as f:
    pool = json.load(f)
lc9 = np.array([len(x["chosen"]) for x in pool])
lr9 = np.array([len(x["rejected"]) for x in pool])
rows.append(row("Length only (984)", lc9 > lr9,
                [x["prompt"] for x in pool], C_LENGTH, "train984"))
LENGTH_PRIOR = (lc9 > lr9).mean()

# Human blind read
with open("blind_eval_binosh_1785755394293.json") as f:
    blind = json.load(f)
ans = blind["answers"]
rows.append(row("Human (all presentations)",
                [bool(a["isCorrect"]) for a in ans],
                [a["clause"] for a in ans], C_HUMAN, "blind"))

seen, uq = set(), []
for a in ans:
    if a["clause"] not in seen:
        seen.add(a["clause"])
        uq.append(a)
rows.append(row("Human (unique clauses)",
                [bool(a["isCorrect"]) for a in uq],
                [a["clause"] for a in uq], C_HUMAN, "blind"))


# -------------------------------------------------------------------- plot

rows.sort(key=lambda r: r["acc"])
fig, ax = plt.subplots(figsize=FIGSIZE)

ax.axvspan(LENGTH_PRIOR - 0.001, LENGTH_PRIOR + 0.001,
           color=C_LENGTH, alpha=0.25, zorder=0)
ax.axvline(0.5, color=C_CHANCE, ls="--", lw=0.8, zorder=1)

for i, r in enumerate(rows):
    ax.plot([r["lo"], r["hi"]], [i, i], color=r["colour"], lw=1.4,
            solid_capstyle="butt", zorder=2)
    ax.plot([r["acc"]], [i], MARK[r["pop"]], color=r["colour"],
            ms=5, zorder=3)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['name']}" for r in rows], fontsize=7.5)
ax.set_xlabel("Pairwise discrimination accuracy", fontsize=8)
ax.set_xlim(0.35, 0.95)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.tick_params(axis="x", labelsize=7)
ax.tick_params(axis="y", length=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color=C_BAND, lw=0.6, zorder=0)
ax.set_axisbelow(True)

handles = [
    Line2D([], [], marker="o", ls="", color="#444441", ms=5,
           label="Held-out pool (n=500)"),
    Line2D([], [], marker="s", ls="", color="#444441", ms=5,
           label="Training pairs (n=984)"),
    Line2D([], [], marker="D", ls="", color="#444441", ms=5,
           label="Blind read"),
    Line2D([], [], color=C_LENGTH, lw=4, alpha=0.35,
           label=f"Length prior ({LENGTH_PRIOR*100:.1f}%)"),
]
ax.legend(handles=handles, fontsize=6.5, frameon=False,
          loc="lower right", handlelength=1.4)

fig.tight_layout(pad=0.4)
os.makedirs(OUTDIR, exist_ok=True)
path = os.path.join(OUTDIR, OUTFILE)
fig.savefig(path, bbox_inches="tight")
print(f"wrote {path}\n")

kind = "anchor/clause-clustered bootstrap" if CLUSTERED_CI else "Wilson"
print(f"CI method: {kind}")
print(f"{'row':28s} {'n':>5s} {'clus':>5s} {'acc':>7s} {'95% CI':>18s}")
for r in sorted(rows, key=lambda r: -r["acc"]):
    print(f"{r['name']:28s} {r['n']:5d} {r['n_clusters']:5d} "
          f"{r['acc']:7.3f}   [{r['lo']:.3f}, {r['hi']:.3f}]")