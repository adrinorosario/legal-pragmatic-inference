#!/usr/bin/env python3
"""
Blind read analysis for the JDAR paper.

Computes every statistic recoverable from the rater's response file and
emits both a console report and LaTeX table bodies ready to paste.

Inputs (working directory):
        rater responses json file
    dpo_dataset_revised.json                 the 984 triplets (for the
                                             population-level length prior)

Outputs:
    tables/blind_read_overall.tex
    tables/blind_read_per_category.tex
    console report

Usage:
    python analyse_blind_read.py
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

BLIND_FILE = "blind_eval_binosh_1785755394293.json"
POOL_FILE = "../dpo_training_env/datasets/dpo_dataset_revised.json"
OUTDIR = "tables"


# ------------------------------------------------------------------ stats

def wilson(k, n, z=1.96):
    """Wilson score interval. Returns (lo, hi). Handles n=0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (c - h) / d, (c + h) / d


def binom_p(k, n, p0=0.5):
    """Two-sided exact binomial test against a null rate p0."""
    if n == 0:
        return float("nan")
    return stats.binomtest(k, n, p0, alternative="two-sided").pvalue


def fmt_ci(lo, hi):
    return f"[{lo*100:.1f}, {hi*100:.1f}]"


# ------------------------------------------------------------------ load

with open(BLIND_FILE) as f:
    blind = json.load(f)

answers = blind["answers"]
n_total = len(answers)

correct = np.array([bool(a["isCorrect"]) for a in answers])
category = np.array([a["category"] for a in answers])
clause = np.array([a["clause"] for a in answers])
your_ans = np.array([a["yourAnswer"] for a in answers])
key_ans = np.array([a["correctAnswer"] for a in answers])

# unique-clause subset: first occurrence of each clause
seen = set()
first_mask = np.zeros(n_total, dtype=bool)
for i, c in enumerate(clause):
    if c not in seen:
        seen.add(c)
        first_mask[i] = True

# repeated clauses: those appearing more than once
counts = Counter(clause)
repeat_clauses = [c for c, k in counts.items() if k > 1]


# ------------------------------------------------- population length prior

length_prior = None
if os.path.exists(POOL_FILE):
    with open(POOL_FILE) as f:
        pool = json.load(f)
    lc = np.array([len(x["chosen"]) for x in pool])
    lr = np.array([len(x["rejected"]) for x in pool])
    k = int((lc > lr).sum())
    n = len(pool)
    lo, hi = wilson(k, n)
    length_prior = dict(k=k, n=n, acc=k / n, lo=lo, hi=hi)


# ------------------------------------------------------------------ report

print("=" * 66)
print("BLIND READ ANALYSIS")
print("=" * 66)
print(f"evaluator        : {blind['evaluator']}")
print(f"started          : {blind['started_at']}")
print(f"completed        : {blind['completed_at']}")
print(f"questions        : {n_total}")
print(f"unique clauses   : {first_mask.sum()}")
print(f"repeated clauses : {len(repeat_clauses)} "
      f"(presented {n_total - first_mask.sum()} extra times)")
print()

# --- overall
k = int(correct.sum())
lo, hi = wilson(k, n_total)
p = binom_p(k, n_total, 0.5)
print("--- OVERALL (all presentations) ---")
print(f"  {k}/{n_total} = {k/n_total*100:.1f}%   95% CI {fmt_ci(lo, hi)}   "
      f"p(vs 50%) = {p:.4f}")

if length_prior:
    p_len = binom_p(k, n_total, length_prior["acc"])
    print(f"  p vs length prior ({length_prior['acc']*100:.1f}%) = {p_len:.4f}")
print()

# --- unique clauses only
ku = int(correct[first_mask].sum())
nu = int(first_mask.sum())
lou, hiu = wilson(ku, nu)
print("--- UNIQUE CLAUSES ONLY (first presentation each) ---")
print(f"  {ku}/{nu} = {ku/nu*100:.1f}%   95% CI {fmt_ci(lou, hiu)}   "
      f"p(vs 50%) = {binom_p(ku, nu, 0.5):.4f}")
print()

# --- per category
print("--- PER CATEGORY ---")
print(f"  {'category':26s} {'n':>3s} {'correct':>8s} {'acc':>7s} "
      f"{'95% CI':>16s} {'p':>8s}")
cat_rows = []
for c in sorted(set(category)):
    m = category == c
    kc, nc = int(correct[m].sum()), int(m.sum())
    l, h = wilson(kc, nc)
    pc = binom_p(kc, nc, 0.5)
    cat_rows.append((c, nc, kc, kc / nc, l, h, pc))
    print(f"  {c:26s} {nc:3d} {kc:8d} {kc/nc*100:6.1f}% "
          f"{fmt_ci(l, h):>16s} {pc:8.4f}")
print()

# --- intra-rater consistency on repeated clauses
print("--- INTRA-RATER CONSISTENCY (repeated clauses) ---")
if repeat_clauses:
    by_clause = defaultdict(list)
    for i, c in enumerate(clause):
        by_clause[c].append(i)
    consistent = 0
    considered = 0
    detail = []
    for c in repeat_clauses:
        idxs = by_clause[c]
        # a presentation is "same choice" if the rater's pick maps to the
        # same underlying option; recover via (yourAnswer == correctAnswer)
        picks = [bool(correct[i]) for i in idxs]
        agree = len(set(picks)) == 1
        consistent += int(agree)
        considered += 1
        detail.append((c[:52], len(idxs), picks, agree))
    print(f"  clauses shown more than once : {considered}")
    print(f"  same outcome every time      : {consistent}/{considered} "
          f"({consistent/considered*100:.0f}%)")
    print()
    for c, k_, picks, agree in detail:
        flag = "consistent" if agree else "INCONSISTENT"
        print(f"    [{flag:12s}] x{k_}  {picks}  {c}...")
else:
    print("  no repeated clauses found")
print()

# --- answer-position check (order effect)
print("--- POSITION / ORDER EFFECT ---")
print(f"  key was 'A' in {int((key_ans=='A').sum())}/{n_total} items")
print(f"  rater chose 'A' in {int((your_ans=='A').sum())}/{n_total} items")
kA = int((your_ans == "A").sum())
print(f"  p(rater A-rate vs 50%) = {binom_p(kA, n_total, 0.5):.4f}")
acc_when_A = correct[key_ans == "A"].mean()
acc_when_B = correct[key_ans == "B"].mean()
print(f"  accuracy when key='A' : {acc_when_A*100:.1f}% "
      f"(n={int((key_ans=='A').sum())})")
print(f"  accuracy when key='B' : {acc_when_B*100:.1f}% "
      f"(n={int((key_ans=='B').sum())})")
tbl = [[int(((key_ans == "A") & correct).sum()),
        int(((key_ans == "A") & ~correct).sum())],
       [int(((key_ans == "B") & correct).sum()),
        int(((key_ans == "B") & ~correct).sum())]]
print(f"  Fisher exact p = {stats.fisher_exact(tbl)[1]:.4f}")
print()

# --- length prior
if length_prior:
    lp = length_prior
    print("--- POPULATION LENGTH PRIOR (984 pairs) ---")
    print(f"  chosen longer than rejected in {lp['k']}/{lp['n']} = "
          f"{lp['acc']*100:.1f}%  95% CI {fmt_ci(lp['lo'], lp['hi'])}")
    print("  NOTE: this is a property of the pair population, not a claim")
    print("  about the rater. Per-item option lengths are not recoverable")
    print("  from the response file (option texts were not stored).")
    print()

print("--- NOT RECOVERABLE FROM THIS FILE ---")
print("  * which underlying passage was shown as option A vs option B")
print("  * per-item character/word lengths of the two options")
print("  => a per-item length control requires re-exporting the 30")
print("     presented pairs with their option texts.")
print()


# ------------------------------------------------------------ latex tables

os.makedirs(OUTDIR, exist_ok=True)

overall = []
overall.append(("All presentations", n_total, k, k / n_total, lo, hi))
overall.append(("Unique clauses", nu, ku, ku / nu, lou, hiu))

with open(os.path.join(OUTDIR, "blind_read_overall.tex"), "w") as f:
    f.write("% Blind read: overall accuracy. Wilson 95% intervals.\n")
    f.write("\\begin{tabular}{lccc}\n\\toprule\n")
    f.write("Subset & $n$ & Accuracy & 95\\% CI \\\\\n\\midrule\n")
    for name, nn, kk, acc, l, h in overall:
        f.write(f"{name} & {nn} & {kk}/{nn} ({acc*100:.1f}\\%) "
                f"& [{l*100:.1f}, {h*100:.1f}] \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUTDIR, "blind_read_per_category.tex"), "w") as f:
    f.write("% Blind read per category. Wilson 95% intervals.\n")
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
    f.write("Category & $n$ & Correct & Accuracy & 95\\% CI \\\\\n\\midrule\n")
    for c, nc, kc, acc, l, h, _pc in cat_rows:
        f.write(f"{c} & {nc} & {kc} & {acc*100:.1f}\\% "
                f"& [{l*100:.1f}, {h*100:.1f}] \\\\\n")
    f.write("\\midrule\n")
    f.write(f"Overall & {n_total} & {k} & {k/n_total*100:.1f}\\% "
            f"& [{lo*100:.1f}, {hi*100:.1f}] \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print(f"wrote {OUTDIR}/blind_read_overall.tex")
print(f"wrote {OUTDIR}/blind_read_per_category.tex")