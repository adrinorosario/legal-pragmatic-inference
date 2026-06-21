"""
Bi-encoder score distribution analysis across all datapoints
in distributed_threshold_results.json.

Focuses exclusively on the numeric bi_encoder_score values —
ignores raw_sentence / cuad_anchor text.
"""

import json
import statistics
from collections import Counter

# ── Load ────────────────────────────────────────────────────────────
with open("./run_5/distributed_threshold_results_version5.json", "r") as f:
    data = json.load(f)  # top-level dict: threshold_key -> list[dict]

# ── Collect every bi_encoder_score ──────────────────────────────────
scores = []
per_threshold_scores: dict[str, list[float]] = {}

for threshold_key, entries in data.items():
    # Skip non-numeric keys (e.g. "reasoning_sentences")
    try:
        float(threshold_key)
    except ValueError:
        continue
    bucket_scores = []
    for entry in entries:
        s = entry["bi_encoder_score"]
        scores.append(s)
        bucket_scores.append(s)
    per_threshold_scores[threshold_key] = bucket_scores

scores.sort()

# ── Global statistics ───────────────────────────────────────────────
n = len(scores)
mean = statistics.mean(scores)
median = statistics.median(scores)
stdev = statistics.stdev(scores) if n > 1 else 0.0
minimum = scores[0]
maximum = scores[-1]
q1 = scores[int(n * 0.25)]
q3 = scores[int(n * 0.75)]
iqr = q3 - q1
p5 = scores[int(n * 0.05)]
p10 = scores[int(n * 0.10)]
p90 = scores[int(n * 0.90)]
p95 = scores[int(n * 0.95)]

print("=" * 70)
print("BI-ENCODER SCORE DISTRIBUTION (Version 5) — GLOBAL SUMMARY")
print("=" * 70)
print(f"  Total datapoints:   {n}")
print(f"  Mean:               {mean:.6f}")
print(f"  Median:             {median:.6f}")
print(f"  Std Dev:            {stdev:.6f}")
print(f"  Min:                {minimum:.6f}")
print(f"  Max:                {maximum:.6f}")
print(f"  Range:              {maximum - minimum:.6f}")
print(f"  Q1 (25th pctl):     {q1:.6f}")
print(f"  Q3 (75th pctl):     {q3:.6f}")
print(f"  IQR:                {iqr:.6f}")
print(f"  5th percentile:     {p5:.6f}")
print(f"  10th percentile:    {p10:.6f}")
print(f"  90th percentile:    {p90:.6f}")
print(f"  95th percentile:    {p95:.6f}")

# ── Histogram (text-based) ──────────────────────────────────────────
print("\n" + "=" * 70)
print("HISTOGRAM — bi_encoder_score distribution")
print("=" * 70)

# Fixed bins from 0.0 to 1.0 in 0.05 increments
bin_width = 0.05
bins = {}
for s in scores:
    b = round(int(s / bin_width) * bin_width, 2)
    bins[b] = bins.get(b, 0) + 1

max_count = max(bins.values()) if bins else 1
for b in sorted(bins):
    bar_len = int(60 * bins[b] / max_count)
    pct = 100 * bins[b] / n
    print(f"  [{b:.2f}, {b + bin_width:.2f})  {bins[b]:>5}  ({pct:5.1f}%)  {'█' * bar_len}")

# ── Per-threshold bucket breakdown ──────────────────────────────────
print("\n" + "=" * 70)
print("PER-THRESHOLD BUCKET BREAKDOWN")
print("=" * 70)
print(f"  {'Threshold':<12} {'Count':>6} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
print("  " + "-" * 68)

for tk in sorted(per_threshold_scores, key=lambda x: float(x)):
    bs = per_threshold_scores[tk]
    bs_sorted = sorted(bs)
    bcount = len(bs)
    bmean = statistics.mean(bs)
    bmedian = statistics.median(bs)
    bmin = bs_sorted[0]
    bmax = bs_sorted[-1]
    bstd = statistics.stdev(bs) if bcount > 1 else 0.0
    print(f"  {tk:<12} {bcount:>6} {bmean:>10.6f} {bmedian:>10.6f} {bmin:>10.6f} {bmax:>10.6f} {bstd:>10.6f}")

# ── Clustering diagnostic ──────────────────────────────────────────
print("\n" + "=" * 70)
print("CLUSTERING DIAGNOSTIC")
print("=" * 70)

# What fraction of scores fall below key thresholds?
thresholds_to_check = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
print(f"  {'Cutoff':<10} {'Count Below':>12} {'% Below':>10}")
print("  " + "-" * 34)
for t in thresholds_to_check:
    count_below = sum(1 for s in scores if s < t)
    print(f"  {t:<10.2f} {count_below:>12} {100 * count_below / n:>9.1f}%")

# ── Suspiciously low clustering check ──────────────────────────────
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

pct_below_05 = 100 * sum(1 for s in scores if s < 0.5) / n
pct_below_04 = 100 * sum(1 for s in scores if s < 0.4) / n
pct_in_narrow = 100 * sum(1 for s in scores if 0.4 <= s < 0.5) / n

print(f"  • {pct_below_05:.1f}% of scores fall below 0.50")
print(f"  • {pct_below_04:.1f}% of scores fall below 0.40")
print(f"  • {pct_in_narrow:.1f}% of scores cluster in [0.40, 0.50)")
print(f"  • Score spread (range): {maximum - minimum:.6f}")
print(f"  • Coefficient of variation: {(stdev/mean)*100:.2f}%")

if mean < 0.5:
    print("  ⚠  Mean is below 0.50 — scores cluster suspiciously low.")
elif mean < 0.6:
    print("  ⚠  Mean is in [0.50, 0.60) — moderately low clustering.")
else:
    print("  ✓  Mean is ≥ 0.60 — no obvious low-clustering concern.")

if stdev < 0.10:
    print("  ⚠  Very low variance (σ < 0.10) — scores are tightly packed.")
elif stdev < 0.15:
    print("  ~  Moderate variance.")
else:
    print("  ✓  Healthy variance in score distribution.")