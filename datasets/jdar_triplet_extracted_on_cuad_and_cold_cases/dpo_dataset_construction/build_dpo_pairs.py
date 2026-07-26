"""
DPO chosen/rejected pairing script for JDAR.

Restricted to the 4 validated categories (Cap On Liability, Third Party
Beneficiary, Competitive Restriction Exception, Non-Compete) per the
dataset audit — IP Ownership Assignment and Covenant Not To Sue are
excluded (confirmed unsound on manual review), as is the unaudited
long tail.

Failing pool definition (finalized):
  A failing/negative candidate is any triplet where D1==1 (the anchor is a
  genuine candidate — real pragmatic vagueness present, same class of
  anchor as the chosen examples) AND total_score < 2 (the reasoning
  window offered is too weak on judicial density and/or subject
  coherence to count as valid interpretive reasoning over that clause).

  This is intentionally coarse for the conference paper: it does not yet
  distinguish *why* the window failed (low D2 judicial-content density
  vs. D3 subject-mismatch vs. both). Each pair is tagged with
  `failure_reason` anyway (informational, not used for selection) so
  this can be filtered/refined without re-deriving anything from raw
  data — deferred refinement for journal-stage work.

Pairing rule (within-category only, per prior decision):
  1. PREFERRED: same-anchor negative — a failing triplet (as defined
     above) that shares the exact same cuad_anchor as a passing triplet,
     but a different raw_sentence. Tightest possible negative: same
     legal question, weaker reasoning window.
  2. FALLBACK: same-category, different-anchor negative — used only
     when a passing triplet's anchor has no failing counterpart.
     Looser signal (model learns category-level discrimination, not
     anchor-level), used only where necessary.

Every emitted pair is tagged with pair_type so the split between
same-anchor and same-category pairs can be reported directly in the
paper, not buried. Every pair is also tagged with failure_reason
(informational only for this pass — see above).

Inputs expected:
  - PASSING_PATH: cleaned/deduped passing triplets (both fixes applied)
  - FAILING_PATH: raw scored triplets (any/all runs) filtered at load
    time in this script to D1==1 AND total_score < 2. Point this at
    your consolidated raw scored output (pre-passing-filter), NOT a
    pre-filtered failing-only file, so the D1==1/total<2 filter is
    applied consistently here rather than assumed upstream.
"""

import json
import random
from collections import defaultdict

PASSING_PATH = "clean_passing_triplets_deduped.json"
FAILING_SOURCE_PATH = "concatenated_runs_json_triplets.json"
OUTPUT_PATH = "dpo_pairs_2.jsonl"

VALIDATED_CATEGORIES = {
    "Cap On Liability",
    "Third Party Beneficiary",
    "Non-Compete",
}
# Competitive Restriction Exception removed after DPO-pair-level manual
# audit (20-sample review) found its category gate matches on generic
# boilerplate ("notwithstanding," "for clarity") rather than
# competitive-restriction-specific substance — same failure pattern as
# club-misty (Sec. 3) and Covenant Not To Sue (Sec. 6.4), confirmed at
# the pairing stage. See audit notes Section 13 for detail.

SEED = 42


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def derive_failure_reason(d):
    """Informational tag only — not used for pair selection in this pass."""
    d2 = d.get("d2_judicial_content", d.get("d2_score"))
    d3 = d.get("d3_semantic_coherence", d.get("d3_score"))
    if d2 == 0 and d3 == 0:
        return "low_density_and_subject_mismatch"
    if d2 == 0:
        return "low_judicial_density"
    if d3 == 0:
        return "subject_mismatch"
    return "unknown"


def get_failing_pool(raw_scored_rows, validated_categories):
    """
    Filters a raw (unfiltered) scored triplet list down to the negative
    candidate pool: D1==1 AND total_score < 2, restricted to the
    validated categories. This filter is applied here, explicitly, so
    the failing-pool definition lives in one place and isn't assumed
    upstream.
    """
    pool = []
    for d in raw_scored_rows:
        if d.get("cuad_category") not in validated_categories:
            continue
        d1 = d.get("d1_anchor_vagueness", d.get("d1_score"))
        total = d.get("total_score")
        if d1 == 1 and total is not None and total < 2:
            d["failure_reason"] = derive_failure_reason(d)
            pool.append(d)
    return pool


def build_pairs(passing, failing_pool):
    """
    passing: list of dicts, each a passing triplet (both fixes applied)
    failing_pool: list of dicts, pre-filtered via get_failing_pool() to
                  D1==1 AND total_score < 2, restricted to validated
                  categories, each already tagged with failure_reason.
    """
    random.seed(SEED)

    passing = [d for d in passing if d["cuad_category"] in VALIDATED_CATEGORIES]

    # Index failing pool by (category, anchor) for same-anchor lookup,
    # and by category alone for the fallback pool.
    failing_by_anchor = defaultdict(list)
    failing_by_category = defaultdict(list)
    for d in failing_pool:
        key = (d["cuad_category"], d["cuad_anchor"])
        failing_by_anchor[key].append(d)
        failing_by_category[d["cuad_category"]].append(d)

    pairs = []
    unpaired = []
    stats = defaultdict(lambda: defaultdict(int))

    for p in passing:
        cat = p["cuad_category"]
        anchor = p["cuad_anchor"]
        key = (cat, anchor)

        same_anchor_candidates = [
            f for f in failing_by_anchor.get(key, [])
            if f["raw_sentence"] != p["raw_sentence"]
        ]

        if same_anchor_candidates:
            negative = random.choice(same_anchor_candidates)
            pair_type = "same_anchor"
        else:
            # Fallback: same category, different anchor, excluding any
            # accidental match to this exact passing example.
            fallback_pool = [
                f for f in failing_by_category.get(cat, [])
                if f["cuad_anchor"] != anchor
            ]
            if fallback_pool:
                negative = random.choice(fallback_pool)
                pair_type = "same_category_diff_anchor"
            else:
                unpaired.append(p)
                stats[cat]["unpaired"] += 1
                continue

        pairs.append({
            "cuad_category": cat,
            "prompt_anchor": anchor,
            "chosen": p["raw_sentence"],
            "chosen_contract_id": p.get("contract_id"),
            "chosen_source_run": p.get("_source_run"),
            "chosen_total_score": p.get("total_score"),
            "rejected": negative["raw_sentence"],
            "rejected_contract_id": negative.get("contract_id"),
            "rejected_source_run": negative.get("_source_run"),
            "rejected_total_score": negative.get("total_score"),
            "failure_reason": negative.get("failure_reason", "unknown"),
            "pair_type": pair_type,
        })
        stats[cat][pair_type] += 1

    return pairs, unpaired, stats


def main():
    if FAILING_SOURCE_PATH is None:
        print("FAILING_SOURCE_PATH is not set. This script needs raw scored "
              "triplets (all runs, unfiltered) so it can derive the D1==1 / "
              "total_score<2 negative pool itself. Set FAILING_SOURCE_PATH "
              "at the top of this file and rerun.")
        return

    passing = load_json(PASSING_PATH)
    raw_scored = load_json(FAILING_SOURCE_PATH)
    failing_pool = get_failing_pool(raw_scored, VALIDATED_CATEGORIES)

    print(f"Derived failing/negative pool: {len(failing_pool)} triplets "
          f"(D1==1, total_score<2, validated categories only)")
    reason_counts = defaultdict(int)
    for d in failing_pool:
        reason_counts[d["failure_reason"]] += 1
    print("  Failure reason breakdown (informational only, not used for "
          "selection this pass):")
    for reason, count in sorted(reason_counts.items()):
        print(f"    {reason}: {count}")
    print()

    pairs, unpaired, stats = build_pairs(passing, failing_pool)

    with open(OUTPUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Total pairs written: {len(pairs)}")
    print(f"Unpaired passing triplets (no valid negative found): {len(unpaired)}")
    print()
    print(f"{'Category':<35} {'same_anchor':>12} {'same_cat_diff_anchor':>22} {'unpaired':>10}")
    for cat in sorted(VALIDATED_CATEGORIES):
        s = stats[cat]
        print(f"{cat:<35} {s.get('same_anchor', 0):>12} "
              f"{s.get('same_category_diff_anchor', 0):>22} {s.get('unpaired', 0):>10}")

    print()
    print(f"Output written to: {OUTPUT_PATH}")
    print("Report the same_anchor vs same_category_diff_anchor split in the "
          "paper's data/methods section — this is the pair_type field on "
          "every row of the output.")


if __name__ == "__main__":
    main()
