"""
Held-out negative pairs for JDAR DPO evaluation.

build_dpo_pairs.py's same-anchor branch draws exactly one random
negative per chosen (passing) triplet via random.choice() over that
anchor's failing pool, then discards the rest. Where more than one
failing candidate shared the anchor, those leftovers are eligible,
same-anchor, same-pipeline negatives that were never selected into
dpo_pairs_2.jsonl and so were never seen during DPO training.

To know exactly which negative build_dpo_pairs.py used (and therefore
which ones are leftover), this script re-runs its selection process
verbatim: same inputs, same SEED, same iteration order, and the same
random.choice() call in the same branch for every passing triplet
(including the no-op draw in the fallback branch) so the random state
stays in lockstep with the original run and the "used" pick matches
dpo_pairs_2.jsonl exactly.

Anchors with only one same-anchor failing candidate (nothing left
over) and passing triplets that fell back to a same-category/
different-anchor negative (no same-anchor pool at all) contribute
nothing here.

Inputs: identical to build_dpo_pairs.py (PASSING_PATH, FAILING_SOURCE_PATH).
Output: held_out_pairs.jsonl, written next to dpo_pairs_2.jsonl.
"""

import json
import random
from collections import defaultdict

from build_dpo_pairs import (
    PASSING_PATH,
    FAILING_SOURCE_PATH,
    VALIDATED_CATEGORIES,
    SEED,
    load_json,
    get_failing_pool,
)

OUTPUT_PATH = "held_out_pairs.jsonl"


def build_held_out_pairs(passing, failing_pool):
    random.seed(SEED)  # must match build_dpo_pairs.build_pairs() exactly

    passing = [d for d in passing if d["cuad_category"] in VALIDATED_CATEGORIES]

    failing_by_anchor = defaultdict(list)
    failing_by_category = defaultdict(list)
    for d in failing_pool:
        key = (d["cuad_category"], d["cuad_anchor"])
        failing_by_anchor[key].append(d)
        failing_by_category[d["cuad_category"]].append(d)

    held_out = []

    for p in passing:
        cat = p["cuad_category"]
        anchor = p["cuad_anchor"]
        key = (cat, anchor)

        same_anchor_candidates = [
            f for f in failing_by_anchor.get(key, [])
            if f["raw_sentence"] != p["raw_sentence"]
        ]

        if same_anchor_candidates:
            # Same draw build_dpo_pairs.py made for this passing triplet.
            used = random.choice(same_anchor_candidates)
            leftovers = [f for f in same_anchor_candidates if f is not used]
            for negative in leftovers:
                held_out.append({
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
                    "pair_type": "held_out_same_anchor",
                })
        else:
            # build_dpo_pairs.py draws once from the fallback pool here too;
            # replicate the draw (result unused) to keep random state in sync.
            fallback_pool = [
                f for f in failing_by_category.get(cat, [])
                if f["cuad_anchor"] != anchor
            ]
            if fallback_pool:
                random.choice(fallback_pool)

    return held_out


def main():
    passing = load_json(PASSING_PATH)
    raw_scored = load_json(FAILING_SOURCE_PATH)
    failing_pool = get_failing_pool(raw_scored, VALIDATED_CATEGORIES)

    held_out = build_held_out_pairs(passing, failing_pool)

    with open(OUTPUT_PATH, "w") as f:
        for pair in held_out:
            f.write(json.dumps(pair) + "\n")

    print(f"Held-out pairs written: {len(held_out)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
