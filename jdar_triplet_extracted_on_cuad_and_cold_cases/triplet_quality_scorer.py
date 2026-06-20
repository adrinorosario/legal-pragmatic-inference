"""
Triplet Quality Scorer — 3-Dimension Annotation Artifact
=========================================================

Scores every triplet in distributed_threshold_results.json across three
quality dimensions:

    D1  Anchor Vagueness
        Does the CUAD anchor contain at least one term from the 206-term
        vagueness seed set?  Binary 0/1.

    D2  Judicial Interpretive Content
        Does the raw_sentence window contain genuine judicial reasoning
        about contractual interpretation (not just procedure or facts)?
        Binary 0/1.

    D3  Semantic Coherence
        Does the vagueness *category* of the anchor match the interpretive
        territory of the raw_sentence?  Binary 0/1.

A triplet's total score is 0–3.  Pass threshold: ≥ 2.

Outputs:
    scored_triplets.json          — full scored dataset (all rows)
    scored_triplets_sample.json   — scored 50-row random sample for manual review
    scoring_summary.txt           — distribution statistics printed to console + file

Usage:
    python triplet_quality_scorer.py                       # score all, sample 50
    python triplet_quality_scorer.py --sample-size 100     # score all, sample 100
    python triplet_quality_scorer.py --full-only            # score all, no sample file
"""

import json
import re
import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# VAGUENESS SEED SET — expanded from notebook Cell 20
# 9 categories + targeted gap-fills from scored sample audit
# ═══════════════════════════════════════════════════════════════════════════

VAGUENESS_CATEGORIES = {
    "effort": [
        "reasonable efforts", "best efforts", "commercially reasonable",
        "due diligence", "reasonable care", "good faith", "workmanlike manner",
        "best practices", "reasonable endeavours", "all reasonable steps",
        "every reasonable effort", "diligent efforts", "reasonable commercial efforts",
        "utmost care", "exercise of judgment", "commercially practicable",
        "economically reasonable", "technically feasible",
        "consistent with good industry practice", "as would a prudent operator",
        "acting reasonably", "using its discretion",
    ],
    "time": [
        "promptly", "in a timely manner", "as soon as practicable",
        "without undue delay", "for a reasonable period",
        "termination of this Agreement", "from time to time", "periodic",
        "duration", "seasonable", "business hours", "within a reasonable time",
        "with all due speed", "expeditiously", "at the earliest opportunity",
        "without unnecessary delay", "within a commercially reasonable period",
        "in due course", "forthwith", "in due time", "on a timely basis",
        "in the near term", "shortly after", "when practicable",
        "upon reasonable notice", "reasonable notice period",
    ],
    "scope": [
        "material", "substantial", "limited", "relevant", "related", "generally",
        "appropriate", "similar", "de minimis", "significant", "incidental",
        "including but not limited to", "inter alia", "and/or",
        "save as otherwise provided", "appreciable", "meaningful", "non-trivial",
        "measurable", "proportionate", "commensurate", "reasonably proportionate",
        "unduly burdensome", "reasonably necessary", "to the extent practicable",
        "to a reasonable extent", "without limitation", "as applicable",
        "where relevant", "as appropriate", "to the extent required",
        # ── gap-fills from scored sample audit ──
        "substantially all",
        "satisfactory to",
        "in form and manner satisfactory",
        "reasonable satisfaction",
    ],
    "harm": [
        "material adverse effect", "material breach", "material adverse change",
        "material adverse impact", "material adverse consequence",
        "materially and adversely", "substantial impairment", "material disruption",
        "material deviation", "materially prejudice", "disproportionate impact",
        "unreasonable hardship", "undue prejudice", "undue harm", "undue risk",
    ],
    "necessity": [
        "necessary", "sole discretion", "need to know", "confidential nature",
        "adequate", "satisfactory", "proper", "intended purpose",
        "not to be unreasonably withheld", "mutual satisfaction", "at its option",
        "consultation", "absolute discretion", "unfettered discretion",
        "not to be unreasonably delayed", "not to be unreasonably conditioned",
        "without arbitrary restriction", "reasonably required",
        "reasonably requested", "if deemed appropriate", "as deemed necessary",
        "in its reasonable opinion", "acting in good faith",
        "in its reasonable judgment", "as it sees fit", "as directed",
        # ── gap-fill from scored sample audit ──
        "prior written consent",
    ],
    "industry_norms": [
        "customary", "ordinary course of business", "industry standard",
        "standard practice", "normally", "comparable", "acceptable",
        "conventional", "fit for purpose", "first-class condition",
        "commercially sensitive", "prevailing market practice",
        "generally accepted practice", "market standard",
        "accepted industry norms", "standard market terms",
        "customary market conditions", "in accordance with accepted methods",
        "consistent with past practice", "as is customary",
        "in accordance with best available techniques",
        "reasonable engineering standards", "professionally acceptable",
        "to a professional standard", "of merchantable quality",
        "of satisfactory quality",
    ],
    "knowledge": [
        "foreseeable", "contemplated", "intended", "anticipated", "applicable",
        "knowledge", "directly or indirectly", "disclosed in confidence",
        "all copies", "survive", "mutual agreement", "substantially similar",
        "reasonable expectations", "actual knowledge", "constructive knowledge",
        "reasonably should have known", "to the best of its knowledge",
        "as far as it is aware", "reasonably foreseeable",
        "unforeseen circumstances", "unanticipated events",
        "beyond reasonable expectation", "reasonable belief", "bona fide belief",
        "reasonable grounds", "having regard to all circumstances",
    ],
    "confidentiality": [
        "proprietary information", "non-public information",
        "sensitive business information", "trade secrets",
        "sufficiently confidential", "reasonably considered confidential",
        "maintained in confidence", "treated as confidential",
        "in accordance with confidentiality obligations",
    ],
    "financial": [
        "commercially attractive", "economically viable",
        "commercially justifiable", "at a reasonable price", "fair market value",
        "arm's length", "at prevailing rates", "on reasonable commercial terms",
        "on competitive terms", "at a rate reflecting market conditions",
        "at cost", "without unreasonable mark-up", "reasonable compensation",
        "reasonable fees",
    ],
    "survival": [
        "notwithstanding the foregoing", "without prejudice to",
        "subject to the foregoing", "except as otherwise agreed",
        "unless otherwise specified", "where not inconsistent", "insofar as",
        "to the fullest extent permitted by law", "as may be amended",
        "as modified from time to time", "by mutual written consent",
    ],
}

# Flatten all terms into a single set for D1
ALL_VAGUENESS_TERMS = set()
for terms in VAGUENESS_CATEGORIES.values():
    ALL_VAGUENESS_TERMS.update(terms)


# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION 2: Interpretive vs. procedural signal patterns
# ═══════════════════════════════════════════════════════════════════════════

INTERPRETIVE_SIGNALS = [
    r'\bwe interpret\b', r'\bwe construe\b', r'\bthe term\b',
    r'\bthe parties intended\b', r'\bambiguous\b', r'\bmeaning of\b',
    r'\bmust be read\b', r'\bshould be construed\b', r'\bwe read\b',
    r'\bthe language\b', r'\bcontractual\b', r'\bobligation\b',
    r'\bthe provision\b', r'\bthe clause\b',
    # additional interpretive signals for judicial reasoning
    r'\bconstrued\b', r'\bconstruction\b', r'\bconstruing\b',
    r'\binterpretation\b', r'\bintended to mean\b',
    r'\bplain meaning\b', r'\bplain language\b',
    r'\bon its face\b', r'\bthe contract\b',
    r'\bthe agreement\b', r'\bparties agreed\b',
    r'\brequires that\b', r'\bprovides that\b',
    r'\bcontract language\b', r'\breasonable interpretation\b',
    # ── appellate court language patterns from scored sample audit ──
    r'\bin interpreting\b',
    r'\brender.{1,20}surplusage\b',           # "render X surplusage" — canonical interpretive phrase
    r'\bgive effect to\b',
    r'\bharmonize\b',
    r'\bread.{1,15}entirety\b',               # "read in its entirety"
    r'\bcontract.{1,20}requires\b',
    r'\bnotice.{1,20}requirement\b',
    r'\bthe agreement.{1,20}provides\b',
    r'\bparties.{1,20}intended\b',
    r'\bconstrued.{1,20}require\b',
    r'\bno work to do\b',                     # courts use this exact phrase for surplusage
    r'\beconomic benefit\b',
    r'\bascertainable value\b',
    r'\bbroad definition\b',
]

PROCEDURAL_NOISE = [
    r'\bplaintiff filed\b', r'\bwe affirm\b', r'\bwe reverse\b',
    r'\bon appeal\b', r'\bdistrict court\b', r'\bsummary judgment\b',
    r'\bappellant argues\b', r'\bthe jury\b',
    # additional procedural markers
    r'\bmotion to dismiss\b', r'\bgranted summary\b',
    r'\bthe record shows\b', r'\bthe trial court\b',
    r'\bcourt of appeals\b', r'\bwe remand\b',
    r'\bdismissed with\b', r'\bcross-appeal\b',
    r'\bprocedural\b', r'\bjurisdiction\b',
]


# ═══════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def score_dimension_1(cuad_anchor: str) -> dict:
    """
    D1 — Anchor Vagueness.
    Binary check: does the CUAD anchor contain at least one vagueness
    seed term?

    Returns:
        dict with 'score' (0 or 1), 'matched_terms' (list of matched terms)
    """
    anchor_lower = cuad_anchor.lower()
    matched = []
    for term in ALL_VAGUENESS_TERMS:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, anchor_lower):
            matched.append(term)
    return {
        "score": 1 if matched else 0,
        "matched_terms": matched,
    }


def score_dimension_2(raw_sentence: str) -> dict:
    """
    D2 — Judicial Interpretive Content.
    Two-part check: count interpretive signal hits and procedural noise hits.
    Pass if interpretive_hits >= 2 AND procedural_hits <= 1.

    Returns:
        dict with 'score' (0 or 1), 'interpretive_hits' (int),
        'procedural_hits' (int), 'interpretive_matched' (list),
        'procedural_matched' (list)
    """
    text = raw_sentence.lower()

    interpretive_matched = []
    for p in INTERPRETIVE_SIGNALS:
        if re.search(p, text):
            interpretive_matched.append(p)

    procedural_matched = []
    for p in PROCEDURAL_NOISE:
        if re.search(p, text):
            procedural_matched.append(p)

    interpretive_hits = len(interpretive_matched)
    procedural_hits = len(procedural_matched)

    score = 1 if (interpretive_hits >= 2 and procedural_hits <= 1) else 0
    return {
        "score": score,
        "interpretive_hits": interpretive_hits,
        "procedural_hits": procedural_hits,
        "interpretive_matched": interpretive_matched,
        "procedural_matched": procedural_matched,
    }


def get_anchor_category(cuad_anchor: str) -> str | None:
    """
    Map a CUAD anchor to its vagueness category by checking which category's
    terms appear in the anchor.  Returns the first matching category name,
    or None if no category matches.
    """
    anchor_lower = cuad_anchor.lower()
    for category, terms in VAGUENESS_CATEGORIES.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', anchor_lower):
                return category
    return None


def score_dimension_3(cuad_anchor: str, raw_sentence: str) -> dict:
    """
    D3 — Semantic Coherence.
    Category-level coherence check: does the anchor's vagueness category
    match terms found in the raw_sentence?

    Returns:
        dict with 'score' (0 or 1), 'anchor_category' (str or None),
        'matched_category_terms' (list of terms from the category
        found in the raw_sentence)
    """
    category = get_anchor_category(cuad_anchor)
    if category is None:
        return {
            "score": 0,
            "anchor_category": None,
            "matched_category_terms": [],
        }

    sentence_lower = raw_sentence.lower()
    matched = []
    for term in VAGUENESS_CATEGORIES[category]:
        if re.search(r'\b' + re.escape(term.lower()) + r'\b', sentence_lower):
            matched.append(term)

    return {
        "score": 1 if matched else 0,
        "anchor_category": category,
        "matched_category_terms": matched,
    }


def score_triplet(row: dict) -> dict:
    """
    Score a single triplet across all three dimensions.
    Total score is 0–3.  Passes at >= 2.
    """
    d1 = score_dimension_1(row["cuad_anchor"])
    d2 = score_dimension_2(row["raw_sentence"])
    d3 = score_dimension_3(row["cuad_anchor"], row["raw_sentence"])
    total = d1["score"] + d2["score"] + d3["score"]

    return {
        # original fields
        "cuad_category": row["cuad_category"],
        "cuad_anchor": row["cuad_anchor"],
        "raw_sentence": row["raw_sentence"],
        "bi_encoder_score": row["bi_encoder_score"],
        "contract_id": row.get("contract_id", "Unknown"),

        # dimension scores
        "d1_anchor_vagueness": d1["score"],
        "d1_matched_terms": d1["matched_terms"],

        "d2_judicial_content": d2["score"],
        "d2_interpretive_hits": d2["interpretive_hits"],
        "d2_procedural_hits": d2["procedural_hits"],
        "d2_interpretive_matched": d2["interpretive_matched"],
        "d2_procedural_matched": d2["procedural_matched"],

        "d3_semantic_coherence": d3["score"],
        "d3_anchor_category": d3["anchor_category"],
        "d3_matched_category_terms": d3["matched_category_terms"],

        # aggregate
        "total_score": total,
        "passes": total >= 2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(scored: list[dict], output_lines: list[str] | None = None):
    """
    Print (and optionally capture) structured summary statistics
    over the scored triplet list.
    """

    def emit(line=""):
        print(line)
        if output_lines is not None:
            output_lines.append(line)

    n = len(scored)
    if n == 0:
        emit("No scored triplets to summarise.")
        return

    # ── Pass / fail ────────────────────────────────────────────────
    passes = sum(1 for s in scored if s["passes"])
    fails = n - passes
    emit("=" * 72)
    emit("TRIPLET QUALITY SCORING — SUMMARY VERSION 3")
    emit("=" * 72)
    emit(f"  Total triplets scored:   {n}")
    emit(f"  Passes (≥ 2/3):         {passes}  ({100*passes/n:.1f}%)")
    emit(f"  Fails  (< 2/3):         {fails}  ({100*fails/n:.1f}%)")

    # ── Score distribution ─────────────────────────────────────────
    score_dist = Counter(s["total_score"] for s in scored)
    emit(f"\n  Score Distribution:")
    for score_val in sorted(score_dist):
        count = score_dist[score_val]
        bar = "█" * int(50 * count / n)
        emit(f"    {score_val}/3:  {count:>5}  ({100*count/n:5.1f}%)  {bar}")

    # ── Per-dimension pass rates ──────────────────────────────────
    d1_pass = sum(1 for s in scored if s["d1_anchor_vagueness"] == 1)
    d2_pass = sum(1 for s in scored if s["d2_judicial_content"] == 1)
    d3_pass = sum(1 for s in scored if s["d3_semantic_coherence"] == 1)
    emit(f"\n  Per-Dimension Pass Rates:")
    emit(f"    D1 (Anchor Vagueness):      {d1_pass:>5} / {n}  ({100*d1_pass/n:5.1f}%)")
    emit(f"    D2 (Judicial Content):       {d2_pass:>5} / {n}  ({100*d2_pass/n:5.1f}%)")
    emit(f"    D3 (Semantic Coherence):      {d3_pass:>5} / {n}  ({100*d3_pass/n:5.1f}%)")

    # ── Failure mode breakdown ────────────────────────────────────
    # For triplets that fail, what pattern of 0s do they have?
    emit(f"\n  Failure Mode Breakdown (among {fails} failures):")
    failure_patterns = Counter()
    for s in scored:
        if not s["passes"]:
            pat = (s["d1_anchor_vagueness"], s["d2_judicial_content"], s["d3_semantic_coherence"])
            failure_patterns[pat] += 1

    for pat in sorted(failure_patterns, key=failure_patterns.get, reverse=True):
        count = failure_patterns[pat]
        d1_label = "✓" if pat[0] else "✗"
        d2_label = "✓" if pat[1] else "✗"
        d3_label = "✓" if pat[2] else "✗"
        emit(f"    D1={d1_label}  D2={d2_label}  D3={d3_label}  →  {count:>5}  ({100*count/max(fails,1):5.1f}%)")

    # ── D2 interpretive hit distribution ──────────────────────────
    emit(f"\n  D2 Interpretive Hits Distribution:")
    interp_dist = Counter(s["d2_interpretive_hits"] for s in scored)
    for hits in sorted(interp_dist):
        count = interp_dist[hits]
        emit(f"    {hits} hits:  {count:>5}  ({100*count/n:5.1f}%)")

    # ── D3 category coverage ─────────────────────────────────────
    emit(f"\n  D3 Anchor Category Distribution:")
    cat_dist = Counter(s["d3_anchor_category"] for s in scored)
    for cat in sorted(cat_dist, key=cat_dist.get, reverse=True):
        count = cat_dist[cat]
        label = cat if cat else "(unmapped)"
        emit(f"    {label:<20}  {count:>5}  ({100*count/n:5.1f}%)")

    # ── Bi-encoder score vs quality score correlation ─────────────
    emit(f"\n  Bi-Encoder Score by Quality Tier:")
    for total_val in sorted(score_dist):
        tier_scores = [s["bi_encoder_score"] for s in scored if s["total_score"] == total_val]
        if tier_scores:
            emit(
                f"    Score {total_val}/3  (n={len(tier_scores):>5}):  "
                f"mean={statistics.mean(tier_scores):.4f}  "
                f"median={statistics.median(tier_scores):.4f}  "
                f"min={min(tier_scores):.4f}  "
                f"max={max(tier_scores):.4f}"
            )

    # ── CUAD category pass rates ──────────────────────────────────
    emit(f"\n  Pass Rate by CUAD Category:")
    cuad_cats = Counter(s["cuad_category"] for s in scored)
    cuad_pass = Counter(s["cuad_category"] for s in scored if s["passes"])
    for cat in sorted(cuad_cats, key=cuad_cats.get, reverse=True):
        total_c = cuad_cats[cat]
        pass_c = cuad_pass.get(cat, 0)
        emit(f"    {cat:<35}  {pass_c:>4}/{total_c:<4}  ({100*pass_c/total_c:5.1f}%)")

    emit("\n" + "=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Score triplets across 3 quality dimensions."
    )
    parser.add_argument(
        "--input", "-i",
        default="distributed_threshold_results_version3.json",
        help="Path to the distributed_threshold_results.json file "
             "(default: distributed_threshold_results_version2.json in cwd)"
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=50,
        help="Number of rows for the random sample output (default: 50)"
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only write the full scored output, skip sample file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--threshold-key",
        default="0.4",
        help="Top-level key in the JSON to read triplets from (default: '0.4')"
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────
    input_path = Path(args.input)
    print(f"Loading triplets from {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get(args.threshold_key, [])
    if not entries:
        print(f"ERROR: No entries found under key '{args.threshold_key}'.")
        return

    print(f"Loaded {len(entries)} triplets under key '{args.threshold_key}'.")

    # ── Score all triplets ─────────────────────────────────────────
    print("Scoring all triplets ...")
    scored_all = []
    for i, row in enumerate(entries):
        scored_all.append(score_triplet(row))
        if (i + 1) % 1000 == 0:
            print(f"  ... scored {i+1}/{len(entries)}")
    print(f"  ... scored {len(scored_all)}/{len(entries)} — done.")

    # ── Write full output ──────────────────────────────────────────
    out_dir = input_path.parent
    full_output_path = out_dir / "scored_triplets_version3.json"
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(scored_all, f, indent=2, ensure_ascii=False)
    print(f"\nFull scored output written to {full_output_path}")

    # ── Write sample output ────────────────────────────────────────
    if not args.full_only:
        random.seed(args.seed)
        sample_size = min(args.sample_size, len(scored_all))
        sample = random.sample(scored_all, sample_size)

        sample_output_path = out_dir / "scored_triplets_sample.json"
        with open(sample_output_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        print(f"Sample ({sample_size} rows) written to {sample_output_path}")

    # ── Print summary ──────────────────────────────────────────────
    summary_lines = []
    print()
    print_summary(scored_all, summary_lines)

    # Write summary to file
    summary_path = out_dir / "scoring_summary_version3.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
