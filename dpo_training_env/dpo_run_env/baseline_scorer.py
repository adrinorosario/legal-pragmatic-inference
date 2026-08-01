"""
Lexical baseline scorer for the JDAR DPO triplets.

Question: on the 984 anchor/chosen/rejected triplets in
dpo_dataset_revised.json, how often does a *lexical* similarity score
(TF-IDF cosine, Okapi BM25) rank the chosen judicial-reasoning passage
above the rejected one, just by matching words with the anchor clause?
That's compared against JDAR's own embedding-gemma bi-encoder score
(the `bi_encoder_score` field computed during triplet construction with
`google/embeddinggemma-300m`, see
`dataset construction notebooks/jdar_construction/hybrid_clause_and_term_extraction_v1.ipynb`)
recovered for the same pairs from the intermediate construction
artifacts under
`datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/`.

Only the standard library + numpy are used (TF-IDF and BM25 are
implemented from scratch below) so the script has no extra
dependencies beyond what's already in this environment.

Usage:
    python baseline_scorer.py
Outputs (written next to this script):
    results.json  - machine-readable per-category + overall numbers
    results.md    - human-readable report
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DPO_DATASET_PATH = REPO_ROOT / "dpo_training_env" / "datasets" / "dpo_dataset_revised.json"
JDAR_CONSTRUCTION_DIR = REPO_ROOT / "datasets" / "jdar_triplet_extracted_on_cuad_and_cold_cases"
OUT_DIR = Path(__file__).resolve().parent

PROMPT_PATTERN = re.compile(
    r"falls in the category of ([^,]+), provide judicial reasoning of the kind "
    r"a court would apply when interpreting clauses that raise similar legal "
    r"questions:\n\n(.*)",
    re.DOTALL,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


# ─────────────────────────────────────────────────────────────────────────
# Load the 984 triplets and pull (category, anchor) back out of the prompt
# ─────────────────────────────────────────────────────────────────────────

def load_triplets():
    with open(DPO_DATASET_PATH) as f:
        rows = json.load(f)

    triplets = []
    for row in rows:
        m = PROMPT_PATTERN.search(row["prompt"])
        if not m:
            raise ValueError(f"Could not parse category/anchor from prompt: {row['prompt'][:200]!r}")
        category, anchor = m.group(1), m.group(2)
        triplets.append({
            "category": category,
            "anchor": anchor,
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        })
    return triplets


# ─────────────────────────────────────────────────────────────────────────
# JDAR embedding-gemma score recovery
#
# The final DPO dataset doesn't carry the bi-encoder score, but the
# intermediate triplet-construction artifacts do (`bi_encoder_score`,
# cosine similarity between `cuad_anchor` and `raw_sentence` under
# google/embeddinggemma-300m). dpo_dataset_revised.json's chosen/rejected
# text is a verbatim, 1:1 positional copy of dpo_pairs_2.jsonl's
# chosen/rejected (verified before writing this script), so we join back
# on the (anchor, passage) text pair across every scored-triplet artifact
# in the JDAR construction tree to recover the original score for each
# side of each pair.
# ─────────────────────────────────────────────────────────────────────────

def build_bi_encoder_lookup():
    lookup = {}
    json_files = sorted(JDAR_CONSTRUCTION_DIR.glob("**/*.json"))
    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for d in data:
            if not isinstance(d, dict) or "bi_encoder_score" not in d:
                continue
            key = (d.get("cuad_anchor"), d.get("raw_sentence"))
            if key not in lookup:
                lookup[key] = d["bi_encoder_score"]
    return lookup


def attach_embedding_gemma_scores(triplets, lookup):
    n_chosen_found = n_rejected_found = 0
    for t in triplets:
        c_score = lookup.get((t["anchor"], t["chosen"]))
        r_score = lookup.get((t["anchor"], t["rejected"]))
        t["gemma_chosen"] = c_score
        t["gemma_rejected"] = r_score
        n_chosen_found += c_score is not None
        n_rejected_found += r_score is not None
    return n_chosen_found, n_rejected_found


# ─────────────────────────────────────────────────────────────────────────
# TF-IDF cosine (fit over the pooled chosen+rejected corpus, anchor scored
# as an out-of-corpus query against that vocabulary/IDF)
# ─────────────────────────────────────────────────────────────────────────

class TfidfScorer:
    def __init__(self, documents):
        self.vocab = {}
        df = Counter()
        doc_tokens = [tokenize(d) for d in documents]
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1
        n_docs = len(documents)
        for term in df:
            self.vocab[term] = len(self.vocab)
        self.idf = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

    def _vector(self, text):
        tokens = tokenize(text)
        counts = Counter(tokens)
        vec = np.zeros(len(self.vocab))
        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = count * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def cosine(self, text_a, text_b):
        va, vb = self._vector(text_a), self._vector(text_b)
        return float(np.dot(va, vb))


# ─────────────────────────────────────────────────────────────────────────
# Okapi BM25 (anchor tokens as query, chosen/rejected as scored documents;
# corpus statistics fit over the pooled chosen+rejected documents)
# ─────────────────────────────────────────────────────────────────────────

class Bm25Scorer:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1
        n_docs = len(documents)
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def score(self, query_text, doc_text):
        query_terms = tokenize(query_text)
        doc_terms = Counter(tokenize(doc_text))
        dl = sum(doc_terms.values())
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            f = doc_terms.get(term, 0)
            if f == 0:
                continue
            idf = self.idf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score


# ─────────────────────────────────────────────────────────────────────────
# Pairwise ranking accuracy
# ─────────────────────────────────────────────────────────────────────────

def pairwise_accuracy(rows, chosen_key, rejected_key, restrict_valid=False):
    """
    Fraction of pairs where score(anchor, chosen) > score(anchor, rejected).
    Ties count as incorrect but are also reported separately.
    If restrict_valid, pairs with a None score on either side are skipped
    (used for the embedding-gemma column, where a handful of rejected
    passages have no recovered score in the construction artifacts).
    """
    wins = ties = losses = skipped = 0
    for r in rows:
        c, rj = r[chosen_key], r[rejected_key]
        if restrict_valid and (c is None or rj is None):
            skipped += 1
            continue
        if c > rj:
            wins += 1
        elif c == rj:
            ties += 1
        else:
            losses += 1
    scored = wins + ties + losses
    accuracy = wins / scored if scored else None
    return {
        "n_scored": scored,
        "n_skipped": skipped,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "accuracy": accuracy,
    }


def main():
    triplets = load_triplets()
    print(f"Loaded {len(triplets)} triplets from {DPO_DATASET_PATH}")

    lookup = build_bi_encoder_lookup()
    n_chosen_found, n_rejected_found = attach_embedding_gemma_scores(triplets, lookup)
    print(f"Recovered embedding-gemma bi_encoder_score for "
          f"{n_chosen_found}/{len(triplets)} chosen and "
          f"{n_rejected_found}/{len(triplets)} rejected passages "
          f"from JDAR construction artifacts.")

    corpus = [t["chosen"] for t in triplets] + [t["rejected"] for t in triplets]
    tfidf = TfidfScorer(corpus)
    bm25 = Bm25Scorer(corpus)

    for t in triplets:
        t["tfidf_chosen"] = tfidf.cosine(t["anchor"], t["chosen"])
        t["tfidf_rejected"] = tfidf.cosine(t["anchor"], t["rejected"])
        t["bm25_chosen"] = bm25.score(t["anchor"], t["chosen"])
        t["bm25_rejected"] = bm25.score(t["anchor"], t["rejected"])

    by_category = defaultdict(list)
    for t in triplets:
        by_category[t["category"]].append(t)

    scorers = {
        "tfidf_cosine": ("tfidf_chosen", "tfidf_rejected", False),
        "bm25": ("bm25_chosen", "bm25_rejected", False),
        "embedding_gemma": ("gemma_chosen", "gemma_rejected", True),
    }

    results = {"overall": {}, "per_category": {}, "category_counts": {}}
    for cat, rows in by_category.items():
        results["category_counts"][cat] = len(rows)

    for name, (ck, rk, restrict) in scorers.items():
        results["overall"][name] = pairwise_accuracy(triplets, ck, rk, restrict)
        results["per_category"][name] = {}
        for cat, rows in by_category.items():
            results["per_category"][name][cat] = pairwise_accuracy(rows, ck, rk, restrict)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    write_markdown_report(results, len(triplets))
    print(f"\nWrote {OUT_DIR / 'results.json'} and {OUT_DIR / 'results.md'}")


def write_markdown_report(results, n_total):
    lines = []
    lines.append("# Lexical baseline vs. embedding-gemma: pairwise ranking accuracy\n")
    lines.append(
        f"984 anchor/chosen/rejected triplets from `dpo_training_env/datasets/dpo_dataset_revised.json`. "
        "Metric: how often does the scorer rank `chosen` above `rejected` when both are scored "
        "against the anchor clause (ties count as a loss).\n"
    )

    cat_counts = results["category_counts"]
    lines.append("## Category distribution\n")
    lines.append("| Category | N | % of dataset |")
    lines.append("|---|---|---|")
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cat} | {n} | {100 * n / n_total:.1f}% |")
    lines.append("")

    scorer_labels = {
        "tfidf_cosine": "TF-IDF cosine",
        "bm25": "Okapi BM25",
        "embedding_gemma": "JDAR embedding-gemma (bi_encoder_score)",
    }

    lines.append("## Overall pairwise accuracy\n")
    lines.append("| Scorer | Accuracy | Wins | Ties | Losses | N scored | N skipped (missing score) |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, label in scorer_labels.items():
        r = results["overall"][key]
        acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a"
        lines.append(
            f"| {label} | {acc} | {r['wins']} | {r['ties']} | {r['losses']} | "
            f"{r['n_scored']} | {r['n_skipped']} |"
        )
    lines.append("")

    lines.append("## Per-category pairwise accuracy\n")
    lines.append("Cap On Liability is 82% of the dataset, so the overall number above is "
                  "essentially the Cap On Liability number. Breaking out the minority "
                  "categories (Third Party Beneficiary, Non-Compete) shows whether the "
                  "scorers generalize or are just picking up Cap-On-Liability-specific "
                  "vocabulary.\n")
    lines.append("| Category | N | TF-IDF cosine | Okapi BM25 | embedding-gemma |")
    lines.append("|---|---|---|---|---|")
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        row = [cat, str(n)]
        for key in scorer_labels:
            r = results["per_category"][key][cat]
            acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a"
            if r["n_skipped"]:
                acc += f" (n={r['n_scored']})"
            row.append(acc)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- TF-IDF and BM25 are implemented from scratch (numpy + stdlib only); both are fit "
        "on the pooled corpus of all 1,968 chosen+rejected passages, with the anchor clause "
        "scored as a query against that vocabulary.\n"
        "- The embedding-gemma column is not recomputed here — it's the original "
        "`bi_encoder_score` (cosine similarity under `google/embeddinggemma-300m`) recovered "
        "from JDAR's own triplet-construction artifacts "
        "(`datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/`) by joining on the exact "
        "(anchor, passage) text pair. A small number of rejected passages "
        "(see 'N skipped') could not be matched back to a scored artifact and are excluded "
        "from that column's accuracy rather than silently zero-filled.\n"
        "- Ties (equal score on both sides) are counted as losses for the ranking-accuracy "
        "metric but reported separately in results.json.\n"
        "- Caveat on the embedding-gemma column: `bi_encoder_score` was JDAR's *retrieval* "
        "signal (used to pull candidate reasoning windows near an anchor clause), not the "
        "chosen/rejected *selection* signal. The pairing script "
        "(`dpo_dataset_construction/build_dpo_pairs.py`) picked chosen vs. rejected purely "
        "from the rule-based D1/D2/D3 judicial-content/coherence scorer, with no bi-encoder "
        "score in the selection criteria. So embedding-gemma landing near chance here isn't "
        "evidence the embedding model is a weak similarity scorer in general — it's evidence "
        "that anchor-passage embedding similarity and the D1/D2/D3 pass/fail label are "
        "close to independent in this dataset. Read this column as 'how well would a "
        "similarity-based DPO scorer have agreed with the judicial-content-based label', "
        "not as a fair fight between TF-IDF/BM25 and embedding-gemma on the same task.\n"
    )

    with open(OUT_DIR / "results.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
