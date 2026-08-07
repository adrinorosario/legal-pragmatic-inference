"""
Four-scorer baseline eval on the JDAR *held-out* pairs — the pool of
eligible negatives never used to build the 984-pair DPO training set
(`datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/held_out_pairs.jsonl`,
235,567 anchor/chosen/rejected rows).

This exists because the training-pairs baseline (`dpo_training_env/dpo_run_env/baseline_scorer.py`,
see results.json/results.md next to it) scored TF-IDF/BM25/embedding-gemma on
the 984 *training* pairs, while the DPO implicit-reward number in the
headline table (`evaluation_results.json`, `inspect_implicit_rewards.py`)
was scored on 500 *held-out* pairs. Comparing those two sets of numbers in
one table mixes data splits. This script closes that gap: it draws the
same kind of 500-pair random sample from the held-out pool and scores it
with all four methods (TF-IDF, BM25, fresh embedding-gemma cosine, and the
ms-marco-MiniLM-L6-v2 cross-encoder used during triplet construction) so
the headline table has a like-for-like held-out panel.

Metric: pairwise ranking accuracy -- does the scorer rank `chosen` above
`rejected` when both are scored against `prompt_anchor`? Ties count as a
loss. 95% CI is the Wald normal approximation p +/- 1.96*sqrt(p(1-p)/n),
matching the CI convention already used elsewhere in this project
(see JDAR_continuity_codex.md Experiment 2).

Usage:
    python held_out_baseline_eval.py

Outputs (written next to this script):
    held_out_baseline_results.json
    held_out_baseline_results.md
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HELD_OUT_PATH = (
    REPO_ROOT
    / "datasets"
    / "jdar_triplet_extracted_on_cuad_and_cold_cases"
    / "dpo_dataset_construction"
    / "held_out_pairs.jsonl"
)
OUT_DIR = Path(__file__).resolve().parent

SAMPLE_SIZE = 500
SEED = 42

BI_ENCODER_MODEL = "google/embeddinggemma-300m"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


# ─────────────────────────────────────────────────────────────────────────
# Load + sample
# ─────────────────────────────────────────────────────────────────────────

def load_held_out_pairs():
    rows = []
    with open(HELD_OUT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_pairs(rows, n, seed):
    rng = random.Random(seed)
    return rng.sample(rows, n)


# ─────────────────────────────────────────────────────────────────────────
# TF-IDF cosine (fit over the pooled chosen+rejected corpus of the sample,
# anchor scored as an out-of-corpus query -- same construction as
# dpo_training_env/dpo_run_env/baseline_scorer.py)
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
# Okapi BM25 (anchor tokens as query, chosen/rejected as scored documents)
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
# Fresh embedding cosine (embedding-gemma-300m, symmetric STS prompt --
# same construction as dpo_training_env/datasets/embed_dpo_cosine.py)
# ─────────────────────────────────────────────────────────────────────────

def compute_embedding_scores(triplets, device=None):
    from sentence_transformers import SentenceTransformer

    print(f"Loading {BI_ENCODER_MODEL} fresh...")
    model = SentenceTransformer(BI_ENCODER_MODEL, device=device)

    anchors = [t["anchor"] for t in triplets]
    chosen = [t["chosen"] for t in triplets]
    rejected = [t["rejected"] for t in triplets]

    def encode(texts):
        return model.encode(
            texts, prompt_name="STS", batch_size=16,
            show_progress_bar=True, convert_to_numpy=True,
        )

    print("Embedding anchors...")
    anchor_emb = encode(anchors)
    print("Embedding chosen...")
    chosen_emb = encode(chosen)
    print("Embedding rejected...")
    rejected_emb = encode(rejected)

    def cosine(a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    for i, t in enumerate(triplets):
        t["embgemma_chosen"] = cosine(anchor_emb[i], chosen_emb[i])
        t["embgemma_rejected"] = cosine(anchor_emb[i], rejected_emb[i])


# ─────────────────────────────────────────────────────────────────────────
# Cross-encoder score (ms-marco-MiniLM-L6-v2, the retrieval cross-encoder
# used during JDAR triplet construction -- anchor, candidate pairs)
# ─────────────────────────────────────────────────────────────────────────

def compute_cross_encoder_scores(triplets, device=None):
    from sentence_transformers import CrossEncoder

    print(f"Loading {CROSS_ENCODER_MODEL} fresh...")
    model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)

    chosen_pairs = [(t["anchor"], t["chosen"]) for t in triplets]
    rejected_pairs = [(t["anchor"], t["rejected"]) for t in triplets]

    print("Scoring anchor/chosen pairs...")
    chosen_scores = model.predict(chosen_pairs, show_progress_bar=True)
    print("Scoring anchor/rejected pairs...")
    rejected_scores = model.predict(rejected_pairs, show_progress_bar=True)

    for i, t in enumerate(triplets):
        t["crossenc_chosen"] = float(chosen_scores[i])
        t["crossenc_rejected"] = float(rejected_scores[i])


# ─────────────────────────────────────────────────────────────────────────
# Pairwise ranking accuracy + Wald 95% CI
# ─────────────────────────────────────────────────────────────────────────

def wald_ci(p, n):
    if n == 0:
        return (None, None)
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (lo, hi)


def pairwise_accuracy(rows, chosen_key, rejected_key):
    wins = ties = losses = 0
    for r in rows:
        c, rj = r[chosen_key], r[rejected_key]
        if c > rj:
            wins += 1
        elif c == rj:
            ties += 1
        else:
            losses += 1
    scored = wins + ties + losses
    accuracy = wins / scored if scored else None
    ci_lo, ci_hi = wald_ci(accuracy, scored) if accuracy is not None else (None, None)
    return {
        "n_scored": scored,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "accuracy": accuracy,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def main():
    print(f"Loading held-out pairs from {HELD_OUT_PATH}")
    rows = load_held_out_pairs()
    print(f"Loaded {len(rows)} held-out pairs")

    sample = sample_pairs(rows, SAMPLE_SIZE, SEED)
    print(f"Sampled {len(sample)} pairs (seed={SEED})")

    triplets = [
        {
            "category": r["cuad_category"],
            "anchor": r["prompt_anchor"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in sample
    ]

    corpus = [t["chosen"] for t in triplets] + [t["rejected"] for t in triplets]
    tfidf = TfidfScorer(corpus)
    bm25 = Bm25Scorer(corpus)
    for t in triplets:
        t["tfidf_chosen"] = tfidf.cosine(t["anchor"], t["chosen"])
        t["tfidf_rejected"] = tfidf.cosine(t["anchor"], t["rejected"])
        t["bm25_chosen"] = bm25.score(t["anchor"], t["chosen"])
        t["bm25_rejected"] = bm25.score(t["anchor"], t["rejected"])

    compute_embedding_scores(triplets)
    compute_cross_encoder_scores(triplets)

    by_category = defaultdict(list)
    for t in triplets:
        by_category[t["category"]].append(t)

    scorers = {
        "tfidf_cosine": ("tfidf_chosen", "tfidf_rejected"),
        "bm25": ("bm25_chosen", "bm25_rejected"),
        "embedding_gemma_fresh": ("embgemma_chosen", "embgemma_rejected"),
        "cross_encoder_msmarco": ("crossenc_chosen", "crossenc_rejected"),
    }

    results = {
        "sample_size": len(triplets),
        "seed": SEED,
        "source_file": str(HELD_OUT_PATH.relative_to(REPO_ROOT)),
        "category_counts": {},
        "overall": {},
        "per_category": {},
        "per_pair": triplets,
    }
    for cat, cat_rows in by_category.items():
        results["category_counts"][cat] = len(cat_rows)

    for name, (ck, rk) in scorers.items():
        results["overall"][name] = pairwise_accuracy(triplets, ck, rk)
        results["per_category"][name] = {}
        for cat, cat_rows in by_category.items():
            results["per_category"][name][cat] = pairwise_accuracy(cat_rows, ck, rk)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "held_out_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    write_markdown_report(results)
    print(f"\nWrote {OUT_DIR / 'held_out_baseline_results.json'} "
          f"and {OUT_DIR / 'held_out_baseline_results.md'}")


def write_markdown_report(results):
    lines = []
    lines.append("# Four-scorer baseline on held-out pairs\n")
    lines.append(
        f"{results['sample_size']} pairs sampled (seed={results['seed']}) from "
        f"`{results['source_file']}` ({235567} total eligible held-out pairs, "
        "never used to build the 984-pair DPO training set). Metric: pairwise "
        "ranking accuracy -- does the scorer rank `chosen` above `rejected` "
        "when both are scored against `prompt_anchor`? Ties count as a loss. "
        "95% CI is the Wald normal approximation.\n"
    )

    cat_counts = results["category_counts"]
    lines.append("## Category distribution\n")
    lines.append("| Category | N | % of sample |")
    lines.append("|---|---|---|")
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cat} | {n} | {100 * n / results['sample_size']:.1f}% |")
    lines.append("")

    scorer_labels = {
        "tfidf_cosine": "TF-IDF cosine",
        "bm25": "Okapi BM25",
        "embedding_gemma_fresh": "Raw embedding cosine (freshly computed, embedding-gemma-300m)",
        "cross_encoder_msmarco": "Cross-encoder (ms-marco-MiniLM-L6-v2)",
    }

    lines.append("## Overall pairwise accuracy\n")
    lines.append("| Scorer | Accuracy | 95% CI | Wins | Ties | Losses | N |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, label in scorer_labels.items():
        r = results["overall"][key]
        acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a"
        ci = f"[{100*r['ci_lo']:.1f}, {100*r['ci_hi']:.1f}]" if r["ci_lo"] is not None else "n/a"
        lines.append(
            f"| {label} | {acc} | {ci} | {r['wins']} | {r['ties']} | {r['losses']} | {r['n_scored']} |"
        )
    lines.append("")

    lines.append("## Per-category pairwise accuracy\n")
    lines.append("| Category | N | TF-IDF | BM25 | embedding-gemma (fresh) | cross-encoder |")
    lines.append("|---|---|---|---|---|---|")
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        row = [cat, str(n)]
        for key in scorer_labels:
            r = results["per_category"][key][cat]
            acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a"
            row.append(acc)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- TF-IDF and BM25 are fit on the pooled chosen+rejected corpus of "
        "this 500-pair sample only (not the full 235,567-row held-out pool), "
        "matching the training-pairs baseline methodology in "
        "`dpo_training_env/dpo_run_env/baseline_scorer.py`.\n"
        "- embedding-gemma and the cross-encoder are both scored fresh here "
        "(no reuse of construction-time scores), anchor vs. chosen and "
        "anchor vs. rejected, matching how the DPO implicit-reward held-out "
        "number was produced.\n"
        "- This panel is directly comparable to the DPO implicit-reward "
        "held-out result (`evaluation_results.json`, "
        "`inspect_implicit_rewards.py`), since both are scored on unseen "
        "pairs from the same held-out pool, unlike the training-pairs "
        "baseline which is scored on the 984 pairs the model trained on.\n"
    )

    with open(OUT_DIR / "held_out_baseline_results.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
