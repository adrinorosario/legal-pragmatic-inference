"""
Control: TF-IDF cosine between 500 random pairs of judicial-reasoning
windows drawn from *different* anchors in *different* categories —
text that has nothing to do with each other. Establishes the noise
floor against which the anchor/chosen/rejected cosine scores should be
read.
"""

import json
import math
import random
import re
from collections import Counter

random.seed(42)

rows = [json.loads(l) for l in open("dpo_pairs_2.jsonl")]

by_category = {}
for r in rows:
    by_category.setdefault(r["cuad_category"], []).append(r["chosen"])
categories = list(by_category)

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

docs = [tokenize(r["chosen"]) for r in rows]
df = Counter()
for toks in docs:
    df.update(set(toks))
n_docs = len(docs)
idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

def vectorize(tokens):
    counts = Counter(tokens)
    vec = {t: c * idf[t] for t, c in counts.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return vec, norm

def cosine(text_a, text_b):
    va, na = vectorize(tokenize(text_a))
    vb, nb = vectorize(tokenize(text_b))
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(v * vb.get(t, 0.0) for t, v in va.items())
    return dot / (na * nb)

pairs = []
for _ in range(500):
    cat_a, cat_b = random.sample(categories, 2)
    text_a = random.choice(by_category[cat_a])
    text_b = random.choice(by_category[cat_b])
    pairs.append({
        "category_a": cat_a,
        "category_b": cat_b,
        "cosine": cosine(text_a, text_b),
    })

scores = sorted(p["cosine"] for p in pairs)
n = len(scores)
summary = {
    "n_pairs": n,
    "mean": sum(scores) / n,
    "median": scores[n // 2],
    "min": scores[0],
    "max": scores[-1],
}

with open("cross_category_cosine_control_results.json", "w") as f:
    json.dump({"summary": summary, "pairs": pairs}, f, indent=2)

print(summary)
