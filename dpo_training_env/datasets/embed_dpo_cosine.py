"""
Embed (prompt, chosen, rejected) triples from a DPO-style dataset using
EmbeddingGemma, and compute direct cosine similarity between:
    - anchor vs chosen
    - anchor vs rejected
    - chosen vs rejected

Run this on a machine with internet access to Hugging Face and (ideally)
a GPU. Everything is embedded fresh, in-memory, on every run -- no caching
of prior embeddings.

Install:
    pip install -U sentence-transformers torch

If google/embeddinggemma-300m is gated on your HF account, run:
    huggingface-cli login
first, or set HF_TOKEN in your environment.
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "google/embeddinggemma-300m"

# EmbeddingGemma supports task-specific prompt prefixes (query / document /
# clustering / etc). For a *direct*, symmetric cosine comparison between
# anchor/chosen/rejected, we embed all three with the SAME prompt so no
# field gets an asymmetric query-vs-document treatment. If you specifically
# want retrieval-style asymmetry (anchor as "query", chosen/rejected as
# "document"), change ANCHOR_PROMPT to "query" below.
ANCHOR_PROMPT = "STS"
CHOSEN_PROMPT = "STS"
REJECTED_PROMPT = "STS"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="dpo_dataset_revised.json",
        help="Path to the DPO dataset JSON (list of {prompt, chosen, rejected})",
    )
    parser.add_argument(
        "--output",
        default="dpo_cosine_scores.jsonl",
        help="Where to write per-example cosine scores (JSONL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device, e.g. 'cuda', 'cpu'. Default: auto-detect.",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    print(f"Loaded {len(data)} examples from {args.input}")

    anchors = [ex["prompt"] for ex in data]
    chosen = [ex["chosen"] for ex in data]
    rejected = [ex["rejected"] for ex in data]

    print(f"Loading {MODEL_NAME} fresh (no cache reuse of embeddings)...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)

    def encode(texts, prompt_name):
        return model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    print("Embedding anchors...")
    anchor_emb = encode(anchors, ANCHOR_PROMPT)
    print("Embedding chosen...")
    chosen_emb = encode(chosen, CHOSEN_PROMPT)
    print("Embedding rejected...")
    rejected_emb = encode(rejected, REJECTED_PROMPT)

    results = []
    for i, ex in enumerate(data):
        sim_anchor_chosen = cosine(anchor_emb[i], chosen_emb[i])
        sim_anchor_rejected = cosine(anchor_emb[i], rejected_emb[i])
        sim_chosen_rejected = cosine(chosen_emb[i], rejected_emb[i])
        results.append(
            {
                "index": i,
                "sim_anchor_chosen": sim_anchor_chosen,
                "sim_anchor_rejected": sim_anchor_rejected,
                "sim_chosen_rejected": sim_chosen_rejected,
                "margin": sim_anchor_chosen - sim_anchor_rejected,
            }
        )

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    margins = np.array([r["margin"] for r in results])
    ac = np.array([r["sim_anchor_chosen"] for r in results])
    ar = np.array([r["sim_anchor_rejected"] for r in results])

    print(f"\nWrote {len(results)} scored examples to {args.output}")
    print(f"Mean cosine(anchor, chosen):   {ac.mean():.4f}")
    print(f"Mean cosine(anchor, rejected): {ar.mean():.4f}")
    print(f"Mean margin (chosen - rejected similarity to anchor): {margins.mean():.4f}")
    print(f"Fraction where chosen is closer to anchor than rejected: {(margins > 0).mean():.4f}")


if __name__ == "__main__":
    main()
