"""
Compute cosine(chosen, rejected) across the full DPO dataset using
EmbeddingGemma, fresh embeddings, no caching. Reports mean, distribution,
and a mechanical read of what that distribution implies about pair
construction.

Install (Python 3.12 recommended -- see notes on 3.14 torch/torchvision
issues):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers

Run:
    python cosine_chosen_rejected.py --input dpo_dataset_revised.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "google/embeddinggemma-300m"
PROMPT_NAME = "STS"  # symmetric, direct comparison -- no query/document asymmetry

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.40  # below this, negatives are essentially unrelated text


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equal-length embedding arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dpo_dataset_revised.json")
    parser.add_argument("--output", default="chosen_rejected_cosine.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    n = len(data)
    print(f"Loaded {n} examples from {args.input}")

    chosen = [ex["chosen"] for ex in data]
    rejected = [ex["rejected"] for ex in data]

    print(f"Loading {MODEL_NAME} fresh...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)

    print("Embedding chosen...")
    chosen_emb = model.encode(
        chosen, prompt_name=PROMPT_NAME, batch_size=args.batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    )
    print("Embedding rejected...")
    rejected_emb = model.encode(
        rejected, prompt_name=PROMPT_NAME, batch_size=args.batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    )

    sims = cosine_matrix(chosen_emb, rejected_emb)

    with open(args.output, "w") as f:
        for i, s in enumerate(sims):
            f.write(json.dumps({"index": i, "cosine_chosen_rejected": float(s)}) + "\n")

    mean = float(np.mean(sims))
    median = float(np.median(sims))
    std = float(np.std(sims))
    p10, p25, p75, p90 = np.percentile(sims, [10, 25, 75, 90])
    frac_high = float(np.mean(sims > HIGH_THRESHOLD))
    frac_low = float(np.mean(sims < LOW_THRESHOLD))

    # simple histogram, 20 bins from -1 to 1 (cosine range), text rendering
    bins = np.linspace(-1, 1, 21)
    counts, edges = np.histogram(sims, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    print(f"\nWrote {n} pairwise cosine scores to {args.output}\n")
    print("=" * 60)
    print("cosine(chosen, rejected) -- distribution")
    print("=" * 60)
    print(f"n:            {n}")
    print(f"mean:         {mean:.4f}")
    print(f"median:       {median:.4f}")
    print(f"std:          {std:.4f}")
    print(f"p10 / p90:    {p10:.4f} / {p90:.4f}")
    print(f"p25 / p75:    {p25:.4f} / {p75:.4f}")
    print(f"min / max:    {sims.min():.4f} / {sims.max():.4f}")
    print(f"frac > {HIGH_THRESHOLD}: {frac_high:.4f}")
    print(f"frac < {LOW_THRESHOLD}: {frac_low:.4f}")

    print("\nHistogram (cosine bins, -1 to 1):")
    for i in range(len(counts)):
        bar = "#" * int(40 * counts[i] / max_count)
        print(f"  [{edges[i]:+.2f}, {edges[i+1]:+.2f}) {counts[i]:4d} {bar}")

    print("\n" + "=" * 60)
    print("Reading")
    print("=" * 60)
    if mean > HIGH_THRESHOLD:
        print(
            f"Mean cosine ({mean:.4f}) is above {HIGH_THRESHOLD}.\n"
            "Chosen and rejected texts sit close together in embedding space.\n"
            "This points to a pair-construction problem: negatives are near-\n"
            "duplicates of positives on surface/semantic content, so no scorer\n"
            "operating on this representation can be expected to separate them\n"
            "reliably. Whatever signal DPO training picks up on is likely a\n"
            "surface artifact (length, formatting, boilerplate) rather than the\n"
            "target legal-reasoning distinction."
        )
    elif mean < LOW_THRESHOLD:
        print(
            f"Mean cosine ({mean:.4f}) is below {LOW_THRESHOLD}.\n"
            "Chosen and rejected texts are far apart in embedding space --\n"
            "trivially separable, but not necessarily along the intended axis\n"
            "(judicial reasoning quality/relevance). The separation may be\n"
            "driven by topic, source document, or style rather than the\n"
            "chosen-vs-rejected distinction you actually want the model to learn.\n"
            "Worth checking what the separating axis actually is (e.g. cluster\n"
            "or classify on a few cheap features: length, case name overlap,\n"
            "clause category) before trusting a downstream scorer's performance."
        )
    else:
        print(
            f"Mean cosine ({mean:.4f}) falls between {LOW_THRESHOLD} and "
            f"{HIGH_THRESHOLD}.\n"
            "Neither near-duplicate nor trivially separable at the aggregate\n"
            "level. Check the distribution shape above -- a bimodal split would\n"
            "still indicate two different pair-construction regimes mixed\n"
            "together in the dataset."
        )
    print(f"\nStd of {std:.4f} -- {'high variance, check per-example outliers via the JSONL output' if std > 0.15 else 'relatively tight distribution around the mean'}.")


if __name__ == "__main__":
    main()