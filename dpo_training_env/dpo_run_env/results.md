# Lexical baseline vs. embedding-gemma: pairwise ranking accuracy

984 anchor/chosen/rejected triplets from `dpo_training_env/datasets/dpo_dataset_revised.json`. Metric: how often does the scorer rank `chosen` above `rejected` when both are scored against the anchor clause (ties count as a loss).

## Category distribution

| Category | N | % of dataset |
|---|---|---|
| Cap On Liability | 811 | 82.4% |
| Third Party Beneficiary | 95 | 9.7% |
| Non-Compete | 78 | 7.9% |

## Overall pairwise accuracy

| Scorer | Accuracy | Wins | Ties | Losses | N scored | N skipped (missing score) |
|---|---|---|---|---|---|---|
| TF-IDF cosine | 57.7% | 568 | 0 | 416 | 984 | 0 |
| Okapi BM25 | 56.6% | 557 | 0 | 427 | 984 | 0 |
| JDAR embedding-gemma (bi_encoder_score) | 50.4% | 488 | 0 | 481 | 969 | 15 |

## Per-category pairwise accuracy

Cap On Liability is 82% of the dataset, so the overall number above is essentially the Cap On Liability number. Breaking out the minority categories (Third Party Beneficiary, Non-Compete) shows whether the scorers generalize or are just picking up Cap-On-Liability-specific vocabulary.

| Category | N | TF-IDF cosine | Okapi BM25 | embedding-gemma |
|---|---|---|---|---|
| Cap On Liability | 811 | 57.7% | 55.7% | 49.0% (n=806) |
| Third Party Beneficiary | 95 | 63.2% | 63.2% | 63.8% (n=94) |
| Non-Compete | 78 | 51.3% | 57.7% | 47.8% (n=69) |

## Notes

- TF-IDF and BM25 are implemented from scratch (numpy + stdlib only); both are fit on the pooled corpus of all 1,968 chosen+rejected passages, with the anchor clause scored as a query against that vocabulary.
- The embedding-gemma column is not recomputed here — it's the original `bi_encoder_score` (cosine similarity under `google/embeddinggemma-300m`) recovered from JDAR's own triplet-construction artifacts (`datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/`) by joining on the exact (anchor, passage) text pair. A small number of rejected passages (see 'N skipped') could not be matched back to a scored artifact and are excluded from that column's accuracy rather than silently zero-filled.
- Ties (equal score on both sides) are counted as losses for the ranking-accuracy metric but reported separately in results.json.
- Caveat on the embedding-gemma column: `bi_encoder_score` was JDAR's *retrieval* signal (used to pull candidate reasoning windows near an anchor clause), not the chosen/rejected *selection* signal. The pairing script (`dpo_dataset_construction/build_dpo_pairs.py`) picked chosen vs. rejected purely from the rule-based D1/D2/D3 judicial-content/coherence scorer, with no bi-encoder score in the selection criteria. So embedding-gemma landing near chance here isn't evidence the embedding model is a weak similarity scorer in general — it's evidence that anchor-passage embedding similarity and the D1/D2/D3 pass/fail label are close to independent in this dataset. Read this column as 'how well would a similarity-based DPO scorer have agreed with the judicial-content-based label', not as a fair fight between TF-IDF/BM25 and embedding-gemma on the same task.
