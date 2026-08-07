# Four-scorer baseline on held-out pairs

500 pairs sampled (seed=42) from `datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/held_out_pairs.jsonl` (235567 total eligible held-out pairs, never used to build the 984-pair DPO training set). Metric: pairwise ranking accuracy -- does the scorer rank `chosen` above `rejected` when both are scored against `prompt_anchor`? Ties count as a loss. 95% CI is the Wald normal approximation.

## Category distribution

| Category | N | % of sample |
|---|---|---|
| Cap On Liability | 492 | 98.4% |
| Non-Compete | 4 | 0.8% |
| Third Party Beneficiary | 4 | 0.8% |

## Overall pairwise accuracy

| Scorer | Accuracy | 95% CI | Wins | Ties | Losses | N |
|---|---|---|---|---|---|---|
| TF-IDF cosine | 60.0% | [55.7, 64.3] | 300 | 0 | 200 | 500 |
| Okapi BM25 | 58.0% | [53.7, 62.3] | 290 | 0 | 210 | 500 |
| Raw embedding cosine (freshly computed, embedding-gemma-300m) | 53.2% | [48.8, 57.6] | 266 | 0 | 234 | 500 |
| Cross-encoder (ms-marco-MiniLM-L6-v2) | 53.8% | [49.4, 58.2] | 269 | 0 | 231 | 500 |

## Per-category pairwise accuracy

| Category | N | TF-IDF | BM25 | embedding-gemma (fresh) | cross-encoder |
|---|---|---|---|---|---|
| Cap On Liability | 492 | 60.2% | 58.1% | 52.8% | 53.5% |
| Non-Compete | 4 | 50.0% | 50.0% | 75.0% | 75.0% |
| Third Party Beneficiary | 4 | 50.0% | 50.0% | 75.0% | 75.0% |

## Notes

- TF-IDF and BM25 are fit on the pooled chosen+rejected corpus of this 500-pair sample only (not the full 235,567-row held-out pool), matching the training-pairs baseline methodology in `dpo_training_env/dpo_run_env/baseline_scorer.py`.
- embedding-gemma and the cross-encoder are both scored fresh here (no reuse of construction-time scores), anchor vs. chosen and anchor vs. rejected, matching how the DPO implicit-reward held-out number was produced.
- This panel is directly comparable to the DPO implicit-reward held-out result (`evaluation_results.json`, `inspect_implicit_rewards.py`), since both are scored on unseen pairs from the same held-out pool, unlike the training-pairs baseline which is scored on the 984 pairs the model trained on.
