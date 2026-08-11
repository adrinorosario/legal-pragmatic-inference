# JDAR — Research Continuity Codex

**Project:** Judicial Decision-Aligned Reasoning (JDAR) — conference paper
**Researcher:** Adrino
**Supervisor:** Dr. Rajesh Khanna (no venue preference stated; requires full draft)
**Compiled:** 2 August 2026
**Status at compile time:** All experiments run. Framing settled. Section 3 not yet written. Draft due to supervisor in 3 days.

---

## SECTION 1 — Executive Research Overview

### What this project is

A conference paper reporting an **audit of corpus-grounded automatic preference construction for legal reasoning**. It began as a positive-results paper about a novel reward model and became a negative-results / analysis paper after four independent measurements converged on the same ceiling. The reframing is deliberate, evidence-driven, and final.

### The originating question (unchanged since day one — preserve this)

> Why do language models reason over legal language the way they reason over any other language, when legal language carries depth, extended-context dependency, factual grounding, and heavy pragmatic inference?

This narrowed to: *can a model pragmatically infer the operative meaning of a contractual clause the way a court or lawyer would?*

The paper's final result is a partial empirical answer to the original question, and this framing should be preserved in the introduction. The answer the data supports:

> Models treat legal language like ordinary language in part because the supervision and evaluation signals we build cannot distinguish judicially apt reasoning from merely legal-sounding prose. Pragmatic inference is invisible to the pipeline because nothing in the pipeline ever constructed a contrast along the pragmatic axis.

### Current thesis (latest form — use verbatim as the spine)

> Corpus-grounded automatic preference construction for contractual clause interpretation produces pairs whose chosen and rejected members are semantically near-equivalent, and preference optimization over them yields a discriminator no better than lexical overlap — indicating that threshold-based pipeline supervision does not encode the pragmatic distinctions it is intended to capture.

### The headline result

Four independent scoring methods, held-out or unsupervised, converge on the same narrow accuracy band:

| Scorer | Pairwise accuracy | 95% CI | Training required |
|---|---|---|---|
| Raw embedding cosine (freshly computed) | 54.7% | [51.5, 57.8] | none |
| Okapi BM25 | 56.6% | [53.5, 59.7] | none |
| TF-IDF cosine | 57.7% | [54.6, 60.8] | none |
| DPO implicit reward (length-normalised, held out) | 58.4% | [54.0, 62.8] | SFT + DPO on 984 pairs |

The full pipeline beats bag-of-words by 0.7 points, well inside both intervals. **Training on judicially-grounded preference pairs bought approximately what word counting gives for free.** The convergence — not any single number — is the argument. One method in the mid-50s is a failed experiment; four is a property of the data.

**Held-out panel — all four scorers on the same held-out split.** The table above mixed splits: three scorers on the 984 *training* pairs, one (the DPO implicit reward) on 500 *held-out* pairs. To close that gap, TF-IDF, BM25, freshly-computed embedding-gemma-300m cosine, and the ms-marco-MiniLM-L6-v2 cross-encoder were all re-run on 500 pairs sampled (seed=42) from `datasets/jdar_triplet_extracted_on_cuad_and_cold_cases/dpo_dataset_construction/held_out_pairs.jsonl`. Script and outputs: [evals/held_out_baseline_eval.py](../../evals/held_out_baseline_eval.py), [evals/held_out_baseline_results.md](../../evals/held_out_baseline_results.md), [evals/held_out_baseline_results.json](../../evals/held_out_baseline_results.json).

| Scorer | Held-out pairwise accuracy | 95% CI |
|---|---|---|
| Raw embedding cosine (freshly computed) | 53.2% | [48.8, 57.6] |
| Cross-encoder (ms-marco-MiniLM-L6-v2) | 53.8% | [49.4, 58.2] |
| Okapi BM25 | 58.0% | [53.7, 62.3] |
| TF-IDF cosine | 60.0% | [55.7, 64.3] |
| DPO implicit reward (length-normalised, held out) | 58.4% | [54.0, 62.8] |

Same band, same convergence, now on data the model never trained on and the lexical scorers were never fit for selection. **Caveat, disclosed not buried:** this held-out panel and the DPO implicit-reward row are not drawn from the identical pool. The DPO number (Section 7, Experiment 5) came from 500 of the 2,053 unused eligible negatives from triplet construction. This new panel is sampled from `held_out_pairs.jsonl`, a much larger (235,567-row) and far more skewed held-out pool — the random 500-sample landed 492 Cap On Liability / 4 Third Party Beneficiary / 4 Non-Compete (98.4% Cap On Liability, vs. 82.4% in the 984-pair training set). The per-category breakdown in `held_out_baseline_results.md` is therefore not reliable for the two minority categories (n=4 each) and should not be read for anything beyond the overall row. The five numbers still converge to the same ~53–60% band, which is the point, but the minority-category comparison a reviewer would want is not available from this sample and would need category-stratified re-sampling to get.

### Intended venue

Two live options, both established during this work:

- **NLLP 2026** (Natural Legal Language Processing workshop, co-located with EMNLP, Budapest, 28 Oct 2026). Direct submission deadline 11 August 2026; ARR-commitment route 27 August 2026; notification 15 September. 8 pages long / 4 pages short. NLP-methods audience. CFP explicitly welcomes work going beyond benchmark performance toward legal validity and reasoning quality — a good fit for an audit paper.
- **JURIX 2026** (Toulouse, December 2026). Paper deadline 5 September 2026, recommended abstract 28 August. Single-blind, IOS Press format, CORE C. AI & Law audience — will scrutinise the legal claims harder than the reward-modelling claims.

**Recommended:** NLLP as a 4-page short paper. The evidence base — one primary model, no human annotation, no downstream benchmark, three CUAD categories with severe imbalance — supports a tight short paper, not a padded long one. JURIX is the fallback if more time is needed.

### Current maturity

- Dataset construction: **complete and audited**
- SFT and DPO training: **complete** (Llama-3-8B-Instruct primary)
- Evaluation: **complete** — all planned measurements run
- Paper: **not started**. Section 3 spine agreed but unwritten.

### Blockers

1. §3 Method unwritten (spine exists, see Section 6 below)
2. Two high-value analyses outstanding: the 20-pair blind read, and the decisive-tail analysis
3. Venue not chosen

### What this project is explicitly NOT trying to do

- It is not claiming JDAR improves legal reasoning quality.
- It is not proposing a process reward model. JDAR scores nothing at training or inference time.
- It is not claiming to avoid Bradley-Terry preference modelling. DPO *is* Bradley-Terry; the assumption is inherited and disclosed.
- It is not a benchmark paper, a resource paper alone, or a downstream-task-performance paper.

### Compromises knowingly accepted

- Severe category imbalance (82.4% Cap On Liability) — disclosed, not corrected
- No human or expert annotation of any kind — no ground truth on judicial aptness
- No downstream legal task benchmark
- Single primary model (Llama-3-8B); Ministral-3-14B ruled out for suspected overfitting; gpt-oss-20b had fp32 loss spikes and is an optional add-on at most
- Root-cause fix (adversarial negatives) deferred to future work

---

## SECTION 2 — Collaborator Profile

**Background:** University researcher. Computer vision and ML background, transitioning into NLP for this project. New to formal research; explicitly building intuition for how research should work. Runs a parallel, unrelated project on Hevea brasiliensis disease detection (different supervisor) — do not conflate.

**Working style:**
- High-intensity sprints combining learning and building simultaneously, not sequential learn-then-build
- Codes while learning; breaks things and debugs; derives insight through guided questioning rather than being handed answers
- Writes first, then iterates with feedback. **Does not work from pre-built templates.** Give spines and claims-per-section, not filled-in prose.
- Maintains `JDAR_dataset_audit_notes.md` as a single source of truth, structured to map onto paper sections
- Runs parallel experiments proactively to preserve optionality, then narrows on evidence
- Flags time pressure explicitly; responds well to being redirected back to the primary deliverable

**Intellectual standards (demonstrated, not claimed):**
- Verifies against actual code and data rather than assuming standard recipes. When asked what rule assigned `rejected`, he read `build_dpo_pairs.py` and `triplet_quality_scorer.py` function signatures and reported them precisely, explicitly noting that reasoning from general DPO conventions would have been wrong.
- Independently identified the most important limitation in the entire project (the threshold-vs-preference distinction) and flagged it as belonging in the methods section before being prompted.
- Explicitly refuses to treat AI-generated explanations as corroborating evidence.
- Pushes back on proposed experiments when the cost/benefit is poor, and has been right to do so.

**What has worked in collaboration:**
- Direct correction of factual errors, without softening. He corrected a factual error, expected the prior framing removed, and it was.
- Confidence intervals attached to every number
- Being told plainly when a hypothesis of the AI's was wrong (the anisotropy hypothesis was refuted by his control run and this was conceded immediately — that concession was productive)
- Naming the second-order effect of a choice before recommending it
- Diverge / Add-on framing for scope decisions

**What has caused friction:**
- Proposing experiments without accounting for retraining cost (the negative re-selection proposal was correctly rejected on those grounds)
- Any framing that would let a weak result be reported as a strong one

**Do not re-explain:** DPO mechanics, LoRA/QLoRA, TRL trainer APIs, cosine similarity, CUAD structure, bi-encoder vs cross-encoder retrieval, what SFT is.

**Do carry forward:** he has *not* yet worked through the DPO→Bradley-Terry derivation line by line. This was assigned and is still outstanding. It is now load-bearing for §3.5.

---

## SECTION 3 — Thesis Evolution (preserve this chronology; it is the paper's honesty)

| Stage | Framing | Why abandoned |
|---|---|---|
| 1 | "Why do models reason over legal language like ordinary language?" | Never abandoned — this is still the introduction's frame |
| 2 | "Can a fine-tuned LLM pragmatically infer contractual meaning as courts do?" | Narrowed to operationalisable form; still the research question |
| 3 | "JDAR is a process reward model / verifier-style similarity scorer, not a Bradley-Terry preference RM" | **False.** JDAR scores nothing during training or inference. It filtered a dataset once, offline. |
| 4 | "DPO lets us avoid Bradley-Terry" | **False.** DPO's loss is the Bradley-Terry negative log-likelihood with reward reparameterised as β[log π_θ − log π_ref]. The sigmoid-over-reward-difference *is* the DPO loss. |
| 5 | "JDAR is a corpus-grounded automatic preference-labeling function" | Still an overstatement, but closer. Superseded by 6. |
| 6 | **Current.** "The supervisory signal is *cleared-threshold vs. did-not-clear*, not adjudicated preference; the pairs are near-paraphrases; four independent scorers converge at ~55–58%." | Current and evidence-supported. |

### Framings explicitly rejected — do not resurrect

- **JDAR as a PRM.** It rewards no process. Never used at train or inference time.
- **"Verifier, not Bradley-Terry."** Internally inconsistent with using DPO.
- **`rewards/margins` as evidence of success.** It is a training diagnostic. Rising margins mean the model learned whatever separates chosen from rejected — which, given 0.604 cosine between them, is a surface property, not reasoning quality. It may appear in an appendix as a sanity check, or not at all.
- **Anisotropy as the explanation for the low scores.** Hypothesised, then refuted by the researcher's own control (random unrelated judicial pairs at 0.073 mean cosine). The embedding space is near-isotropic. The instrument is fine. Whitening was cancelled as unnecessary — this decision is final.
- **Asking the trained DPO adapter what the failed triplets lack.** Circular: the model was trained on threshold-defined preferences and would report the threshold function back in fluent prose.

### What would falsify the current thesis

- Expert annotation showing chosen genuinely is judicially superior to rejected at high agreement, despite the 0.604 cosine. (Untested; no annotation exists.)
- A scorer of any kind achieving substantially above ~65% on held-out pairs. Nothing tested has.
- Evidence that the decisive-tail minority (see Section 7) reflects genuine interpretive distinctions rather than threshold artifacts. **This is currently untested and is the single largest open question.**

### What the thesis depends on that is not established

- That the 20-pair blind read will confirm a human also cannot reliably separate chosen from rejected. **If a human separates them easily, the conclusion narrows substantially** — from "the pairs carry no quality signal" to "the pairs carry a signal no automatic scorer tested can see." Both are publishable; they are different papers. **Run this before writing §5.**

---

## SECTION 4 — Retrospective Walkthrough

Read this in order. It is the arc of the project as it actually happened.

**Phase 1 — Motivation.** Started from a question about why legal language is treated as generic language by LLMs. Narrowed to pragmatic inference over contractual clauses.

**Phase 2 — Construction.** Built a pipeline pairing CUAD contractual clauses (anchors) with retrieved judicial reasoning windows from the Harvard COLD Cases corpus (~8.36M opinions, streamed from HuggingFace). Scored candidate windows with embedding-gemma-300m (bi-encoder) and ms-marco-MiniLM-L6-v2 (cross-encoder), plus three heuristic dimensions D1/D2/D3.

**Phase 3 — Audit and pruning.** Found a pipeline bug allowing triplets to pass on D1+D2 while failing D3. Fixed by requiring AND logic across critical dimensions. Dropped three categories (IP Ownership Assignment, Covenant Not To Sue, Competitive Restriction Exception) for confirmed bugs and quality failures. Established that deeply negative cross-encoder scores flagged real false positives, and that demoting the cross-encoder to metadata-only had been a mistake. Final dataset: 984 pairs.

**Phase 4 — Training.** SFT then DPO on three models in parallel. Llama-3-8B-Instruct selected as primary. Ministral-3-14B-Reasoning ruled out (suspected overfitting). gpt-oss-20b showed loss spikes, diagnosed as likely fp32 precision. A checkpoint-loss incident on Kaggle (no explicit `save_model` call) required an SFT rerun.

**Phase 5 — Literature scan and first correction.** Discovered that similarity-based reward for legal reasoning already exists (SyLeR) and that its critique is already published (LegalΔ: reliance on surface-level matching). Discovered the domain-PRM slot is filling (Med-PRM medical, Fin-PRM financial, CorVer factual QA), so "a domain PRM for law" is a template fill, not a novelty claim. Simultaneously established that DPO *is* Bradley-Terry, invalidating the paper's planned central distinction.

**Phase 6 — The audit experiments.** Five measurements, in order:

1. *Lexical baseline vs stored bi_encoder_score.* TF-IDF 57.7%, BM25 56.6%, stored gemma score 50.4%. 15 rows had no stored score.
2. *Fresh recompute from scratch.* cos(anchor,chosen)=0.6371, cos(anchor,rejected)=0.6318, margin 0.0053, ranking accuracy 54.7%. **This ruled out circularity** — if anchor-candidate cosine had been the selection rule, this would have returned ~100%.
3. *Code verification of pair construction.* Established that all scoring is anchor→candidate; nothing ever compared chosen to rejected; the supervisory signal is threshold-clearing.
4. *cos(chosen, rejected) distribution.* Mean 0.6042, std 0.0540, unimodal, no near-duplicates (frac>0.85 = 0), no far outliers (frac<0.4 = 0).
5. *Random-pair anisotropy control.* Unrelated judicial windows: mean 0.0733, max 0.2320. **Near-isotropic. Refuted the anisotropy explanation and made the 0.604 finding much stronger** — chosen and rejected are near-paraphrases in a space with full dynamic range.
6. *DPO implicit reward on held-out unused negatives.* 500 pairs drawn from the 2,053 unused eligible negatives. Raw accuracy 57.6%, length-normalised 58.4%, mean gap 0.00345, median gap 0.00109.

**Phase 7 — Reframing.** The mean gap being three times the median revealed a right-skewed distribution: the model is indifferent on most held-out pairs and decisive on a minority tail. This is the most interesting unexplored observation in the project.

---

## SECTION 5 — Question Sets (write answers to these before drafting)

These are the outstanding exercises from the whole conversation, consolidated and ordered by value. Several were assigned and not completed. **The two marked CRITICAL should be done before §4 and §5 are written.**

### Set A — CRITICAL: the blind read (≈45 min)

Sample 20 triplets stratified across the three categories. Strip labels. Read anchor, then both reasonings. Record which you judge more judicially apt. Unblind.

1. What was your accuracy?
2. If you could tell easily, what feature were you using? Name it concretely.
3. If you could not tell, what does that say about D2/D3 as quality gates?
4. Does your answer differ by category? Non-Compete sits at chance for every scorer tested — can you separate Non-Compete pairs yourself?
5. Which of these does your result support: (a) the pairs carry no quality signal, or (b) the pairs carry a signal no tested scorer can see? These are different papers.

### Set B — CRITICAL: the decisive tail (≈1 hr)

Take the 50 held-out pairs with the largest positive implicit-reward gap and the 50 nearest zero.

1. What distinguishes them? Check: category membership, length difference between chosen and rejected, `total_score` gap, cos(chosen, rejected).
2. If the decisive pairs are systematically those with a large `total_score` gap — what have you shown? (Answer: that the model learned the threshold function, and exactly how much contrast it needs to fire.)
3. If the decisive pairs are just those where chosen is longer — what have you shown? (Answer: length bias. This goes in Limitations, not Results.)
4. If the decisive tail is disproportionately Third Party Beneficiary — the only category above chance for every scorer — what does that imply about whether Cap On Liability, at 82% of the data, contributed any signal at all?

### Set C — the axes exercise (≈20 min)

Similarity is always similarity *along an axis*, determined by what is compared to what. For each of the three axes below, write one sentence on what a HIGH score means and one on what a LOW score means:

1. anchor → candidate (measures topicality)
2. reference → generated (measures reasoning alignment)
3. chosen → rejected (measures negative difficulty)

If you cannot fill six clean sentences, you do not yet know what your results section claims. **This cleaned-up paragraph belongs in §3 and preempts the sharpest reviewer question in the paper.**

### Set D — the negative definition exercise (≈30 min)

Complete this sentence three different ways, one per CUAD category:

> "A rejected reasoning is correctly rejected when it ______."

Then for each: does `total_score < 2` actually detect that? If not for a given category, you have found the boundary of what the dataset supports, and that sentence goes in Limitations. If you cannot complete the sentence at all for Non-Compete, that is why Non-Compete sits at chance for every scorer.

### Set E — the Bradley-Terry assumption (≈30 min, still outstanding)

Bradley-Terry assumes a single latent scalar quality score over responses with a total ordering.

1. Does legal interpretive reasoning satisfy that assumption?
2. In your triplets, is `rejected` worse in one consistent direction, or wrong in mutually incomparable ways across the three categories?
3. If the latter, BT is mis-specified for your data. Write that as a citable limitation (Azar et al. 2023 is the source).

### Set F — framing questions for the draft

1. In one sentence with no hedging: what is the paper's contribution?
2. What is the single strongest objection a reviewer will raise, and what is your one-paragraph answer?
3. If a reviewer says "this is just a failed experiment," what is the sentence that distinguishes your paper from that?
4. Which venue, and why? What does that audience scrutinise hardest?
5. What is the one figure or table a reader will remember? (Candidate: the four-scorer convergence table.)

---

## SECTION 6 — §3 Method spine (agreed, unwritten)

Write against this. Each subsection has a claim it must land.

**3.1 Anchor selection.** CUAD, three categories, counts (811/95/78). Vagueness criterion. **State the imbalance here, not in Limitations** — 82.4% Cap On Liability conditions every result in the paper.

**3.2 Judicial passage retrieval.** COLD Cases corpus, window extraction, embedding-gemma-300m bi-encoder, ms-marco-MiniLM-L6-v2 cross-encoder. **One sentence stating both are anchor→candidate.** Load-bearing; everything in §4 depends on the reader holding it.

**3.3 Quality dimensions.** D1 anchor-only (vagueness of the clause). D2 candidate-only (judicial-content density). D3 anchor×candidate (whether the candidate contains terms from the anchor's assigned vagueness category — scope / knowledge / necessity / etc.). State the AND-logic gate fix and why OR produced false positives.

**3.4 Pair construction.** `chosen` = cleared threshold. `rejected` = same-anchor window with `total_score < 2` and `d1_anchor_vagueness == 1`. 98.5% same-anchor (969/984), 1.5% same-category fallback (15/984). Selection among eligible negatives by `random.choice()`.

Then the honesty paragraph, unsoftened: **no step compared chosen against rejected.** Both are independently-scored anchor-window matches, one of which cleared threshold. The supervisory signal is *cleared-threshold vs. did-not-clear*, not adjudicated judicial preference. A reviewer who reads this in the method section trusts §4. One who discovers it themselves rejects the paper.

**3.5 Training.** SFT then DPO on Llama-3-8B-Instruct, LoRA, hyperparameters, Kaggle. State plainly that DPO's loss is the Bradley-Terry NLL with implicit reward r(x,y) = β[log π_θ(y|x) − log π_ref(y|x)], so the BT assumption is inherited, not avoided. Then note that this same implicit reward is what §4 uses as a scorer — putting that in §3 makes §4 read as designed rather than improvised.

### Suggested full structure

| Section | Status | Content |
|---|---|---|
| 1 Introduction | not started | Write LAST. Frame on the originating question. |
| 2 Related Work | not started | Position against SyLeR, LegalΔ, Med-PRM, Math-Shepherd |
| 3 Method | spine agreed, unwritten | As above |
| 4 Audit | data complete | Four-scorer table, cos distributions, isotropy control |
| 5 Analysis | needs Sets A & B | Blind read, decisive tail, what D2/D3 capture |
| 6 Limitations | outline exists | Imbalance, no annotation, BT mis-specification, length bias |
| 7 Future Work | decided | Adversarial negatives; entailment-based scoring |

---

## SECTION 7 — Empirical State (all numbers, verbatim)

### Dataset
- File: `dpo_training_env/datasets/dpo_dataset_revised.json`
- 984 anchor/chosen/rejected triplets
- Cap On Liability 811 (82.4%), Third Party Beneficiary 95 (9.7%), Non-Compete 78 (7.9%)
- Eligible negative pool: 3,037. Used: 984. **Unused: 2,053** (not discarded for cause — simply not selected by `random.choice()`)

### Pipeline facts (verified against source, not assumed)
- `build_dpo_pairs.py`: `rejected` is a retrieved judicial `raw_sentence` window with `total_score < 2` and `d1_anchor_vagueness == 1`. Not model-generated. Not a perturbation of `chosen`.
- Selection: `same_anchor` 969/984 (98.5%); `same_category_diff_anchor` fallback 15/984 (1.5%)
- `triplet_quality_scorer.py`: `score_dimension_1(anchor)`; `score_dimension_2(sentence)`; `score_dimension_3(anchor, sentence)`
- `bi_encoder_score` and `cross_encoder_score` fields: computed `cuad_anchor` vs `raw_sentence` — anchor→candidate, never chosen→candidate
- **No step in the pipeline ever compared `chosen` against `rejected`**

### Experiment 1 — Lexical baselines vs stored bi_encoder_score
Metric: pairwise ranking accuracy, chosen above rejected against anchor. Ties count as loss.

| Scorer | Acc | Wins | Ties | Losses | N | Skipped |
|---|---|---|---|---|---|---|
| TF-IDF cosine | 57.7% | 568 | 0 | 416 | 984 | 0 |
| Okapi BM25 | 56.6% | 557 | 0 | 427 | 984 | 0 |
| Stored `bi_encoder_score` | 50.4% | 488 | 0 | 481 | 969 | 15 |

Per category (with 95% CIs):

| Category | N | TF-IDF | BM25 | embedding-gemma |
|---|---|---|---|---|
| Cap On Liability | 811 | 57.7% [54.3,61.1] | 55.7% [52.3,59.1] | 49.0% [45.5,52.5] (n=806) |
| Third Party Beneficiary | 95 | 63.2% [53.1,73.3] | 63.2% | 63.8% [53.7,73.9] (n=94) |
| Non-Compete | 78 | 51.3% [40.2,62.4] | 57.7% [46.6,68.8] | 47.8% [36.0,59.6] (n=69) |

Only two statistically defensible statements: TF-IDF weakly beats chance on the majority category; the stored bi-encoder score does not discriminate at all. TPB and Non-Compete are underpowered — all three scorers are mutually indistinguishable there.

### Experiment 2 — Fresh recompute
- mean cos(anchor, chosen) = 0.6371
- mean cos(anchor, rejected) = 0.6318
- mean margin = 0.0053
- fraction chosen closer = 0.5467 [51.5, 57.8]

**Interpretation: not circular.** If anchor→candidate cosine had been the selection criterion, this would return ~100%.

### Experiment 3 — cos(chosen, rejected)
n=984 · mean 0.6042 · median 0.6041 · std 0.0540 · p10/p90 0.5366/0.6665 · p25/p75 0.5700/0.6385 · min/max 0.4385/0.8476 · frac>0.85 = 0.0000 · frac<0.4 = 0.0000. Unimodal, bulk in [0.50, 0.70).

### Experiment 4 — Isotropy control (the pivotal run)
500 random pairs of unrelated judicial windows, different anchors, different categories:
- mean 0.0733 · median 0.0702 · min 0.0170 · max 0.2320

**Near-isotropic space. Available dynamic range ≈ 0.07 to 0.85.** This refuted the anisotropy hypothesis and simultaneously strengthened the main finding: `chosen` and `rejected` sit at 0.604 in a space where unrelated legal text sits at 0.073. They are near-paraphrases, and the encoder is right about that.

### Experiment 5 — DPO implicit reward, held out
500 pairs from the 2,053 unused eligible negatives (model never saw them). r(x,y) = β[log π_θ(y|x) − log π_ref(y|x)], SFT checkpoint as reference.
- raw accuracy 57.6% [53.2, 62.0]
- length-normalised accuracy 58.4% [54.0, 62.8]
- mean gap 0.003454 · median gap 0.001092 · % positive 58.4%

**Unexplored: mean is 3× median.** Right-skewed. Half the held-out pairs have essentially zero gap; the aggregate is carried by a minority tail. The model is indifferent on most pairs and decisive on a few. Set B investigates this.

### Infrastructure
Kaggle notebooks (GPU-limited; `trainer.save_model()` and `tokenizer.save_pretrained()` must be wired immediately after `trainer.train()`, followed by reload-and-verify). TRL (`SFTTrainer`, `DPOConfig`), Unsloth, LoRA/QLoRA.

---

## SECTION 8 — Open Questions Registry

| # | Question | Priority | Next action |
|---|---|---|---|
| 1 | Can a human separate chosen from rejected? | **Critical** | Set A blind read |
| 2 | What drives the decisive tail in implicit rewards? | **Critical** | Set B |
| 3 | Is BT mis-specified for this data (incomparable negatives)? | High | Set E; cite Azar et al. |
| 4 | Does length bias explain the implicit-reward gap? | High | Falls out of Set B |
| 5 | Venue: NLLP short vs JURIX | High | Decide before writing §1 |
| 6 | Is "implicature" the right term, given legal drafting violates Gricean cooperation? | Medium | Read Marmor before §1 |
| 7 | Did Cap On Liability contribute any signal at all? | Medium | Set B question 4 |
| 8 | Would score-gap-maximising negative selection raise contrast? | Low / Future Work | Requires retraining |
| 9 | Do the three dropped categories generalise? | Low | Optional out-of-domain probe |

---

## SECTION 9 — Memory Compression Layer

**Thesis in one sentence:** Corpus-grounded automatic preference construction for contractual clauses yields near-paraphrase pairs, and four independent scorers — geometric, lexical, and learned — all top out at 55–58%, showing the ceiling is in the pairs, not the methods.

**Top 5 sources:** LegalΔ (arXiv 2508.12281) · Rafailov et al. 2023 (DPO) · Med-PRM (EMNLP 2025) · "When Data is the Algorithm" (arXiv 2511.10985) · Ethayarajh 2019.

**Top 3 blockers:** §3 unwritten · Sets A and B not run · venue undecided.

**Immediate next action:** Run Set A (blind read), then write §3 against the spine in Section 6.

**Most fragile assumption:** That a human also cannot separate the pairs. Untested. If false, the paper's conclusion narrows from "no quality signal" to "no scorer tested can see the signal."

**Researcher in 5 bullets:** verifies against source code, not conventions · writes first, rejects templates · wants pushback over validation · new to formal research, strong ML background · three days to a supervisor draft.
