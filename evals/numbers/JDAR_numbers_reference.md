# JDAR — Master Numbers Reference

**Paper:** *Longer, Not Better: What Corpus-Grounded Legal Preference Pairs Actually Encode* (title candidate; current draft title under revision)
**Compiled:** 10 August 2026
**Purpose:** Single authoritative source for every number that appears in the paper. Ordered by the section-revision sequence: §3 → §4 → §5 → §6 → §2 → §1 → §7 → Limitations → Appendix.

**Tag legend**


| Tag      | Meaning                                                                     |
| -------- | --------------------------------------------------------------------------- |
| `NEW`    | Not in either uploaded draft. Must be added.                                |
| `ADD-ON` | Already in the draft; needs an additional number or qualifier attached.     |
| `REVISE` | In the draft with a wrong, stale, or unsupported value. Must be corrected.  |
| `REMOVE` | In the draft and must be deleted; the data does not support it.             |
| `KEEP`   | In the draft, correct, no change. Listed so it is not accidentally dropped. |
| `VERIFY` | Must be checked against source before submission.                           |


**Rounding convention adopted for the paper.** Accuracies to one decimal place as percentages. Correlation coefficients to three decimals. Character counts to one decimal where the source provides it, otherwise integer. p-values below 1e-3 in scientific notation to two significant figures. Confidence intervals to one decimal place as percentages. Apply this uniformly; the current drafts mix conventions.

---



## SECTION 0 — Global constants

These recur across multiple sections. Cite them once, consistently.


| ID  | Quantity                                                   | Exact value              | Tag    | Note                                                                            |
| --- | ---------------------------------------------------------- | ------------------------ | ------ | ------------------------------------------------------------------------------- |
| G01 | CUAD clause categories screened                            | 30                       | KEEP   |                                                                                 |
| G02 | Categories passing screening                               | 6                        | KEEP   |                                                                                 |
| G03 | Categories excluded at screening                           | 24                       | KEEP   | insufficient volume or unresolved quality issues                                |
| G04 | Categories dropped for pipeline bugs                       | 3                        | KEEP   | IP Ownership Assignment, Covenant Not To Sue, Competitive Restriction Exception |
| G05 | Categories reported                                        | 3                        | KEEP   | Cap On Liability, Third Party Beneficiary, Non-Compete                          |
| G06 | Anchor clauses after deduplication and vagueness filtering | 117                      | VERIFY | **Conflicts with G07.** See verification queue V1.                              |
| G07 | Unique prompts in `dpo_dataset_revised.json`               | 85                       | VERIFY | **Conflicts with G06.**                                                         |
| G08 | COLD Cases corpus size                                     | ~8.36M US legal opinions | KEEP   |                                                                                 |
| G09 | Retrieval window size                                      | 150,000 characters       | KEEP   | **Confirmed 10 Aug.** Previously flagged unverified; now resolved.              |
| G10 | Total scored rows                                          | 12,294                   | KEEP   | The population for all D2/D3 correlation analyses                               |
| G11 | Rows passing the AND gate                                  | 1,069                    | KEEP   |                                                                                 |
| G12 | Valid preference pairs constructed                         | 984                      | KEEP   |                                                                                 |
| G13 | Eligible negative pool                                     | 3,037                    | KEEP   |                                                                                 |
| G14 | Negatives used                                             | 984                      | KEEP   |                                                                                 |
| G15 | Negatives unused                                           | 2,053                    | KEEP   | Not excluded for cause; forms the held-out pool                                 |
| G16 | Same-anchor pairs                                          | 969 (98.5%)              | KEEP   |                                                                                 |
| G17 | Same-category different-anchor fallback pairs              | 15 (1.5%)                | KEEP   |                                                                                 |
| G18 | Held-out evaluation set size                               | 500                      | KEEP   |                                                                                 |
| G19 | Passing threshold                                          | total_score ≥ 2          | KEEP   |                                                                                 |
| G20 | Gate logic                                                 | D1 ∧ D2 ∧ D3             | KEEP   | Replaced an earlier OR gate                                                     |
| G21 | DPO β                                                      | 0.1                      | KEEP   |                                                                                 |
| G22 | LoRA α                                                     | 16                       | KEEP   |                                                                                 |
| G23 | LoRA dropout                                               | 0                        | KEEP   |                                                                                 |
| G24 | Precision                                                  | fp16                     | KEEP   |                                                                                 |
| G25 | Primary model                                              | Llama-3-8B-Instruct      | KEEP   |                                                                                 |


**Category composition of the 984 training pairs** — `KEEP`


| Category                | Pairs   | Share      |
| ----------------------- | ------- | ---------- |
| Cap On Liability        | 811     | 82.4%      |
| Third Party Beneficiary | 95      | 9.7%       |
| Non-Compete             | 78      | 7.9%       |
| **Total**               | **984** | **100.0%** |


---



## SECTION 1 OF THE REVISION — §3 Constructing Judicially-Grounded Preference Pairs



### 3.1 Anchor selection and judicial retrieval


| ID   | Item                   | Value                                                                                        | Tag    | Interpretation / instruction                                                                                                                                                                         |
| ---- | ---------------------- | -------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.01 | Screening funnel       | 30 → 6 → 3 categories                                                                        | KEEP   | Already in draft. Keep as is.                                                                                                                                                                        |
| 3.02 | Anchors after dedup    | 117                                                                                          | VERIFY | Reconcile with the 85 unique prompts in the dataset file before submission. If 117 is the pre-pairing count and 85 is the post-pairing count, say so explicitly — a reviewer who counts will notice. |
| 3.03 | Cap On Liability share | 82.4%                                                                                        | KEEP   | The sentence "all per-category results in this paper are conditioned on this imbalance" stays.                                                                                                       |
| 3.04 | Retrieval window size  | 150,000 characters                                                                           | KEEP   | Confirmed. Remove any hedging language around it.                                                                                                                                                    |
| 3.05 | Bi-encoder             | `embedding-gemma-300m`                                                                       | KEEP   |                                                                                                                                                                                                      |
| 3.06 | Cross-encoder          | `ms-marco-MiniLM-L6-v2`                                                                      | KEEP   |                                                                                                                                                                                                      |
| 3.07 | Scoring axis statement | "Scoring is always pairwise between (anchor, candidate), and not (anchor, chosen, rejected)" | KEEP   | This sentence is load-bearing. Do not soften it.                                                                                                                                                     |




### 3.2 Quality scoring and pair construction


| ID   | Item                                      | Value                                                                                                | Tag  | Interpretation / instruction                                                                                                                                                                                                                                                                                        |
| ---- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.08 | Scored rows                               | 12,294                                                                                               | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.09 | Rows passing AND gate                     | 1,069                                                                                                | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.10 | Pairs yielded                             | 984                                                                                                  | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.11 | **Flag configuration of** `chosen`        | **(D1, D2, D3) = (1, 1, 1)**                                                                         | NEW  | Derived: the AND gate requires all three; total_score = 3 ≥ 2. Makes Eq. 1 exact rather than approximate. **VERIFY against source (V2).**                                                                                                                                                                           |
| 3.12 | **Flag configuration of** `rejected`      | **(D1, D2, D3) = (1, 0, 0)**                                                                         | NEW  | Derived: `rejected` requires D1 = 1 and total_score < 2; if total_score is the sum of three binary flags, D2 + D3 < 1 forces both to zero. **This is the single most important structural fact added to §3.** It is what makes the mechanism in §6 provable rather than suggestive. **VERIFY against source (V2).** |
| 3.13 | Eligible negative pool                    | 3,037                                                                                                | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.14 | Negatives used / unused                   | 984 / 2,053                                                                                          | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.15 | Selection method                          | `random.choice()`, uniform over eligible pool                                                        | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.16 | Same-anchor / fallback split              | 969 / 15                                                                                             | KEEP |                                                                                                                                                                                                                                                                                                                     |
| 3.17 | D2 definition                             | binary flag fires when `(interpretive_hits ≥ 1) OR (procedural_hits ≥ 1)`                            | NEW  | Currently the draft says only "judicial-content density". State the actual firing rule. The OR-at-one-hit design is why D2 is only weakly length-permeable — a short passage with one hit passes. This detail is needed for §6 to explain why D2 is the weaker channel.                                             |
| 3.18 | D3 definition                             | binary flag fires when `raw_sentence` matches any term from the anchor's inferred vagueness category | NEW  | Ten categories: effort, time, scope, harm, necessity, industry_norms, knowledge, confidentiality, financial, survival. State this list. It is the fact that makes D3's length permeability intuitive to a reader.                                                                                                   |
| 3.19 | D3 positive rate in the scored population | 2,223 / 12,294 = **18.1%**                                                                           | NEW  | Establishes that D3 is the binding constraint of the three.                                                                                                                                                                                                                                                         |
| 3.20 | D3 negative count                         | 10,071 (81.9%)                                                                                       | NEW  |                                                                                                                                                                                                                                                                                                                     |
| 3.21 | Honesty paragraph                         | "No step in the pipeline compared chosen against rejected."                                          | KEEP | Stays unhedged.                                                                                                                                                                                                                                                                                                     |




### 3.3 Training


| ID   | Item                                  | Value                                                                                                                                                                                                          | Tag    | Interpretation / instruction                                                                                                                                                                                                                              |
| ---- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.22 | SFT data                              | 984 (anchor, chosen) pairs                                                                                                                                                                                     | KEEP   |                                                                                                                                                                                                                                                           |
| 3.23 | DPO data                              | 984 (anchor, chosen, rejected) triplets                                                                                                                                                                        | KEEP   |                                                                                                                                                                                                                                                           |
| 3.24 | Reference policy                      | the SFT checkpoint                                                                                                                                                                                             | KEEP   |                                                                                                                                                                                                                                                           |
| 3.25 | β                                     | 0.1                                                                                                                                                                                                            | KEEP   |                                                                                                                                                                                                                                                           |
| 3.26 | **Prompt, verbatim**                  | *"Given the following contractual clause which falls in the category of {category}, provide judicial reasoning of the kind a court would apply when interpreting clauses that raise similar legal questions:"* | NEW    | **Must be quoted in full.**                                                                                                                                                                                                                               |
| 3.27 | **Analogical supervision disclosure** | —                                                                                                                                                                                                              | NEW    | One sentence stating that the elicited reasoning concerns clauses raising similar legal questions, not the anchor clause itself, and that the supervision is therefore analogical rather than direct. Currently disclosed nowhere. Repeat in Limitations. |
| 3.28 | LoRA citation                         | —                                                                                                                                                                                                              | REMOVE | The draft reads "with Low Rank Adaptation (LoRA) [11], with LoRA [11]". Delete the duplicate.                                                                                                                                                             |




### Artefacts to build in §3


| Artefact              | Content                                                                                                                                                                                                                                                                                                                                                                  | Tag    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| **Eq. 1** (revise)    | Pair-construction predicate, rewritten to state the exact flag configurations from 3.11 and 3.12 rather than the threshold inequality alone. Replace `\textsc` with `\operatorname` or `\mathrm` so it renders in math mode.                                                                                                                                             | REVISE |
| **Algorithm 1** (new) | Pseudocode for pair construction. Inputs: anchor set A, candidate windows C, scorers D1/D2/D3, threshold τ = 2. Body: score, apply AND gate, partition into passing and failing sets per anchor, uniform-sample one failing candidate. **The point of the pseudocode is that no line of it compares** `chosen` **to** `rejected` — that must be visible from reading it. | NEW    |
| **Figure 4** (new)    | Pipeline schematic. Every arrow anchor→candidate. The absent chosen↔rejected edge drawn explicitly as a dashed non-edge. **D2 and D3 annotated as length-permeable gates.**                                                                                                                                                                                              | NEW    |


---



## SECTION 2 OF THE REVISION — §4 Audit Protocol


| ID   | Item                                    | Value                                                                      | Tag    | Interpretation / instruction                                                                                                                                                                                                                    |
| ---- | --------------------------------------- | -------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.01 | Number of scorers                       | **6** (was 5)                                                              | REVISE | Five content-reading scorers plus the length baseline.                                                                                                                                                                                          |
| 4.02 | Number of method families               | **4 content families + 1 surface baseline**                                | REVISE | The draft's "four families / five scorers" ambiguity must be stated once and consistently: sparse lexical (TF-IDF, BM25), dense embedding, discriminative reranking, trained policy — plus a surface baseline that is not a family.             |
| 4.03 | Evaluation set                          | 500 held-out pairs from the 2,053 unused negatives                         | KEEP   |                                                                                                                                                                                                                                                 |
| 4.04 | **Same-set claim**                      | —                                                                          | REVISE | The draft states all five scorers were evaluated on the same 500. **This is false.** `held_out_baseline_results.json` and `evaluation_results.json` are different 500-pair draws from the same pool. Disclose with a footnote on the DPO row.   |
| 4.05 | Unique anchors, baseline file           | **19**                                                                     | NEW    |                                                                                                                                                                                                                                                 |
| 4.06 | Unique anchors, DPO evaluation file     | **20**                                                                     | NEW    |                                                                                                                                                                                                                                                 |
| 4.07 | Anchors shared between the two files    | **12**                                                                     | NEW    |                                                                                                                                                                                                                                                 |
| 4.08 | Largest anchor cluster, held-out 500    | **277 pairs = 55.4%**                                                      | NEW    | The single most important disclosure in §4.                                                                                                                                                                                                     |
| 4.09 | Top three anchor clusters, held-out 500 | **446 pairs = 89.2%**                                                      | NEW    |                                                                                                                                                                                                                                                 |
| 4.10 | Unique anchors, training 984            | **85**                                                                     | NEW    |                                                                                                                                                                                                                                                 |
| 4.11 | Largest anchor cluster, training 984    | **214 pairs**                                                              | NEW    |                                                                                                                                                                                                                                                 |
| 4.12 | Top three clusters, training 984        | **394 pairs = 40.0%**                                                      | NEW    |                                                                                                                                                                                                                                                 |
| 4.13 | Held-out category composition           | Cap On Liability **492**, Non-Compete **4**, Third Party Beneficiary **4** | NEW    | **98.4% one category.** More extreme than the training split's 82.4%. Must be disclosed here, not only in Limitations.                                                                                                                          |
| 4.14 | CI method                               | anchor-clustered bootstrap, 4,000 resamples                                | REVISE | Replaces the Wilson/bootstrap contradiction. Report clustered intervals as primary.                                                                                                                                                             |
| 4.15 | **Few-clusters caveat**                 | 19 clusters, max share 55.4%                                               | NEW    | Cluster bootstrap is a large-cluster-count procedure. At 19 clusters with one holding 55.4% of the data it under-covers. One sentence stating that the reported intervals should be read as optimistic. This is the Cameron & Miller §VI point. |
| 4.16 | Tie handling                            | ties counted as losses                                                     | KEEP   |                                                                                                                                                                                                                                                 |
| 4.17 | Pairwise ranking accuracy               | formal definition                                                          | NEW    | See Eq. 4 below. Satisfies the guide's "performance metrics in more detail".                                                                                                                                                                    |
| 4.18 | Length baseline definition              | s_len(a, c) = |c|                                                          | NEW    | See Eq. 5. State explicitly that it ignores its first argument.                                                                                                                                                                                 |
| 4.19 | McNemar                                 | applied pairwise across all six scorers                                    | ADD-ON | The claim already exists in the draft; the numbers now exist too (see §5).                                                                                                                                                                      |




### Artefacts to build in §4


| Artefact        | Content                                                                                   | Tag |
| --------------- | ----------------------------------------------------------------------------------------- | --- |
| **Eq. 4** (new) | Acc(s) = (1/N) Σᵢ 𝟙[ s(aᵢ, cᵢʷ) > s(aᵢ, cᵢˡ) ], with ties resolving to 0.                | NEW |
| **Eq. 5** (new) | s_len(a, c) = |c|. One clause of surrounding prose: this scorer does not read the anchor. | NEW |


---



## SECTION 3 OF THE REVISION — §5 Results



### 5A. Table 1 — Held-out discrimination accuracy (REVISE)

All accuracies on 500 held-out pairs. Naive intervals retained here for internal reference only; **the paper reports clustered intervals.**


| Scorer                | Family                | Accuracy  | Naive Wilson 95% CI | **Clustered 95% CI (report this)** | Excludes 50%?    |
| --------------------- | --------------------- | --------- | ------------------- | ---------------------------------- | ---------------- |
| Embedding cosine      | dense                 | 53.2%     | [48.8, 57.6]        | **[44.6, 58.2]**                   | No               |
| Cross-encoder         | discriminative rerank | 53.8%     | [49.4, 58.2]        | **[51.1, 60.1]**                   | Yes (marginally) |
| Okapi BM25            | sparse lexical        | 58.0%     | [53.7, 62.3]        | **[47.5, 64.6]**                   | No               |
| DPO implicit reward † | trained policy        | 58.4%     | [54.0, 62.8]        | **[56.2, 67.1]**                   | Yes              |
| TF-IDF cosine         | sparse lexical        | 60.0%     | [55.7, 64.3]        | **[51.5, 64.3]**                   | Yes (marginally) |
| **Length only**       | surface baseline      | **63.0%** | —                   | **[60.6, 67.5]**                   | Yes              |


† Computed on a separate 500-pair draw; 12 of 19 anchors shared.

**Interpretation instructions**

- `REVISE` — The draft sentence "TF-IDF and BM25 are the only two scorers whose 95% confidence intervals exclude 50%" is **false under clustered intervals**. Under clustering, BM25 spans chance, and the cross-encoder and DPO reward no longer do. The pattern reverses. Rewrite the sentence entirely rather than adjusting it.
- `REVISE` — Every occurrence of "55–58%" and "54.7–58.4%" must become **53.2–60.0%**. These appear in the abstract, §1, and §7.
- `NEW` — The load-bearing sentence of the table: **no content-reading scorer exceeds the length baseline.**
- Note the ordering inversion worth one sentence: a bag-of-words method (TF-IDF, 60.0%) outscores the DPO implicit reward (58.4%) trained on 984 judicially-grounded pairs, and a character count outscores both.



### 5B. Length asymmetry in the pairs (NEW)


| ID   | Quantity                                     | Exact value           | Note                                                                                                                                                                 |
| ---- | -------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.01 | Pairs where `chosen` is longer, training 984 | **615 / 984 = 62.5%** | Wilson [59.4, 65.5]                                                                                                                                                  |
| 5.02 | Pairs where `chosen` is longer, held-out 500 | **63.0%** (315/500)   | Clustered [60.6, 67.5]                                                                                                                                               |
| 5.03 | `chosen` mean characters                     | **649.0**             |                                                                                                                                                                      |
| 5.04 | `chosen` median characters                   | **622.0**             |                                                                                                                                                                      |
| 5.05 | `chosen` mean words                          | **97.2**              |                                                                                                                                                                      |
| 5.06 | `rejected` mean characters                   | **571.3**             |                                                                                                                                                                      |
| 5.07 | `rejected` median characters                 | **537.0**             |                                                                                                                                                                      |
| 5.08 | `rejected` mean words                        | **85.6**              |                                                                                                                                                                      |
| 5.09 | **Mean difference**                          | **77.7 characters**   | 649.0 − 571.3                                                                                                                                                        |
| 5.10 | **Median of the per-pair difference**        | **82.0 characters**   |                                                                                                                                                                      |
| 5.11 | Difference of the medians                    | **85.0 characters**   | 622.0 − 537.0. **Not the same quantity as 5.10.** Report one or the other with the correct label; do not conflate.                                                   |
| 5.12 | Mean word difference                         | **11.6 words**        | 97.2 − 85.6                                                                                                                                                          |
| 5.13 | Wilcoxon signed-rank p                       | **3.6 × 10⁻¹⁹**       | Reported as p < 1e-18. Note in prose that at n = 984 the p-value establishes the asymmetry is not noise but says nothing about its magnitude; the magnitude is 5.09. |


**Length baseline accuracy per category (training 984)** — `NEW`


| Category                | Length-baseline accuracy |
| ----------------------- | ------------------------ |
| Cap On Liability        | **63.3%**                |
| Non-Compete             | **62.8%**                |
| Third Party Beneficiary | **55.8%**                |




### 5C. Length is not a proxy for lexical overlap (NEW)


| ID   | Quantity                                       | Exact value        | Interpretation                                                              |
| ---- | ---------------------------------------------- | ------------------ | --------------------------------------------------------------------------- |
| 5.14 | Length–TF-IDF decision agreement, training 984 | **48.4%**          | Below chance agreement. The two scorers are tracking different things.      |
| 5.15 | McNemar, length vs TF-IDF, training 984        | **p ≈ 5.2 × 10⁻⁵** | The difference in their per-pair decisions is not attributable to sampling. |


This pair of numbers pre-empts the obvious objection that the length baseline is just lexical overlap in disguise. Keep them adjacent in the prose.

### 5D. Table 3 — Scorer agreement and McNemar (NEW)

Held-out 500. Lower triangle = fraction of pairs decided identically. Upper triangle = McNemar p.


| Pair                      | Agreement | McNemar p  | Interpretation                                                                                                                                                             |
| ------------------------- | --------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TF-IDF × BM25             | **0.848** | **0.302**  | Indistinguishable. Confirms the shared term-weighting lineage; they are one family, not two.                                                                               |
| TF-IDF × Embedding        | 0.628     | **0.015**  | Distinguishable                                                                                                                                                            |
| TF-IDF × Cross-encoder    | 0.642     | **0.025**  | Distinguishable                                                                                                                                                            |
| TF-IDF × Length           | **0.594** | **0.326**  | **Indistinguishable in decision pattern on the held-out set**, despite 5.14/5.15 on the training set. Flag this tension explicitly rather than picking the convenient one. |
| BM25 × Embedding          | 0.628     | 0.091      | Not distinguishable                                                                                                                                                        |
| BM25 × Cross-encoder      | 0.646     | 0.133      | Not distinguishable                                                                                                                                                        |
| BM25 × Length             | 0.586     | 0.095      | Not distinguishable                                                                                                                                                        |
| Embedding × Cross-encoder | 0.626     | **0.884**  | Indistinguishable                                                                                                                                                          |
| Embedding × Length        | **0.518** | **0.0019** | Distinguishable                                                                                                                                                            |
| Cross-encoder × Length    | **0.524** | **0.0034** | Distinguishable                                                                                                                                                            |


**Interpretation note.** 5.14/5.15 (training 984) and the TF-IDF × Length row (held-out 500) point in opposite directions. Do not suppress either. The honest statement is that length and lexical overlap are separable on the training pairs and not separably distinguishable on the held-out 500, and that the held-out set's 19-anchor structure is the likely reason. This belongs in §5 prose and again in Limitations.

### 5E. Circularity and geometry (KEEP — already in draft, still valid)


| ID   | Quantity                                             | Exact value         |
| ---- | ---------------------------------------------------- | ------------------- |
| 5.16 | cos(anchor, chosen), 984                             | **0.6371**          |
| 5.17 | cos(anchor, rejected), 984                           | **0.6318**          |
| 5.18 | Margin                                               | **0.0053**          |
| 5.19 | Fraction chosen closer                               | **54.67%**          |
| 5.20 | cos(chosen, rejected) mean, 984                      | **0.6042**          |
| 5.21 | cos(chosen, rejected) median                         | **0.6041**          |
| 5.22 | cos(chosen, rejected) std                            | **0.0540**          |
| 5.23 | p10 / p90                                            | **0.5366 / 0.6665** |
| 5.24 | p25 / p75                                            | **0.5700 / 0.6385** |
| 5.25 | min / max                                            | **0.4385 / 0.8476** |
| 5.26 | Fraction > 0.85                                      | **0.0000**          |
| 5.27 | Fraction < 0.40                                      | **0.0000**          |
| 5.28 | Isotropy control, 500 random unrelated windows: mean | **0.0733**          |
| 5.29 | Isotropy control: median                             | **0.0702**          |
| 5.30 | Isotropy control: min / max                          | **0.0170 / 0.2320** |
| 5.31 | Usable dynamic range                                 | **0.07 – 0.85**     |




### 5F. DPO implicit reward internals (ADD-ON)


| ID   | Quantity                             | Exact value  | Note                                                                                                                                                                    |
| ---- | ------------------------------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.32 | Raw accuracy, held-out               | **57.6%**    | Wilson [53.2, 62.0]                                                                                                                                                     |
| 5.33 | Length-normalised accuracy, held-out | **58.4%**    | The reported figure                                                                                                                                                     |
| 5.34 | Mean reward gap                      | **0.003454** |                                                                                                                                                                         |
| 5.35 | Median reward gap                    | **0.001092** |                                                                                                                                                                         |
| 5.36 | Mean / median ratio                  | **≈ 3.16×**  | Right-skewed. The model is indifferent on most held-out pairs and decisive on a minority tail. Never investigated (Set B). State as an open observation, not a finding. |
| 5.37 | Percent positive gap                 | **58.4%**    |                                                                                                                                                                         |


**Interpretation instruction.** Length-normalisation raised accuracy from 57.6% to 58.4% — a 0.8 point change. State that normalisation mitigates but does not remove the length dependence, and that the normalised figure still sits below a raw character count at 63.0%. Cite Park et al. (2024) here.

---



## SECTION 4 OF THE REVISION — §6 What the Supervision Actually Encodes

**This section changes more than any other. The old claim is retired; a new and stronger one replaces it.**

### 6A. The mechanism — D2 and D3 length permeability (ALL NEW)

Population for both analyses: **N = 12,294** scored rows. Length field: `len(raw_sentence)`.

**D2 — judicial-content density**


| Statistic                                     | Exact value       |
| --------------------------------------------- | ----------------- |
| Spearman ρ (D2 hit count × chars)             | **+0.1262**       |
| Spearman p                                    | **7.6 × 10⁻⁴⁵**   |
| Pearson r                                     | **+0.1198**       |
| Pearson p                                     | **1.6 × 10⁻⁴⁰**   |
| ρ² (rank variance explained)                  | **0.0159 = 1.6%** |
| Point-biserial r_pb (D2 flag × chars)         | **+0.0584**       |
| r_pb p                                        | **8.9 × 10⁻¹¹**   |
| Mean chars, D2 = 0                            | **577**           |
| Mean chars, D2 = 1                            | **610**           |
| **Δ mean chars**                              | **+32**           |
| Δ as share of the observed 77.7-char pair gap | **41.2%**         |


**D3 — semantic coherence with the anchor's vagueness category**


| Statistic                                     | Exact value            |
| --------------------------------------------- | ---------------------- |
| Spearman ρ (D3 hit count × chars)             | **+0.1498**            |
| Spearman p                                    | **1.25 × 10⁻⁶²**       |
| Pearson r                                     | **+0.1621**            |
| Pearson p                                     | **3.64 × 10⁻⁷³**       |
| ρ² (rank variance explained)                  | **0.0224 = 2.2%**      |
| Point-biserial r_pb (D3 flag × chars)         | **+0.1495**            |
| r_pb p                                        | **2.33 × 10⁻⁶²**       |
| Mean chars, D3 = 0                            | **568.4** (n = 10,071) |
| Mean chars, D3 = 1                            | **641.5** (n = 2,223)  |
| **Δ mean chars**                              | **+73.0**              |
| Δ as share of the observed 77.7-char pair gap | **94.0%**              |
| D3 positive rate                              | **18.1%**              |


**The decomposition — the paper's central new result**


| Comparison                                | Value                                     |
| ----------------------------------------- | ----------------------------------------- |
| Observed `chosen` mean length             | **649.0 chars**                           |
| Population mean length of D3 = 1 passages | **641.5 chars**                           |
| Discrepancy                               | **7.5 chars (1.2% of the chosen mean)**   |
| Observed `rejected` mean length           | **571.3 chars**                           |
| Population mean length of D3 = 0 passages | **568.4 chars**                           |
| Discrepancy                               | **2.9 chars (0.5% of the rejected mean)** |
| Observed pair gap                         | **77.7 chars**                            |
| Gap predicted by the D3 marginal alone    | **73.0 chars**                            |
| Unexplained residual                      | **4.7 chars (6.0% of the gap)**           |


**Interpretation — write this carefully, it is the paper's argument**

1. The gate configuration derived in 3.11/3.12 means `chosen` is exactly the D3 = 1 condition and `rejected` is exactly the D3 = 0 condition.
2. D3 fires when a candidate contains any term from the anchor's inferred vagueness category. Longer passages have more opportunity to contain such a term. The gate is therefore length-permeable by construction.
3. The population length difference between D3 = 1 and D3 = 0 passages (73.0 chars) reproduces the observed `chosen`–`rejected` gap (77.7 chars) to within 4.7 characters.
4. **Consequence: Claim 4 (the supervisory signal is threshold-clearing) now predicts Claim 3 (length dominance).** The pipeline's supervision correlates with length because the gate that defines it is a length-permeable keyword match. This converts a reported oddity into an explained mechanism.
5. **Required caveat on ρ².** Both correlations are weak in variance-explained terms (1.6% and 2.2%). State plainly that variance explained is the wrong statistic for a selection mechanism: the relevant quantity is the group-mean difference induced by conditioning on the gate, not the population correlation. Cite Elwert & Winship (2014) and Sullivan & Feinn (2012). **A reviewer who sees ρ² = 0.022 without this paragraph will conclude the mechanism claim is overreached.**
6. **Required precision on the vocabulary.** Describe this as *selection on a length-correlated criterion*, not as *collider bias*. Length is a genuine cause of gate passage; the association is causal, not an artefact of conditioning on a common effect. Using "collider" here would be technically imprecise.
7. **Required caveat on the marginal.** The D3 = 1 / D3 = 0 marginal comparison is a population statistic. `chosen` and `rejected` are drawn within-anchor. The near-exact reproduction (4.7 chars) is strong evidence but not a within-anchor decomposition. Disclose this and place the (D2 × D3) cell-mean decomposition in Future Work. **See verification queue V3.**



### 6B. The human probe — reported honestly (REVISE, heavy)


| ID   | Slice             | n   | Correct | Accuracy  | Wilson 95% CI | Clause-clustered 95% CI |
| ---- | ----------------- | --- | ------- | --------- | ------------- | ----------------------- |
| 6.01 | All presentations | 30  | 21      | **70.0%** | [52.1, 83.3]  | **[50.0, 87.1]**        |
| 6.02 | Unique clauses    | 20  | 14      | **70.0%** | [48.1, 85.5]  | **[50.0, 90.0]**        |
| 6.03 | Repeats only      | 10  | 7       | **70.0%** | [38.6, 90.9]  | —                       |


**Per category, unique anchors — this is the honest view; raw counts are inflated by repeats** — `REVISE`


| Category                | Unique anchors | Correct | Accuracy | Raw (with repeats) |
| ----------------------- | -------------- | ------- | -------- | ------------------ |
| Third Party Beneficiary | 4              | 4       | **100%** | 8/10 (80%)         |
| Non-Compete             | 8              | 4       | **50%**  | 5/10 (50%)         |
| Cap On Liability        | 8              | 6       | **75%**  | 8/10 (80%)         |
| **Total**               | **20**         | **14**  | **70%**  | 21/30 (70%)        |


**Significance block** — `NEW`


| ID   | Test                               | Value      | Interpretation                                                                                                           |
| ---- | ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| 6.04 | p vs 50%, all presentations, naive | **0.0428** | Nominally significant, but treats 30 presentations as 30 independent items when they rest on 20 clauses.                 |
| 6.05 | p vs 50%, unique clauses           | **0.1153** | Not significant.                                                                                                         |
| 6.06 | **p vs the 62.5% length prior**    | **0.4549** | **The number that retires the old claim.** The human result is not distinguishable from what a character count achieves. |
| 6.07 | Bootstrap P(mean ≤ 0.500)          | **0.030**  |                                                                                                                          |
| 6.08 | Bootstrap P(mean ≤ 0.625)          | **0.235**  |                                                                                                                          |


**Reliability and design** — `NEW`


| ID   | Item                                      | Value                                             | Interpretation                                                                                                                                                                                 |
| ---- | ----------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.09 | Rater                                     | single non-expert in law                          |                                                                                                                                                                                                |
| 6.10 | Session duration                          | ~60 minutes (10:09:34 – 11:09:54 UTC, 3 Aug 2026) |                                                                                                                                                                                                |
| 6.11 | Presentations / unique clauses / repeats  | 30 / 20 / 10                                      | Repeats confirmed exact clause repeats                                                                                                                                                         |
| 6.12 | **Intra-rater inconsistency**             | **3 / 10 repeats = 30%**                          | Items: Q14 vs Q10, Q29 vs Q27, Q7 vs Q2. One clause answered wrong → right → wrong across three presentations. This bounds how much signal any single-rater result could carry.                |
| 6.13 | Key was option A                          | 19 / 30                                           |                                                                                                                                                                                                |
| 6.14 | Rater chose A                             | 20 / 30                                           | Binomial p vs 0.5 = **0.0987**                                                                                                                                                                 |
| 6.15 | Accuracy when key was A                   | **78.9%**                                         |                                                                                                                                                                                                |
| 6.16 | Accuracy when key was B                   | **54.5%**                                         |                                                                                                                                                                                                |
| 6.17 | Fisher exact p on the position split      | **0.2252**                                        | Not significant, but the direction is consistent with an order effect and the design was not counterbalanced. Disclose.                                                                        |
| 6.18 | Option texts stored in the response file? | **No**                                            | `blind_eval_binosh_1785755394293.json` records id, category, clause, yourAnswer, correctAnswer, isCorrect only. **A per-item length control is therefore impossible from existing artefacts.** |


**Claims to REMOVE from §6 and §5 entirely**

- "showing the ceiling is in the scorers, not the data"
- "If a non-expert can separate the pairs at 70%, the signal is in general pragmatic inference, not in legal domain expertise"
- "a pragmatic judgment that term-frequency and geometric methods do not attempt"
- Any sentence implying the rater accessed something the scorers could not.

**Framing instruction.** State the null result first and the point estimate second. If §6 says "the human reached 70%, suggesting a signal the scorers miss, though this is not significant", a reviewer reads the first clause and ignores the second.

### 6C. Figure 1 — DPO reward margins (KEEP with fixes)


| ID   | Item       | Value / instruction                                                                                                                   |
| ---- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 6.19 | Framing    | Generalisation-gap contrast only. Margins rise on training pairs; held-out accuracy remains 58.4%; the length baseline reaches 63.0%. |
| 6.20 | Legend fix | Currently displays the raw path `meta-llama_Meta-Llama-3-8B-dpo-from-sft`. Replace with a readable model label.                       |
| 6.21 | Axis fix   | TRL defaults (`train/global_step`, `train/rewards/margins`). Replace with proper axis labels.                                         |


---



## SECTION 5 OF THE REVISION — §2 Related Work


| ID   | Item                                                                                          | Tag    | Instruction                                                                                                                                                                                                                                                                                                        |
| ---- | --------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2.01 | SyLeR (Zhang et al., CIKM 2025, pp. 4117–4127)                                                | KEEP   | Direct precedent for similarity-as-legal-reward.                                                                                                                                                                                                                                                                   |
| 2.02 | LegalΔ (Dai et al., arXiv 2508.12281)                                                         | KEEP   | The pre-written critique this paper quantifies.                                                                                                                                                                                                                                                                    |
| 2.03 | Med-PRM (Yun et al., EMNLP 2025, pp. 16554–16571)                                             | KEEP   |                                                                                                                                                                                                                                                                                                                    |
| 2.04 | Math-Shepherd (Wang et al., ACL 2024, pp. 9426–9439)                                          | KEEP   | The verifiable-outcome contrast.                                                                                                                                                                                                                                                                                   |
| 2.05 | Fin-PRM                                                                                       | VERIFY | The bib entry and the rendered reference disagree. See V4.                                                                                                                                                                                                                                                         |
| 2.06 | CorVer (Fan et al., arXiv 2605.29648)                                                         | VERIFY | Confirm posting date precedes submission.                                                                                                                                                                                                                                                                          |
| 2.07 | Djuhera et al. (ICLR 2026, arXiv 2511.10985), preference coherence **70–80%**                 | ADD-ON | **Reframe.** The old use was "our scorers fall below this band, so the gap is in the instruments." That inference is retired. The new use: this establishes that construction pipelines pass structural and qualitative defects through without human oversight, and this paper identifies a specific such defect. |
| 2.08 | **Singhal, Goyal, Xu, Durrett (2024), COLM 2024, arXiv 2310.03716**                           | NEW    | **Highest-priority addition.** Their finding that a purely length-based reward reproduces most downstream RLHF gains over SFT models is the direct precedent for the length baseline. It converts the baseline from a self-inflicted wound into a recognised method.                                               |
| 2.09 | **Park, Rafailov, Ermon, Finn (2024), Findings of ACL 2024, pp. 4998–5017, arXiv 2403.19159** | NEW    | Length exploitation specifically in DPO; explains why the implicit reward inherits the confound and why length-normalisation is partial.                                                                                                                                                                           |
| 2.10 | **McCoy, Pavlick, Linzen (2019), ACL**                                                        | NEW    | Establishes the genre: a surface heuristic explaining apparent model competence.                                                                                                                                                                                                                                   |
| 2.11 | **Bowman & Dahl (2021), NAACL**                                                               | NEW    | Measurement failure rather than model failure as the bottleneck. Supports §1.                                                                                                                                                                                                                                      |
| 2.12 | **Cameron & Miller (2015), J. Human Resources 50(2):317–372**                                 | NEW    | Grounds the clustered-inference disclosure in §4.                                                                                                                                                                                                                                                                  |
| 2.13 | **Card et al. (2020), EMNLP**                                                                 | NEW    | Grounds the underpowered-probe disclosure in §6 and Limitations.                                                                                                                                                                                                                                                   |
| 2.14 | **Artstein & Poesio (2008), Computational Linguistics 34(4)**                                 | NEW    | Grounds the intra-rater reliability disclosure.                                                                                                                                                                                                                                                                    |
| 2.15 | Azar et al. (2023), IPO, arXiv 2310.12036                                                     | NEW    | Bradley-Terry mis-specification, for Limitations.                                                                                                                                                                                                                                                                  |


**New §2 paragraph required.** A length-in-preference-learning cluster covering 2.08, 2.09, and 2.10, positioned after the domain-PRM paragraph and before the Djuhera paragraph. Its claim: length exploitation is a documented failure mode of preference optimisation in general-domain settings; it has not been examined in corpus-grounded legal preference construction, where the pairs are retrieved rather than generated.

---



## SECTION 6 OF THE REVISION — §1 Introduction


| ID   | Item                                                                                                                                                                         | Tag    | Instruction                                                                                                                                                                                                                                                                               |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.01 | Move 1 — the originating question                                                                                                                                            | KEEP   | Unchanged since day one.                                                                                                                                                                                                                                                                  |
| 1.02 | Move 2 — standard approach and gap                                                                                                                                           | ADD-ON | Add the length-exploitation literature to the gap statement.                                                                                                                                                                                                                              |
| 1.03 | **"54.7–58.4% accuracy band"c0**                                                                                                                                             | REVISE | Stale. Computed on the 984 training pairs. → **53.2–60.0% on 500 held-out pairs.**                                                                                                                                                                                                        |
| 1.04 | **"four scorers"**                                                                                                                                                           | REVISE | → six scorers: five content-reading methods across four families, plus a length baseline.                                                                                                                                                                                                 |
| 1.05 | cos(chosen, rejected) 0.604 vs 0.073                                                                                                                                         | KEEP   | Still correct and still load-bearing.                                                                                                                                                                                                                                                     |
| 1.06 | **"A single-rater blinded evaluation separated 21 of 30 pairs correctly, indicating the pairs encode a quality signal that no tested automatic evaluation method captures"** | REMOVE | The inference is not supported.                                                                                                                                                                                                                                                           |
| 1.07 | Contribution (1) — auditable pipeline                                                                                                                                        | KEEP   |                                                                                                                                                                                                                                                                                           |
| 1.08 | Contribution (2) — convergence result                                                                                                                                        | REVISE | Restate as: five scorers across four families converge at 53.2–60.0%, and none exceeds a length-only baseline at 63.0%.                                                                                                                                                                   |
| 1.09 | **Contribution (3)**                                                                                                                                                         | REVISE | Old: a measurement-gap finding. New: **a mechanism** — the D3 gate is length-permeable, and the population length difference it induces (73.0 chars) reproduces the observed `chosen`–`rejected` asymmetry (77.7 chars), so the supervisory signal is a surface property by construction. |
| 1.10 | Contribution (4) — optional fourth                                                                                                                                           | NEW    | Consider adding: an exploratory single-rater probe that is reported as underpowered, with the design failures documented so the follow-up study can be specified. Only add if the paper has room; it is honest but weak.                                                                  |


---



## SECTION 7 OF THE REVISION — §7 Conclusion


| ID   | Item                                                                                                   | Tag    | Instruction                                                                                                                                                                                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.01 | **Sentence 1**                                                                                         | REVISE | Currently claims a human detects at 70% what the scorers miss. Replace with the mechanism finding.                                                                                                                                                                         |
| 7.02 | **"four automatic scorer families — geometric, lexical, and learned — converge at 55–58%"**            | REVISE | Names three families after promising four, and the band is stale. → six scorers, four content families, 53.2–60.0%, none above 63.0%.                                                                                                                                      |
| 7.03 | "entailment-based or cross-encoder scoring that jointly encodes the pair to close the measurement gap" | REMOVE | The cross-encoder is a tested and failing method at 53.8%. Do not propose it as the fix.                                                                                                                                                                                   |
| 7.04 | Future work                                                                                            | REVISE | Two directions: (a) length-matched negative reselection from the 3,037-item eligible pool, so the interpretive axis is isolated by construction; (b) a counterbalanced, multi-rater, length-controlled human study powered against a 62.5% baseline rather than a 50% one. |
| 7.05 | Length                                                                                                 | KEEP   | ~100 words. Do not expand.                                                                                                                                                                                                                                                 |


---



## SECTION 8 OF THE REVISION — Limitations

Does not count against the page limit. Each item below is a required disclosure with its exact numbers.


| ID  | Limitation                                                   | Numbers to state                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Tag |
| --- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| L01 | **Category imbalance**                                       | Training: Cap On Liability 82.4% (811/984). Held-out: **98.4%** (492/500), with Non-Compete and Third Party Beneficiary at 4 pairs each.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | NEW |
| L02 | **Anchor clustering**                                        | 984 training pairs on **85** anchors; top three hold **394 (40.0%)**. 500 held-out pairs on **19** anchors; the largest holds **277 (55.4%)**, top three hold **446 (89.2%)**.                                                                                                                                                                                                                                                                                                                                                                                                                                          | NEW |
| L03 | **Few-clusters caveat on the intervals**                     | Cluster bootstrap with 19 clusters and a 55.4% maximum cluster share under-covers. Reported intervals should be read as optimistic, not definitive.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | NEW |
| L04 | **Non-commensurable DPO draw**                               | Baseline scorers and the DPO implicit reward were computed on different 500-pair draws from the same pool: **19 vs 20 unique anchors, 12 shared**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | NEW |
| L05 | **Length asymmetry**                                         | `chosen` longer in **62.5%** of the 984 (615/984, Wilson [59.4, 65.5]) and **63.0%** of the held-out 500; mean difference **77.7 characters**, median per-pair difference **82.0**; Wilcoxon p ≈ 3.6 × 10⁻¹⁹.                                                                                                                                                                                                                                                                                                                                                                                                           | NEW |
| L06 | **Partial length-matched analysis**                          | A length-matched subset of the training pairs was constructed: **154 pairs with |Δ length| ≤ 50 characters**. On this subset the length baseline falls to **55.8%** (Wilson 95% CI **[47.9, 63.4]**) and TF-IDF falls to **46.8%** (Wilson 95% CI **[39.1, 54.7]**) — both intervals include chance. **The remaining four scorers were not evaluated on this subset.** A complete length-matched audit of all six scorers is left to future work. *(Decision recorded 10 Aug: this is a Limitations disclosure, not a body result. Not to be reopened until the full draft with Limitations and Appendix is complete.)* | NEW |
| L07 | **Human probe is not a study**                               | Single non-expert rater, 30 presentations across 20 unique clauses, ~60-minute session. **Intra-rater inconsistency 3/10 (30%)** on exact repeats. Clause-clustered CI **[50.0, 87.1]** with a lower bound exactly at chance. **Not distinguishable from the 62.5% length prior (p = 0.4549).**                                                                                                                                                                                                                                                                                                                         | NEW |
| L08 | **No length control on the probe, and none is now possible** | Option texts were not retained in the response file; a per-item length control cannot be recovered from existing artefacts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | NEW |
| L09 | **Order effect not controlled**                              | Key was A in 19/30; the rater chose A in 20/30 (p = 0.0987). Accuracy **78.9%** when the key was A versus **54.5%** when B (Fisher exact p = 0.2252). Not significant, but the presentation order was not counterbalanced.                                                                                                                                                                                                                                                                                                                                                                                              | NEW |
| L10 | **Analogical supervision**                                   | The prompt elicits reasoning about clauses raising similar legal questions, not the anchor clause itself.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | NEW |
| L11 | **Mechanism established at the population level only**       | The D3 marginal comparison (73.0 chars) is computed over the 12,294 scored rows; `chosen` and `rejected` are selected within-anchor. A within-anchor (D2 × D3) cell decomposition was not performed.                                                                                                                                                                                                                                                                                                                                                                                                                    | NEW |
| L12 | **Bradley-Terry mis-specification**                          | DPO inherits a single-latent-scalar total ordering. If negatives fail in mutually incomparable ways, the assumption is violated. Cite Azar et al. (2023).                                                                                                                                                                                                                                                                                                                                                                                                                                                               | NEW |
| L13 | **No expert legal annotation**                               | Anywhere in the pipeline or the evaluation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | NEW |
| L14 | **Single primary model**                                     | Llama-3-8B-Instruct. Ministral-3-14B ruled out for suspected overfitting; gpt-oss-20b showed fp32 loss spikes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | NEW |
| L15 | **No downstream legal task benchmark**                       | The audit measures pair discrimination, not legal reasoning quality.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | NEW |
| L16 | **TF-IDF preprocessing sensitivity**                         | A recomputation on the 984 returned **53.2%** against the original script's **57.7%**; the difference is attributable to preprocessing (sublinear tf, English stopwords, anchor extraction from the prompt template). Disclose so a reimplementation that differs is not read as a contradiction.                                                                                                                                                                                                                                                                                                                       | NEW |
| L17 | **Contradictory length/lexical separability**                | Length and TF-IDF are distinguishable on the 984 (agreement 48.4%, McNemar p ≈ 5.2 × 10⁻⁵) but not on the held-out 500 (agreement 0.594, McNemar p = 0.326).                                                                                                                                                                                                                                                                                                                                                                                                                                                            | NEW |


---



## SECTION 9 OF THE REVISION — Appendix

Does not count against the page limit.


| ID  | Appendix item                                | Contents                                                                                                                                                                          | Tag |
| --- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| A01 | **Training hyperparameters**                 | LoRA α = 16, dropout = 0, fp16, gradient accumulation and checkpointing, DPO β = 0.1, π_ref = SFT checkpoint. **The body currently points here twice and the appendix is empty.** | NEW |
| A02 | **Models explored and ruled out**            | Ministral-3-14B-Reasoning (suspected overfitting); gpt-oss-20b (fp32 loss spikes). The body promises this.                                                                        | NEW |
| A03 | **Full D2 and D3 correlation tables**        | All values from §6A.                                                                                                                                                              | NEW |
| A04 | **Full scorer agreement and McNemar matrix** | All ten pairwise cells from §5D.                                                                                                                                                  | NEW |
| A05 | **Per-category results, all scorers**        | Including the minority categories with their n = 4 held-out counts stated so the reader sees why no inference is drawn.                                                           | NEW |
| A06 | **Blind read instrument**                    | The instruction text shown to the rater, the response schema, and a note that option texts were not retained.                                                                     | NEW |
| A07 | **Length distribution statistics**           | Full quantiles for chosen and rejected, not just means and medians.                                                                                                               | NEW |
| A08 | **Cosine distribution statistics**           | Values 5.20–5.31.                                                                                                                                                                 | NEW |
| A09 | **Dropped categories**                       | IP Ownership Assignment, Covenant Not To Sue, Competitive Restriction Exception, with the specific pipeline bug for each.                                                         | NEW |
| A10 | **Prompt template, verbatim**                | Full string as in 3.26.                                                                                                                                                           | NEW |


---



## NUMBERS THAT MUST NEVER APPEAR AGAIN

These are stale, computed on the wrong population, or unsupported. Search the LaTeX for each.


| Stale value                                                | Where it appears    | Replacement                                                                            |
| ---------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------- |
| 54.7%                                                      | §1                  | 53.2%                                                                                  |
| 56.6%                                                      | earlier tables      | 58.0% (BM25, held-out)                                                                 |
| 57.7%                                                      | earlier tables      | 60.0% (TF-IDF, held-out) — but see L16                                                 |
| "54.7–58.4% band"                                          | §1                  | 53.2–60.0%                                                                             |
| "55–58%"                                                   | abstract, §5, §7    | 53.2–60.0%                                                                             |
| "four scorers" / "four automatic scorer families converge" | abstract, §1, §7    | six scorers; five content-reading methods across four families, plus a length baseline |
| "the ceiling is in the scorers, not the data"              | abstract, §5        | the mechanism statement from 1.09                                                      |
| Wilson intervals as the primary reported CI                | §4, Table 1 caption | anchor-clustered bootstrap, 4,000 resamples                                            |


---



## VERIFICATION QUEUE


| ID  | Item                                                                                                                        | Cost   | Blocking?                                               |
| --- | --------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------- |
| V1  | Reconcile 117 anchor clauses (G06) against 85 unique prompts (G07)                                                          | 10 min | Yes — §3                                                |
| V2  | Confirm from source that `total_score` is the sum of three binary flags, making `chosen` = (1,1,1) and `rejected` = (1,0,0) | 15 min | **Verified** — flags are strictly binary 0/1            |
| V3  | (D2 × D3) cell means and counts, within-anchor if feasible                                                                  | 20 min | No — improves §6, otherwise goes to L11                 |
| V4  | Fin-PRM citation: bib has Yang/Ming/Wang with arXiv 2502.14825; rendered reference has Zhu et al. with arXiv 2508.15202     | 5 min  | Yes — one is wrong                                      |
| V5  | `zhou2024bootstrapping` full author list                                                                                    | 5 min  | Yes                                                     |
| V6  | `ma2025stepreward` full author list                                                                                         | 5 min  | Yes                                                     |
| V7  | CorVer arXiv posting date precedes submission                                                                               | 5 min  | Yes                                                     |
| V8  | Held-out 500: confirm 315/500 for the 63.0% length figure                                                                   | 2 min  | No                                                      |
| V9  | LaTeX still in `[preprint]` mode                                                                                            | 1 min  | **Yes — desk-reject risk under any double-blind venue** |


---



## FIGURE AND TABLE MANIFEST


| ID          | Artefact                                                                                     | Section  | Status                                                                            |
| ----------- | -------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------- |
| Table 1     | Held-out discrimination, six scorers, family column, clustered CIs, DPO dagger               | §5       | REVISE                                                                            |
| Table 2     | Dataset description with per-category length statistics                                      | §3 or §5 | NEW                                                                               |
| Table 3     | Scorer agreement and McNemar                                                                 | §5       | NEW                                                                               |
| Table 4     | D2 and D3 length permeability with the decomposition                                         | §6       | NEW                                                                               |
| Table 5     | Blind read, per-category unique-anchor, with the significance block                          | §6       | NEW                                                                               |
| Figure 1    | DPO reward margins                                                                           | §6       | KEEP, fix legend and axes                                                         |
| Figure 2    | Discrimination forest plot                                                                   | §5       | BUILT, polish                                                                     |
| Figure 3    | Length distribution, chosen vs rejected, median difference annotated                         | §5       | NEW                                                                               |
| Figure 4    | Pipeline schematic with the absent chosen↔rejected edge and D2/D3 marked as length-permeable | §3       | NEW                                                                               |
| Figure 5    | Cosine scale                                                                                 | —        | DROPPED — raw arrays unavailable, and length has displaced cosine as the headline |
| Eq. 1       | Pair-construction predicate, restated with exact flag configurations                         | §3       | REVISE                                                                            |
| Eq. 2       | Bradley-Terry                                                                                | §3       | KEEP                                                                              |
| Eq. 3       | Implicit reward                                                                              | §3       | KEEP                                                                              |
| Eq. 4       | Pairwise ranking accuracy                                                                    | §4       | NEW                                                                               |
| Eq. 5       | Length scorer s_len(a, c) = |c|                                                              | §4       | NEW                                                                               |
| Algorithm 1 | Pair construction pseudocode                                                                 | §3       | NEW                                                                               |


