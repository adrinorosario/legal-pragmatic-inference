# JDAR — Research Continuity Transfer (v2)

**Paper:** *Threshold-Clearing Is Not Preference: Auditing Automatic Preference Pairs for Contractual Interpretation* (title under revision — see §3)
**Researcher:** Adrino Rosario James
**Co-authors:** Rajesh Khanna, Deepthi Das
**Institution:** CHRIST (Deemed to be University)
**Compiled:** 10 August 2026
**Supersedes:** the 2 August codex and the 7 August transfer file. Where they conflict with this, **this file wins.**

**Why this file exists.** Between 7 and 10 August, analysis of three uploaded result files changed a load-bearing claim in the paper. The measurement-gap framing — "a human sees signal the scorers cannot" — is no longer supported at the strength the current draft asserts. §6 and §7 need rewriting. The rest of the paper survives intact and is arguably stronger. The next conversation's job is that rewrite.

---
---

# ARTIFACT 1 — RESEARCH CONTINUITY CODEX

---

## SECTION 1 — Executive Research Overview

### What this project is

A conference paper reporting an **audit of corpus-grounded automatic preference construction for legal reasoning**. It has been reframed three times, each time on evidence:

1. **Positive results** — a novel legal reward model. Abandoned when four scorers converged in a narrow band.
2. **Negative results** — "the ceiling is in the data; the pairs carry no quality signal." Abandoned when a human blind read returned 70%.
3. **Measurement gap** — "the ceiling is in the instruments; the pairs carry signal no scorer can see." **Now itself under revision** (see below).
4. **Current, forming** — an audit finding that no method tested, trained or untrained, exceeds what a character count achieves; the human probe was underpowered to separate pragmatic discrimination from a length confound.

### The originating question (unchanged since day one — preserve this)

> Why do language models reason over legal language the way they reason over any other language, when legal language carries depth, extended-context dependency, factual grounding, and heavy pragmatic inference?

Narrowed to: *can a model pragmatically infer the operative meaning of a contractual clause the way a court or lawyer would?*

### What changed on 10 August, and why it matters

Three files were analysed: `held_out_baseline_results.json`, `evaluation_results.json`, `dpo_dataset_revised.json`, plus `blind_eval_binosh_1785755394293.json`.

**Finding 1 — a length-only classifier beats every scorer tested.**
`chosen` is longer than `rejected` in **62.5%** of the 984 training pairs (615/984, Wilson [59.4, 65.5]) and **63.0%** of the 500 held-out pairs. Wilcoxon signed-rank p ≈ 3.6e-19 on the 984; median length difference 82 characters. A classifier that reads nothing — picks whichever passage has more characters — outperforms TF-IDF (60.0%), the DPO implicit reward (58.4%), BM25 (58.0%), the cross-encoder (53.8%), and embedding cosine (53.2%).

**Finding 2 — the human's 70% is not distinguishable from the length prior.**
Rater scored 21/30 (70.0%). Against the 62.5% length prior, **p = 0.4549**. Under clause-clustered bootstrap (20 unique clauses, not 30 independent items), the 95% CI is **[0.500, 0.871]** — the lower bound sits exactly on chance.

**Finding 3 — the option texts were not stored in the blind-read response file.**
`blind_eval_binosh_1785755394293.json` records `id`, `category`, `clause`, `yourAnswer` (A/B), `correctAnswer` (A/B), `isCorrect`. It does **not** record which passage was shown as A vs B, nor the option texts. **A per-item length control is therefore impossible from existing artefacts.** It would require re-exporting the 30 presented pairs with their option texts.

**Finding 4 — the "same held-out 500" claim in §4 is false.**
`held_out_baseline_results.json` (4 baselines) and `evaluation_results.json` (DPO implicit reward) are different draws. 19 vs 20 unique anchors, only 12 shared. The codex disclosed this; the paper does not.

**Finding 5 — severe anchor clustering everywhere.**
Held-out 500 → **19 unique anchors**; top anchor has 277 pairs (55.4%), top three have 446 (89.2%).
Training 984 → **85 unique anchors**; top anchor has 214 pairs, top three have 394 (40%).
Held-out category split → Cap On Liability 492, Non-Compete 4, Third Party Beneficiary 4 (98.4% one category).
Every CI in the current draft is a naive binomial over n and is therefore too narrow.

**Finding 6 — intra-rater inconsistency of 30%.**
The 10 repeats were confirmed exact clause repeats. 3 of 10 were answered inconsistently with their anchor presentation (Q14 vs Q10, Q29 vs Q27, Q7 vs Q2). One clause (Q2's) was answered wrong, then right, then wrong across three presentations.

### Researcher's stated position on the length finding (10 August)

Adrino's view: the rater reported relying on pragmatic understanding and inference; treating the result as a length comparison would frame the rater's ability quantitatively rather than qualitatively and undermine the point about human intelligence. He proposes deferring a length-controlled human study to future GRPO work over a larger corpus.

**The agreed middle path** (his decision to accept or reject):
- Report length as a **scorer row** in the results table — a property of the pairs, not a claim about the rater.
- One Limitations sentence disclosing the length asymmetry and that the blind read was not length-controlled.
- Say nothing about the rater's cognition.
- Defer the length-controlled multi-rater study to Future Work.

**What cannot be deferred:** the sentence in the current draft claiming the human accessed signal the scorers cannot. That specific claim is not supported at p = 0.45.

### Intended venue

**NLLP 2026** (co-located with EMNLP, Budapest, 28 Oct 2026). Long papers 8 pages + references; short 4 pages. **Appendices, references, Limitations, ethics, acknowledgements do not count against the limit.** Review is **double-blind** — anonymisation violations are desk-rejected. Original direct deadline 11 Aug 2026; ARR commit 27 Aug. **Researcher reports the deadline has been extended — UNVERIFIED, check `https://nllpw.org/workshop/call/` first thing.**

Fallback: **JURIX 2026** (Toulouse, Dec 2026), paper deadline 5 Sep, abstract 28 Aug, single-blind, IOS Press, CORE C.

### Current state

- Body: ~5 pages, all seven sections drafted, deduplicated
- Target: 8-page long paper
- Limitations: **empty**
- Appendix: **empty**
- `custom.bib`: exists, ~26 entries
- Figures: Figure 1 (DPO margins) exists; discrimination forest plot built 10 Aug; pipeline schematic not built
- Scripts built 10 Aug: `make_figure2.py`, `analyse_blind_read.py`, `make_discrimination_figure.py`

### What this project is explicitly NOT trying to do

- Not claiming JDAR improves legal reasoning quality
- Not proposing a process reward model — JDAR scores nothing at train or inference time
- Not claiming to avoid Bradley-Terry — DPO *is* BT; inherited and disclosed
- Not a benchmark, resource-only, or downstream-performance paper
- **Not a human study.** The blind read is a single-rater, non-length-controlled, underpowered probe

---

## SECTION 2 — Collaborator Profile

**Background.** University researcher at a Centre for AI. Computer vision and ML background, transitioning into NLP. New to formal research; explicitly building intuition for how research should work. Parallel unrelated project on *Hevea brasiliensis* disease detection under a different supervisor — **do not conflate.**

**Standing instructions (verbatim intent, follow these):**

1. Four lenses in every response: **tutor** (fundamentals grasped?), **researcher** (method sound?), **supervisor** (serves goals and timeline?), **layman** (could he explain it simply?).
2. Before answering a constrained task, state what the work is *for*.
3. Reason explicitly about **second-order effects**; flag when a shortcut now creates a gap later, before recommending.
4. When a task pulls away from the core approach, offer labelled **Diverge** and **Add-on** options. Never silently redirect.
5. On detecting a knowledge gap: name it plainly, give **1–3 specific named readings**, and **one exercise** combining a critical question with a practical task.
6. **Push back and probe. Rigor over comfort.**
7. Ground claims in sources; search when unsure.
8. Direct tone. **No emojis, no flashy language.**
9. Confidence intervals on every number.
10. Only ask questions when something material is missing or has changed.
11. **Writes first, then iterates. No templates.** Give spines and per-section claims, not filled-in prose.

**Demonstrated intellectual standards:**

- Verifies against actual source code rather than assuming conventions. Read `build_dpo_pairs.py` and `triplet_quality_scorer.py` and reported signatures precisely, noting that reasoning from general DPO practice would have been wrong.
- Independently identified the threshold-vs-preference limitation before being prompted.
- Refuses AI-generated explanations as corroborating evidence.
- Correctly rejected a proposed experiment (negative re-selection) on retraining-cost grounds.
- Improved the blind read design beyond what was assigned (30 random with 10 repeats and a non-expert rater, rather than 20 stratified self-rated).
- **Pushed back correctly on 10 August** when told the length finding transferred from held-out to the 984 — that inference was unjustified until computed, and he said so.

**Requests scripts and outputs before interpretation.** On 10 August he explicitly asked for the analysis script and table construction *first*, so he could state his own reading and be corrected. Honour this pattern: produce the artefact, report the numbers plainly, withhold interpretation until asked.

**What works:** direct correction without softening; naming second-order effects before recommending; Diverge/Add-on framing; precise locatable edits ("paragraph 2, sentence 3"); conceding when the AI was wrong.

**What causes friction:** experiments proposed without accounting for retraining cost; any framing that lets a weak result be reported as strong; generic advice not anchored to the text in front of him.

**Do NOT re-explain:** DPO mechanics, LoRA/QLoRA, TRL APIs, cosine similarity, CUAD structure, bi-encoder vs cross-encoder, SFT, Bradley-Terry, ACL page limits.

**Outstanding assigned reading:** Rafailov et al. 2023 §4, Eq. 1 → Eq. 7 derivation. Assigned three times, completion unconfirmed.

---

## SECTION 3 — Thesis and Argument Architecture

### Thesis evolution

| Stage | Framing | Status |
|---|---|---|
| 1 | "Why do models reason over legal language like ordinary language?" | Never abandoned — still the introduction's frame |
| 2 | "Can a fine-tuned LLM pragmatically infer contractual meaning as courts do?" | Narrowed; still the research question |
| 3 | "JDAR is a process reward model" | **False.** Scores nothing at train or inference. |
| 4 | "DPO avoids Bradley-Terry" | **False.** DPO loss *is* the BT NLL. |
| 5 | "Threshold-clearing, not preference; pairs are near-paraphrases; scorers converge 53–60%" | **Correct and survives.** |
| 6 | "Human separates at 70% → ceiling is in the instruments" | **Not supported at p = 0.45 vs the length prior.** Must be softened. |
| 7 | **Forming.** "No method tested — trained, untrained, lexical, dense, or learned — exceeds a length heuristic. The supervisory signal correlates with a surface property requiring no reading." | Current best-supported framing. |

### Candidate thesis statement (draft, for the next conversation to refine)

> Corpus-grounded automatic preference construction for contractual clause interpretation produces pairs whose chosen and rejected members are near-paraphrases by every geometric and lexical measure, and whose most predictive property is passage length. Five automatic scorers spanning four method families converge at 53–60%, none exceeding a length-only baseline at 62.5%, indicating that threshold-based pipeline supervision encodes surface properties rather than the interpretive distinctions it was intended to capture.

### Argument tree, with current status

```
THESIS: Threshold-based pipeline supervision encodes surface properties,
        not interpretive distinctions.
│
├── CLAIM 1: The pairs are near-paraphrases to automatic measures.
│   ├── cos(chosen, rejected) = 0.604, std 0.054, unimodal
│   ├── random unrelated judicial text = 0.073; usable range 0.07–0.85
│   └── STATUS: SUPPORTED
│
├── CLAIM 2: No automatic scorer separates them meaningfully.
│   ├── TF-IDF 60.0%, BM25 58.0%, DPO 58.4%, cross-enc 53.8%, cosine 53.2%
│   ├── Under anchor-clustered CIs, BM25 spans chance; TF-IDF lower bound 0.515
│   └── STATUS: SUPPORTED, but CIs must be clustered
│
├── CLAIM 3 (NEW, replaces the old Claim 3): The most predictive property
│           of the pairs is length.
│   ├── 62.5% on the 984 [59.4, 65.5]; 63.0% on held-out
│   ├── Wilcoxon p ≈ 3.6e-19; median diff 82 chars
│   ├── Beats TF-IDF on the 984 (McNemar p ≈ 5.2e-05)
│   ├── Agreement with TF-IDF only 48.4% — tracking something different
│   └── STATUS: SUPPORTED, and it is the paper's sharpest finding
│
├── CLAIM 4: The supervisory signal is threshold-clearing.
│   ├── Verified against build_dpo_pairs.py, triplet_quality_scorer.py
│   ├── All scoring anchor→candidate; nothing compares chosen to rejected
│   └── STATUS: SUPPORTED
│
├── CLAIM 5: The instrument (embedding space) is not at fault.
│   ├── Isotropy control: random pairs at 0.073
│   └── STATUS: SUPPORTED
│
└── RETIRED CLAIM: "A human accesses signal the scorers cannot."
    ├── Human 70%, length prior 62.5%, p = 0.4549
    ├── Clause-clustered CI [0.500, 0.871] — lower bound at chance
    ├── Per-item length control IMPOSSIBLE (option texts not stored)
    └── STATUS: NOT SUPPORTED. Report as an exploratory probe.
```

### Framings explicitly rejected — do not resurrect

- **JDAR as a PRM.** Rewards no process.
- **"Verifier, not Bradley-Terry."** Inconsistent with using DPO.
- **`rewards/margins` as evidence of success.** Training diagnostic. Appears as Figure 1 in §6 framed as a generalisation-gap contrast, at the researcher's explicit request. That framing is the condition of its inclusion.
- **Anisotropy as the explanation.** Refuted by the isotropy control. Whitening cancelled, final.
- **"Standard NLP lacks the capability to understand legal reasoning."** Overclaim — no legal-reasoning-comprehension method was tested.
- **"Cross-encoders correct the mathematical inaccuracy of cosine."** Cosine is not inaccurate; it is a relevance failure.
- **"The model is fooled by texts sharing legal vocabulary."** Not established.
- **"The human accessed pragmatic signal the scorers cannot."** Not supported at p = 0.45.

### Title

Current title foregrounds the mechanism (threshold-clearing), which was the stage-5 headline. With the length finding now central, the title should be revisited. Options previously surfaced, plus one new:

- *Near-Paraphrase to Every Metric, Separable to a Reader* — **now inaccurate**, retire it
- *Threshold-Clearing Is Not Preference* — mechanism only; still defensible
- **New candidate:** *Longer, Not Better: What Corpus-Grounded Legal Preference Pairs Actually Encode*
- **New candidate:** *A Length Heuristic Beats Every Scorer We Tested: Auditing Automatic Preference Construction for Contractual Interpretation*

---

## SECTION 4 — Literature Map

### Cluster A — Similarity-based legal reward (direct precedent)

**SyLeR** — Zhang, Yu, Sun, Xu (2025), CIKM 2025, pp. 4117–4127. `zhang2025syler`
Structure-aware similarity-based reward for legal syllogistic reasoning. The direct precedent. Cite in §1 and §2; distinguish — SyLeR targets syllogistic structure, JDAR targeted pragmatic implicature, both hit the same wall.

**LegalΔ** — Dai, Xu, Liu, Yan, Xie, Yi, Wang, Yu (2025), arXiv 2508.12281. `dai2025legaldelta`
Critiques SyLeR: surface-level matching cannot capture the complexity of legal reasoning. **This is Reviewer 2, pre-written.** The paper converts the assertion into a quantified result. Cite §1, §2.

### Cluster B — Corpus-grounded process supervision

**Med-PRM** — Yun et al. (2025), EMNLP 2025, pp. 16554–16571. `yun2025medprm`
Nearest structural competitor. Honest distinction: curated, propositionally-structured, verifiable guideline corpus vs raw judicial opinions with no propositional structure. Cite §2.

**Math-Shepherd** — Wang et al. (2024), ACL 2024, pp. 9426–9439. `wang2024mathshepherd`
Automatic labels from *verifiable outcome correctness*; JDAR's from a heuristic threshold with no verification. That contrast is a §2 paragraph and half of Limitations. Cite §2.

**Fin-PRM** — Zhu, Zhou, Jiang, Li, Guo, Chen, Zhang (2025), arXiv 2508.15202. `zhou2025finprm` — cite-only, §2.

**CorVer** — Fan, Hao, Min, Liu, Yu, Cheng (2026), arXiv 2605.29648. `fan2026corver` — cite-only, §2. **Verify posting date precedes submission.**

### Cluster C — Preference dataset quality (calibration benchmark)

**"When Data is the Algorithm"** — Djuhera, Ahmed, Kadhe, Zawad, Ludwig, Boche (2026), ICLR 2026, arXiv 2511.10985. `djuhera2026whendata`
Audited five major DPO corpora with an independent reward model; preference coherence 70–80%. **The calibration benchmark.** JDAR's scorers fall below that band. Cite §1, §2, §6.

### Cluster D — Method and mechanics

- **Rafailov et al. (2023)**, DPO, NeurIPS 36. `rafailov2023dpo` — **§4 derivation still outstanding.**
- **Bradley & Terry (1952)**, Biometrika 39(3/4):324–345. `bradleyterry1952`
- **Rafailov, Hejna, Park, Finn (2024)**, "From r to Q*", arXiv 2404.12358. `rafailov2024rtoq`
- **"Bootstrapping LMs with DPO Implicit Rewards"**, arXiv 2406.09760. `zhou2024bootstrapping` — **verify author list**
- **Hu et al. (2022)**, LoRA, ICLR. `hu2022lora`
- **Dubey et al. (2024)**, Llama 3, arXiv 2407.21783. `llama3`
- **Hendrycks, Burns, Chen, Ball (2021)**, CUAD, NeurIPS D&B. `hendrycks2021cuad`
- **COLD Cases**, Harvard Library Innovation Lab. `coldcases`
- **Reimers & Gurevych (2019)**, Sentence-BERT, EMNLP-IJCNLP, pp. 3982–3992. `reimers-gurevych-2019-sentence`
- **EmbeddingGemma Team et al. (2025)**, arXiv 2509.20354. `embeddinggemma2025`

### Cluster E — Measurement and reporting discipline

- **Robertson & Zaragoza (2009)**, BM25 and Beyond, FnTIR 3(4). `robertson2009bm25` — grounds the TF-IDF/BM25 shared-lineage claim
- **Dietterich (1998)**, Neural Computation 10(7):1895–1923. `dietterich1998approximate`
- **McNemar (1947)**, Psychometrika 12(2):153–157. `mcnemar1947note`
- **Ethayarajh (2019)**, EMNLP, pp. 55–65. `ethayarajh2019contextual` — isotropy vocabulary
- **Azar et al. (2023)**, IPO, arXiv 2310.12036. `azar2023ipo` — BT mis-specification, for Limitations
- **Ma et al. (2025)**, AAAI. `ma2025stepreward` — **verify author list**

### NEW — required for the length finding and the human-probe rewrite

These are **not yet in `custom.bib`** and are needed for the revised §5, §6, and Limitations:

1. **Singhal, Goyal, Xu, Durrett (2024), "A Long Way to Go: Investigating Length Correlations in RLHF."** The canonical reference that reward models and RLHF pipelines latch onto length. Directly supports the length finding and makes it a contribution to a known problem rather than an isolated oddity. **Highest priority citation.**
2. **Dubois, Galambosi, Liang, Hashimoto (2024), "Length-Controlled AlpacaEval."** Method for length-debiasing an evaluation. This is the citation for the Future Work proposal.
3. **Park, Rafailov, Ermon, Finn (2024), "Disentangling Length from Quality in Direct Preference Optimization."** Length exploitation specifically under DPO. Explains why the implicit reward would inherit the confound.

Verify all three before adding — titles and venues are from working memory, not confirmed.

### Flagged, not yet read

- **Bowman & Dahl (2021)**, "What Will it Take to Fix Benchmarking in NLU?", NAACL — structurally the paper's argument
- **Dodge et al. (2019)**, "Show Your Work", EMNLP — reporting discipline for small effects
- **Marmor**, "The Pragmatics of Legal Language" — binary vocabulary decision, timebox 20 min
- **Robinson et al. (2021)**, Hard Negative Samples, ICLR — justifies `random.choice()` as a modelling decision
- **Karpukhin et al. (2020)**, DPR §3 — negative selection as a reportable choice
- **Cumming & Finch (2005)**, "Inference by Eye" — CI overlap fallacy
- **Thakur et al. (2021)**, BEIR — sparse/dense family separation

---

## SECTION 5 — Methodology and Empirical State

### Pipeline (the object under audit)

**Anchor selection.** CUAD. 30 clause categories screened, 6 passed, 24 excluded for insufficient volume or unresolved quality issues. After deduplication and vagueness filtering, 117 anchor clauses. Three categories reported: Cap On Liability 811 (82.4%), Third Party Beneficiary 95 (9.7%), Non-Compete 78 (7.9%). Three excluded for confirmed pipeline bugs: IP Ownership Assignment, Covenant Not To Sue, Competitive Restriction Exception.

**Retrieval.** COLD Cases (~8.36M US opinions, streamed from HuggingFace), filtered to contractual disputes. Windowed; bi-encoder `embedding-gemma-300m` for initial ranking, cross-encoder `ms-marco-MiniLM-L6-v2` for reranking. **Window size stated as 150,000 characters in the draft — UNVERIFIED, check source.**

**Quality dimensions.** D1 anchor-only vagueness against a seed set. D2 candidate-only judicial-content density. D3 anchor × candidate vagueness-category match. Threshold total_score ≥ 2, AND-gate across all three. An earlier OR gate passed pairs on D1+D2 while failing D3, producing false positives. From 12,294 scored rows, 1,069 passed; 984 yielded valid negative pairings.

**Pair construction.** `chosen` cleared the threshold. `rejected` = same-anchor window with total_score < 2 and d1_anchor_vagueness == 1 — **not model-generated, not a perturbation of chosen**. Selection uniform via `random.choice()`. 969/984 same-anchor, 15/984 same-category fallback. Eligible negative pool 3,037; used 984; unused 2,053.

**Verified against source:** `bi_encoder_score` and `cross_encoder_score` both computed `cuad_anchor` vs `raw_sentence`. `score_dimension_1(anchor)`, `score_dimension_2(sentence)`, `score_dimension_3(anchor, sentence)`. **No step ever compared `chosen` against `rejected`.**

**Training.** SFT then DPO on Llama-3-8B-Instruct, LoRA α=16, dropout 0, fp16, gradient accumulation and checkpointing. SFT on 984 (anchor, chosen). DPO on 984 triplets from the SFT checkpoint, which also served as π_ref, β = 0.1.

**Prompt — analogical supervision, still undisclosed in the paper.** The prompt reads: *"Given the following contractual clause which falls in the category of {category}, provide judicial reasoning of the kind a court would apply when interpreting clauses that raise similar legal questions."* The supervision is **analogical, not direct** — reasoning about clauses raising similar questions, not the anchor clause itself. **This must appear in §3 and Limitations. It currently appears in neither.**

### Data files and what is in them

| File | Contents | Notes |
|---|---|---|
| `dpo_dataset_revised.json` | 984 items, keys `prompt`/`chosen`/`rejected` | 85 unique prompts; top-3 anchors = 394 pairs (40%) |
| `held_out_baseline_results.json` | 500 per-pair records, 4 baselines scored | 19 unique anchors; top anchor 277 pairs; categories 492/4/4 |
| `evaluation_results.json` | 500 records, DPO implicit reward | 20 unique anchors; **only 12 shared with the baseline file** |
| `blind_eval_binosh_1785755394293.json` | 30 answers | **No option texts stored** — per-item length control impossible |

### Results — current, all verified 10 August

**Automatic scorers, held-out 500 (naive Wilson vs anchor-clustered bootstrap):**

| Scorer | Family | Acc | Naive 95% CI | Clustered 95% CI |
|---|---|---|---|---|
| TF-IDF cosine | sparse lexical | 60.0% | [55.7, 64.3] | [51.5, 64.3] |
| DPO implicit reward | trained policy | 58.4% | [54.0, 62.8] | [56.2, 67.1] |
| Okapi BM25 | sparse lexical | 58.0% | [53.7, 62.3] | [47.5, 64.6] |
| Cross-encoder | discriminative rerank | 53.8% | [49.4, 58.2] | [51.1, 60.1] |
| Embedding cosine | dense | 53.2% | [48.8, 57.6] | [44.6, 58.2] |

**Length-only baselines:**

| Population | n | Acc | Clustered 95% CI |
|---|---|---|---|
| Held-out 500 | 500 | 63.0% | [60.6, 67.5] |
| Training 984 | 984 | 62.5% | [58.2, 66.9] |

Length stats on the 984: chosen mean 649.0 chars / median 622.0 / 97.2 words; rejected mean 571.3 / median 537.0 / 85.6 words. Mean diff 77.7 chars, median 82. Wilcoxon p ≈ 3.6e-19. Per category: Cap On Liability 63.3%, Non-Compete 62.8%, Third Party Beneficiary 55.8%.

**Human blind read (evaluator "binosh", 3 August, ~60 min):**

| Slice | n | Correct | Acc | Wilson CI | Clause-clustered CI |
|---|---|---|---|---|---|
| All presentations | 30 | 21 | 70.0% | [52.1, 83.3] | [50.0, 87.1] |
| Unique clauses | 20 | 14 | 70.0% | [48.1, 85.5] | [50.0, 90.0] |
| Repeats only | 10 | 7 | 70.0% | [38.6, 90.9] | — |

Per category, **unique anchors** (this is the honest view — raw counts are inflated by repeats):

| Category | Unique anchors | Score | Acc | Raw |
|---|---|---|---|---|
| Third Party Beneficiary | 4 | 4/4 | 100% | 8/10 (80%) |
| Non-Compete | 8 | 4/8 | 50% | 5/10 (50%) |
| Cap On Liability | 8 | 6/8 | 75% | 8/10 (80%) |
| **Total** | **20** | **14/20** | **70%** | **21/30 (70%)** |

Significance: p vs 50% = 0.0428 (all presentations, naive); p vs 50% = 0.1153 (unique clauses); **p vs the 62.5% length prior = 0.4549**. Clause-clustered bootstrap: P(boot mean ≤ 0.500) = 0.030; **P(boot mean ≤ 0.625) = 0.235**.

Intra-rater consistency: **3/10 repeats inconsistent (30%)**. Q14 vs Q10, Q29 vs Q27, Q7 vs Q2. One clause answered wrong→right→wrong across three presentations.

Position check: key was A in 19/30; rater chose A in 20/30 (p = 0.0987). Accuracy 78.9% when key was A, 54.5% when key was B; Fisher exact p = 0.2252. **Not significant, but the direction is worth a Limitations clause.**

**Scorer agreement matrix (held-out 500, fraction of pairs with the same decision):**

| | TF-IDF | BM25 | Embedding | Cross-enc | Length |
|---|---|---|---|---|---|
| **TF-IDF** | — | **0.848** | 0.628 | 0.642 | 0.594 |
| **BM25** | 0.848 | — | 0.628 | 0.646 | 0.586 |
| **Embedding** | 0.628 | 0.628 | — | 0.626 | 0.518 |
| **Cross-enc** | 0.642 | 0.646 | 0.626 | — | 0.524 |
| **Length** | 0.594 | 0.586 | 0.518 | 0.524 | — |

McNemar, held-out: TF-IDF vs BM25 p = 0.302 (indistinguishable — same family, confirmed). TF-IDF vs embedding p = 0.015*. TF-IDF vs cross-enc p = 0.025*. Embedding vs length p = 0.0019*. Cross-enc vs length p = 0.0034*. TF-IDF vs length p = 0.326 (indistinguishable).

On the 984: TF-IDF vs length McNemar p ≈ 5.2e-05, agreement only 48.4%.

**Length-controlled subset of the 984:** 154 pairs with |length difference| ≤ 50 chars. Length baseline there drops to 55.8%; TF-IDF drops to 46.8%.

### Earlier experiments (retained, still valid)

- **E2 fresh cosine recompute (984):** cos(anchor, chosen) 0.6371, cos(anchor, rejected) 0.6318, margin 0.0053, frac chosen closer 0.5467. **Rules out circularity.**
- **E3 cos(chosen, rejected) (984):** mean 0.6042, median 0.6041, std 0.0540, p10/p90 0.5366/0.6665, min/max 0.4385/0.8476, frac>0.85 = 0, frac<0.4 = 0.
- **E4 isotropy control (PIVOTAL):** 500 random unrelated judicial windows, mean 0.0733, median 0.0702, min 0.0170, max 0.2320. Near-isotropic; refutes anisotropy; whitening cancelled.
- **E5 DPO implicit reward held out:** raw 57.6%, length-normalised 58.4%, mean gap 0.003454, median 0.001092. **Mean is 3× median — right-skewed, never investigated (Set B).**

### Discrepancy to resolve

Recomputed TF-IDF on the 984 gave **53.2%**; the original script reported **57.7%**. Preprocessing differs (sublinear tf, English stopwords, anchor extracted by string-split from the prompt template). **Trust the researcher's original number**, but resolve the discrepancy — a reviewer who reimplements will get a different figure. Note this does not affect the conclusion: length beats both.

---

## SECTION 6 — Writing State and Required Changes

### Current structure

| Section | Status | Required change |
|---|---|---|
| Abstract | Drafted ~155w | **Rewrite.** Currently claims the human shows the ceiling is in the scorers. |
| §1 Introduction | Drafted | **Rewrite Move 3 and contribution (3).** |
| §2 Related Work | Drafted | Add the length-in-RLHF cluster. |
| §3 Constructing Pairs | Drafted, 5 subsections | Add analogical-supervision disclosure. Fix Eq. 1 rendering. |
| §4 Audit Protocol | Drafted | **Fix the false "same 500" claim.** Add clustering disclosure. Add length as a scorer. Resolve Wilson-vs-bootstrap. |
| §5 Results | Drafted | Add length row. Clustered CIs. Add agreement matrix. |
| §6 Supervision Encodes | Drafted | **Heaviest rewrite.** Blind read becomes an exploratory probe. |
| §7 Conclusion | Drafted ~100w | **Rewrite sentence 1.** |
| Limitations | **EMPTY** | Write. Free pages. |
| Acknowledgments | **EMPTY** | Must be anonymised. |
| Appendix | **EMPTY** | Free pages. |

### The table — required changes

Current:

```latex
\begin{table}[t]
  \centering \small
  \begin{tabular}{lcc}
    \toprule
    \textbf{Scorer} & \textbf{Accuracy} & \textbf{95\% CI} \\
    \midrule
    Embedding cosine     & 53.2\% & [48.8, 57.6] \\
    Cross-encoder        & 53.8\% & [49.4, 58.2] \\
    Okapi BM25           & 58.0\% & [53.7, 62.3] \\
    DPO implicit reward  & 58.4\% & [54.0, 62.8] \\
    TF-IDF cosine        & 60.0\% & [55.7, 64.3] \\
    \bottomrule
  \end{tabular}
  \caption{Held-out pairwise accuracy across five scoring functions,
           ranked by accuracy. Confidence intervals are Wilson 95\%
           binomial proportion intervals.}
  \label{tab:held-out-accuracy}
\end{table}
```

Six problems: (1) no length baseline, so the reader cannot see that every scorer loses to a character count; (2) no method-family column, so the convergence-across-families argument is invisible; (3) no `n` column, so the reader cannot see the DPO row is a different draw; (4) Wilson intervals ignore that 500 pairs sit on 19 anchors; (5) caption does not disclose the different DPO sample; (6) caption states a conclusion the table now contradicts.

Replacement:

```latex
\begin{table}[t]
  \centering \small
  \begin{tabular}{llcc}
    \toprule
    \textbf{Scorer} & \textbf{Family} & \textbf{Acc.} & \textbf{95\% CI} \\
    \midrule
    Embedding cosine     & dense      & 53.2\% & [44.6, 58.2] \\
    Cross-encoder        & rerank     & 53.8\% & [51.1, 60.1] \\
    Okapi BM25           & lexical    & 58.0\% & [47.5, 64.6] \\
    DPO implicit reward$^{\dagger}$ & learned & 58.4\% & [56.2, 67.1] \\
    TF-IDF cosine        & lexical    & 60.0\% & [51.5, 64.3] \\
    \midrule
    Length only          & --         & 63.0\% & [60.6, 67.5] \\
    \bottomrule
  \end{tabular}
  \caption{Pairwise discrimination accuracy on 500 held-out pairs.
    Intervals are anchor-clustered bootstrap (4{,}000 resamples); the
    500 pairs derive from only 19 distinct anchor clauses, so naive
    binomial intervals understate uncertainty. The length-only baseline
    predicts the longer passage and reads no content.
    $^{\dagger}$Computed on a separate 500-pair draw from the same pool
    (12 of 19 anchors shared); see \S\ref{sec:audit-protocol}.}
  \label{tab:held-out-accuracy}
\end{table}
```

Notes on this replacement, for the next conversation to interrogate rather than accept:
- The length row is separated by `\midrule` because it is a floor, not a competitor.
- Clustered intervals are wider and **BM25 now spans chance** — the §5 sentence claiming TF-IDF and BM25 both exclude 50% must change.
- If the researcher prefers Wilson intervals, the clustering must instead be disclosed in §4 prose. Do not silently keep naive intervals.
- The `\dagger` disclosure is mandatory; the current §4 claim is false as written.

### Figures

| ID | Content | Status |
|---|---|---|
| Fig 1 | DPO reward margins over training | **Exists.** Legend still shows the raw path `meta-llama_Meta-Llama-3-8B-dpo-from-sft`; axis labels are TRL defaults. Framing condition: generalisation-gap contrast only, never as evidence the method worked. |
| Fig 2 | Discrimination forest plot — all scorers + length + human, marker shape by population, shaded length prior | **Built 10 Aug**, `make_discrimination_figure.py` |
| Fig 3 | Pipeline schematic with the absent chosen↔rejected edge | **Not built.** TikZ scaffold exists in prior conversation. High value — makes the mechanism visible. |
| Fig 4 | Cosine scale (0.073 / 0.604 / 0.637–0.632 superimposed) | **Not built.** Needs the raw cosine arrays, which are not in any uploaded file. |

### Scripts produced 10 August

- `make_figure2.py` — scorer-only forest plot, `INCLUDE_LENGTH` and `CLUSTERED_CI` flags
- `analyse_blind_read.py` — full blind-read report + two LaTeX tables (`blind_read_overall.tex`, `blind_read_per_category.tex`)
- `make_discrimination_figure.py` — combined figure with population markers and length band

### Prose style decisions

Past tense for the pipeline (artifact under examination), present for results. No justification tone in §3. The honesty paragraph in §3.4 stays unhedged. Every number carries a CI. No inline paper titles — citation keys only. ACL template, `custom.bib`.

---

## SECTION 7 — Open Questions Registry

| # | Question | Priority | Status | Next action |
|---|---|---|---|---|
| 1 | Has the NLLP deadline actually been extended? | **Critical** | Reported, unverified | Check the CFP before planning |
| 2 | Is the LaTeX still `\usepackage[preprint]{acl}`? | **Critical** | Was, as of last seen source | Change to `[review]`. Desk-reject risk. |
| 3 | Wilson or bootstrap for the reported CIs? | **Critical** | Draft has contradicted itself | Decide; disclose clustering either way |
| 4 | How does §6 report the blind read now? | **Critical** | Unresolved | The main task for the next conversation |
| 5 | Does the researcher accept the length row in the table? | **Critical** | Middle path proposed, not accepted | His call |
| 6 | Resolve TF-IDF 53.2% vs 57.7% discrepancy | High | Open | Compare preprocessing |
| 7 | Is the 150,000-char window figure correct? | High | Unverified | Check source. 5 min. |
| 8 | Disclose analogical supervision? | High | Undisclosed | Yes — §3 and Limitations |
| 9 | Rerun DPO implicit reward on the baseline file's exact 500? | High | Not done | Makes the §4 claim true |
| 10 | Set B — decisive tail (mean gap 3× median) | High | Never run | ~1 hr, data in hand |
| 11 | Why is Non-Compete at 50% for the human and at chance for scorers? | Medium | Two hypotheses: no differential signal, or defective pipeline handling | Inspect Non-Compete pairs |
| 12 | Is Bradley-Terry mis-specified (incomparable negatives)? | Medium | Open | Cite Azar et al. in Limitations |
| 13 | Is "pragmatic inference" the right vocabulary? | Medium | Open | Marmor, timeboxed 20 min |
| 14 | Title revision | Medium | Candidates listed §3 | Decide after §6 rewrite |
| 15 | Score-gap negative re-selection | Low | Requires retraining | Journal version |

---

## SECTION 8 — Active Work State

### Where things stood at the end of the 10 August conversation

The researcher had just supplied the blind-read analysis document confirming that the 10 repeats were **exact clause repeats** (validating the 30% inconsistency figure) and giving the per-category unique-anchor breakdown. He asked for: the chart figure script, the table changes, and this continuity file.

He had also stated his position that the length comparison should not be applied to the human result, and proposed deferring a length-controlled human study to future GRPO work.

### The disagreement, stated fairly for the next AI

**Researcher's position:** the rater reported using pragmatic inference; framing the result against a length baseline reduces a qualitative human capability to a quantitative artefact and undermines the paper's point about human intelligence. Defer to future work.

**The counter-position put to him:** a length control measures the *pairs*, not the rater. The 62.5% asymmetry is a data property that exists independently of any human. It creates an alternative explanation for the 70% rather than dissolving one. And the specific number — p = 0.4549 against the length prior, clause-clustered CI reaching exactly 0.500 — means this experiment cannot distinguish the two hypotheses. Deferring the *study* is reasonable; deferring the *number*, when it is already computed, is not.

**Agreed middle path, not yet accepted:** length as a scorer row (a claim about pairs), one Limitations sentence, nothing about the rater's cognition, study deferred to Future Work.

**The next AI should not re-argue this from scratch.** Ask once whether he has decided, then work with whichever he chooses. If he declines the length row entirely, the honest fallback is a Limitations paragraph that states the asymmetry and that the probe was not length-controlled — that minimum is not negotiable, because the claim currently in §6 and §7 is not supported by the data.

### Immediate next steps

1. Verify the NLLP deadline.
2. Fix `[preprint]` → `[review]`.
3. Settle the length-row question with the researcher.
4. **Rewrite §6.** This is the main task. Spine below.
5. Rewrite §7 sentence 1 and the abstract's Move 4.
6. Write Limitations (free pages).
7. Update Table 1 per §6 above.
8. Build the pipeline schematic (Fig 3).

### §6 rewrite spine (for the next conversation to work against)

The section's claim changes from *"the human accessed signal the scorers cannot"* to *"the supervisory signal is a surface property, and our human probe was underpowered to test whether anything deeper is present."*

- **¶1 — What the scorers found.** Convergence at 53–60% across four families. One sentence.
- **¶2 — The length result.** 62.5% on the 984, 63.0% held-out, Wilcoxon p ≈ 3.6e-19. Every scorer loses to a character count. Agreement with TF-IDF only 48.4% — length is not a proxy for lexical overlap. Cite Singhal et al. and Park et al. This paragraph is the paper's sharpest finding and should read as such.
- **¶3 — What the threshold encodes.** D2/D3 gate on judicial-content density and topical match. The model learns topical appropriateness and legal register. No contrast was ever constructed along the pragmatic axis. Figure 1 supports this: margins rise on training pairs while held-out accuracy stays at 58.4%.
- **¶4 — The human probe, reported honestly.** 21/30 across 20 unique clauses, single non-expert rater, 60-minute session. Per-category unique-anchor: TPB 4/4, Cap On Liability 6/8, Non-Compete 4/8. Intra-rater inconsistency 3/10 on exact repeats. Clause-clustered CI [50.0, 87.1]. **Not significantly above the length prior (p = 0.45).** Frame as exploratory; state plainly that it cannot separate pragmatic discrimination from a length confound because option texts were not retained.
- **¶5 — What this licenses and what it does not.** It licenses: the pairs encode surface properties. It does not license: any claim about what a human can or cannot perceive in these pairs. The required follow-up is a length-matched, multi-rater study powered against a 62.5% baseline, not a 50% one.

**Failure mode to avoid:** hedging in a way that leaves the old claim standing implicitly. If §6 says "the human reached 70%, suggesting a signal the scorers miss, though this is not significant," a reviewer reads the first clause and ignores the second. State the null result first and the point estimate second.

---

## SECTION 9 — Operational Continuity Instructions

**How to continue.** Four lenses in every response — tutor, researcher, supervisor, layman. Push back rather than validate. Ground in sources; search when unsure. Direct, no emojis, no flourish. Spines and per-section claims, never filled-in prose. Diverge/Add-on labels for scope decisions. Knowledge gaps get 1–3 named readings plus one exercise.

**Produce artefacts before interpretation.** He asked explicitly on 10 August for the script and the table first, so he could state his own reading and be corrected. Honour that: run the analysis, report the numbers plainly, hold the interpretation until he offers his.

**Finalised — do not re-litigate:**
1. JDAR is not a process reward model.
2. DPO is Bradley-Terry; inherited and disclosed.
3. Anisotropy refuted; whitening cancelled.
4. `rewards/margins` appears only as Figure 1 in §6, framed as a generalisation-gap contrast, at his explicit request.
5. The paper is an audit paper. The reframing away from positive results is final.
6. The blind read is a probe, not a study.

**Genuinely open:** how §6 reports the human probe; whether the length row goes in the table; the title; what fills the remaining three pages of the long-paper expansion; whether to rerun the DPO scorer on the baseline file's 500.

**Wordings to avoid:** "standard NLP lacks the capability to understand legal reasoning"; "cross-encoders correct the mathematical inaccuracy of cosine"; "the model is fooled by texts sharing legal vocabulary"; "hard negatives" as a description of the adversarial-negatives proposal (they are opposites); "human study" for the blind read; "unfortunately"; "surprisingly"; apologetic framing.

**Do not re-explain:** DPO mechanics, LoRA/QLoRA, TRL APIs, cosine similarity, CUAD structure, bi-encoder vs cross-encoder, SFT, Bradley-Terry, ACL page limits.

**On handling the disagreement:** he is the researcher and it is his paper. State the evidence once, clearly, name what is and is not supported, offer the middle path, then work with his decision. Do not relitigate across turns. But do not help him write a sentence the data does not support — the minimum non-negotiable is a Limitations disclosure of the length asymmetry.

---

## SECTION 10 — Memory Compression Layer

**Thesis in one sentence (current best-supported):** Corpus-grounded preference pairs for contractual clause interpretation are near-paraphrases whose most predictive property is length; five scorers across four families converge at 53–60% and none beats a 62.5% length-only baseline, so the supervision encodes surface properties rather than interpretive distinctions.

**Top 5 sources:** Djuhera et al. 2026 (ICLR) · Rafailov et al. 2023 (DPO) · LegalΔ arXiv 2508.12281 · SyLeR CIKM 2025 · Singhal et al. 2024 (length in RLHF — **not yet in the bib, highest-priority addition**).

**What changed:** the human's 70% is not significantly above the 62.5% length prior (p = 0.45); clause-clustered CI reaches exactly 0.500. The measurement-gap claim in §6 and §7 is unsupported and must be rewritten.

**Top 3 blockers:** §6 rewrite; the length-row decision; `[preprint]` mode still in the LaTeX.

**Immediate next action:** verify the deadline, then rewrite §6 against the five-paragraph spine in Section 8.

**Most fragile remaining assumption:** that the 984 pairs behave as 984 independent observations. They sit on 85 anchors, with 40% on three. Every reported CI is affected.

**Researcher in 5 bullets:** verifies against source code · writes first, rejects templates · wants pushback over validation · asks for scripts before interpretation · expanding 5 pages to 8 with one working day.

---
---

# ARTIFACT 2 — RESEARCH STATE JSON

```json
{
  "project_metadata": {
    "title": "Threshold-Clearing Is Not Preference: Auditing Automatic Preference Pairs for Contractual Interpretation",
    "title_status": "under revision; length finding now central",
    "researcher": "Adrino Rosario James",
    "coauthors": ["Rajesh Khanna", "Deepthi Das"],
    "institution": "CHRIST (Deemed to be University)",
    "compiled": "2026-08-10",
    "supersedes": ["2026-08-02 codex", "2026-08-07 transfer"],
    "paper_type": "audit paper",
    "body_pages_current": 5,
    "body_pages_target": 8,
    "venue_primary": "NLLP 2026",
    "venue_fallback": "JURIX 2026",
    "deadline_status": "reported extended, UNVERIFIED"
  },
  "critical_change_2026_08_10": {
    "summary": "A length-only classifier outperforms every scorer tested, and the human blind read is not significantly above the length prior. The measurement-gap framing is unsupported.",
    "length_prior_984": {"k": 615, "n": 984, "acc": 0.625, "wilson": [0.594, 0.655]},
    "length_prior_heldout": {"acc": 0.630, "n": 500},
    "human_vs_length_p": 0.4549,
    "human_clause_clustered_ci": [0.500, 0.871],
    "consequence": "Rewrite section 6 and section 7 sentence 1 and abstract move 4"
  },
  "results_heldout_500": [
    {"scorer": "embedding cosine", "family": "dense", "acc": 0.532, "wilson": [0.488, 0.576], "clustered": [0.446, 0.582]},
    {"scorer": "cross-encoder", "family": "discriminative rerank", "acc": 0.538, "wilson": [0.494, 0.582], "clustered": [0.511, 0.601]},
    {"scorer": "okapi bm25", "family": "sparse lexical", "acc": 0.580, "wilson": [0.537, 0.623], "clustered": [0.475, 0.646]},
    {"scorer": "dpo implicit reward", "family": "trained policy", "acc": 0.584, "wilson": [0.540, 0.628], "clustered": [0.562, 0.671], "note": "DIFFERENT 500-pair draw; 12 of 19 anchors shared"},
    {"scorer": "tf-idf cosine", "family": "sparse lexical", "acc": 0.600, "wilson": [0.557, 0.643], "clustered": [0.515, 0.643]},
    {"scorer": "length only", "family": "surface", "acc": 0.630, "clustered": [0.606, 0.675]}
  ],
  "length_statistics_984": {
    "chosen_chars_mean": 649.0, "chosen_chars_median": 622.0, "chosen_words_mean": 97.2,
    "rejected_chars_mean": 571.3, "rejected_chars_median": 537.0, "rejected_words_mean": 85.6,
    "mean_diff_chars": 77.7, "median_diff_chars": 82.0,
    "wilcoxon_p": 3.57e-19,
    "per_category": {"Cap On Liability": 0.6326, "Non-Compete": 0.6282, "Third Party Beneficiary": 0.5579},
    "length_controlled_subset": {"n_within_50_chars": 154, "length_acc": 0.5584, "tfidf_acc": 0.4675}
  },
  "blind_read": {
    "file": "blind_eval_binosh_1785755394293.json",
    "evaluator": "binosh",
    "started": "2026-08-03T10:09:34.892Z",
    "completed": "2026-08-03T11:09:54.289Z",
    "duration_min": 60,
    "rater_type": "non-expert in law",
    "presentations": 30,
    "unique_clauses": 20,
    "repeats": 10,
    "repeats_are_exact_clause_repeats": true,
    "correct_all": 21, "acc_all": 0.700, "wilson_all": [0.521, 0.833], "clustered_all": [0.500, 0.871],
    "correct_unique": 14, "acc_unique": 0.700, "wilson_unique": [0.481, 0.855], "clustered_unique": [0.500, 0.900],
    "correct_repeats": 7, "acc_repeats": 0.700,
    "per_category_unique": {
      "Third Party Beneficiary": {"n": 4, "correct": 4, "acc": 1.00},
      "Non-Compete": {"n": 8, "correct": 4, "acc": 0.50},
      "Cap On Liability": {"n": 8, "correct": 6, "acc": 0.75}
    },
    "per_category_raw": {
      "Third Party Beneficiary": {"n": 10, "correct": 8},
      "Non-Compete": {"n": 10, "correct": 5},
      "Cap On Liability": {"n": 10, "correct": 8}
    },
    "p_vs_chance_all": 0.0428,
    "p_vs_chance_unique": 0.1153,
    "p_vs_length_prior": 0.4549,
    "bootstrap_p_le_chance": 0.030,
    "bootstrap_p_le_length_prior": 0.235,
    "intra_rater_inconsistent": 3, "intra_rater_total_repeats": 10, "inconsistency_rate": 0.30,
    "inconsistent_items": ["Q14 vs Q10", "Q29 vs Q27", "Q7 vs Q2"],
    "position_effect": {"key_A": 19, "rater_A": 20, "p_rater_A_vs_half": 0.0987, "acc_key_A": 0.789, "acc_key_B": 0.545, "fisher_p": 0.2252},
    "option_texts_stored": false,
    "per_item_length_control_possible": false
  },
  "agreement_matrix_heldout": {
    "tfidf_bm25": 0.848, "tfidf_embgemma": 0.628, "tfidf_crossenc": 0.642, "tfidf_length": 0.594,
    "bm25_embgemma": 0.628, "bm25_crossenc": 0.646, "bm25_length": 0.586,
    "embgemma_crossenc": 0.626, "embgemma_length": 0.518, "crossenc_length": 0.524
  },
  "mcnemar_heldout": {
    "tfidf_vs_bm25": 0.3019, "tfidf_vs_embgemma": 0.01531, "tfidf_vs_crossenc": 0.02467,
    "tfidf_vs_length": 0.3258, "bm25_vs_embgemma": 0.09144, "bm25_vs_crossenc": 0.1325,
    "bm25_vs_length": 0.09506, "embgemma_vs_crossenc": 0.8838,
    "embgemma_vs_length": 0.001927, "crossenc_vs_length": 0.003448
  },
  "clustering": {
    "heldout_500_unique_anchors": 19,
    "heldout_top_anchor_pairs": 277,
    "heldout_top3_pairs": 446,
    "heldout_categories": {"Cap On Liability": 492, "Non-Compete": 4, "Third Party Beneficiary": 4},
    "training_984_unique_anchors": 85,
    "training_top_anchor_pairs": 214,
    "training_top3_pairs": 394,
    "eval_file_unique_anchors": 20,
    "anchors_shared_between_files": 12,
    "consequence": "all naive binomial CIs in the draft are too narrow"
  },
  "prior_experiments": {
    "E2_fresh_cosine_984": {"cos_anchor_chosen": 0.6371, "cos_anchor_rejected": 0.6318, "margin": 0.0053, "frac_chosen_closer": 0.5467, "interpretation": "rules out circularity"},
    "E3_cos_chosen_rejected_984": {"mean": 0.6042, "median": 0.6041, "std": 0.0540, "p10": 0.5366, "p90": 0.6665, "min": 0.4385, "max": 0.8476},
    "E4_isotropy_control": {"n": 500, "mean": 0.0733, "median": 0.0702, "min": 0.0170, "max": 0.2320, "interpretation": "near-isotropic; refutes anisotropy; whitening cancelled"},
    "E5_dpo_implicit_reward": {"raw_acc": 0.576, "norm_acc": 0.584, "mean_gap": 0.003454, "median_gap": 0.001092, "unexplored": "mean is 3x median; right-skewed; Set B never run"}
  },
  "dataset": {
    "file": "dpo_dataset_revised.json",
    "n_pairs": 984,
    "keys": ["prompt", "chosen", "rejected"],
    "categories": {"Cap On Liability": 811, "Third Party Beneficiary": 95, "Non-Compete": 78},
    "dropped_categories": ["IP Ownership Assignment", "Covenant Not To Sue", "Competitive Restriction Exception"],
    "negative_pool_eligible": 3037, "negatives_used": 984, "negatives_unused": 2053,
    "selection_same_anchor": 969, "selection_same_category_fallback": 15,
    "scored_rows_total": 12294, "rows_passing_and_gate": 1069,
    "anchor_clauses_after_dedup": 117, "categories_screened": 30, "categories_passed": 6
  },
  "prompt_template": {
    "text": "Given the following contractual clause which falls in the category of {category}, provide judicial reasoning of the kind a court would apply when interpreting clauses that raise similar legal questions:",
    "issue": "ANALOGICAL supervision, not direct -- reasoning about clauses raising similar questions, not the anchor clause itself",
    "disclosure_status": "NOT DISCLOSED anywhere in the paper; must appear in section 3 and limitations"
  },
  "draft_errors": [
    {"error": "LaTeX in [preprint] mode reveals author identity", "severity": "desk-reject", "fix": "change to [review]"},
    {"error": "section 4 claims all five scorers on the same 500 held-out pairs", "severity": "critical", "fix": "false; disclose the separate DPO draw or rerun"},
    {"error": "section 6 and 7 claim the human accessed signal scorers cannot", "severity": "critical", "fix": "not supported at p=0.45; rewrite"},
    {"error": "CI method contradiction Wilson vs bootstrap", "severity": "high", "fix": "decide and disclose clustering"},
    {"error": "no length baseline anywhere", "severity": "high", "fix": "add as a table row"},
    {"error": "analogical supervision undisclosed", "severity": "high", "fix": "section 3 and limitations"},
    {"error": "150,000-char window figure unverified", "severity": "medium", "fix": "check source"},
    {"error": "TF-IDF 53.2 vs 57.7 discrepancy", "severity": "medium", "fix": "compare preprocessing"}
  ],
  "limitations_spine": [
    "category imbalance 82.4% Cap On Liability; held-out set 98.4%",
    "anchor clustering: 984 pairs on 85 anchors, 500 held-out on 19",
    "systematic length asymmetry: chosen longer in 62.5% of pairs",
    "blind read not length-controlled; option texts not retained",
    "blind read single non-expert rater, 30 presentations, 20 clauses, 30% intra-rater inconsistency",
    "no expert legal annotation anywhere",
    "single primary model Llama-3-8B",
    "Bradley-Terry mis-specification if negatives are incomparable (cite Azar et al.)",
    "no downstream legal task benchmark",
    "analogical supervision -- prompt elicits reasoning about similar clauses, not the anchor"
  ],
  "scripts_produced": [
    {"name": "make_figure2.py", "purpose": "scorer forest plot", "flags": ["INCLUDE_LENGTH", "CLUSTERED_CI"]},
    {"name": "analyse_blind_read.py", "purpose": "blind read report + LaTeX tables"},
    {"name": "make_discrimination_figure.py", "purpose": "combined figure: scorers + length + human, population markers"}
  ],
  "next_actions": [
    {"order": 1, "action": "verify NLLP deadline at nllpw.org", "cost": "2 min"},
    {"order": 2, "action": "change [preprint] to [review]", "cost": "1 min"},
    {"order": 3, "action": "settle the length-row decision with the researcher", "cost": "one exchange"},
    {"order": 4, "action": "rewrite section 6 against the five-paragraph spine", "cost": "half day"},
    {"order": 5, "action": "rewrite section 7 sentence 1 and abstract move 4", "cost": "1 hr"},
    {"order": 6, "action": "write limitations", "cost": "1 hr, free pages"},
    {"order": 7, "action": "update table 1", "cost": "30 min"},
    {"order": 8, "action": "build the pipeline schematic figure", "cost": "1 hr"},
    {"order": 9, "action": "add Singhal/Dubois/Park length citations to custom.bib", "cost": "20 min"}
  ]
}
```

---
---

# ARTIFACT 3 — RESTART PROMPT

*Paste this into a new conversation together with this file.*

---

I am writing a conference paper and I want you as a research mentor and thinking partner, not a task-executor. Combine four lenses in every response: a tutor checking whether I have grasped the fundamentals, a researcher checking whether the method and reasoning are sound, a supervisor checking whether this serves my actual goals and timeline, and a layman checking whether I could explain it simply. Push back and probe rather than validating me by default — rigor matters more than comfort. Ground claims in actual sources rather than answering from memory; search when you are unsure. Be direct. No emojis, no flashy language. I write first and iterate — give me section spines and the claim each subsection must land, not filled-in prose. When you detect a knowledge gap, name it plainly, give one to three specific named readings, and give one exercise combining a critical question with a practical task. When a task pulls away from the core approach, give me two labelled options — Diverge and Add-on — and let me choose. When I ask for a script or an analysis, produce it and report the numbers plainly first; hold your interpretation until I have given mine.

**Read the attached continuity file before responding. It supersedes anything you may infer from older documents.**

## Where things stand

The paper is an audit of corpus-grounded automatic preference construction for contractual clause interpretation, targeting NLLP 2026. I built 984 preference pairs from CUAD anchors and COLD Cases judicial text, trained SFT+DPO on Llama-3-8B, and audited the pairs with five scorers spanning four method families.

**What survives:** the scorers converge at 53–60% on held-out pairs. `chosen` and `rejected` sit at 0.604 cosine where unrelated judicial text sits at 0.073 — near-paraphrases. No step in my pipeline ever compared `chosen` against `rejected`; all scoring was anchor→candidate, so the supervisory signal is cleared-threshold vs did-not-clear.

**What changed:** a length-only classifier — pick the longer passage, read nothing — scores 62.5% on the 984 and 63.0% on held-out, beating every scorer I tested. And my human blind read (21/30, 70%) is not significantly above that length prior (p = 0.45); under clause-clustered bootstrap its CI lower bound is exactly 0.500. My §6 and §7 currently claim the human accessed signal the scorers cannot. That claim is not supported.

## Settled — do not re-litigate

- JDAR is not a process reward model.
- DPO is Bradley-Terry; inherited and disclosed, not avoided.
- Anisotropy was refuted by my own isotropy control. Whitening is cancelled.
- `rewards/margins` appears only as Figure 1 in §6, framed as a generalisation-gap contrast. That framing is the condition of its inclusion.
- This is an audit paper. The move away from positive results is final.
- The blind read is an exploratory probe, not a study.

## What I need now

The body is at 5 pages and I want an 8-page long paper. But before expanding, §6 has to be rewritten against what the data actually supports, and §7 sentence 1 and the abstract follow from it. Start there — the continuity file has a five-paragraph spine for §6 in Section 8; tell me whether you agree with it before we write against it.

I also need your view on one open decision. I do not want the length comparison applied to my human rater's result — he reported using pragmatic inference, and I think framing it against a character count misrepresents what was being tested. The counter-argument put to me is that a length control measures the pairs rather than the rater, and that the number should be reported as a scorer row regardless. Tell me where you land, once, and then work with whatever I decide.

Three housekeeping items I have not confirmed: whether the NLLP deadline extension is real, whether my LaTeX is still in `[preprint]` mode, and whether my original TF-IDF preprocessing explains the 57.7% vs 53.2% discrepancy in the continuity file.