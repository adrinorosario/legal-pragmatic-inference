# JDAR — Reading List and Literature Map

Every reading token surfaced during this project, organised by **which section of the paper it serves**. Read in the order given within each block.

---

## PART I — TOKENS BY PAPER SECTION

### For §1 Introduction — the framing

**1. Bowman & Dahl (2021), "What Will it Take to Fix Benchmarking in Natural Language Understanding?" (NAACL)**
The argument that measurement failure, not model failure, is the field's bottleneck. This is structurally the argument your paper makes for legal reasoning. Use it to establish that "our evaluation could not see the thing" is a legitimate and important finding, not an apology.

**2. Marmor, "The Pragmatics of Legal Language" (USC Legal Studies working paper); also Poggi on legal implicature**
Read to decide whether "implicature" is even the right term for what CUAD clauses do. Marmor and Poggi both argue legal drafting violates Gricean cooperative assumptions in ways ordinary conversation does not — drafters are frequently adversarial, deliberately vague, or writing for future disputes rather than present understanding. **If that holds, your framing needs adjusting.** Better to adjust it yourself than to have a JURIX reviewer do it. This is the one reading that could change your introduction's vocabulary.

**3. "Insights from Negative Results in NLP" (ACL workshop proceedings) — any two papers**
Read purely for structure and posture. Note how they front-load the mechanism and treat the negative as a finding rather than an apology. That posture is most of what separates an accepted analysis paper from a rejected one.

---

### For §2 Related Work — positioning

**4. LegalΔ (arXiv 2508.12281) — READ FIRST**
Contains the critique of similarity-based legal reward, aimed at SyLeR: reliance on surface-level matching makes it difficult to capture the complexity and rigor of legal reasoning. **This is your Reviewer 2, pre-written and citable.** Your paper's job is to convert this from an assertion into a demonstrated, quantified result — which your 0.604-vs-0.073 measurement does. Read the related-work paragraph closely.

**5. SyLeR (Zhang et al., 2025)**
Structure-aware similarity-based reward for legal syllogistic reasoning. The direct precedent for similarity-as-legal-reward. You must cite it and distinguish yourself from it: SyLeR targets syllogistic structure; you targeted pragmatic implicature; both hit the same wall.

**6. Med-PRM (EMNLP 2025) — "Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards"**
The closest structural precedent for a domain-specific PRM grounded in an external authoritative corpus (clinical guidelines). Read for how they justify the grounding choice. **Also your nearest competitor for framing** — reviewers will ask how you differ. Note the honest answer: they had a curated, propositionally-structured guideline corpus; you had raw judicial opinions with no propositional structure, and that difference is a large part of why the approach transferred badly.

**7. Math-Shepherd (Wang et al., ACL 2024) — "Verify and Reinforce LLMs Step-by-step without Human Annotations"**
The structural template for framing automatic supervision as the contribution when the optimizer is standard. Also useful as a contrast: their automatic labels came from verifiable outcome correctness; yours came from a heuristic threshold with no verification. **That contrast is a paragraph in your Related Work and half your Limitations.**

**8. Fin-PRM; CorVer (corpus-grounded process supervision for factual QA)**
Cite briefly to show the domain-PRM landscape is filling and law was the open cell. Establishes that your problem is timely without claiming novelty you don't have.

---

### For §3 Method — the mechanics you must state correctly

**9. Rafailov et al. (2023), "Direct Preference Optimization" — Section 4, the derivation. BLOCKING.**
Not the abstract. Follow Eq. 1 (Bradley-Terry) through to Eq. 7 (the DPO loss). You need to be able to point at the BT term in your own loss function and write one sentence about it in §3.5. **This was assigned twice and is still outstanding.** It is now the single reading that blocks §3.

**10. Azar et al. (2023), "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO)**
What the Bradley-Terry assumption buys and what it breaks. Your Limitations paragraph on BT mis-specification comes from here. Specifically relevant if Set E shows your negatives are wrong in mutually incomparable ways rather than along one axis.

**11. Robinson et al. (2021), "Contrastive Learning with Hard Negative Samples" (ICLR)**
The theory of negative difficulty — negatives too easy give no signal, negatives too hard give noise. Your §3.4 justification for discussing `random.choice()` as a modelling decision rather than a sampling detail comes from here. Also the reason not to naively select least-similar negatives in future work.

**12. Karpukhin et al. (2020), "Dense Passage Retrieval" — Section 3**
Practical negative selection (random vs BM25 vs in-batch) in a retrieval pipeline structurally similar to yours. The precedent that negative selection strategy is a reportable design decision.

---

### For §4 Results — how to report a null

**13. Dodge et al. (2019), "Show Your Work: Improved Reporting of Experimental Results" (EMNLP)**
Reporting standards for exactly your situation: small effects, wide intervals, comparisons that do not separate. Your results section needs this discipline to be credible. **Read before writing §4.**

**14. Lambert et al., "RewardBench"**
The held-out pairwise accuracy protocol and its length-bias caveats. Gives you the reporting conventions so your implicit-reward evaluation reads as standard rather than improvised.

**15. "Bootstrapping Language Models with DPO Implicit Rewards" (arXiv 2406.09760)**
Demonstrates using the DPO implicit reward directly as a reward model. **Your methodological citation for the held-out evaluation** — you are doing the evaluation half of what they do.

**16. Rafailov et al. (2024), "From r to Q*: Your Language Model is Secretly a Q-Function"**
Why the implicit reward is a legitimate scorer, and what it is a scorer *of*. Read alongside 15.

---

### For §5 Analysis and §6 Limitations

**17. "When Data is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets" (arXiv 2511.10985) — Section 4.3**
Audited major DPO datasets with an independent reward model and found only 70–80% of samples where the chosen completion actually outranked the rejected, suggesting preference decisions are sometimes arbitrary or based on near-identical completions. **This is your calibration benchmark and your framing.** Your numbers are below that band, and citing this makes your finding a contribution to a known problem rather than an isolated failure. Your before/after or distribution table should mirror their Figure 4.

**18. "Towards Data-Centric RLHF: Simple Metrics for Preference Dataset Comparison" (arXiv 2409.09603)**
Off-the-shelf dataset-level diagnostics you can run in an afternoon. **The cheapest way to add two or three substantive table rows** and turn a single bad number into a proper dataset audit.

**19. Ma et al. (AAAI 2025), "What are step-level reward models rewarding? Counterintuitive findings from MCTS-boosted mathematical reasoning"**
The conceptual precedent for "the step-level signal is not measuring what you assumed." Cite when you argue the model learned the threshold function rather than reasoning quality.

**20. Ethayarajh (2019), "How Contextual are Contextualized Word Representations?"**
Gives you the vocabulary (anisotropy, narrow cone, random-pair baseline) to report your isotropy control precisely. **Note: your control refuted anisotropy as an explanation** — cite this to show you tested for it and ruled it out. That ruling-out is what makes the 0.604 finding strong rather than ambiguous. Su et al. (2021) on whitening and Li et al. (2020) BERT-flow can be cited in the same sentence as the standard corrections you did not need.

---

## PART II — READING ORDER UNDER TIME PRESSURE

If you can only read four things before drafting:

1. **Rafailov et al. 2023 §4** — blocks §3.5
2. **LegalΔ related work** — blocks §2 and §5
3. **"When Data is the Algorithm" §4.3** — blocks §4's framing
4. **Dodge et al. 2019** — blocks how §4 reports its numbers

Everything else can be cited from abstract-level familiarity for a first draft to your supervisor.

---

## PART III — ARGUMENT TREE

```
THESIS: Threshold-based pipeline supervision does not encode the
        pragmatic distinctions it intends to capture.
│
├── CLAIM 1: The pairs are semantically near-equivalent.
│   ├── Evidence: cos(chosen,rejected) = 0.604, std 0.054, unimodal
│   ├── Evidence: random unrelated judicial text = 0.073
│   ├── Inference: 0.604 in a 0.07–0.85 range means near-paraphrase
│   └── Vulnerability: no human confirmation → Set A
│
├── CLAIM 2: No scorer separates them meaningfully.
│   ├── Fresh cosine 54.7% [51.5,57.8]
│   ├── BM25 56.6% [53.5,59.7]
│   ├── TF-IDF 57.7% [54.6,60.8]
│   ├── DPO implicit reward, held out 58.4% [54.0,62.8]
│   └── Argument: convergence across independent method families
│       is evidence about the DATA, not about any one method
│
├── CLAIM 3: The supervisory signal is threshold-clearing, not preference.
│   ├── Evidence: verified in build_dpo_pairs.py, triplet_quality_scorer.py
│   ├── All scoring is anchor→candidate; nothing compares chosen to rejected
│   └── Consequence: rewards/margins measures learning of a surface property
│
├── CLAIM 4: The instrument is not at fault.
│   ├── Isotropy control rules out anisotropy
│   ├── Whitening therefore unnecessary
│   └── Strengthens Claim 1 substantially
│
└── OPEN: the decisive tail (mean gap 3× median) → Set B
```

---

## PART IV — COUNTER-ARGUMENT REGISTRY

| Objection | Response | Status |
|---|---|---|
| "Similarity-based legal reward is known to fail (LegalΔ)" | Agreed — we quantify it rather than assert it, and rule out the instrument as the cause | **Answered** |
| "This is just Med-PRM for law" | Med-PRM had a curated propositional guideline corpus; raw judicial opinions have no such structure, and that difference is the finding | **Answered** |
| "You claim to avoid Bradley-Terry but use DPO" | Withdrawn. BT is inherited and disclosed in §3.5 | **Resolved** |
| "The embedding space is degenerate" | Isotropy control: 0.073 on random pairs | **Answered** |
| "Rising rewards/margins shows it works" | Training diagnostic only; near-paraphrase pairs mean it learned a surface property | **Answered** |
| "82% one category invalidates the aggregate" | Disclosed in §3.1; all results reported per-category with CIs | **Mitigated** |
| "Maybe a human can separate these" | Untested | **OPEN — Set A** |
| "Maybe the negatives are just badly chosen" | `random.choice()` over 3,037 eligible; score-gap selection untested | **OPEN — Future Work** |
| "n=95 and n=78 are too small to conclude anything" | Correct. Explicitly reported as underpowered; no claims rest on them | **Conceded** |

---

## PART V — GAP REGISTRY

- No expert or human annotation anywhere in the project — the largest single gap
- No downstream legal task benchmark
- No qualitative generation comparison yet run (SFT vs DPO on unseen anchors)
- Three dropped categories never used as an out-of-domain probe
- Decisive-tail composition unanalysed
- Cross-encoder scores present in the data but never used as a discriminative baseline — a cheap missing row in the four-scorer table
