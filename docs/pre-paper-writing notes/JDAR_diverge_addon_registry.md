# JDAR — Diverge / Add-on Registry

Every scope option surfaced during this project, with the reason it was proposed, its cost, and its current status. **Add-on** = extends current tools/methods without abandoning the core approach. **Diverge** = a genuinely new direction.

Nothing here is a recommendation. These are decisions for you to make.

---

## STILL LIVE FOR THIS PAPER (three-day window)

### A1 — Add-on: the blind read
**What:** 20 stratified triplets, labels stripped, human judgement, then unblind.
**Why it matters:** It is the only human validation anywhere in the project and it determines which of two papers you are writing — "the pairs carry no quality signal" or "the pairs carry a signal no automatic scorer can see."
**Cost:** 45 minutes.
**Status:** Not run. Highest value remaining item.

### A2 — Add-on: decisive-tail analysis
**What:** Top 50 vs bottom 50 held-out pairs by implicit-reward gap; compare on category, length delta, `total_score` gap, cos(chosen,rejected).
**Why it matters:** Mean gap is 3× median. The model is indifferent on most pairs and decisive on a minority. This is the only unexplained structure in your results and could be the figure people remember.
**Cost:** ~1 hour, data already in hand.
**Status:** Not run. Second-highest value.

### A3 — Add-on: cross-encoder as a fourth baseline row
**What:** `cross_encoder_score` is already in the triplet JSON but was never used as a discriminative scorer. Add it to the convergence table.
**Why it matters:** Your own audit notes flagged that deeply negative cross-encoder scores caught real false positives the bi-encoder missed. If the cross-encoder scores markedly higher than 58%, that is a significant result and a direct pointer to your future work. If it also lands at ~57%, the convergence argument gets a fifth independent member.
**Cost:** Under an hour — the scores already exist.
**Status:** Never proposed explicitly during the conversation. **Probably the cheapest high-value item on this list.**

### A4 — Add-on: qualitative generation comparison
**What:** Unseen anchors, generate with SFT checkpoint and DPO checkpoint, present 5–6 side by side.
**Constraint:** Must use anchors *not* in the SFT or DPO datasets. Candidate sources: unused CUAD clauses in the three categories, or the three dropped categories (IP Ownership Assignment, Covenant Not To Sue, Competitive Restriction Exception) as an out-of-domain probe — the raw clauses are fine, only the pipeline was buggy.
**Framing constraint:** This is a case study of *model behaviour shift*. It is NOT evidence about what the failed triplets lack, and must not be labelled as such.
**Cost:** Half a day including generation.
**Status:** Live, lower priority than A1–A3.

### A5 — Add-on: data-centric dataset metrics
**What:** Apply the diagnostics from arXiv 2409.09603 to the 984 pairs.
**Why:** Turns a single low number into a proper dataset audit; adds two to three table rows.
**Cost:** An afternoon.
**Status:** Live if time permits after A1–A3.

---

## DEFERRED TO FUTURE WORK (name explicitly in the paper)

### D1 — Diverge: adversarially constructed negatives
**What:** Take the chosen judicial reasoning and perturb the *interpretive conclusion* while holding topic, register, and vocabulary fixed.
**Why it is the right fix:** It produces pairs that differ on exactly the pragmatic axis you care about and nowhere else. Your entire results section is the argument for why this is necessary — which is the strongest possible position from which to write a future-work section.
**Cost:** New generation pipeline plus validation. Weeks.
**Status:** Deferred. **Say this in the paper.**

### D2 — Diverge: entailment-based scoring instead of geometric similarity
**What:** Replace cosine with an NLI or cross-encoder entailment formulation — "does this reasoning entail this reading of the clause" rather than "is this reasoning near this clause."
**Why:** Entailment models are trained to discriminate rather than to embed, so they have dynamic range where cosine does not. Your own audit already flagged the cross-encoder catching false positives the bi-encoder missed.
**Status:** Deferred. Natural successor paper. Pairs with A3 — if A3 shows the cross-encoder separating better, D2 has a running start.

### D3 — Diverge: human agreement study
**What:** Recruit law students or a plant-... (no — a law-qualified annotator: KAU-equivalent for law would be a law faculty member or practising advocate) to annotate a subsample for which reasoning a court would accept, and validate against that rather than against corpus structure.
**Why:** It is the largest gap in the project and would substantially strengthen legal credibility — especially for a JURIX audience.
**Cost:** Annotator recruitment, guidelines, IAA measurement. Not compatible with a three-day window.
**Status:** Deferred. Note that A1 (the blind read) is a one-person, non-rigorous proxy for this and should be described as such, not as a study.

### D4 — Add-on: score-gap negative re-selection, then retrain
**What:** For each passing triplet, select from the same-anchor failing pool the candidate with the *lowest* `total_score` rather than a random one. Then recompute cos(chosen,rejected) and retrain.
**Why:** 2,053 eligible negatives were discarded by `random.choice()`. Nobody decided that; it was a default. Maximising the score gap maximises contrast along the only quality axis the pipeline measures.
**Caution:** Do NOT select the least-similar negative instead. That produces easy negatives — trivially separable off-topic text — which is a different failure and just as bad. Score gap, not cosine distance.
**Why it was dropped:** It changes the training data, so learning anything from it requires retraining. Without retraining you get only a dataset-level contrast table, not a claim about the model. Correctly rejected as a poor three-day trade.
**Status:** Deferred. **The right first experiment for the journal version.**

### D5 — Diverge: JDAR as a live scorer at inference
**What:** Best-of-N reranking — generate N reasonings, score each against retrieved judicial passages with JDAR, select the best.
**Why:** This would make JDAR a genuine verifier and vindicate the original framing. It is the only version of the project in which the name "Judicial Decision-Aligned Reasoning" describes a scoring mechanism rather than a data filter.
**Status:** Deferred. Was the original vision; note in the paper that the current work does not implement it.

### D6 — Diverge: GRPO with JDAR as a live reward
**What:** Online RL against JDAR rather than SFT+DPO on static JDAR-derived pairs.
**Status:** Long-standing deferral to the journal version. **Associated unfinished exercise:** write out crisply the difference between training with GRPO against JDAR as a live reward versus SFT+DPO on static JDAR-derived pairs. Inability to answer this crisply should be named as a limitation in the paper.

### D7 — Diverge: broaden into a cognitive-science-flavoured pragmatic inference framing
**What:** Reposition around pragmatic inference as a cognitive phenomenon rather than a legal-NLP engineering problem.
**Why it stayed deferred:** Higher experimental cost, and the empirical results do not yet support the broader claim.
**Status:** Deferred. Worth revisiting only after D1 produces pairs that isolate the pragmatic axis.

---

## CLOSED — do not revisit

| Item | Why closed |
|---|---|
| Whitening / BERT-flow correction | Refuted by the isotropy control (random pairs at 0.073). The space is not anisotropic. Final. |
| Asking the DPO adapter what failed triplets lack | Circular — model trained on threshold-defined preferences would report the threshold function back |
| Three-way ablation "verifier vs Bradley-Terry vs lexical" | Premised on JDAR being a verifier, which it is not. Superseded by the four-scorer convergence table. |
| gpt-oss-20b as a reportable second model | fp32 loss spikes; add-on at most, never a dependency |
| Ministral-3-14B-Reasoning | Ruled out, suspected overfitting |
| Journal version now | Explicitly deferred; conference paper first |

---

## DECISION FRAME FOR THE THREE DAYS

If you do nothing else: **A1, A2, A3.** They cost roughly three hours combined, use data you already have, require no retraining, and each one materially changes what §5 can claim.

A4 and A5 are genuinely optional. D1–D7 all belong in Future Work as prose, not as experiments.
