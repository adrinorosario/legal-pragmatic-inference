# JDAR Dataset Audit — Complete Working Notes
**Purpose:** Reference document for conference paper writing (data section, methodology, limitations, future work). Captures the full audit trail from initial dilemma through final dataset decision.

**Context at time of writing:** 2–4 week window to conference submission. Decision made to submit extraction pipeline + partial DPO results as a conference paper, positioned as a stepping stone toward a fuller journal submission. Full JDAR training/ablations deferred to the journal version.

---

## 1. Strategic Framing Decision (Conference vs. Journal)

**Original dilemma:** Deep journal-track work on JDAR was falling behind university deadlines. Journal review cycles run 2+ months; university timeline does not allow that flexibility.

**Decision:** Submit current JDAR extraction work to a conference first, positioned explicitly as a stepping stone to a more detailed journal publication later. Confirmed as fitting supervisor's (Dr. Rajesh Khanna's) view of scope.

**Rationale:**
- Conference cycles are faster and match the university timeline constraint.
- The existing extraction pipeline (triplet extraction, AND-gate/phrase-gate architecture, bi-encoder-non-discriminative finding) is a complete, defensible methods contribution on its own, independent of whether full JDAR training/ablations are finished.
- Conference → journal staged publication is a standard, well-regarded pattern in ML/NLP; not perceived as "rushed" if framed correctly.

**Three possible conference paper framings considered:**
1. **Pipeline + dataset paper** — contribution is the extraction methodology and a curated dataset for reward model training; no trained model required.
2. **Pipeline + preliminary/pilot results** — pipeline plus a small-scale proof-of-concept (e.g., SFT warmup only, or DPO on a verified subset) to show the signal is usable, without claiming a final trained JDAR.
3. **Position/architecture paper** — the verifier-vs-reward-model framing as the core contribution, with full training/ablations explicitly scoped as future/journal work.

**Note for paper framing:** Whichever framing is chosen, the data-quality audit documented below (Sections 3–7) is itself a legitimate methodological contribution and should be written up as such — not hidden as an implementation detail. Reviewers familiar with CUAD/legal-NLP work are likely to recognize rigorous, disclosed data auditing as a strength rather than a weakness, especially given the reduced dataset size relative to original volume estimates.

---

## 2. Training Method Decision: SFT + DPO (not GRPO)

**Question raised:** DPO vs. GRPO vs. SFT-only, given the 2–4 week window.

**Constraints considered:**
- Nothing trained yet as of decision point.
- Realistic available experiment time within the 2–4 week window: likely 5–7 days of actual training/debugging time after accounting for paper writing and inevitable reruns.

**Comparison:**

| Approach | Assessment |
|---|---|
| **SFT warmup alone** | Fastest, most predictable, lowest implementation risk. But does not test JDAR's core research claim — SFT teaches imitation of reference reasoning, not discrimination between good/bad reasoning steps. Reward signal itself goes untested. Treated as a fallback, not the target. |
| **SFT + DPO** | **Selected approach.** DPO is simpler to implement/stabilize than PPO/GRPO — no separate rollout loop or value function, works directly on (chosen, rejected) pairs, which map naturally onto the passing/failing triplet structure already in the pipeline. Directly tests the core research question: does training against JDAR's reference-similarity signal shift model reasoning toward judicial-style inference vs. baseline. Previously flagged as favored given compute constraints (see memory: DPO favored over PPO). |
| **GRPO** | Requires a working reward model actively serving rewards during rollouts (not just offline preference pairs), more hyperparameter sensitivity (group size, KL calibration under rollout noise), higher wall-clock cost per iteration. Appropriate for the journal version with full ablations; too much undebugged infrastructure risk for a 2–4 week conference timeline. |

**Decision:** SFT warmup + DPO for the conference paper. GRPO explicitly scoped as future work / journal-version content — this should be stated in the paper as a deliberate design choice signaling awareness of the design space, not an omission.

**Open question flagged, not yet resolved:** KL penalty β calibration during DPO training runs — noted as needing attention (see also memory: KL penalty β functions as an active penalty term subtracted from reward before the optimizer sees it, not passive monitoring).

---

## 3. The Triggering Example — club-misty-inc-v-laski-james

**How it surfaced:** User shared a single triplet from the dataset while discussing DPO pairing strategy, prompting an unplanned but necessary detour into data quality auditing.

**Full triplet as shared:**
```json
{
  "cuad_category": "Ip Ownership Assignment",
  "cuad_anchor": "Company acknowledges and agrees that all Intellectual Property created by Company, its affiliates, representatives, or agents in connection with or resulting from any work or services related to the Products, including the Deliverables (\"Work Product\"), but excluding the Neutral Alcohol Beverage Base and excluding the Company's general know-how and independently developed production processes not specifically related to the Products, have been specially ordered and commissioned by Reed's, are works-made-for-hire from the moment of creation and that all such Work Product is and will be the sole and exclusive property of Reed's.",
  "raw_sentence": "And so those decisions provide the framework for our analysis. Reed holds that an Illinois liquor license is a property right within the meaning of the due process clause of the Fourteenth Amendment. The license is revocable during its term only for cause, just like a public school teacher's tenure contract--a familiar example of \"property\" as the Supreme Court has defined the term in the due process clauses of the Fifth and Fourteenth Amendments.",
  "bi_encoder_score": 0.44649824500083923,
  "contract_id": "club-misty-inc-v-laski-james",
  "cross_encoder_score": -5.63739013671875
}
```

**Why it's a mismatch:** The anchor is a contractual IP work-for-hire/ownership-assignment clause. The raw_sentence is judicial reasoning about an Illinois liquor license as a constitutionally protected property right under Fourteenth Amendment due process. These are unrelated legal questions (contractual IP assignment vs. constitutional due process in licensing). The only surface connection is the shared word "property" — a lexical-overlap false positive, structurally the same disease already diagnosed and fixed in the bi-encoder.

**Cross-encoder score of −5.64:** Strongly negative — the cross-encoder itself was flagging this pair as irrelevant. This score was being logged as metadata only (from Run 4 onward) rather than used as a gating criterion, so it never blocked the triplet despite correctly signaling the problem.

---

## 4. Root Cause Analysis — Verified Against Actual Code

### 4.1 First hypothesis (from an external/attached AI-generated explanation) — REJECTED

An externally-sourced explanation (shared by the user, written in a similar tone/idiom to this assistant) claimed:
- D3 "sees scope + scope and calls it coherent" — i.e., D3's vagueness-vector matching was too coarse and let lexical overlap on "property" pass as conceptual coherence.
- Proposed fix: add a cross-encoder floor (`CE_FLOOR = -4.0`) as a post-hoc gate on passing triplets.

**This was flagged explicitly as unverified elaboration** — plausible-sounding, confirmatory of the assistant's own prior diagnosis, using real numbers from the run-comparison table but constructing an unverified causal story around them (a specific cross-encoder-by-category breakdown, a specific threshold value) without checking against actual code.

**Verification against actual `triplet_quality_scorer.py` code proved this explanation wrong.** Running the actual `score_dimension_3` function against the club-misty triplet showed:
```
Assigned anchor category: scope
D1: 1  matched_terms=['related']
D2: 1  interp_hits=2  matched=['\\bthe term\\b', '\\bmeaning of\\b']  proc_hits=0
D3: 0  category=scope  matched_terms=[]
TOTAL: 2   PASSES: True
```
**D3 correctly scored 0** — its scope-category term list genuinely does not appear in the raw_sentence. D3 did its job correctly. The external explanation's core claim was factually incorrect.

### 4.2 Actual root cause #1 — Threshold logic gap (scorer-level)

The pipeline's pass criterion was `total_score >= 2` (out of D1+D2+D3, each 0 or 1). This means **a triplet can pass purely on D1+D2 with D3 failing outright**, since 1+1+0 = 2 ≥ 2. D3 — the one dimension explicitly designed to check anchor-window legal coherence — is not actually required to pass for a triplet to be certified as "passing."

This is exactly what happened with club-misty: D1=1, D2=1, D3=0, total=2, passes=True.

**Fix (verified, minimal, one line):**
```python
# Before:
"passes": total >= 2,
# After:
"passes": total >= 2 and d3["score"] == 1,
```

### 4.3 Actual root cause #2 — Category-key casing mismatch (extraction-level, notebook)

Found in `jdar-extraction-accelerated.ipynb`. The category gating logic:
```python
category = matching_cuad_metadata["clause_category"]
category_pattern = CATEGORY_SPECIFIC_SIGNALS.get(category)
if category_pattern and not category_pattern.search(matching_raw_window):
    continue  # window lacks category-specific legal reasoning signal
```
`.get(category)` returns `None` if the key doesn't match exactly. When `category_pattern` is `None`, `category_pattern and ...` short-circuits to `False`, so the `continue` (rejection) never fires — **the category-specific gate silently no-ops and the window passes through unchecked.**

**Verified exact string mismatch:**
- `RUN4_TARGET_CATEGORIES` (Cell 22) uses CUAD's native label casing: `"Ip Ownership Assignment"`.
- `CATEGORY_SPECIFIC_SIGNALS` (Cell 45) was hand-typed with: `"IP Ownership Assignment"` (capital IP as acronym).
- These are different strings in Python; the dict lookup fails silently for every IP Ownership Assignment triplet across every run that used this gate (Run 4 onward).

**Exhaustive check across all 6 target categories (verified via direct key-set comparison, not sampling):**
```
RUN4_TARGET_CATEGORIES: ['Cap On Liability', 'Competitive Restriction Exception', 'Covenant Not To Sue',
                          'Ip Ownership Assignment', 'Non-Compete', 'Third Party Beneficiary']
CATEGORY_SPECIFIC_SIGNALS keys: ['Cap On Liability', 'Competitive Restriction Exception', 'Covenant Not To Sue',
                          'IP Ownership Assignment', 'Non-Compete', 'Third Party Beneficiary']
Mismatch: ONLY 'Ip Ownership Assignment' / 'IP Ownership Assignment' — all other 5 categories match exactly.
```
**Conclusion: this bug is scoped to exactly one category (IP Ownership Assignment). All other categories' phrase gates were live and functioning throughout Runs 4/4.1/5.**

**Why this bug was invisible until now:** IP Ownership Assignment consistently underperformed as a category across every run (17.6%–52.4% pass rate depending on run — see Section 6 table), which was previously attributed to the category being inherently harder. In fact its phrase gate — which should have been one of the *strongest* discriminators, since work-for-hire/inventorship/moral-rights language is narrow and legally specific — was never running at all.

**Fix location:** Cell 45 of the notebook, correct the key casing to `"Ip Ownership Assignment"` to match `clause_category`. This is an extraction-time fix; requires only rerunning Phase 3 (gating pass over already-computed embeddings), not full re-extraction, since Phases 1–2 (windowing, embedding) don't depend on this dict.

**Historical context on when components were added (from run comparison table):**
- Cross-encoder (ms-marco) added Run 3 (as gate), demoted to metadata-only Run 4/4.1/5.
- Category restriction (6 categories) and category-specific phrase gates first introduced Run 4.
- This means Runs 1–3 never had the category-specific phrase gate concept at all (broader, unrestricted category scope) — their reliability must be assessed independently, not assumed comparable to Run 4+.

---

## 5. Quantified Impact — Verified Against Real Data (Run 4.1, n=982)

Run 4.1's full scored file (`scored_triplets_version4_1.json`) was the first dataset actually inspected row-by-row (982 total triplets). All numbers below are computed directly from this file, not estimated.

### 5.1 Category distribution in Run 4.1
```
Ip Ownership Assignment:          651
Cap On Liability:                 149
Third Party Beneficiary:           77
Competitive Restriction Exception: 50
Non-Compete:                       50
Covenant Not To Sue:                5
```

### 5.2 Impact of D3-required fix alone (Section 4.2 fix, before casing fix)
```
Current pass count (total>=2):          366
  of which IP Ownership Assignment:     197
D3-required pass count:                 229
  of which IP Ownership Assignment:     112
Of 197 currently-passing IP triplets, 85 (43.1%) fail D3 and would be removed by the D3-required fix.
```

### 5.3 Impact of BOTH fixes combined (D3-required + corrected IP phrase gate) — decisive finding

Retroactively applying the corrected IP Ownership Assignment phrase gate (work for hire, shop rights, assignment of invention, ownership vests, moral rights, inventorship, etc.) to Run 4.1's IP data:
```
Total IP Ownership Assignment triplets (all, pre-scoring):        651
Of all 651, how many pass the corrected phrase gate:                 2
Of currently-passing (n=197), how many pass the phrase gate:         0
Of D3-required-passing (n=112), how many pass the phrase gate:       0
BOTH fixes applied — surviving IP Ownership Assignment triplets:     0
```
**Conclusion: zero of 651 IP Ownership Assignment windows in Run 4.1 contain genuine IP-ownership-specific judicial reasoning language.** This is not "some noise to trim" — the entire category's extraction in this run surfaced no real positive examples. Two interpretations considered: (a) the gate is correctly strict and the category is genuinely rare/hard to surface from COLD Cases at current windowing settings, or (b) the regex is under-covering plain-language ownership clauses. Both were tested (see Section 5.5) — evidence supports (a) combined with a deeper D1/D2/D3 discrimination failure, not (b) alone.

### 5.4 Other categories — confirmed structurally sound (Run 4.1)

Retroactive gate-hit check across all other categories:
```
Category                          Total  CurPass  D3Pass  GateHits(all)  GateHits(CurPass)  GateHits(D3Pass)  BOTH-fixes-survive
Third Party Beneficiary             77      49       41         77            49                 41                41
Non-Compete                         50      22       16         50            22                 16                16
Competitive Restriction Exception   50      22       15         50            22                 15                15
Cap On Liability                   149      76       45        149            76                 45                45
Covenant Not To Sue                  5       0        0          5             0                  0                 0
```
**Every non-IP category shows 100% gate-hit rate on all rows** — because the generic interpretive-signal regex used elsewhere in the pipeline substantially overlaps with these categories' specific-signal patterns (unlike IP's narrow, non-overlapping legal vocabulary). This confirms the category phrase gate discriminates meaningfully *only* for IP Ownership Assignment; for the other categories it was effectively redundant with existing checks, which is why the casing bug had no equivalent effect elsewhere.

**Run 4.1 "both fixes applied" usable totals by category:**
| Category | Usable (both fixes), Run 4.1 only |
|---|---|
| Third Party Beneficiary | 41 |
| Cap On Liability | 45 |
| Non-Compete | 16 |
| Competitive Restriction Exception | 15 |
| Covenant Not To Sue | 0 (negligible N regardless) |
| IP Ownership Assignment | 0 |
| **Total (Run 4.1 only)** | **117** |

### 5.5 Manual inspection of the 3 IP "survivor" rows in the final all-runs cleaned dataset — confirms false positives

After the user ran the fix across all 5 runs and deduped (Section 6), exactly 3 IP Ownership Assignment rows survived both fixes. Manual inspection:
- Row 1/2 (same anchor, different sentence): sentences about "ownership of the subject property has been transferred" (real property/land law) and statutory definition of "intellectual property" in a copyright infringement case — not reasoning about the anchor's actual work-for-hire assignment question.
- Row 3: sentence about land rezoning ("Landowners," "Rural Estates Residential," "avigation") — entirely unrelated to IP assignment; matched only on the bare phrase "ownership of the."

**Conclusion: even the 3 gate-surviving rows are false positives.** This rules out the hypothesis that the gate was simply too strict (under-covering good examples) — the rows it does let through are still wrong. Confirmed by also checking the 112–196 rows that passed D1+D2+D3 but were rejected by the corrected gate (a sample was pulled from Run 4.1): these too were substantively unrelated to IP (press-release privilege disputes, arbitration-subpoena procedure, ADEA age-discrimination litigation, Superfund cost-allocation litigation) — i.e., not "good IP examples wrongly excluded," but further confirmation that D1/D2/D3 alone cannot reliably surface genuine IP-ownership reasoning from this corpus at current settings.

**Final decision: IP Ownership Assignment is excluded from the dataset entirely, not salvaged by gate-tuning.** This should be stated as a disclosed limitation, evidenced by the above audit trail, in the paper's data/limitations section.

---

## 6. Full Multi-Run Audit — Final Cleaned Dataset

### 6.1 Process

User independently built and ran a consolidation script across all 5 runs' full scored JSON files (not just Run 4.1), applying:
1. The D3-required pass rule (`passes = total_score >= 2 and d3_semantic_coherence == 1`) uniformly across all rows in all 5 files.
2. The corrected IP Ownership Assignment category phrase gate, retroactively applied to all rows in all 5 files.
3. Deduplication by `(contract_id, cuad_anchor, raw_sentence)`, keeping first-seen occurrence in Run 1 → Run 5 order (explicitly flagged as an arbitrary tie-break choice, not implied by any prior spec — noted here for reproducibility/methods-section purposes).

### 6.2 Process-integrity findings (self-reported by the audit script, worth recording for methods rigor)

- **`d3_semantic_coherence` was present and non-null in every row across all 5 files.** This is fully expected and requires no caveat: **D1/D2/D3 scoring is, and always was, a standalone post-hoc layer (`triplet_quality_scorer.py`) entirely decoupled from the extraction notebook.** It was never part of the per-run extraction pipeline changes documented in the run-comparison table (Section 4.3/changes-per-run) — that table describes changes to the *extractor* (prompt fixes, AND-gate, category restriction, phrase gates, etc.), not the scorer. Every run's raw triplet JSON output was passed through the same scorer, applied uniformly, after extraction. There is no version-drift or historical-scorer-mismatch question for the paper's reproducibility statement. The two corrections made in this audit (D3-required `passes` criterion; corrected IP Ownership Assignment category-key casing) were applied retroactively and uniformly across all five runs' scored output as a single consolidation pass — not layered inconsistently across runs.
- **Stored `passes` field in every source file equals `total_score >= 2` exactly** — i.e., none of the original run outputs ever used a D3-aware rule; the new rule is a genuine methodological change, not a re-derivation of existing logic. Recomputation diverges from stored `passes` specifically and only where `total_score >= 2` but `d3 == 0` (i.e., D1=1, D2=1, D3=0, total=2) — the exact class of error demonstrated by club-misty.
- **Checkpoint math verified:** 12,294 input rows → 2,027 clean-pass + 10,267 clean-fail + 0 unverified = 12,294 (sums correctly).
- **300 duplicate rows removed** across the 5 runs (expected, given overlapping COLD Cases source material across runs).
- **1,727 final unique passing triplets** after both fixes and deduplication.
- **1,482 rows flip from stored `passes=True` to recomputed `False`** — entirely attributable to the new D3 condition and the IP gate fix. **Zero rows flip the other direction**, confirming the new rule is strictly more conservative than the original, as expected/intended.

### 6.3 Final category distribution (all 5 runs, both fixes applied, deduped) — n=1,727

```
Cap On Liability                     811
Covenant Not To Sue                  166
Insurance                            147
Third Party Beneficiary               95
Anti-Assignment                       92
Competitive Restriction Exception     85
Non-Compete                           78
Post-Termination Services             41
Audit Rights                          34
No-Solicit Of Employees               29
Exclusivity                           28
Rofr/Rofo/Rofn                        25
No-Solicit Of Customers               15
Non-Disparagement                     11
Minimum Commitment                    11
Revenue/Profit Sharing                10
License Grant                         10
Joint Ip Ownership                     9
Most Favored Nation                    7
Non-Transferable License               7
Warranty Duration                      5
Volume Restriction                     5
Ip Ownership Assignment                3   ← confirmed false positives, see Section 5.5; EXCLUDE
Price Restrictions                     2
Termination For Convenience            1
```

By source run:
```
Run 1: 884
Run 2: 624
Run 5: 186
Run 4:  26
Run 3:   7
```

### 6.4 Covenant Not To Sue spot-check — second confirmed-bad category

Rationale for checking: second-largest category (166 rows, ~10% of total dataset) that was **never part of `RUN4_TARGET_CATEGORIES`** — meaning it never had a category-specific phrase gate at all in any run, relying solely on generic D1/D2/D3 checks, the same checks already shown once to let a false positive through.

**Method:** Random stratified sample of 15 rows (seed=42) manually inspected for anchor-window topical coherence.

**Findings:**
- **10 of 15 sampled rows share the identical anchor** ("Notwithstanding any prior termination of the Owner or this Agreement, the Servicer shall not at any time...") paired against 10+ different, unrelated judicial windows — suggesting the windowing step pairs one anchor against many nearby-but-topically-random case-law paragraphs, with D1/D2/D3 passing several regardless of actual subject match.
- **0 of 15 sampled rows contain genuine covenant-not-to-sue reasoning** (forbearance, release of claims, non-assertion doctrine). Actual content found: arbitration severability doctrine, bankruptcy jurisdiction, VPA volunteer-immunity statutes, insurance claims-notice conditions precedent, civil procedure rules, agency-action finality review, foreclosure/eviction terms, specific-performance defenses, and one entirely unrelated immigration/USCIS appellate procedure case.
- **Verdict: worse failure rate than IP Ownership Assignment** — at least some of IP's rejected rows were topically adjacent (patent litigation, arbitration of IP disputes); Covenant Not To Sue's sample showed no topical adjacency at all, just generic contract/procedural litigation language.

**Decision: Covenant Not To Sue (166 rows) excluded from the dataset, same as IP Ownership Assignment.**

### 6.5 Categories NOT individually audited — explicit residual risk

Because Covenant Not To Sue (a D1/D2/D3-only category with no phrase gate) failed decisively on inspection, **the same doubt logically extends to every other category that also lacked a category-specific phrase gate** — i.e., every category outside the original 6-category `RUN4_TARGET_CATEGORIES` set. This includes at minimum:

```
Insurance                    147
Anti-Assignment                92
Post-Termination Services      41
Audit Rights                   34
No-Solicit Of Employees        29
Exclusivity                    28
Rofr/Rofo/Rofn                 25
No-Solicit Of Customers        15
Non-Disparagement              11
Minimum Commitment             11
Revenue/Profit Sharing         10
License Grant                  10
Joint Ip Ownership               9
Most Favored Nation              7
Non-Transferable License         7
Warranty Duration                5
Volume Restriction               5
Price Restrictions               2
Termination For Convenience      1
```
**These were explicitly NOT individually spot-checked**, due to time constraints — a deliberate, disclosed scoping decision, not an oversight. Given the established pattern (2 for 2 categories checked so far failing on manual inspection when relying on D1/D2/D3 alone), these should **not** be treated as reliable without further audit, and were excluded from the final usable dataset on that basis.

### 6.6 FINAL VALIDATED, USABLE DATASET

Only categories with a **verified, correctly-keyed, category-specific phrase gate** (the original 4 of the 6 `RUN4_TARGET_CATEGORIES` that were confirmed both present and correctly matched — Cap On Liability, Third Party Beneficiary, Competitive Restriction Exception, Non-Compete) are treated as validated for use in DPO pairing and any conference-paper results.

| Category | Count (all runs, both fixes, deduped) |
|---|---|
| Cap On Liability | 811 |
| Third Party Beneficiary | 95 |
| Competitive Restriction Exception | 85 |
| Non-Compete | 78 |
| **Total validated, usable** | **1,069** |

**Explicitly excluded, with reasons, for the paper's limitations/data section:**
- IP Ownership Assignment (3 surviving rows, all confirmed false positives on manual review) — category-key casing bug meant its phrase gate never ran across Runs 4/4.1/5; even after correction, D1/D2/D3 do not reliably surface genuine IP-ownership judicial reasoning from this corpus at current settings.
- Covenant Not To Sue (166 rows, 0/15 sampled rows topically valid) — never had a category-specific gate in any run; relying on D1/D2/D3 alone is confirmed insufficient.
- Insurance, Anti-Assignment, and 16 other long-tail categories (~300+ rows combined) — never had a category-specific gate; not individually audited due to time constraints; excluded on the precautionary logic established by the two confirmed failures above.

---

## 7. Key Methodological Findings — For Paper's Data/Methods Section

1. **Bi-encoder confirmed non-discriminative across all runs and quality tiers** (pre-existing finding, reconfirmed): mean bi-encoder score frozen at ~0.41–0.42 regardless of D1/D2/D3 quality tier or run, with 3–5% coefficient of variation throughout. This holds even within the newly-verified clean dataset.
2. **D1/D2/D3 rubric alone (i.e., without a category-specific phrase gate) is insufficient to guarantee anchor-window topical/legal coherence.** Demonstrated concretely twice: IP Ownership Assignment (Section 5) and Covenant Not To Sue (Section 6.4). The `total_score >= 2` threshold in particular allowed D3 (the coherence-specific dimension) to fail outright while the triplet still passed on D1+D2 alone — this was the proximate mechanism in the club-misty case.
3. **A category-specific phrase gate is the actual load-bearing discriminator when a category's legal vocabulary is narrow/non-overlapping with generic interpretive language** (e.g., IP's work-for-hire, moral rights, inventorship terms). For categories whose specific vocabulary substantially overlaps with the pipeline's generic interpretive-signal list (Cap On Liability, Third Party Beneficiary, Non-Compete, Competitive Restriction Exception), the category gate is close to redundant with existing checks — which is precisely why a silent gate failure (the casing bug) only manifested visibly in one category and not the others.
4. **A single-character-class casing mismatch in a dictionary key** (`"IP Ownership Assignment"` vs. `"Ip Ownership Assignment"`) silently disabled an entire category's most precise filter across three full extraction runs (Run 4, 4.1, 5), with no runtime error — `dict.get()` returning `None` combined with short-circuit `and` logic meant the failure was silent rather than crashing. This is worth a methods-section note on defensive coding practices for pipeline gates (e.g., asserting all target categories have a corresponding signal-dict entry at pipeline startup, rather than silently no-op-ing on a missing key).
5. **Dominant failure mode across every run (D1✓ D2✗ D3✗, 83–100% of failures in every run)** confirms D2 (interpretive/judicial-content density) is the primary structural bottleneck of the pipeline in terms of raw yield — most windows fail because they lack sufficient interpretive-signal density, not because of thematic mismatch. This is a separate, larger-scale finding from the specific false-positive bugs above, and was the original motivation for the Run 4→5 windowing changes (window size 3→4, step size 1→2).
6. **AND-gate structural filter remains the single highest-leverage pipeline change** of those tested (pre-existing finding, not contradicted by this audit).
7. **Cross-encoder score, though demoted to metadata-only from Run 4 onward, correctly flagged the club-misty triplet as irrelevant (score −5.64)** — raising a design question for the paper/future work: whether cross-encoder score should be reinstated as an active gate (as it was, partially, in Run 3) rather than logged-only, especially for categories lacking a strong category-specific phrase gate.

---

## 8. Dataset Yield Summary — Historical vs. Verified

**Originally planned/estimated (pre-audit, per project memory):** ~2,700–2,900 unique passing triplets after deduplication across Runs 1–5, under the original (non-D3-required) `passes` logic, intended for tiered allocation: Run 4/4.1/5 passing triplets → reward model training; Run 2/3 passing triplets → SFT warmup; failed triplets from high-quality runs → hard negatives.

**Actual, verified, audit-corrected numbers:**
- Total passing under corrected D3-required + IP-gate-fixed logic, all runs, deduped: **1,727** (already ~40% below the original estimate before further category exclusions).
- Total usable after excluding categories without a verified, correctly-functioning category-specific gate (Covenant Not To Sue, and the ~300-row long tail that was never individually audited): **1,069**, concentrated in 4 categories.
- **This is the number to plan DPO pairing around, not the original ~2,700–2,900 estimate.** The original tiered-allocation plan (Run 4/4.1/5 → reward model, Run 2/3 → SFT warmup) should be revisited given that the validated 1,069 triplets are pooled across Runs 1, 2, 4, and 5 for the 4 sound categories (exact per-run breakdown for just these 4 categories was not separately re-tabulated in this conversation — recommend a final pass to confirm before formal tiering).

---

## 9. Per-Run Historical Summary Tables (Raw, Pre-Correction) — For Appendix/Reproducibility Reference

These are the original run summaries as scored by the pipeline's own (pre-D3-required, pre-casing-fix) logic. Included for completeness/reproducibility — **do not use these pass counts directly for paper results; use Section 6.6's corrected figures.**

### Run 1 (n=5,985)
- Passes (≥2/3): 1,323 (22.1%) — note: this figure appears to differ slightly from the "1,266" cited in project memory notes; use the scored-summary file's figure (1,323) as authoritative, or reconcile before publication.
- D1: 94.2%, D2: 8.4%, D3: 15.4%
- Dominant fail mode: D1✓D2✗D3✗, 92.5%
- IP Ownership Assignment pass rate: 17.6% (6/34)
- Cap On Liability pass rate: 27.5% (682/2,476)

### Run 2 (n=3,879)
- Passes: 1,210 (31.2%)
- D1: 98.0%, D2: 16.0%, D3: 17.5%
- Dominant fail mode: D1✓D2✗D3✗, 97.1%
- IP Ownership Assignment pass rate: 44.1% (30/68)
- Cap On Liability pass rate: 34.7% (320/922)

### Run 3 (n=1,005)
- Passes: 353 (35.1%)
- D1: 96.1%, D2: 16.3%, D3: 23.1%
- Dominant fail mode: D1✓D2✗D3✗, 94.0%
- Widest category spread of any run (26 categories, unrestricted)
- IP Ownership Assignment pass rate: 52.4% (11/21)
- Cap On Liability pass rate: 38.7% (118/305)

### Run 4 (n=73)
- Passes: 34 (46.6%) — highest pass rate of any run, but smallest N by far (pilot/validation run for the new category-restricted + phrase-gate approach)
- D1: 100.0%, D2: 15.1%, D3: 35.6%
- Dominant fail mode: D1✓D2✗D3✗, 100%
- Only 5 categories represented (no IP Ownership Assignment in this particular scored file)

### Run 4.1 (n=982) — see Section 5 for full row-level audit
- Passes: 366 (37.3%)
- D1: (not separately restated; see Section 5 for row-level detail)
- IP Ownership Assignment pass rate (original logic): 30.3% (see project memory) — **superseded by Section 5's finding of 0 valid rows after correction**

### Run 5 (n=1,352)
- Passes: 589 (43.6%)
- D1: 91.8%, D2: 24.2%, D3: 27.1%
- Dominant fail mode: D1✓D2✗D3✗, 85.5%
- IP Ownership Assignment pass rate: 38.6% (320/828) — largest category by far in this run
- Cap On Liability: 48.3% (112/232)
- Third Party Beneficiary: 67.9% (72/106) — consistently the best-performing category across every run
- Covenant Not To Sue: 7.1% (1/14)

**Cross-run constants worth noting for methods section:**
- Third Party Beneficiary is consistently the highest-passing category in every run (25.4%→69.0% range) — likely because its interpretive vocabulary (intended beneficiary, privity, standing to enforce) is both distinctive and well-represented in general case law.
- Bi-encoder score means are flat (~0.41–0.42) across every single quality tier (0/3 through 3/3) and every run — the clearest, most consistent evidence of the bi-encoder's non-discriminative behavior across the entire project.
- D3 anchor category distribution shows `scope` dominant at 47–65% across every run, with `industry_norms` collapsing to 0.0% from Run 3 onward (cross-encoder penalization + category restriction effects) — relevant context if discussing the D3 vagueness-vector taxonomy's coverage limitations in the paper, though note this is a distributional observation, not itself a confirmed bug (unlike the two issues in Sections 4 and 6.4).

---

## 10. Open Items / Not Yet Resolved (Carry Forward)

### Resolved in follow-up session (see Section 12 for full detail)

1. ~~Per-run breakdown of the final validated 1,069 triplets~~ — **RESOLVED.** See Section 12.1 for the full per-run × per-category table. Key implication: original run-based tiering plan (Run 4/4.1/5 → reward model, Run 2/3 → SFT warmup) no longer applies cleanly, since Runs 3/4 barely contributed to 3 of the 4 validated categories. Recommend tiering by triplet quality score (3/3 → reward model, 2/3 → SFT warmup) instead.
2. ~~Whether the pre-D3 scorer version genuinely existed for Runs 1–2~~ — **RESOLVED, non-issue.** D1/D2/D3 scoring was always a standalone post-hoc layer (`triplet_quality_scorer.py`), fully decoupled from the extraction notebook and its per-run changes. Every run's raw output was scored uniformly, after extraction, by the same scorer. No version-drift or reproducibility caveat needed. See Section 12.2.
3. ~~Within-category DPO pairing negative-selection strategy~~ — **RESOLVED AND EXECUTED.** `build_dpo_pairs.py` built, run, and reconfirmed after the Competitive Restriction Exception drop. Final output: `dpo_pairs_2.jsonl`, 984 pairs across 3 categories (Cap On Liability, Third Party Beneficiary, Non-Compete). Same-anchor-preferred / same-category-fallback logic confirmed working as specified; `pair_type` and `failure_reason` logged on every row. See Sections 12.3 and 13 for full detail, Section 13.3 for final counts.

### Still open

4. **KL penalty β calibration** for DPO training — flagged as needing attention, not yet addressed.
5. **Category imbalance in the final 4-category set** (Cap On Liability alone is 811/1,069 ≈ 76% of the validated dataset) — decided to address via (a) weighted/stratified sampling during DPO training (e.g., `WeightedRandomSampler`) to prevent majority-category domination of gradient updates, and (b) mandatory category-stratified evaluation reporting in the results section (not just an aggregate DPO-vs-baseline number), so any category-specific weakness is disclosed rather than hidden. Neither yet implemented.
6. **Whether to reinstate the cross-encoder as an active gate** (as in Run 3) rather than metadata-only (Run 4/4.1/5), given it correctly flagged club-misty — open design question for future work / journal version, not resolved for the conference paper.
7. **Long-tail categories (Insurance, Anti-Assignment, and 16 others, ~300+ rows) remain unaudited** — excluded from the validated set on precautionary grounds but not definitively proven bad; could be revisited for the journal version with more time.
8. **Run the pairing script** once `FAILING_PATH` is supplied, then review the same_anchor vs. same_category_diff_anchor split it reports — if the fallback rate is high for the thinner categories (Non-Compete, Competitive Restriction Exception), that's worth flagging in the paper as a limitation of pairing precision for those categories specifically.

---

## 12. Follow-Up Session — Detail on Items 1–3

### 12.1 Per-run × per-category breakdown of the validated 1,069 triplets

Computed directly from `clean_passing_triplets_deduped.json`, filtered to the 4 validated categories:

| Category | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Total |
|---|---|---|---|---|---|---|
| Cap On Liability | 505 | 211 | 2 | 15 | 78 | 811 |
| Competitive Restriction Exception | 2 | 56 | 0 | 1 | 26 | 85 |
| Non-Compete | 18 | 32 | 0 | 3 | 25 | 78 |
| Third Party Beneficiary | 6 | 26 | 3 | 6 | 54 | 95 |

**Notes for methods/limitations sections:**
- Cap On Liability is 62% sourced from Run 1 (505/811), which predates category restriction and phrase-gate refinements. The category's phrase gate is confirmed correctly keyed and was applied retroactively/uniformly, so this is not a correctness issue — but it means the majority of this category's data was extracted under the least-refined windowing/step settings, worth a methods-section sentence.
- Competitive Restriction Exception and Non-Compete have near-zero presence in Run 3 (0 each) and thin presence in Run 4 (1, 3) — both categories' totals lean heavily on Runs 1, 2, and 5.
- **Practical consequence:** the original tiering plan (Run 4/4.1/5 triplets → reward model training; Run 2/3 → SFT warmup) does not map cleanly onto this data, since Runs 3 and 4 contributed minimally to 3 of the 4 validated categories. **Recommended replacement:** tier by `total_score` instead (3/3-scoring triplets → reward model training pool; 2/3-scoring → SFT warmup pool) — a criterion still meaningful post-consolidation, unlike source-run, which now mixes extraction settings within each category.

### 12.2 D1/D2/D3 scorer versioning — clarified, not actually an open question

**Correction to the original framing in Section 6.2:** the scorer (`triplet_quality_scorer.py`) was never part of the extraction notebook and never changed alongside the per-run extraction changes (prompt fixes, AND-gate, category restriction, phrase gates — all documented in the Section 4.3 changes-per-run table). It is, and always was, a **separate, standalone, post-hoc scoring pass** applied uniformly to each run's raw triplet JSON output after extraction. There is no scenario in which Run 1/2's D3 scores reflect a different scorer version than later runs — they were all scored by the same logic, at the time scoring was performed, not at the time of extraction.

**Practical consequence:** no hedge or caveat is needed in the paper's reproducibility statement regarding scorer-version consistency across runs. A clean, defensible sentence for the methods section: *"All five extraction runs' triplet outputs were scored using a single, consistent post-hoc quality rubric (D1/D2/D3), applied uniformly across runs. The rubric's pass criterion was subsequently corrected during dataset auditing to require D3 (semantic coherence) explicitly — previously D1+D2 alone could satisfy the ≥2/3 threshold — and this corrected criterion, along with a fix to a category-key mismatch affecting the IP Ownership Assignment phrase gate, was applied retroactively and uniformly to all five runs' scored output prior to final dataset consolidation."*

### 12.3 DPO pairing script — built, not yet executable

**File:** `build_dpo_pairs.py`

**Logic implemented, matching the prior decision exactly:**
1. For each passing triplet (restricted to the 4 validated categories), look for a same-category, same-`cuad_anchor` failing triplet with a different `raw_sentence` — this is the preferred, tightest negative (same legal question, weaker reasoning window).
2. If no same-anchor failing triplet exists, fall back to any same-category failing triplet with a different anchor.
3. If neither exists, the passing triplet is left unpaired and counted separately (not silently dropped without a record).
4. Every emitted pair is tagged with `pair_type` (`same_anchor` or `same_category_diff_anchor`), plus provenance fields (`contract_id`, `_source_run`, `total_score` for both sides) for full traceability back to source.

**Blocking issue:** the script requires a failing-triplet pool (D1=1, real-candidate-but-failing triplets — your originally planned hard-negative source) as input. Only the passing/cleaned dataset (`clean_passing_triplets_deduped.json`) has been supplied so far. `FAILING_PATH` is deliberately left unset (`None`) in the script rather than guessed, so it will not run against an assumed or reconstructed file — this mirrors the audit principle established earlier in this project (verify against real data, don't extrapolate from what's plausible).

**Status: executed.** Script was run against the raw scored output (all 5 runs) with `FAILING_SOURCE_PATH` pointed at the consolidated raw scored files. Initial run (4 categories, before the Competitive Restriction Exception drop) produced 1,069 pairs — see Section 13 for the pair-level audit that followed and the resulting category drop. Final run (3 categories, post-drop) produced 984 pairs, written to `dpo_pairs_2.jsonl`. Printed per-category breakdown from the final run:

```
Derived failing/negative pool: 3037 triplets (D1==1, total_score<2, validated categories only)
  Failure reason breakdown: low_density_and_subject_mismatch: 3037 (100% — expected, see above)
Total pairs written: 984
Unpaired passing triplets: 0

Category                    same_anchor   same_cat_diff_anchor   unpaired
Cap On Liability                    806                      5          0
Non-Compete                          69                      9          0
Third Party Beneficiary              94                      1          0
```

This exactly matches Section 13.3's figures, confirming consistency between the pre- and post-drop runs (failing pool size changed from 3,278 → 3,037, i.e., 241 fewer failing candidates, consistent with Competitive Restriction Exception's own failing-pool contribution being removed).

**Watch for once run:** if the `same_category_diff_anchor` fallback rate is high for Non-Compete (78 total) or Competitive Restriction Exception (85 total) — the two thinnest validated categories — that's a real limitation worth disclosing, since it means DPO's negative signal for those categories is weaker (topic-level rather than clause-level discrimination) than for Cap On Liability or Third Party Beneficiary.

---

## 11. Recommended Paper Section Mapping

- **Data / Methods section:** Sections 4, 6.1–6.3, 6.6, 7 (methodology of the audit, the two root-caused bugs, the corrected pipeline logic, final validated dataset composition).
- **Limitations section:** Sections 5.3–5.5, 6.4, 6.5 (excluded categories, why, evidence trail — frame as rigor, not weakness).
- **Results section:** Should report DPO/SFT results stratified by the 4 validated categories individually, not only pooled (per Section 10, item 5), given the volume imbalance.
- **Future work / journal-version framing:** Section 2 (GRPO, full ablations), Section 10 items 6–7 (cross-encoder reinstatement, long-tail category recovery), and re-extraction of IP Ownership Assignment and Covenant Not To Sue with corrected/strengthened gates as a distinct future contribution.
- **Reproducibility appendix:** Section 9 (raw per-run tables), with an explicit note that these reflect pre-correction pipeline logic and pointing readers to the corrected Section 6.6 figures for actual reported results.

---

## 13. Pair-Level Audit and Final Category Drop (post-pairing)

After running `build_dpo_pairs.py` against the corrected failing pool (Section 12.3 logic finalized: D1==1 AND total_score<2), the script produced 1,069 pairs across the 4 validated categories:

```
Cap On Liability:                   811 pairs (806 same_anchor, 5 same_category_diff_anchor)
Third Party Beneficiary:             95 pairs (94 same_anchor, 1 same_category_diff_anchor)
Competitive Restriction Exception:   85 pairs (82 same_anchor, 3 same_category_diff_anchor)
Non-Compete:                         78 pairs (69 same_anchor, 9 same_category_diff_anchor)
Unpaired: 0 across all categories.
```

**Failing pool size:** 3,278 candidate negatives (D1==1, total_score<2, validated categories only). `failure_reason` breakdown showed 100% `low_density_and_subject_mismatch` — this is a mathematical necessity of the pool definition (D1=1 contributes 1 point; total<2 forces D2+D3 to sum to <1, meaning both must be 0), not a data anomaly. The finer D2-only vs. D3-only distinction intended for journal-stage refinement will require loosening the pool definition beyond `total<2`, not just re-reading this field.

**Same-anchor pairing dominance (1,051/1,069 = 98.3% same_anchor overall)** is a strong, reportable methods-section result: the failing pool (3,278) was large enough relative to the passing set (1,069) that same-anchor negatives — the tightest, most informative type — were available for nearly every passing triplet, with the same_category fallback needed only 18 times total.

### 13.1 Manual pair-level audit — the critical finding

Despite both fixes (D3-required threshold, corrected IP category-key casing) being applied at the **triplet** level before pairing, a manual read of actual chosen/rejected pairs surfaced a **new, pairing-stage-visible problem in Competitive Restriction Exception** that triplet-level auditing had not caught.

**Sample (2 pairs, both sharing the same anchor):**
- Anchor: "For clarity and notwithstanding anything contained herein, nothing in this Section 2.1(e)(i)(A) is intended to be inconsistent with Section 2.4(e)(i) or to otherwise indicate that Customer is subject..."
- Pair 1 — chosen: bankruptcy safe-harbor doctrine (11 U.S.C. §546(e)); rejected: insurance claims-made-and-reported notice-prejudice rule.
- Pair 2 — chosen: TVPRA (Trafficking Victims Protection Reauthorization Act) civil-remedy provision interpretation; rejected: motion-for-clarification procedural standard.

**Neither chosen side is substantively about competitive restrictions or exceptions to them.** Both are generic legal-interpretation windows that happened to co-occur with an anchor whose own text is generic boilerplate ("for clarity," "notwithstanding anything contained herein") — i.e., **the anchor itself is a weak/generic instance of this category**, and the category-specific phrase gate for Competitive Restriction Exception evidently matches on this kind of generic connective language rather than requiring true competitive-restriction substance (safe harbor, carve-out, excepted activity, etc. — the gate's intended terms per Section 4.3's `CATEGORY_SPECIFIC_SIGNALS`).

**This is the same underlying failure pattern as club-misty (Section 3) and Covenant Not To Sue (Section 6.4) — lexical/structural overlap passing as topical coherence — but it slipped past both the D3-required fix and the category-gate-casing fix**, because unlike the IP Ownership Assignment bug, this is not a casing/key-lookup error. The Competitive Restriction Exception gate is correctly keyed and does run — its *pattern itself* is under-specific, matching connective/boilerplate phrasing rather than category-specific substance. This is a **gate-precision issue, not a gate-activation issue**, and is a genuinely different root cause from Sections 4 and 6.4, worth distinguishing clearly in the paper's limitations section.

**A 20-pair stratified manual audit of Cap On Liability (the largest category, 811 pairs) was also conducted for comparison:**
- Result: mixed. Approximately 70–80% of sampled pairs (estimate from 20-sample review, not exhaustive) showed a clear reasoning-quality gap between chosen and rejected sides consistent with the category's actual subject (indemnification, consequential/punitive damages exclusion, liability limitation clauses).
- A minority (~20–30%, roughly 4–6 of 20 sampled) were weaker pairs where neither chosen nor rejected side clearly engaged with liability-limitation reasoning — notably a cluster of pairs drawing on **BP Deepwater Horizon settlement-program claims-administration text** (Court-Supervised Settlement Program, Economic Loss claims, Claimant R&D expense calculations), which is damages-adjacent but is claims-processing/administrative text, not judicial interpretation of a liability-cap contract clause.
- **Not yet checked:** whether this BP-settlement text represents a small number of contract_ids contributing disproportionately many windows (i.e., whether Cap On Liability's 811 pairs reflect fewer genuinely independent anchors than the count suggests). Flagged as a follow-up item, not resolved.
- **Decision:** Cap On Liability was retained despite this softer finding, on the grounds that (a) the quality gap, while inconsistent, is present in the clear majority of sampled pairs, unlike Competitive Restriction Exception where it was absent in both sampled pairs, and (b) the finding is disclosable as an honest, bounded limitation rather than indicating the category is fundamentally unsound.

### 13.2 Decision: Competitive Restriction Exception dropped

**Competitive Restriction Exception (85 pairs) is excluded from the final training/paper dataset**, on the same evidentiary standard applied to IP Ownership Assignment and Covenant Not To Sue: a manual sample (2 of 85 pairs, both sharing one anchor) showed 0/2 pairs with genuine category-relevant substance on either the chosen or rejected side. Given the established pattern (3 for 3 categories now failing when their gate/pairing output is manually inspected closely, whether due to casing bugs, generic-vocabulary D1/D2/D3 passes, or under-specific phrase-gate patterns), this was treated as sufficient evidence to drop rather than pursue further without-time-to-spare investigation into whether a larger sample would show a better rate.

**This is an explicit, disclosed, time-constrained judgment call** — a 2-pair sample is thin evidence on its own, and it remains possible a larger audit would show Competitive Restriction Exception performing better than this snapshot suggests. The decision to drop rather than investigate further reflects the project's remaining timeline, not a claim of exhaustive proof. This should be stated plainly if referenced in the paper's limitations section (e.g., "Competitive Restriction Exception was excluded following a small-sample manual audit that raised concerns about gate precision; a fuller audit is deferred to future work").

### 13.3 FINAL, FINAL validated dataset for conference paper

| Category | Pairs | Same-anchor | Same-category-fallback |
|---|---|---|---|
| Cap On Liability | 811 | 806 | 5 |
| Third Party Beneficiary | 95 | 94 | 1 |
| Non-Compete | 78 | 69 | 9 |
| **Total** | **984** | **969 (98.5%)** | **15 (1.5%)** |

**This 984-pair, 3-category set supersedes the 1,069-pair, 4-category figure used earlier in this document (Sections 6.6, 10, 12.1). Confirmed via actual script execution — output file `dpo_pairs_2.jsonl`.** `build_dpo_pairs.py` was updated to `VALIDATED_CATEGORIES = {"Cap On Liability", "Third Party Beneficiary", "Non-Compete"}` and rerun; printed output matched this table exactly (failing pool 3,278 → 3,037 candidates after Competitive Restriction Exception's removal, consistent with expectations).

**Category imbalance is now more pronounced, not less** — Cap On Liability is 811/984 ≈ 82% of the final dataset (up from 76% in the 4-category version). The weighted-sampling and category-stratified-evaluation mitigations discussed in Section 10, item 5 are now more important, not optional, given this concentration.

---

## 14. Current Status Summary (as of dataset-phase close)

**Dataset construction and DPO pairing are complete.** This is the state to work from going forward — sections above are the audit trail, not open work.

- **Final validated dataset:** 984 chosen/rejected pairs, 3 categories (Cap On Liability 811, Third Party Beneficiary 95, Non-Compete 78), file `dpo_pairs_2.jsonl`.
- **Excluded categories, all confirmed via manual evidence, not assumption:** IP Ownership Assignment, Covenant Not To Sue, Competitive Restriction Exception, plus ~300 rows across 16 unaudited long-tail categories (Insurance, Anti-Assignment, etc. — excluded on precautionary grounds, not individually disproven).
- **Known, disclosed soft spot:** Cap On Liability (82% of final dataset) showed a ~20–30% weak-pair rate in a 20-sample manual audit, concentrated around BP Deepwater Horizon settlement-administration text. Retained, not dropped — the majority of sampled pairs were sound, and the finding is written up as an honest limitation (Section 13.1) rather than hidden.
- **Category imbalance is real and unresolved as an engineering matter:** weighted/stratified sampling during DPO training and mandatory category-stratified evaluation reporting (Section 10, item 5) are the agreed mitigations — neither implemented yet. This is the next concrete task before training starts.
- **Also still open, unchanged from Section 10:** KL penalty β calibration (item 4); cross-encoder-as-active-gate design question, deferred to journal version (item 6); long-tail category recovery, deferred to journal version (item 7).

**Next step:** build/confirm the DPO training loop with weighted sampling by category wired in before the first training run, given Cap On Liability's dominance of the final dataset.