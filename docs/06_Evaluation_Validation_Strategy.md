# Evaluation & Validation Strategy

## Redrob Intelligent Candidate Discovery & Ranking Engine

**Document:** `docs/06_Evaluation_Validation_Strategy.md`
**Version:** 1.0
**Inputs:** `01_PRD.md`, `02_Data_Understanding_Report.md`, `03_System_Architecture_Document.md`, `04_Ranking_Scoring_Explainability_Methodology.md`, `05_API_Module_Spec.md`
**Status:** Proposed — evaluation plan for the build and demo phase

---

## 0. Why This Document Exists

There is no labeled ground truth anywhere in this dataset. No `correct_rank` column, no human-annotated "this candidate is a 9/10 fit" field, nothing. That means we cannot report a single clean accuracy number and call it done — and it means a judge cannot either, which is exactly why the brief frames Judging Criterion 1 as a question ("are the top 20 _genuinely_ the best matches?") rather than a number to hit.

The honest position is: **we cannot prove the ranking is optimal, but we can build enough converging evidence — from multiple independent angles — that a skeptical judge has to actively work to find a counterexample.** That is the bar this document is designed to clear, not "achieve 0.91 NDCG," because no such ground-truth NDCG exists to achieve.

This document is deliberately scoped to what is runnable by one or two people inside a hackathon timeline (realistically 3-5 hours of evaluation work, spread across build phases, not a dedicated research sprint). Every check below produces either a number, a table, or a side-by-side comparison that goes directly into the methodology presentation — nothing here is evaluation for its own sake.

**The four pillars, mapped to judging criteria:**

| Pillar                                   | Judging Criterion                         | What it answers                                           |
| ---------------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| §1–2: Golden test sets + ranking metrics | Criterion 1 (Ranking Quality)             | "Are the top 20 defensible?"                              |
| §3: Ablations                            | Criterion 2 (Methodology Clarity)         | "Does each scoring module actually do something?"         |
| §4: Explainability checks                | Criterion 3 (Explainability)              | "Is the AI's reasoning accurate, not just fluent?"        |
| §5: Fairness checks                      | Criterion 1 (Ranking Quality, robustness) | "Is the system rewarding fit, or rewarding demographics?" |

---

## 1. Constructing an Evaluation Set Without Ground Truth

There are two complementary sources of "reference" judgment available to us, and we use both rather than picking one. They fail in different ways, so triangulating across them is more defensible than leaning on either alone.

### 1.1 Source A — Hand-Picked Golden JDs with Manually Identified Anchors

This is the primary evaluation set and the one judges will find most credible, because a human (us) made the call, not another model.

**Construction procedure:**

1. **Pick 5 JDs that stress-test different parts of the system**, not 5 similar ones. The PRD's own constraint — 10 near-uniformly distributed job titles, IT-services-heavy dataset — means the golden set must deliberately span the dataset's diversity, or we will only ever validate the "easy" case (an obvious technical JD against an IT-heavy dataset).

   | #   | JD Type                                                                                                               | Why it's in the set                                                                                                                                                                                                                                                 |
   | --- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | 1   | Senior technical role, skills-heavy (e.g., Senior Data Engineer — the worked example in doc 04)                       | Tests B1 semantic matching and the `requires_technical_github_signals` gate at full strength                                                                                                                                                                        |
   | 2   | Non-technical role (e.g., HR Manager or Accountant)                                                                   | Tests that GitHub/technical signals correctly _don't_ leak into scoring (Flag 8 mitigation) and that semantic matching still works on non-engineering free text                                                                                                     |
   | 3   | Mid-level role with an explicit "growth into senior" framing                                                          | Directly tests the B2 "stretch readiness" logic — the PRD's headline innovation                                                                                                                                                                                     |
   | 4   | A JD using heavy implicit/behavioral language ("own the roadmap," "drive alignment") with few explicit skill keywords | Tests whether the JD parser extracts soft skills and seniority correctly _without_ keyword crutches — this is the test that most directly answers "did we actually solve the Problem Statement, or just relocate the keyword-matching problem to a different field" |
   | 5   | A JD with an explicit hard constraint stated (salary ceiling or location)                                             | Tests `hard_filter.py` fires correctly and _only_ removes candidates who violate the stated constraint — never on `open_to_work`, notice period, or education tier                                                                                                  |

2. **For each JD, manually construct (not retrieve) 3 "obvious-fit" anchors and 3 "obvious-non-fit" anchors directly from the dataset, by reading raw profiles — not by running the pipeline first.** This ordering matters: if we pick anchors _after_ seeing pipeline output, we are unconsciously grading the pipeline against itself. The procedure:
   - Pick a JD.
   - Before running anything, `grep`/filter the raw JSONL for candidates whose `skills[]` and `career_history[].description` plainly and substantially match the JD (e.g., for the Data Engineer JD: candidates with Kafka + GCP/BigQuery + Airflow at advanced/expert proficiency and multiple years of duration). Read 2-3 full profiles to confirm, by eye, that these are genuinely strong matches a recruiter would be glad to see.
   - Separately pick 3 candidates from a clearly mismatched domain (e.g., for the Data Engineer JD: a Content Writer or Civil Engineer with no overlapping skills) as obvious-non-fit anchors. These should not be edge cases — they exist purely to catch catastrophic failures (a Civil Engineer ranking in the top 50 for a Data Engineer role is a five-alarm bug, not a tuning nuance).
   - This gives **5 JDs × 6 anchors = 30 anchor judgments**, achievable in an afternoon, each one defensible because it's traceable to a specific human reading a specific profile.

3. **Add 2-3 "hard" anchors per JD where possible** — these are the cases that actually exercise the system's claimed differentiation, and they matter more than the easy ones:
   - A **stretch-readiness candidate**: mid-level title, strong promotion history, applying to the senior JD. Manually confirm this candidate looks like a defensible stretch hire by reading their full career history.
   - A **keyword-stuffer**: a candidate whose `current_title` or skill names superficially match the JD but whose `skill_strength_score` inputs (duration, endorsements) are weak. Doc 04 §6.2 already gives the exact arithmetic for why this candidate should score lower on B2 — the eval set should contain a real instance of this pattern, not just the synthetic worked example.
   - A **passive high-signal candidate**: `open_to_work=False` but high `search_appearances_30d` / `offer_acceptance_rate`. Confirm this candidate's profile content is otherwise a strong domain match, so we can check the pipeline doesn't bury them.

   These hard anchors don't need a fixed "this candidate should rank #N" target — that's over-precise given no ground truth exists. They need a directional, defensible claim: _"this candidate should appear somewhere in the top 20-30, not be invisible past rank 200."_ That's a claim we can actually stand behind in front of a judge.

**What this source is good for:** Catastrophic-failure detection (Validation 6 in doc 02 — the obvious-non-fit anchors must never appear near the top) and validating the system's headline differentiators (stretch readiness, keyword-stuffer resistance, passive-candidate fairness) against real data rather than only the single worked example already in doc 04.

**What this source cannot do:** Tell us anything about ranking quality _between_ two strong-but-different candidates at, say, rank 8 vs. rank 14. Human-constructed anchors are reliable at the extremes (obviously good, obviously bad) and unreliable in the middle — we should not pretend otherwise by inventing a precise rank order for 20 similar mid-tier candidates we didn't carefully read.

### 1.2 Source B — LLM-as-Judge Reference Ranking (Secondary, Bias-Aware)

**Why use this at all, given doc 03 explicitly rejected LLM-as-judge as the _production_ architecture?** Doc 03's Option B critique is about using an LLM to **score 500 candidates live, per query, in production** — that's a latency/cost/stability argument about the _system_. Using an LLM offline, once, on a small shortlist, purely to generate an independent reference point for _evaluating_ the deterministic system, is a different and much cheaper use of the same tool, and it doesn't inherit those problems because it never runs at production scale or under the 60-second budget.

**Construction procedure:**

1. For each of the 5 golden JDs, take the deterministic pipeline's top-40 (not top-20 — overlap padding lets us check whether the LLM judge and the deterministic scorer agree on the boundary, not just the obvious top).
2. Feed the JD text and the 40 full candidate profiles (skills, career history, redrob_signals — the raw JSON, not our computed sub-scores, so the judge forms an independent opinion) to Groq (llama-3.3-70b-versatile) in a single large context call, asking it to rank all 40 and give a one-line rationale per candidate.
3. Use a **rubric-constrained prompt** (not "who's best?") that mirrors the PRD's own four explainability dimensions — skill match, seniority/trajectory fit, stability, platform reliability — so the LLM judge is evaluating on the same axes the deterministic system claims to optimize, not some other implicit criterion.
4. **Run it twice** with the candidate order shuffled between runs. If the LLM judge's top-10 changes substantially between the two runs, that is itself a finding to report: it's direct empirical confirmation of doc 03's stability critique of Option B, and it strengthens (not weakens) the case for the deterministic architecture — "we built a hybrid system specifically because we tested pure LLM judgment and found it order-sensitive; here's the data."

**What this source is good for:** A second, independent opinion that uses _holistic_ judgment rather than a fixed formula — useful as a sanity check for candidates our formula might score poorly for reasons that don't generalize well (e.g., an unusual but clearly strong career path that doesn't fit the promotion-detection heuristic).

**What this source is explicitly not:** Ground truth. It is one more fallible opinion, generated by a model with its own blind spots and prompt sensitivity. We use it as _one input to rank correlation_ (§2.3), not as an oracle the deterministic system is "graded against." Any place this document or the presentation describes LLM-judge agreement, it should say "agreement with an independent LLM assessment," never "accuracy" or "correctness."

### 1.3 Why Two Sources, Not One

|          | Source A (Human anchors)                                                         | Source B (LLM judge)                                                                |
| -------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Strength | Ground-truth-adjacent at the extremes; fully defensible provenance               | Captures holistic nuance across the full middle of the ranking, not just extremes   |
| Weakness | Sparse — only ~6-9 labeled points per JD; can't validate middle-of-pack ordering | Can hallucinate, has its own biases, is order/batch-sensitive (doc 03 §1, Option B) |
| Used for | Catastrophic-failure checks, headline-feature validation                         | Rank correlation, secondary explainability spot-check                               |

If both sources agree the system is doing something wrong, that is a strong signal worth fixing before submission. If they disagree, Source A wins by construction (it's the one with a traceable human judgment behind it) — but the disagreement itself is worth a sentence in the presentation, since "here's a case where our system disagreed with an LLM holistic judgment, and here's why we believe our system is right" is a more credible argument than silence on the topic.

---

## 2. Metrics

All metrics below are computed only against the two reference sources from §1 — there is no dataset-wide "true" metric to report, and the presentation should never imply otherwise.

### 2.1 Precision@K Against Human Anchors (Primary Metric)

**Definition:** Of the obvious-fit anchors identified in §1.1 for a given JD, what fraction appear in the system's top-20 output? Separately: do any obvious-non-fit anchors appear in the top-20 (or even top-100)?

```
precision_at_20_anchors = (obvious_fit_anchors found in top 20) / (total obvious_fit_anchors for that JD)
contamination_check = count(obvious_non_fit_anchors found in top 100)
```

**What "good" looks like:**

| Metric                                                | Good                           | Acceptable                                              | Concerning           |
| ----------------------------------------------------- | ------------------------------ | ------------------------------------------------------- | -------------------- |
| Obvious-fit anchors in top 20                         | 3/3                            | 2/3 (with a defensible reason for the miss — see below) | 1/3 or 0/3           |
| Obvious-non-fit anchors in top 100                    | 0/3                            | —                                                       | Any                  |
| Stretch-readiness anchor rank                         | Top 30                         | Top 60                                                  | Outside top 100      |
| Keyword-stuffer rank vs. genuine-equivalent candidate | Stuffer ranks measurably lower | Stuffer ranks marginally lower                          | Stuffer ranks higher |

A miss on an obvious-fit anchor is not automatically a failure — it's a debugging trigger. The correct response is to pull that candidate's full B1-B5 breakdown (every sub-score is persisted per doc 03 §6, so this is a single lookup) and identify _which_ sub-score under-performed and why. If the diagnosis is "the candidate's career description is unusually short" (thin-profile cap, doc 04 §6.1) or "they have the right skills, but with low endorsements and short duration" (B1's deliberate behavior per Flag 3), that's the system working as designed, and the presentation should say so explicitly rather than treating every miss as a defect. If the diagnosis is "the skill clearly exists in their profile and the formula still scored it near zero," that's a real bug, and this is exactly the kind of failure this evaluation step exists to catch _before_ the judges see it.

### 2.2 NDCG / Relevance-Graded Ranking Quality

**Why a relevance-graded metric, not just precision:** Precision@20 treats every anchor as equally important and only checks set membership, not position. NDCG rewards the system more for surfacing a strong-fit candidate at rank 2 than at rank 19, which is the behavior we actually care about — a recruiter reads top-to-bottom and loses patience.

**Construction:** Assign each anchor a relevance grade rather than a binary label:

| Grade | Meaning                                                                                       |
| ----- | --------------------------------------------------------------------------------------------- |
| 3     | Obvious-fit anchor                                                                            |
| 2     | Hard anchor (stretch-readiness, passive-high-signal) — should rank well but isn't a slam-dunk |
| 1     | Present in the shortlist but not specifically curated                                         |
| 0     | Obvious-non-fit anchor                                                                        |

```python
def ndcg_at_k(ranked_candidate_ids: list[str], relevance_map: dict[str, int], k: int = 20) -> float:
    def dcg(ids):
        return sum(
            relevance_map.get(cid, 0) / np.log2(i + 2)
            for i, cid in enumerate(ids[:k])
        )
    actual_dcg = dcg(ranked_candidate_ids)
    ideal_order = sorted(relevance_map, key=relevance_map.get, reverse=True)
    ideal_dcg = dcg(ideal_order)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

This only scores the handful of anchored candidates per JD (graded relevance is only defined for candidates we manually labeled) — it is not a metric over the full 100K pool, and the report should say so plainly: _"NDCG@20 computed over the manually-graded anchor set, not the full candidate pool, since no full-pool ground truth exists."_

**What "good" looks like:** NDCG@20 ≥ 0.85 across the anchor sets is a reasonable target — high enough to show obvious-fit anchors are front-loaded, not so close to 1.0 that it implies precision we can't actually claim given the anchor set's small size. Report the number per JD, not just an average — a single weak JD (e.g., the non-technical one, where semantic matching is hardest to sanity-check by eye) hidden inside a good average is exactly the kind of thing a sharp judge will ask about if we don't surface it first.

### 2.3 Rank Correlation Against the LLM-Judge Reference

**Definition:** Spearman's rank correlation (ρ) between the deterministic system's ranking and the LLM judge's ranking, computed over the candidates both methods ranked (the top-40 union from §1.2).

```python
from scipy.stats import spearmanr

def rank_correlation(deterministic_ranking: list[str], llm_judge_ranking: list[str]) -> float:
    common_ids = set(deterministic_ranking) & set(llm_judge_ranking)
    det_ranks = {cid: i for i, cid in enumerate(deterministic_ranking) if cid in common_ids}
    llm_ranks = {cid: i for i, cid in enumerate(llm_judge_ranking) if cid in common_ids}
    ids = list(common_ids)
    rho, p_value = spearmanr([det_ranks[i] for i in ids], [llm_ranks[i] for i in ids])
    return rho
```

**What "good" looks like:** ρ in the 0.5-0.75 range is the honest target, not 0.9+. Two independently-reasoning systems — one deterministic-formula-based, one holistic-LLM-based — agreeing on the rough shape of who belongs near the top while disagreeing on fine-grained order is the _expected and healthy_ outcome, and is consistent with doc 03's own argument that LLM judgment is less stable than formula-based scoring. A correlation near 1.0 would be slightly suspicious (it would suggest the LLM judge is just reading off the same surface signals rather than forming an independent view); a correlation near 0 or negative would indicate the two systems disagree fundamentally and needs investigation into why (check whether disagreement clusters around a specific sub-score, e.g. candidates the deterministic system ranks high mainly on platform signals that the LLM judge — which wasn't shown those signals as a "score," just raw data — may be weighting completely differently).

Run this per-JD, not pooled, and report the per-JD spread — a single pooled correlation can hide one badly-disagreeing JD (most likely candidate: the non-technical JD, where the LLM judge has less structured signal to anchor on).

### 2.4 Summary Metrics Table (what goes in the presentation)

| Metric                              | Computed against  | Reported as                                                             |
| ----------------------------------- | ----------------- | ----------------------------------------------------------------------- |
| Precision@20 (obvious-fit)          | Source A          | Fraction per JD + aggregate, e.g. "14/15 anchors recovered"             |
| Contamination@100 (obvious-non-fit) | Source A          | Count per JD, target = 0                                                |
| NDCG@20                             | Source A (graded) | Per-JD score + average                                                  |
| Spearman ρ                          | Source B          | Per-JD score + average, framed as "directional agreement," not accuracy |
| Stretch-readiness rank              | Source A          | Specific rank number per JD #3, with sub-score breakdown                |

---

## 3. Ablation Tests — Proving Each Component Earns Its Weight

This section answers Judging Criterion 2's deeper question: not just "is the code modular" (which doc 05's folder structure already demonstrates structurally) but "does the modularity _do_ anything, or could we delete B3 tomorrow and nobody would notice?" Ablations are the only way to answer that honestly, and they're cheap to run because the B1-B5 functions are already pure, independent functions per doc 04/05 — no retraining, no reindexing, just re-running `composite_score()` with different inputs zeroed out.

### 3.1 Method — Leave-One-Out Re-Weighting

For each of the 5 golden JDs, compute the full ranking 6 times:

1. **Baseline** — full composite score, all weights as specified (B1=0.35, B2=0.25, B3=0.15, B4=0.20, B5=0.05).
2. **No B1 (semantic)** — redistribute B1's 0.35 proportionally across B2-B5, i.e. scale the remaining weights up so they still sum to 1.0.
3. **No B2 (trajectory)** — same redistribution pattern.
4. **No B3 (stability)**
5. **No B4 (platform)**
6. **No B5 (certs)** — trivial; the cert bonus is additive and capped at 0.10, so this ablation mostly tests whether removing it changes anything material (it shouldn't, by design — if it does shift rankings a lot, that's itself a finding worth flagging).

This is a few lines of code given the existing `composite_score()` function in doc 04/05 — it's a config-level change, not a code change, which is itself a small demonstration of the modularity claim.

### 3.2 What to Measure Per Ablation

For each ablated ranking vs. the baseline ranking, compute:

- **Top-20 overlap (Jaccard):** `|baseline_top20 ∩ ablated_top20| / |baseline_top20 ∪ ablated_top20|`. A low overlap means that component is doing meaningful, non-redundant work — it's pulling distinct candidates into (or out of) the top 20 that the other four components alone wouldn't surface.
- **Anchor sensitivity:** Does removing this component cause one of the §1.1 obvious-fit anchors to drop out of the top 20, or an obvious-non-fit anchor to enter it? This is the most presentation-friendly result, because it converts an abstract ablation into a concrete, narratable story: _"Without B2 (trajectory), our keyword-stuffer test case — Senior title, 2 years experience, no promotions — climbs from rank 47 to rank 12. This demonstrates the trajectory scorer is the specific mechanism preventing the exact failure mode described in the Problem Statement."_
- **Rank correlation to baseline:** Spearman ρ between the full ranking and the ablated ranking. Lower ρ = bigger contribution from that component (it's reordering things, not just nudging scores within noise).

### 3.3 Expected Pattern and What It Would Mean

| Ablation | Expected effect                                                           | What a _surprising_ result would mean                                                                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No B1    | Largest top-20 overlap drop; domain-mismatched candidates enter top 20    | If overlap barely changes, semantic scoring isn't actually differentiating — investigate whether `embed_sim` is too compressed (a known risk if the embedding model under-performs, per PRD Risk 1)                                                             |
| No B2    | Stretch-readiness anchor and keyword-stuffer test case move the most      | If trajectory removal barely moves the keyword-stuffer case, the seniority-keyword-vs-substance distinction (doc 04 §6.2) isn't working as documented — worth re-checking the `promotion_rate` detection logic against that specific candidate's career_history |
| No B3    | Smallest overlap change (by design — B3 has the lowest weight, 0.15)      | If removing B3 changes the top 20 _more_ than removing B2, the weights as configured don't match the stated rationale (doc 04 §3.3 says stability should "influence... but cannot override" trajectory) and the weighting needs revisiting before submission    |
| No B4    | Moderate overlap change; passive-high-signal anchor likely drops somewhat | If a passive candidate's rank is _unaffected_ by removing B4 entirely, B4 isn't actually rewarding platform signals the way doc 04 §6.3's worked arithmetic claims — check whether B4 inputs are wired correctly                                                |
| No B5    | Minimal change (capped, additive, 0.05 weight by design)                  | If removing the 0.05-weighted, 0.10-capped cert bonus visibly reshuffles the top 20, something is wrong with the cap enforcement in `cert_bonus()`                                                                                                              |

The "surprising result" column is the actual point of this table: an ablation isn't just a number to report when it confirms what the formula weights already imply by construction — its real value is as a bug-detection tool. If an ablation's effect size doesn't roughly track its configured weight, that's evidence of a wiring bug between the documented formula and the implemented one, and it's far better to find that during evaluation than to have a judge notice the methodology presentation's claims don't match the ranking behavior.

### 3.4 Presenting Ablations Compactly

A single results table across all 5 golden JDs and 5 ablations (25 rows of Jaccard overlap + anchor sensitivity) is enough — this does not need 25 separate analyses, just one consistently-computed table plus 2-3 called-out anchor-sensitivity stories like the keyword-stuffer example above, which are far more persuasive in a live presentation than a table of overlap coefficients alone.

---

## 4. Explainability Quality Checks

Judging Criterion 3 is explicitly about whether the AI explains its decisions well — and doc 04 §5.1 already draws the critical distinction this evaluation must verify empirically: narration of a computed decision (correct) vs. independent re-judgment dressed up as explanation (wrong, and exactly what an evaluator should be hunting for). Fluency is not the bar. Grounding and usefulness are.

### 4.1 Grounding Check (Automated, Already Partially Specified)

Doc 05 §3.21 already specifies `validate_grounding()` as a pipeline component — this check exists in production, not just in evaluation. The evaluation task is to **stress-test it**, not just trust that it exists:

1. Run the grounding validator (doc 04 §5.4's literal-substring check: does the explanation text contain at least one of the candidate's actual skill names, company names, titles, cert names, or numeric platform values) across all top-20 explanations for all 5 golden JDs = 100 explanations.
2. Report the pass rate. Per doc 02 Validation 7, the bar is 10/10 (100%) on the original small test — at 100 explanations, expect and report the actual rate; if it's not 100/100, that's not necessarily disqualifying, but it must be visible, with the failing cases routed to the fallback template per doc 05 §6.2's error-handling design (this is the system "never silently dropping a slot" — the eval should confirm that promise holds, not just that the happy path works).
3. **Go one level deeper than the literal-substring check**: manually read 15-20 explanations (a sample across JDs and ranks, not just the top 3, since top-3 explanations get the most scrutiny during development and are least likely to reveal problems) and check for **plausible-sounding but unsupported claims** that a substring match wouldn't catch — e.g., an explanation that says "shows strong leadership potential" when nothing in the prompt's evidence block (doc 04 §5.2) actually supports a leadership inference. The substring check catches hallucinated _facts_; it does not catch hallucinated _inferences_ dressed in confident language, which is the more insidious failure mode for an LLM narrating someone else's computed scores. This step has no automatable pass criterion — it's a structured manual read, and "zero unsupported inferential claims found in the 15-20 sample" is the target.

### 4.2 Accuracy Check — Does the Explanation Match the Numbers?

This is a check the existing pipeline doesn't automate, and it should be added as an evaluation step: **pick 10 explanations and manually cross-reference every specific claim in the prose against the actual sub-score breakdown and source profile fields.**

Concretely, for each sampled explanation, check:

- Does the "skill alignment" paragraph's claimed proficiency/duration figures match the candidate's actual `skill_records` entries, not just mention skills that exist somewhere in the profile?
- Does the "seniority assessment" correctly reflect `latest_seniority` vs. `jd_seniority`, including correctly identifying whether this is an exact match or a stretch case (these read very differently and the explanation should say which one it is)?
- Does the "trajectory signal" paragraph's promotion narrative match what `promotion_rate` and the actual `career_history` chronology show — not just a plausible-sounding promotion story?
- Do the "flags" correctly surface what the system's own flag logic (doc 04 §5.2 — job hopping, notice period, passive status) actually triggered for this candidate, with no flags invented and no real flags omitted?

**What "good" looks like:** Zero factual contradictions between prose and underlying data across the 10-explanation sample. A factual contradiction (e.g., explanation claims "5 years of Kafka experience" when the skill record says 3.5) is a more serious finding than a missing nuance, and should be reported as a specific count, not glossed over — "9/10 explanations fully consistent with underlying scores; 1 explanation slightly overstated tenure (3.5 years reported as 'nearly 4 years' — borderline rounding, not a contradiction)" is exactly the right level of honesty for the presentation.

### 4.3 Usefulness Check — Would a Recruiter Actually Trust This?

Grounding and accuracy are necessary but not sufficient — an explanation can be perfectly factual and still be useless if it doesn't help Priya (the PRD's recruiter persona) defend the candidate to a hiring manager. This check is the PRD's own success metric for Goal 3 turned into a test: _"A recruiter can forward the explanation to a hiring manager verbatim without editing."_

**Procedure:** Take 5-10 explanations and run them through this checklist:

| Check                   | Pass condition                                                                                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Specificity             | Contains specific numbers/names (years, skill names, companies) — not generic phrases like "relevant experience" or "good fit" that could apply to any candidate        |
| Non-redundant structure | The six sections (match summary, skill alignment, seniority, trajectory, platform, flags) say genuinely different things — not the same sentence restated five ways     |
| Decision-relevant flags | Flags are things a recruiter would actually want to probe in a screening call (notice period, thin tenure, unverified self-reported skill) — not noise                  |
| Forwardable as-is       | Reading it cold, would a hiring manager understand why this candidate is ranked here without needing the raw JSON open beside it?                                       |
| Honest about weaknesses | Does the explanation surface genuine gaps (a missing nice-to-have skill, a borderline stretch case) rather than reading like uniform praise for every top-20 candidate? |

This is inherently a qualitative judgment call best made by 2 people independently rating the same 5-10 explanations and comparing notes — cheap, fast, and exactly the kind of check that's more convincing in a demo ("we had two people independently review explanations against this rubric") than a number with no human behind it.

### 4.4 Counterfactual Spot-Check — "Why Not Candidate X?"

One additional check worth including because it directly demonstrates white-box behavior (doc 03 §6's central claim): pick one strong-but-not-top-20 candidate from the shortlist (rank ~25-40) and generate the same explanation structure for them. Confirm the explanation correctly and specifically identifies _why_ they fell short of the top 20 (a real, named sub-score gap) rather than producing a vague "good candidate but not the best" non-answer. This is cheap (one extra explanation call) and directly answers a question judges are likely to ask live: "show me a near-miss and explain why."

---

## 5. Fairness & Bias Checks

This is not a full fairness audit — PRD §6 explicitly places "bias auditing or fairness certification" out of scope for this hackathon. What follows is narrower and specific to this dataset: a check that the system is rewarding _job fit_, not _proxies that correlate with job fit for reasons we'd be uncomfortable defending._ This matters for credibility even where it isn't formally required, and several of these checks are quick correlational scans, not a research-grade audit.

### 5.1 Institution Tier

**Risk:** Doc 02 Flag 5 already documents that tier_1 institutions are only 4.9% of the dataset and that institution tier is designed to be a marginal tiebreaker (capped at +0.05 in B3), never a primary filter. The check is to confirm the implementation actually respects that design intent.

**Test:** Across the top-20 results for all 5 golden JDs (100 candidates total, with overlaps), compute the institution-tier distribution and compare it to the dataset's overall tier distribution (tier_1: 4.9%, tier_2: 19.9%, tier_3: 38.1%, tier_4: 37.1%, per doc 02 §1).

**What "good" looks like:** The top-20 tier distribution should be only mildly shifted toward tier_1/2 relative to the baseline — some shift is expected and fine, since tier correlates loosely with other genuine signals (e.g., it might mildly correlate with `domain_experience` if certain fields of study cluster by tier), but a _dramatic_ shift (e.g., tier_1 candidates going from 4.9% baseline to 40%+ of top-20 slots across multiple unrelated JDs) would indicate institution tier is functioning as a much larger factor than its 0.05-weighted, capped design — worth checking the `edu_bonus` implementation against doc 04 §3.3's formula if so.

### 5.2 Location / City

**Risk:** Doc 02 Flag 9 notes geographic distribution is artificially uniform across ~10 Indian cities, and the System Architecture doc (§3.3) explicitly says there should be no continuous geographic scoring — only binary hard filters, and only when the JD states one.

**Test:** For JDs with no stated location requirement, check the top-20 location distribution against the dataset baseline (~4.2% per city). Any city should not be dramatically over- or under-represented in the top 20 across multiple JDs with no location constraint — since location isn't supposed to be a scoring input at all in the no-hard-constraint case, the top-20 distribution should look close to random draws from the shortlist, not systematically tilted toward any specific city.

**What this would catch:** A bug where location accidentally leaked into the embedding (doc 04 §2.2 explicitly excludes location from `build_candidate_doc()` — this check verifies that exclusion actually holds in the shipped implementation) or into scoring through some unintended path.

### 5.3 Industry / Job-Title Domain

**Risk:** Doc 02 Flag 1 — near-uniform title distribution across very different domains (Civil Engineer through Content Writer) is, per the same doc, a deliberate test of generalization. The fairness question here is narrower than tier or location: _does the system perform comparably well (not identically — the JDs are different) at surfacing strong, specific matches regardless of which of the 10 domains the JD falls into, or does it secretly only work well for IT/software JDs because that's the dominant industry (52% combined IT Services + Software per the Problem Statement) and most "natural" for an engineer building this to test against?_

**Test:** This is effectively a re-framing of the golden-JD-diversity requirement from §1.1 — JD #2 (the non-technical role) is the primary instrument here. Compute the same precision@20 and NDCG metrics from §2 separately for the technical JDs vs. the non-technical JD, and report them side by side rather than only as a pooled average. A material quality gap between domains (e.g., NDCG@20 of 0.9 on the Data Engineer JD vs. 0.5 on the HR Manager JD) is the single most important fairness-adjacent finding this whole document can surface, because it would mean the system's quality claims don't transfer to 9 of the dataset's 10 job-title domains — directly relevant to whether the top 20 for a non-IT JD are "genuinely the best matches," which is Judging Criterion 1 itself, not a separate fairness add-on.

### 5.4 Gender / Name-Based Proxies

**Note for completeness, scoped appropriately:** The dataset's data dictionary (doc 02 §1) does not list a `name` or `gender` field among the structured profile fields actually used by any scoring function — `candidate_id` is explicitly called out as never used as a feature, and no other identity-correlated field appears in the B1-B5 inputs. If candidate names exist anywhere in the raw JSONL outside the documented schema, confirm they are excluded from `build_candidate_doc()` (doc 04 §2.2's explicit exclusion list) and never embedded — this is a one-time code-inspection check, not an ongoing statistical test, since there's no field in the documented schema for a demographic correlation to run against in the first place. State this plainly in the presentation rather than presenting an elaborate disparity analysis the dataset doesn't actually support — claiming a rigorous fairness audit on a field that isn't in the schema would be less credible than briefly noting it was checked and is a non-issue by construction.

### 5.5 Reporting Fairness Checks Honestly

None of §5.1-5.4 should be presented as "we ran a bias audit and the system is unbiased" — that overclaims relative to both the scope (PRD explicitly excludes formal bias auditing) and the depth of these checks (correlational scans on a synthetic dataset, not a causal fairness analysis). The honest framing for the presentation is: _"We checked for the specific distortions this synthetic dataset's known quirks (Data Understanding Report, Flags 1/5/9) could plausibly introduce, found [result], and designed the scoring formulas (institution tier capped at 0.05, no geographic scoring, no company-name signal) specifically to prevent them by construction — these checks confirm the design intent holds in the implementation."_ That is both accurate and still a meaningfully positive thing to be able to say.

---

## 6. Presenting This in the Methodology Presentation

The PDF deliverable (PRD §5, deliverable 2) has limited space and a panel that will skim before they read deeply. The goal is to make the evaluation section feel like rigor without making it feel like a wall of numbers nobody will read live. Concretely:

### 6.1 Structure (one slide/section per pillar, ~4-5 total)

1. **"How do you know it's good without labeled data?"** — One slide stating the two-source approach (§1) plainly. This question is coming regardless of whether we pre-empt it; addressing it head-on in the first evaluation slide, before a judge has to ask, signals methodological self-awareness rather than evasion.
2. **Ranking quality** — The summary metrics table (§2.4), but lead with the narrative, not the table: "Across 5 deliberately diverse test JDs spanning technical and non-technical roles, we recovered 14 of 15 manually-identified obvious-fit candidates in the top 20, with zero obvious-mismatches contaminating the list." Put the full per-JD table as a backup/appendix slide, not the headline.
3. **Ablations as a story, not a table** — Lead with the single most narratable result: the keyword-stuffer case moving from rank 47 to rank 12 when trajectory scoring is removed (§3.3). This one example does more work in 30 seconds of presentation time than the full 25-cell ablation table, because it's a direct, concrete callback to the exact failure mode named in the Problem Statement's opening paragraph — closing that loop explicitly is a strong narrative move. Follow with the compact table for the judges who want to verify it's not cherry-picked.
4. **Explainability** — Show one real explanation output side-by-side with the sub-scores that produced it (this is literally what doc 04 §7.5's worked example already is — reuse it), then state the grounding pass rate and the accuracy-check finding from §4.2 in one sentence each. This is the section where showing > telling matters most: a judge reading one good explanation is more convinced than being told "our explanations are grounded and accurate."
5. **Fairness/robustness, briefly** — One slide, scoped honestly per §5.5's framing. Don't oversell; a short, precise "here's what we checked and why, given the dataset's known synthetic quirks" reads as more credible than a longer section straining to sound comprehensive.

### 6.2 The Single Most Important Presentation Principle

**Every evaluation claim should point back to a specific, inspectable artifact** — a `jd_intent.json`, a row in `ranked_output.csv` with its full sub-score breakdown, a specific explanation paragraph, a specific ablation diff. This isn't presentation polish; it's the direct continuation of doc 03 §6's "white-box" argument into the evaluation itself. A panel evaluating "clarity of methodology" should come away believing that every number in the evaluation section was pulled from a file they could open themselves, not generated specifically for the slide — because in this design, that's true, and saying so plainly is more persuasive than any chart.

### 6.3 What Not to Do

- Don't report a single aggregate "system accuracy: 94%" figure. There's no ground truth to compute accuracy against, and a precise-looking number with no defensible basis is a worse look than an honestly-scoped precision@20 with a stated denominator of 15 anchors.
- Don't hide the ablation or fairness section if a result is mediocre (e.g., the non-technical JD scoring lower than the technical ones). A judge who finds a gap we didn't mention is a bigger credibility cost than a judge who sees us name the gap ourselves and explain the likely cause.
- Don't present the LLM-judge correlation as validation that "the AI agrees with itself" — be explicit that it's a second, independently fallible opinion, exactly as scoped in §1.2-1.3.

---

## 7. Time Budget (Hackathon-Realistic)

| Task                                                               | Estimated time   | Can be parallelized?                                                               |
| ------------------------------------------------------------------ | ---------------- | ---------------------------------------------------------------------------------- |
| Pick 5 golden JDs + read profiles to hand-pick anchors (§1.1)      | 60-90 min        | Yes — split JDs across team members                                                |
| Run pipeline on golden JDs, compute precision@20 / NDCG (§2.1-2.2) | 20-30 min        | No — depends on pipeline being done                                                |
| LLM-as-judge reference ranking + rank correlation (§1.2, §2.3)     | 30-40 min        | Yes, runs alongside other checks                                                   |
| Ablation runs (5 JDs × 5 ablations) + diff analysis (§3)           | 30-45 min        | No — single script run, mostly compute time                                        |
| Explainability grounding + accuracy + usefulness checks (§4)       | 45-60 min        | Yes — split the manual-read sample across reviewers                                |
| Fairness scans (§5)                                                | 20-30 min        | Yes                                                                                |
| Assembling presentation slides from results (§6)                   | 30-45 min        | No — needs all prior results                                                       |
| **Total**                                                          | **~4-5.5 hours** | Roughly 2.5-3 hours of wall-clock time with 2 people splitting parallelizable work |

This assumes the core pipeline (indexing + query) is already working end-to-end — this document is the validation layer on top of a working system, not a prerequisite for building one.

---

_Document owner: ML Evaluation lead_
_Last updated: June 2026_
_Status: Proposed — ready for execution alongside build phase_
