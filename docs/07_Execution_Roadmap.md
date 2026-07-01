# Project Execution Plan / Roadmap

## Redrob Intelligent Candidate Discovery & Ranking Engine

**Team size:** 2 people
**Timeline assumption:** 7 working days, ~2–4 focused hrs/day per person (≈40–55 person-hours total)
**Deliverables:** (1) GitHub repo with working code, (2) Methodology Presentation PDF, (3) Ranked Output File
**If your actual window is longer than 7 days:** the phase order and dependencies below don't change — just stretch the "days" column proportionally and use the slack for the Full Version items in each phase rather than compressing toward MVP.

---

## 0. How to Read This Plan

Every phase ends with a **"Done looks like"** bar — a checkable state, not a vibe. Every phase also has an **MVP cut line** — the version of that phase you ship if the week goes sideways. The MVP path is not "do it badly" — it's "do less of it, correctly, and never skip the explainability or evaluation work entirely," because those are two of the three judging criteria and a thin-but-honest version of each beats a skipped one.

Two people means real parallelization is possible from Day 2 onward (data/scoring track vs. JD/explainability track), which is reflected in the day estimates below.

---

## 1. Phased Breakdown

### Phase 1 — Data Pipeline & JD Parser

**Maps to:** `data_loader/`, `jd_parser/`, `embedder/candidate_doc_builder.py`, indexing groundwork

**What happens:**

- Stand up repo skeleton per doc 05 §1 folder structure, `config/settings.yaml`, `Settings` dataclass.
- Build `data_loader/loader.py` + `validator.py` — stream JSONL, skip/log malformed records.
- Build `candidate_doc_builder.py` — the concatenation logic, including the thin-profile fallback (Flag 7 mitigation).
- Build `jd_parser/parser.py` + `prompt_templates.py` — single Groq (llama-3.3-70b-versatile) call → validated `JDIntent`, with the retry-on-validation-failure path.
- Manually inspect 20–30 random career descriptions (PRD Risk 4) to confirm the candidate-doc strategy is the right one before committing to it.
- Smoke-test the JD parser against 2–3 throwaway JDs and eyeball the JSON.

**Done looks like:**

- `python -m data_loader` (or equivalent test script) loads the full 100K JSONL with a skip-rate logged and under 10%.
- `parse_job_description()` returns a schema-valid `JDIntent` for at least 5 different JD styles (technical, non-technical, implicit-seniority, hard-constraint, vague).
- Candidate doc builder output has been eyeballed on ≥20 real profiles; thin-profile flag fires on a sane (not 0%, not 50%) fraction of them.

**MVP cut line:** Skip the retry-on-validation-failure path (just raise and log); skip batch loading, do it in one pass since 100K JSONL fits in memory. Do **not** cut the manual description-quality inspection — it's 20 minutes and determines whether B1 will even work.

---

### Phase 2 — Scoring & Ranking Engine

**Maps to:** `embedder/embedder.py` + `index_builder.py`, `feature_extractor/*`, `retriever/*`, `scorer/b1–b5`, `composite.py`, `ranker/ranker.py`

**What happens:**

- Load embedding model, batch-encode all 100K candidate docs, build FAISS flat index, persist `.npy` + `.faiss` + `candidate_id_map.json`.
- Implement the four `feature_extractor` modules (skill, trajectory, stability, platform) as pure functions per doc 04 §2–3 formulas — these are JD-independent and run once at index time.
- Implement `retriever.py` (ANN top-500) and `hard_filter.py` (salary/location, gated correctly on `salary_stated` / `location_is_hard_requirement`).
- Implement `b1_semantic.py` through `b5_cert.py` and `composite.py` exactly per doc 04's formulas — these are the highest-scrutiny files since Judging Criterion 1 lives here.
- Implement `ranker.py` with the tiebreaker rule.
- Run **Validation 1 (embedding recall check)** from doc 02 §5 here, not later — if recall is bad, this is the phase to catch and fix it (fallback model or BM25 hybrid), before any scoring work is built on top of a broken retrieval layer.

**Done looks like:**

- Indexing run completes end-to-end on the full 100K set and produces all four persisted artifacts.
- A test JD run through retriever → scorer → ranker produces a top-20 list with all five sub-scores populated and a composite score that matches a hand-computed value for at least one candidate (replicate doc 04 §7's worked example arithmetic against your own implementation as a unit check).
- Validation 1 passes (≥90% of manually-identified strong matches land in top-500) or the documented fallback (stronger model / BM25) has been applied and re-tested.

**Done = the system can take a JD and produce a numerically-ranked top 20, with zero LLM involvement in this phase.**

**MVP cut line:** Use `all-MiniLM-L6-v2` directly (skip the `bge-small` vs. fallback comparison) if recall is "good enough" on the first try — don't burn time chasing marginal recall gains. Hard filters can be reduced to salary-only (drop location/relocation logic) if time is short; document the simplification rather than silently dropping it. Do **not** cut any of B1–B5 — five sub-scores are a named requirement of both the PRD and the architecture doc's modularity claim, and removing one undermines Judging Criterion 2 directly.

---

### Phase 3 — Explainability Layer

**Maps to:** `explainer/prompt_builder.py`, `explainer.py`, `grounding_validator.py`, `output_writer/writer.py`

**What happens:**

- Build the per-candidate evidence-extraction logic (matched skills, career summary, platform summary, flags) per doc 04 §5.2.
- Build the batched explanation prompt and the Groq (llama-3.3-70b-versatile) call (4–5 candidates per call).
- Build `validate_grounding()` — the literal-substring check.
- Build the fallback path: failed grounding → retry once → template fallback, so no top-20 slot is ever silently dropped.
- Build `output_writer.py` — writes both CSV and JSON in the exact column order from doc 05 §3.22.
- Run the full pipeline end-to-end on one real JD and read the 20 explanations in full — not just check that grounding passed, but check they're actually good (this is also a preview of Phase 4's explainability checks, done early and cheaply here).

**Done looks like:**

- A single `python main.py run` (or `query`, if index already built) produces `ranked_output.csv` and `ranked_output.json` with all 20 rows fully populated — scores, sub-scores, and a six-part explanation each.
- Grounding validator passes on all 20 for at least one test JD; if any fail, the fallback path has actually been exercised and produces a non-empty, non-generic explanation (not just "True" in a test — read the fallback text).
- Total query-pipeline latency logged and under the 60s target (or close enough that you know exactly which stage is slow and why).

**This phase is what turns "a ranked list with numbers" into the actual product** — it's the one PRD Goal (3) and judging criterion (3) the system has done nothing for yet at the end of Phase 2.

**MVP cut line:** Batch size can drop to 1 candidate per call (slower, simpler, easier to debug) if batching logic is eating time — latency budget has slack since explanation generation only touches 20 candidates regardless. The grounding-failure retry can be cut to "fail once → straight to fallback template" (skip the corrective re-prompt) — this is a reasonable simplification, not a corner cut, because the fallback already guarantees no dropped slot. Do **not** cut the fallback path itself, and do **not** ship templated explanations as the primary path — that's the one thing doc 03 explicitly says disqualifies an architecture against Goal 3.

---

### Phase 4 — Evaluation & Validation

**Maps to:** doc 06 §1–5 — golden JDs, anchors, precision/NDCG, ablations, explainability checks, fairness scans

**What happens:**

- Pick and construct the 5 golden JDs + manually-read anchors (doc 06 §1.1) — this is the single highest-leverage block of work in the whole evaluation phase, because it's the thing a judge will most directly test you on ("are the top 20 genuinely the best?").
- Run precision@20 / contamination@100 / NDCG against the anchors (§2.1–2.2).
- Run the LLM-as-judge secondary reference on at least 2–3 of the 5 JDs if time allows (§1.2–2.3) — this is explicitly the most cuttable evaluation item since it's framed as secondary throughout doc 06.
- Run the leave-one-out ablations (§3) — at minimum, get the one narratable story (keyword-stuffer rank shift when B2 is removed) since doc 06 §6.1 flags this as the single best presentation moment in the whole evaluation section.
- Run the explainability grounding stress-test + manual accuracy spot-check on 10–15 explanations (§4.1–4.2).
- Run the institution-tier and location distribution checks (§5.1–5.2) — these are fast correlational scans, not a research project.

**Done looks like:**

- A results table or two (anchors recovered per JD, ablation overlap/Jaccard per ablation) that could be dropped directly into slides.
- At least one concrete "before/after" story from the ablations, with rank numbers, ready to narrate live.
- A documented pass rate on grounding + a documented count of any factual contradictions found in the manual accuracy check (even "9/10 explanations consistent, 1 minor rounding issue" is a good, honest result to have ready).

**MVP cut line:** Run only 3 golden JDs (not 5) — one technical, one non-technical, one stretch-readiness — and document why those three were chosen to cover the dataset's diversity claim. Skip the LLM-as-judge entirely (§1.2/§2.3) — doc 06 itself frames it as secondary and bias-aware, not core. Do the ablation as a single "remove B2" run with the keyword-stuffer story rather than the full 5×5 grid. Do **not** skip the human-anchor precision check entirely, even in miniature — it is the most direct, most judge-legible answer to Judging Criterion 1, and skipping it means you have nothing to say when asked "how do you know the top 20 are good?"

---

### Phase 5 — Presentation & Packaging

**Maps to:** methodology PDF, README, final repo cleanup, final output file generation

**What happens:**

- Write the methodology presentation following doc 06 §6.1's five-section structure (no-ground-truth framing → ranking quality → ablations-as-story → explainability side-by-side → fairness briefly).
- Write/finalize the repo `README.md`: setup instructions, CLI usage (doc 05 §5), architecture summary, and a pointer to the methodology PDF.
- Generate the final `ranked_output.csv`/`.json` against whatever JD the challenge specifies (or your best golden JD if none is specified — see Open Question 1 in the PRD, resolve this **before** this phase starts, ideally on Day 1).
- Final repo pass: remove dead code, confirm `.gitignore` excludes `data/raw/candidates.jsonl` and large artifacts, confirm `requirements.txt` is accurate, confirm logging isn't spamming DEBUG-level noise in the demo run.
- Dry-run the full submission as if you were a judge: clone fresh, follow the README, run `index` then `query`, confirm it works with no undocumented manual steps.

**Done looks like:**

- All three deliverables exist as final files in their submission-ready form.
- A clean clone of the repo, following only the README, successfully reproduces the ranked output file.
- The PDF is under whatever page/slide limit the brief specifies (check the challenge brief for this — not specified in the docs provided here) and every claim in it points to a specific file or row, per doc 06 §6.2's "inspectable artifact" principle.

**MVP cut line:** The PDF can be a clean, well-organized slide-style document built directly from the docs you already have (this conversation's outputs) rather than a separately-designed deck — content over production polish. The README can be terse (setup + run commands + one paragraph architecture summary) rather than exhaustive. Do **not** skip the clean-clone dry run — a repo that doesn't actually run for a judge following your own README is a Criterion 2 failure regardless of how good the code looks in the editor.

---

## 2. Time Estimates (7-day window, 2 people, 2–4 hrs/day each)

| Phase                                   | Person-hours (target) | Calendar days | Can run in parallel with...                                                                                            |
| --------------------------------------- | --------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** — Data pipeline & JD parser | 8–10 hrs              | Day 1–2       | — (foundational; both people can split: one on data_loader, one on jd_parser)                                          |
| **Phase 2** — Scoring & ranking engine  | 12–14 hrs             | Day 2–4       | Phase 3's prompt-design work (doesn't need the scorer finished, just the data shapes)                                  |
| **Phase 3** — Explainability layer      | 8–10 hrs              | Day 3–4       | Phase 2 (once `ScoreBreakdown` + `CandidateFeatureRow` shapes are fixed, explainer can be built against mocked scores) |
| **Phase 4** — Evaluation & validation   | 6–8 hrs               | Day 5–6       | Phase 5's PDF _outline_ (writing slide structure while eval numbers are still being generated)                         |
| **Phase 5** — Presentation & packaging  | 6–8 hrs               | Day 6–7       | — (needs Phase 4's results as input; can't fully finish until eval numbers exist)                                      |
| **Total**                               | **~40–50 person-hrs** | **7 days**    |                                                                                                                        |

**Why Phase 2 is the biggest block:** it's where all five judging-criterion-1-relevant formulas live, it has the only "rebuild everything" risk (embedding recall, per PRD Risk 1), and it gates Phase 3 (explainer needs real `ScoreBreakdown` objects) and Phase 4 (nothing to evaluate without a working scorer).

**Suggested 2-person split:** One person owns the Phase 1→2 critical path (data → embeddings → scoring → ranking) end-to-end since it's the longest dependency chain. The other person starts on JD parser (Phase 1) in parallel, then moves to explainer scaffolding (Phase 3) against mocked score objects as soon as the `ScoreBreakdown`/`CandidateFeatureRow` schemas are agreed (this can happen Day 1, before either implementation is done — doc 05 §2 already gives you the exact dataclass shapes to agree on up front). This person then leads Phase 4/5 while the other finishes any Phase 2 rough edges.

**If your real window is longer than 7 days:** add the time to Phase 2 (chase better embedding recall, add BM25 hybrid if needed) and Phase 4 (run the full 5-JD × 5-ablation grid, add the LLM-as-judge secondary reference) — these are the two phases with the most legitimate "more time = better" levers. Phases 1, 3, and 5 have less to gain from extra time once their MVP bar is cleared.

---

## 3. Explicit Dependencies (What Blocks What)

```
Phase 1 (Data Loader)  ──┬──▶ Phase 2 (needs CandidateProfile objects to embed/extract features from)
                          │
Phase 1 (JD Parser)    ──┴──▶ Phase 2 (retriever needs JDIntent to build query vector)
                          │
                          └──▶ Phase 3 (explainer needs JDIntent for prompt context)

Phase 2 (Embedder + Index) ──▶ Phase 2 (Scorer — B1 needs candidate_vectors + jd_vector)
Phase 2 (Feature Extractor) ─▶ Phase 2 (Scorer — B2/B3/B4 need CandidateFeatureRow)
Phase 2 (Scorer)           ──▶ Phase 2 (Ranker — needs ScoreBreakdown objects)
Phase 2 (Ranker)           ──▶ Phase 3 (Explainer — needs the top-20 ScoreBreakdown list)

Phase 2 + Phase 3 (full query_pipeline working) ──▶ Phase 4 (nothing to evaluate without real output)
Phase 4 (golden JD results, ablation results)    ──▶ Phase 5 (PDF needs real numbers, not placeholders)

Phase 1 + 2 + 3 (working end-to-end pipeline)    ──▶ Phase 5 (final ranked_output file generation)
```

**The one dependency that's easy to miss:** Phase 3 (Explainer) only _needs_ the `ScoreBreakdown` and `CandidateFeatureRow` **schemas**, not the finished scorer — so it can start in parallel with Phase 2's implementation as long as both people agree on the dataclass shapes from doc 05 §2 on Day 1. This is the main lever for real parallelization with 2 people; without that early schema agreement, Phase 3 is fully blocked on Phase 2 finishing.

**The one dependency that's easy to violate by accident:** Phase 5's PDF should not be drafted with placeholder numbers "to save time" and back-filled later — doc 06 §6.2's core presentation principle is that every claim traces to a real artifact. Drafting the structure early (fine, encouraged) is different from writing fake numbers into it (don't).

---

## 4. Minimum Viable Version vs. Full Version

Both versions below satisfy all three judging criteria — the difference is depth and polish, not coverage. The MVP is a legitimate fallback, not a compromised submission.

| Component                       | MVP Version                                                       | Full Version                                                                   |
| ------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Embedding model**             | `all-MiniLM-L6-v2` only                                           | `bge-small-en-v1.5` with documented fallback comparison                        |
| **Retrieval recall check**      | One quick manual check on 1 JD                                    | Full 3-JD Validation 1 with pass/fail criterion documented                     |
| **Hard filters**                | Salary only                                                       | Salary + location/relocation                                                   |
| **Scoring (B1–B5)**             | All five, formulas as specified — **not cuttable**                | Same, plus weight-tuning pass against golden anchors before final run          |
| **Explanation batching**        | 1 candidate per call (simpler, slower)                            | 4–5 per call (per doc 04 §5.3)                                                 |
| **Grounding retry**             | Fail once → fallback template                                     | Fail → corrective retry → fallback template                                    |
| **Golden JDs for eval**         | 3 (technical, non-technical, stretch-readiness)                   | 5 (full diversity set per doc 06 §1.1)                                         |
| **LLM-as-judge reference**      | Skipped entirely                                                  | Run on 2–3 JDs, rank correlation reported                                      |
| **Ablations**                   | 1 (remove B2 — the keyword-stuffer story)                         | Full 5×5 grid with Jaccard overlap table                                       |
| **Explainability manual check** | 5 explanations read                                               | 15–20 explanations read, per doc 06 §4.1                                       |
| **Fairness checks**             | Institution tier only                                             | Tier + location + industry-domain comparison                                   |
| **Methodology PDF**             | Content-complete, built from existing docs, minimal design polish | Same content, designed slide layout, diagrams from doc 03 §4 rendered visually |
| **README**                      | Setup + run commands + 1-paragraph summary                        | Full architecture walkthrough + troubleshooting section                        |

**The non-negotiable floor, regardless of time pressure:**

1. All five B1–B5 scoring dimensions implemented and weighted per doc 04 — this is Criterion 1 and 2's foundation; a 3-dimension scorer is a different, weaker product, not a smaller version of this one.
2. Every top-20 candidate has a real, grounded explanation — even via fallback template, never silently missing.
3. At least one golden-JD precision check with real anchors you personally read — this is the minimum credible answer to "how do you know the top 20 are good?"
4. A clean-clone dry run of the repo actually works end to end.

If Phase 4 or 5 has to be cut short, cut breadth (fewer JDs, fewer ablations, simpler PDF design) — never cut to zero on any of the three judging criteria.

---

## 5. Pre-Submission Checklist

Mapped to the three deliverables and three judging criteria, so the final pass is a lookup, not a re-think.

### Deliverable 1 — GitHub Repository

- [ ] Repo structure matches (or sensibly approximates) doc 05 §1's module layout
- [ ] `requirements.txt` is accurate and installs cleanly in a fresh virtualenv
- [ ] `data/raw/candidates.jsonl` and other large generated artifacts are git-ignored, not committed
- [ ] `README.md` has setup steps, the exact `index`/`query`/`run` commands, and a short architecture summary
- [ ] A clean clone + fresh `pip install` + documented commands reproduces the ranked output with **no undocumented manual steps**
- [ ] No hardcoded API keys committed; `GROQ_API_KEY` is read from environment only
- [ ] Logging output is sane at default level (not flooding stdout with DEBUG noise during the demo run)
- [ ] Unit tests exist for at least the B1–B5 scoring functions and pass

### Deliverable 2 — Methodology Presentation (PDF)

- [ ] States the architecture decision (Option C / hybrid funnel) and _why_ the alternatives were rejected, briefly
- [ ] Includes the system diagram (text or rendered version of doc 03 §4)
- [ ] States the "no ground truth" framing up front (doc 06 §6.1, slide 1) — don't let a judge ask this first
- [ ] Reports evaluation results with real numbers and stated denominators (e.g., "14/15 anchors," not "94% accuracy")
- [ ] Includes at least one ablation story with before/after rank numbers
- [ ] Shows at least one full explanation example side-by-side with its sub-score breakdown
- [ ] Mentions fairness checks performed, scoped honestly (not oversold as a full audit)
- [ ] Every quantitative claim traces back to a specific file/row a judge could ask to see
- [ ] Under whatever length limit the challenge brief specifies

### Deliverable 3 — Ranked Output File

- [ ] Format matches the challenge brief's specified format (CSV/JSON column names/order) — **confirm this against the actual brief before final generation**, per PRD Open Question 3
- [ ] Contains exactly the required number of ranked candidates (top 20, or whatever the brief specifies)
- [ ] Every row has all sub-scores (`semantic_score`, `trajectory_score`, `stability_score`, `platform_score`, `cert_bonus`) and the composite
- [ ] Every row has a complete, non-empty explanation across all six structured fields
- [ ] Generated against the JD the challenge specifies, if one is provided (PRD Open Question 1) — otherwise generated against your strongest golden JD with that choice noted
- [ ] File was generated by an actual pipeline run you can reproduce, not hand-edited after the fact

### Cross-Cutting — Judging Criterion 1: Ranking Quality

- [ ] At least 3 (ideally 5) golden JDs tested with manually-read, manually-selected anchors
- [ ] Precision@20 and contamination@100 numbers exist and are documented
- [ ] No obvious-non-fit anchor appears anywhere near the top of any golden JD's results
- [ ] The stretch-readiness case (mid-level + promotions vs. senior JD) has been tested and produces a defensible rank
- [ ] The keyword-stuffer case has been tested and scores measurably lower than a genuine equivalent

### Cross-Cutting — Judging Criterion 2: Methodology Clarity

- [ ] Each of B1–B5 is an independently callable, independently testable pure function
- [ ] A new reader of the repo could plausibly modify `stability_score()` without touching any other module (this is the doc 03 §3 success metric — actually try it once as a self-check)
- [ ] The technical choices (embedding model, FAISS flat index, two-LLM-touchpoint architecture) are each justified in the PDF with a stated rationale, not just stated as facts
- [ ] Config (weights, thresholds) lives in `settings.yaml`, not hardcoded inside scoring functions

### Cross-Cutting — Judging Criterion 3: Explainability

- [ ] Grounding validator pass rate is measured and reported (not just assumed to be 100%)
- [ ] At least 10 explanations have been manually cross-checked against their underlying sub-scores for factual accuracy
- [ ] At least one explanation has been read by someone who didn't build the system, to sanity-check the "forwardable to a hiring manager as-is" bar
- [ ] The counterfactual spot-check ("why not candidate at rank ~30?") has been run at least once and produces a specific, named-sub-score answer — not a vague non-answer
- [ ] No explanation in the final output file is a raw template fallback for a candidate where the LLM path should have worked (spot-check `grounding_validated=False` rows specifically, if any exist, before submitting)

---

_Plan owner: Hackathon team_
_Built for: 2-person team, 7-day window, 2–4 focused hrs/day per person_
_Status: Ready to execute — re-confirm the actual deadline and challenge-specified output format (PRD Open Questions 1 & 3) on Day 1 before locking Phase 5's final output generation_
