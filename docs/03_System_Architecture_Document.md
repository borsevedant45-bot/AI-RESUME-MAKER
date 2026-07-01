# System Architecture Document

## Redrob Intelligent Candidate Discovery & Ranking Engine

**Document:** `docs/03_System_Architecture_Document.md`
**Version:** 1.0
**Inputs:** `01_PRD.md`, `02_Data_Understanding_Report.md`
**Status:** Proposed — architecture decision for build phase

---

## 0. Purpose

The PRD defines _what_ the system must do (deep JD understanding, contextual signal integration, white-box explainability) and the Data Understanding Report defines _what features can be computed_ from the dataset (B1–B5 sub-scores). This document closes the loop: it decides _how the system is physically built_ — which components exist, what each one is responsible for, where the LLM sits versus where deterministic code sits, and why.

The central engineering tension is this: the system must reason about JD intent the way a human recruiter would (which favors an LLM), but it must do so consistently across 100,000 candidates within a tight latency and cost budget (which an LLM is poorly suited for at that scale). The architecture below is built around resolving that tension rather than picking a side.

---

## 1. Architecture Options Considered

### Option A — Pure Embedding Similarity + Rule-Based Signal Scoring

The JD is embedded directly as raw text (no LLM parsing step). Candidate ranking is a cosine-similarity search against pre-computed candidate embeddings, combined with deterministic rule-based scores for trajectory, stability, and platform signals (computed via regex/keyword heuristics on titles, e.g. matching "Senior," "Lead," "Manager"). No LLM is in the pipeline at all, or it's used only as an optional, swappable component. Explanations are templated strings populated with the computed numbers.

**Why it's tempting:** It's the cheapest and fastest option, and it's fully deterministic — the same JD always produces the same ranking.

**Why it's risky:** The Problem Statement's entire premise is that _keyword-based extraction fails to read between the lines_. Regex-based seniority/intent extraction from the JD ("lead a team of engineers" → leadership expectation) reintroduces exactly the keyword-brittleness this system exists to replace — just one layer deeper, on the JD side instead of the resume side. Templated explanations also conflict directly with PRD Goal 3, which explicitly requires explanations to not be "templated string substitutions."

---

### Option B — LLM-as-Judge Re-Ranking Over a Shortlist

ANN retrieval narrows 100K candidates to a shortlist (e.g., top 200–500). Instead of deterministic scoring formulas, each shortlisted candidate's full profile is passed directly to an LLM alongside the JD, and the LLM itself produces a fit score, rank, and explanation in one step — replacing the B1–B5 formulas with LLM judgment.

**Why it's tempting:** It's the most "human-like" option. An LLM reading a full profile holistically can pick up on nuance that a fixed formula might miss, and the explanation is a natural byproduct of the same reasoning that produced the score, not a separate post-hoc step.

**Why it's risky:**

- **Cost and latency don't survive contact with 100K-candidate retrieval.** Even a conservatively-sized shortlist of 500 candidates means 500 individual judgments (or large batched prompts covering many candidates each, which trades latency for accuracy as context grows). Sequential calls blow well past the PRD's 60-second budget; parallelizing introduces rate-limit and orchestration complexity disproportionate to a hackathon timeline.
- **LLM scoring at this volume is not stable.** Position-in-batch and prompt-recency effects mean the same candidate can receive a different score depending on what else is in the same call or what order candidates appear in. That's a hard property to defend to a judging panel asking "are the top 20 genuinely the best of 100,000?" — the honest answer would be "best of however they happened to be batched."
- **It collapses explainability into a black box at the worst possible point.** The PRD's Goal 2 requires a new engineer to extend one scoring module without touching others. If scoring is "ask the LLM," there are no modules — there's one large, unauditable judgment call standing in for five named, independently-testable dimensions.

---

### Option C — Hybrid Retrieval-Then-Score-Then-Explain Pipeline (Recommended)

ANN retrieval narrows 100K → ~500 candidates for recall. A deterministic, fully-specified multi-signal scoring engine (the B1–B5 formulas already defined in the Data Understanding Report) computes a composite score for all 500, consistently and reproducibly. The LLM is used at exactly two narrow points: once to parse the JD into a structured intent object at the start of the pipeline, and once to generate grounded explanations for only the final top 20 — and in both cases, it is given structured data to work _from_, not asked to produce a judgment from nothing.

This is the architecture implied throughout the PRD and Data Understanding Report; this document formalizes it as the chosen design and states explicitly why it beats the alternatives.

---

### Tradeoff Table (at 100K-candidate scale, per JD query)

| Dimension                        | A — Pure Embedding + Rules                                                                              | B — LLM-as-Judge Re-Rank                                                                                                        | C — Hybrid Retrieve→Score→Explain                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Ranking accuracy**             | Medium. Strong on explicit skill overlap; blind to implicit JD intent (no LLM reasoning on the JD side) | Potentially high per-candidate nuance, but **unstable** — score depends on batching/ordering, not just content                  | High. LLM nuance is captured _once_ (JD parsing) and then applied _consistently_ across every candidate via deterministic formulas |
| **Reproducibility**              | Perfect — fully deterministic                                                                           | Poor — re-running the same JD can shift individual scores                                                                       | Near-perfect — only the JD parse step has any LLM variance; scoring itself is pure functions                                       |
| **Explainability**               | Low — templated text, explicitly disallowed by PRD Goal 3                                               | High in principle, but ungrounded — explanation and score come from the same unverifiable judgment, risk of hallucinated detail | High and **grounded** — explanation generator is fed the actual computed sub-scores and required to cite them                      |
| **Methodology clarity (Goal 2)** | High — simple, but the "intelligence" is mostly illusory (regex dressed as understanding)               | Low — "ask the LLM" is not a decomposable, independently testable module                                                        | High — five named, independently testable scoring functions (B1–B5), each documented with formula and rationale                    |
| **Cost per JD query**            | Lowest — near-zero LLM usage                                                                            | Highest — hundreds of LLM calls (or few very large ones) per query, repeated for every JD a recruiter pastes                    | Low — one JD-parse call plus a handful of batched explanation calls; bulk of work is vectorized math, not API calls                |
| **Latency @ 100K, 60s budget**   | Comfortably under budget                                                                                | Very likely over budget without heavy parallelization                                                                           | Comfortably under budget: embed JD (<1s) + ANN retrieve (<1s) + score 500 (<10s) + explain top 20 (<30s)                           |
| **Engineering complexity**       | Low                                                                                                     | High (batching strategy, rate limits, score-stability mitigation)                                                               | Moderate — clear separation of concerns, no exotic infrastructure                                                                  |

---

## 2. Recommended Architecture: The Funnel (Retrieve → Score → Explain)

**Decision: Option C.** The system is a progressively narrowing funnel — 100,000 candidates → ~500 via retrieval → 20 via deterministic scoring → 20 explained in natural language — where LLM involvement increases as the candidate pool shrinks. This is not a compromise between A and B; it is the architecture that puts each technology where it is strongest and nowhere it is weakest.

### Why this wins on Ranking Quality (Judge Criterion 1)

The two failure modes the PRD is explicitly worried about — keyword brittleness (Option A's weakness) and unstable, unauditable judgment (Option B's weakness) — are both avoided by construction. JD nuance is interpreted once, by the LLM, which is exactly the kind of single, reviewable judgment LLMs are reliable at. That interpretation then becomes a fixed structured target (`jd_intent.json`) and is applied identically to all 500 candidates through deterministic math, which is exactly the kind of repeated, scale-sensitive judgment formulas are reliable at and LLMs are not. The PRD's own "stretch readiness" and "semantic equivalence" requirements are handled in the embedding layer and trajectory formulas (Data Understanding Report §3), not invented per-candidate by an LLM.

### Why this wins on Clarity of Methodology (Judge Criterion 2)

Every box in the pipeline corresponds to a named, independently testable Python module: `jd_parser.py`, `embedder.py`, `retriever.py`, `scorer.py` (containing five pure functions, one per B1–B5), `ranker.py`, `explainer.py`. A new engineer can rewrite `stability_score()` without touching `trajectory_score()` or the retrieval layer — this is the PRD's literal success metric for Goal 2. None of this modularity survives in Option B, where scoring is implicit inside an LLM call.

### Why this wins on Explainability (Judge Criterion 3)

Because the explanation generator is handed the actual B1–B5 sub-scores and the specific fields that produced them (not the raw 100K-candidate pool, not a blank slate), it is structurally biased toward citing real data rather than inventing plausible-sounding generalities. This directly satisfies the PRD's Risk 3 mitigation: a post-processing check can verify each explanation references at least one concrete field (skill name, company, tenure figure) precisely because the generator was given those fields as required context.

### Why this is the only option that meets the practical constraint

100,000 candidates against an arbitrary JD, repeated on demand, on a standard laptop CPU, within 60 seconds, is a hard constraint that **rules out Option B outright** (LLM judgment doesn't scale to hundreds of candidates per query under that budget) and makes **Option A look attractive only because it under-delivers on the actual ask** (contextual, not keyword-based, understanding). Option C is the only one of the three that satisfies both the functional requirement and the runtime budget simultaneously.

---

## 3. Component-Level Breakdown

The system splits cleanly into an **offline indexing pipeline** (runs once, when the dataset loads) and an **online query pipeline** (runs once per JD a recruiter pastes). This split is what makes the 60-second budget achievable — the expensive part (embedding 100K candidates) is amortized to a one-time cost, not paid per query.

### 3.1 Candidate Representation & Embedding Layer _(offline)_

**Responsibility:** Convert each of the 100,000 unstructured candidate profiles into a single dense vector and a parallel set of structured numeric features, both keyed by `candidate_id`.

- Builds a "candidate document" per profile by concatenating skills (with proficiency), all role titles, all role descriptions, and certifications — per the Data Understanding Report's mitigation for Flag 7 (thin career-history text), embedding the concatenation rather than the description field alone.
- Encodes all 100K documents through a sentence-embedding model in batch, producing a `(100000, 384)` matrix persisted to disk.
- In parallel, computes every deterministic feature defined in the Data Understanding Report (`skill_strength_score`, `trajectory_score`, `stability_score`, `active_intent_score`, `hire_reliability_score`, `cert_bonus`, etc.) and persists them as a structured table joined on `candidate_id`.
- Runs once. Re-run only if the candidate dataset changes (out of scope per PRD §6, since the dataset is static).

### 3.2 JD Parser _(online, 1 LLM call per query)_

**Responsibility:** Convert free-text JD into the same structured vocabulary the candidate features use, so the two sides of the match are comparable.

- Single LLM call, structured-JSON output: seniority level (mapped to the same 0–1 scale as `trajectory_score`), must-have vs. nice-to-have skills, domain/problem space, implicit soft-skill requirements, work-mode/location/salary constraints if stated, and a `requires_technical_github_signals` flag (per Data Understanding Report Flag 8, so GitHub activity doesn't pollute scoring for an HR Manager JD).
- This is the system's _only_ point of open-ended interpretation. Everything downstream consumes its structured output, never the raw JD text again — which is what keeps the rest of the pipeline deterministic.

### 3.3 Retrieval / Filtering Stage _(online, <2s)_

**Responsibility:** Cut 100,000 candidates down to a scoring-feasible shortlist without discarding strong matches.

- Embeds the JD intent object (not the raw JD string — the structured object produces a cleaner vector than noisy free text) using the same model as the candidate index.
- Runs a nearest-neighbor search against the pre-built candidate index to retrieve the top ~500 by cosine similarity.
- Applies hard filters only where the JD explicitly states a non-negotiable constraint (e.g., a stated salary ceiling or a hard location requirement combined with `willing_to_relocate = false`). Per Data Understanding Report Flag 9, no continuous geographic scoring — binary compatibility only, and only when the JD makes it a hard requirement.
- This stage exists purely for compute economy. It is the only stage with recall risk (see §5, embedding model choice and the validation plan already defined in the Data Understanding Report §5).

### 3.4 Multi-Signal Scoring & Ranking Stage _(online, <10s)_

**Responsibility:** Score every retrieved candidate against the JD on all five PRD dimensions, deterministically.

- Five independent, pure functions — one per B1–B5 — each taking the candidate's structured feature row plus the JD intent object and returning a `[0,1]` sub-score (B5 capped at 0.10, additive).
- A fixed weighted-sum formula combines them into `composite_score` (weights documented and justified in the Data Understanding Report §3 — semantic fit weighted highest at 0.35 since it's the system's core differentiator over keyword ATS, platform signals second-highest among the "soft" dimensions since they're this dataset's unique advantage over a traditional ATS).
- Sorts by `composite_score`, applies the tiebreaker rule (per PRD Open Question 4 — recommend: higher `platform_score` wins ties, since it reflects a more reliably hireable candidate among otherwise-equal matches), and selects the top 20.
- Every sub-score is retained alongside the composite — nothing is discarded once computed. This is what makes the next stage possible without re-deriving anything.

### 3.5 Explanation Generator _(online, 1 batched LLM call or a few small ones)_

**Responsibility:** Turn the top 20 candidates' numeric sub-scores and source fields into the six-part justification structure the PRD specifies (match summary, skill alignment, seniority assessment, trajectory signal, platform summary, flags).

- Receives, per candidate: the five sub-scores, the specific fields that produced them (skill names + proficiency, role titles + companies + durations, the relevant platform numbers), and the JD intent object.
- Explicitly prompted to cite only what it was given — it is not shown the other 99,980 candidates and has no ability to invent comparative claims it can't support.
- Batches multiple candidates per call (e.g., 4–5 at a time) to reduce call count without overloading any single prompt with so many profiles that grounding degrades.
- A post-processing check (per PRD Risk 3) verifies each explanation references at least one literal field value before accepting it; candidates that fail are re-generated once or fall back to a structured-field summary so the pipeline never silently drops a top-20 candidate's explanation.

### 3.6 Output Writer

**Responsibility:** Produce the deliverable file.

- Writes the CSV/JSON specified in PRD §7: `[rank, candidate_id, composite_score, semantic_score, trajectory_score, stability_score, platform_score, cert_bonus, explanation]`.
- The full sub-score breakdown ships in the same row as the explanation — a recruiter or judge can spot-check the narrative against the numbers without re-running anything.

---

## 4. System Diagram (Text)

```
══════════════════════════════════════════════════════════════════════════
 OFFLINE — INDEXING PIPELINE  (runs once, at dataset load)
══════════════════════════════════════════════════════════════════════════

   ┌──────────────────────┐
   │ 100K Candidate         │
   │ Profiles (JSONL)       │
   └───────────┬────────────┘
               │
       ┌───────┴────────┐
       ▼                 ▼
┌─────────────────────┐  ┌──────────────────────────────┐
│ Candidate Document    │  │ Structured Feature Extractor  │
│ Builder                │  │ (B2 / B3 / B4 / B5 — pure     │
│ (skills + titles +     │  │  deterministic formulas)      │
│  descriptions + certs) │  └───────────────┬────────────────┘
└───────────┬───────────┘                  │
            ▼                               ▼
┌─────────────────────┐         ┌────────────────────────────┐
│ Embedding Model        │         │ candidate_features.parquet │
│ (bge-small-en-v1.5)    │         │ trajectory / stability /    │
└───────────┬───────────┘         │ platform / cert sub-scores  │
            ▼                      └──────────────┬───────────────┘
┌─────────────────────┐                            │
│ candidate_vectors.npy  │                          │
│ (100,000 × 384-dim)    │                          │
└───────────┬───────────┘                           │
            ▼                                       │
┌─────────────────────┐    candidate_id is the join │
│ FAISS Flat Index       │    key across both stores ◄┘
│ (in-memory, persisted  │
│  to disk after build)  │
└───────────────────────┘

══════════════════════════════════════════════════════════════════════════
 ONLINE — QUERY PIPELINE  (runs once per JD, target < 60s end-to-end)
══════════════════════════════════════════════════════════════════════════

   ┌────────────────┐
   │ Recruiter pastes │
   │ a Job Description │
   └────────┬──────────┘
            ▼
   ┌──────────────────────────────┐
   │ 1. JD Parser  (1 LLM call)      │   llama-3.3-70b-versatile
   │    → seniority, must-have       │   structured JSON output
   │      skills, domain, soft       │
   │      skills, salary, work mode  │
   └────────────┬───────────────────┘
                │  jd_intent.json
                ▼
   ┌──────────────────────────────┐
   │ 2. JD Embedder                  │   same model as the index
   └────────────┬───────────────────┘
                │  jd_vector (384-dim)
                ▼
   ┌──────────────────────────────┐     ┌─────────────────────────────┐
   │ 3. ANN Retrieval                │────▶│ Hard Filters (optional)        │
   │    FAISS top-500 by cosine sim  │     │ salary ceiling / location,     │
   │                                  │     │ applied only if JD states one  │
   └────────────┬───────────────────┘     └──────────────┬──────────────┘
                │  500 candidate_ids                       │
                ◄────────────────────────────────────────────┘
                ▼
   ┌─────────────────────────────────────────────────┐
   │ 4. Multi-Signal Scoring Engine  (deterministic)    │
   │    B1 semantic_score    (embed sim + skill cover)  │
   │    B2 trajectory_score  (seniority + promotion)     │
   │    B3 stability_score   (tenure + hopping flag)      │
   │    B4 platform_score    (intent + reliability)       │
   │    B5 cert_bonus        (capped additive)             │
   │    → composite_score = weighted sum                   │
   └──────────────────────┬──────────────────────────────┘
                           │  500 scored candidates
                           ▼
   ┌──────────────────────────────┐
   │ 5. Ranker                       │
   │    sort by composite_score       │
   │    apply tiebreaker → top 20      │
   └────────────┬───────────────────┘
                │  top 20 candidate_ids + full sub-score breakdown
                ▼
   ┌──────────────────────────────┐
   │ 6. Explanation Generator         │  llama-3.3-70b-versatile
   │    (batched LLM calls, grounded   │   fed sub-scores + source fields
   │     in computed sub-scores)        │   per candidate, not free reasoning
   └────────────┬───────────────────┘
                ▼
   ┌──────────────────────────────┐
   │ 7. Output Writer                  │
   │    ranked_output.csv / .json       │
   │    [rank, candidate_id, scores,     │
   │     sub-scores, explanation]         │
   └────────────┬───────────────────┘
                ▼
         Recruiter reviews top 20
```

---

## 5. Technology Choices

| Component                        | Choice                                                                           | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding model**              | `BAAI/bge-small-en-v1.5` (384-dim), with `all-MiniLM-L6-v2` as a faster fallback | CPU-friendly, runs the full 100K-candidate index in minutes on a laptop, no GPU dependency. `bge-small` generally edges out MiniLM on retrieval recall for this kind of asymmetric query-to-document matching (short JD phrase vs. long candidate document), which directly mitigates the PRD's Risk 1. The benchmark described in the Data Understanding Report §5 (embed 3 test JDs, check whether obvious matches land in the top-500) is the gating check — if recall comes in under the 90% pass criterion, the documented fallback is `bge-base-en-v1.5` or adding a BM25 sparse-retrieval pass as a hybrid recall safety net, not a re-architecture. |
| **Vector index**                 | FAISS, flat (exact) L2/cosine index                                              | At 100K × 384-dim (~150MB in memory), an exact flat index searches in well under a second — there is no recall-vs-speed tradeoff to make at this scale, so there's no reason to introduce the added complexity and approximation error of IVF/HNSW. This is also the more defensible choice in front of a judging panel: "exact nearest-neighbor search" needs no caveats about approximation parameters. If the dataset were to grow into the millions, swapping to IVF-PQ is a contained change inside the retrieval module only — nothing else in the pipeline needs to know.                                                                            |
| **Candidate feature store**      | Parquet file, joined on `candidate_id`                                           | Columnar, fast to load, trivially inspectable with pandas — important for the validation checks the Data Understanding Report already specifies (distribution plots for `skill_strength_score`, `trajectory_score`, etc.).                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **LLM — JD parsing**             | llama-3.3-70b-versatile, single call per query, structured JSON output           | One call, fixed cost regardless of dataset size. This is the system's single point of genuine open-ended language understanding, so it's worth spending the highest-quality call here rather than on a high-volume step.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **LLM — explanation generation** | llama-3.3-70b-versatile, batched calls (top 20 only, ~4–5 candidates per call)   | Bounded to a constant 20 candidates regardless of pool size — this is what keeps LLM cost and latency from scaling with the 100K candidate count. Batching trades a small amount of per-candidate context isolation for fewer round-trips; 4–5 per call keeps each prompt focused enough that grounding doesn't degrade.                                                                                                                                                                                                                                                                                                                                    |
| **Orchestration**                | Plain Python, no agent framework                                                 | The pipeline is a fixed, linear sequence of well-defined steps, not an open-ended task requiring dynamic tool selection. Introducing an agent framework would add indirection without adding capability, and would directly work against PRD Goal 2 (clean, modular, production-readable code a new engineer can extend).                                                                                                                                                                                                                                                                                                                                   |

### Keeping inference cost bounded as the dataset scales

The architecture's cost profile is dominated by the offline indexing step (linear in candidate count, but paid once) and almost flat at query time (the LLM only ever sees the JD once and the top 20 candidates, never the full 500 or the full 100K). Three deliberate choices make this hold even if the candidate pool grew well past 100K:

1. **The expensive, per-candidate work (embedding, feature extraction) is amortized** — it happens at index-build time, not per query. A recruiter pasting 50 JDs in a day pays that cost zero additional times.
2. **LLM calls are bounded by the _output_ size (top 20), not the _input_ size (100K or 500)** — the funnel shape means LLM cost is structurally decoupled from candidate-pool size.
3. **The 500-candidate shortlist is scored with vectorized math, not API calls** — this is the step that scales with pool size, and it's the cheapest possible operation per candidate (array arithmetic), not the most expensive (a network call to a hosted model).

---

## 6. White-Box Design: Explainability as a Pipeline Property, Not a Final Step

A system is only a genuine white box if every stage produces an artifact a human can inspect — not if a black-box score gets a paragraph of LLM-generated prose stapled onto it afterward. This pipeline is built so that an audit trail exists at every hop, not just at the end:

- **The JD parser's output is itself inspectable.** `jd_intent.json` is not consumed and discarded — it's the literal object a recruiter (or a judge) can read to confirm the system understood the JD correctly _before_ any candidate is touched. If the ranking looks wrong, this is the first thing to check, and it's a structured JSON object, not a buried prompt.
- **Retrieval is a similarity score, not a discard.** Every candidate in the 500-shortlist carries the cosine similarity that earned its place — retrieval isn't a black hole that candidates either survive or vanish into.
- **Every sub-score is named, persisted, and independently computed.** B1–B5 are not internal intermediate variables collapsed into a single number before anyone can see them — they are first-class columns in the output file. A recruiter can see _that_ a candidate scored low on stability and high on semantic fit, not just a single composite number.
- **Each scoring function is a documented, independently testable unit.** `stability_score()` has a fixed formula, a stated rationale, and can be unit-tested against known inputs without invoking the embedding model, the LLM, or any other module — satisfying the PRD's literal Goal 2 success metric.
- **The explanation generator is structurally prevented from free-associating.** It receives the exact sub-scores and source fields that produced them, is instructed to cite them, and is checked post-hoc for grounding (Validation 7 in the Data Understanding Report). The explanation is therefore a _narration of an already-computed, auditable decision_ — not an independent judgment that happens to be rendered as text.
- **The final output file puts numbers and narrative side by side.** Every row in `ranked_output.csv` carries both the explanation and the full sub-score breakdown that produced it. A skeptical hiring manager — or a judge — never has to choose between trusting the prose or trusting the math; both are right there, and they're guaranteed to be derived from the same computation.

The result is that explainability isn't a feature bolted onto stage 6 — it's a property the pipeline can't lose, because every upstream stage already produces a labeled, persisted artifact rather than an opaque intermediate value.

---

## 7. Summary

| Question                          | Answer                                                                                                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Which architecture?**           | Hybrid Retrieve → Score → Explain (Option C)                                                                                                                                                           |
| **Where does the LLM sit?**       | Exactly two points: JD parsing (1 call) and top-20 explanation generation (a handful of batched calls)                                                                                                 |
| **Where is everything else?**     | Deterministic: embedding similarity, ANN retrieval, and five named pure-function scoring formulas (B1–B5)                                                                                              |
| **Why not more LLM (Option B)?**  | Doesn't fit the 60s/100K budget, and per-candidate LLM scoring isn't stable enough to defend as "the genuine top 20 of 100,000"                                                                        |
| **Why not less LLM (Option A)?**  | Reintroduces keyword brittleness on the JD side and produces templated explanations the PRD explicitly rules out                                                                                       |
| **How does it stay a white box?** | Every stage persists a named, inspectable artifact — sub-scores are never collapsed before they're shown, and the explanation is grounded in those same persisted numbers, not generated independently |

---

_Document owner: Hackathon team lead_
_Last updated: June 2026_
_Status: Proposed — ready for component implementation_
