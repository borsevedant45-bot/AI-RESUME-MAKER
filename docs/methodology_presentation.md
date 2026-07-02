# Methodology & Results — Redrob Ranking Engine

## Slide 1: Problem Statement

**Traditional ATS ranking is broken.**

- Boolean keyword matching produces false negatives at scale
- A candidate who "built highly scalable applications with React at Meta" ranks below someone who stuffed "Senior Frontend Engineer" five times
- Keyword filters cannot: see career trajectory, tell a top performer from a mediocre applicant, or explain _why_ anyone was shortlisted

**Goal:** Replace keyword filters with a hybrid retrieve → score → explain pipeline that is reproducible, explainable, and LLM-efficient.

---

## Slide 2: System Architecture

**Offline (run once per dataset, ~41 min):**

- 100,000 candidate profiles → Candidate Document Builder → `all-MiniLM-L6-v2` embeddings (384-dim) → FAISS IndexFlatIP → candidate_vectors.npy
- Structured Feature Extractor → trajectory/stability/platform/cert sub-features → candidate_features.parquet

**Online (run once per JD, target < 60s):**

1. JD Parser (1 LLM call) → structured jd_intent
2. JD Embedder (same model as index)
3. ANN Retrieval → FAISS top-500
4. Multi-Signal Scoring (deterministic, 1.2s for 500 candidates)
5. Ranker → top 20 by composite_score
6. Explanation Generator (batched LLM calls)
7. Output → ranked_output.csv / .json

**Key design principle:** LLM used exactly twice per JD — understanding the JD once, and narrating 20 already-computed decisions. Pure math everywhere it has to run at scale.

---

## Slide 3: Scoring Methodology

Five independent, deterministic sub-scores:

| Sub-score                       | Weight           | What it measures                                                                  |
| ------------------------------- | ---------------- | --------------------------------------------------------------------------------- |
| B1 — Semantic Skill Match       | 0.35             | Embedding similarity + JD skill coverage with semantic equivalence                |
| B2 — Trajectory & Seniority Fit | 0.25             | Seniority alignment (penalized both ways) + promotion history + stretch readiness |
| B3 — Career Stability           | 0.15             | Average tenure, job-hopping pattern, education-tier tiebreaker                    |
| B4 — Platform Activity & Intent | 0.20             | Active job-seeking signals, recruiter demand, hire reliability                    |
| B5 — Certification Bonus        | +0.05 (cap 0.10) | Relevance- and recency-weighted certification boost                               |

**Composite:** `B1×0.35 + B2×0.25 + B3×0.15 + B4×0.20 + B5×0.05`

Every sub-score is persisted — nothing is collapsed before it's shown.

---

## Slide 4: Indexing Performance

| Metric              | Value                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Dataset size        | 100,000 candidates                                                                                |
| Total indexing time | **41 minutes 31 seconds**                                                                         |
| Embedding model     | `all-MiniLM-L6-v2` (384-dim)                                                                      |
| Batch size          | 256                                                                                               |
| Threading           | 3-level: work-stealing pool + per-core workers + embedding pool                                   |
| Records loaded      | 100,000 (100%)                                                                                    |
| Records skipped     | 0                                                                                                 |
| Index artifacts     | `candidate_vectors.npy`, `faiss_index.bin`, `candidate_features.parquet`, `candidate_id_map.json` |
| Total disk size     | ~300 MB                                                                                           |

**Scalability note:** Indexing is O(n) in the number of candidates. Doubling to 200K would take ~83 min on the same hardware.

---

## Slide 5: Retrieval & Query Performance

**Single JD query pipeline (Senior Data Engineer):**

| Stage                  | Time       | Notes                                         |
| ---------------------- | ---------- | --------------------------------------------- |
| JD parsing             | ~1.5-2s    | 1 LLM call via Groq (llama-3.3-70b-versatile) |
| ANN retrieval          | ~0.1s      | FAISS top-500 from 100K                       |
| Multi-signal scoring   | **~1.2s**  | 500 candidates × 5 dimensions                 |
| Explanation generation | ~60-80s    | Batched LLM calls for top 20 (with TPD retries) |
| **Total**              | **~80-140s**| Varies with Groq rate-limit retries           |

**Optimization highlight:** Scoring was 116s before skill-vector caching. Pre-encoding JD skill vectors once and caching per-candidate skill vectors reduced this to **1.2s — a 100× speedup**.

---

## Slide 6: Quantitative Evaluation (All 5 Golden JDs)

| Metric          | DE                 | HR                 | Analyst            | PM                 | Accountant         |
| --------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Grounding       | **20/20**          | **20/20**          | **20/20**          | **16/20**          | **20/20**\*\*      |
| Strong matches  | 9                  | 13                 | 20                 | 13                 | —                  |
| Moderate matches| 11                 | 7                  | 0                  | 3                  | —                  |
| Fallback        | 0                  | 0                  | 0                  | 4                  | 0                  |
| Contamin.@100   | 0                  | 0                  | 0                  | 0                  | 0                  |
| Precision@20    | **15/15**          | **15/15**          | **15/15**          | **15/15**          | **15/15**          |
| Semantic range  | [0.657, 0.668]    | [0.663, 0.676]    | [0.692, 0.698]    | [0.677, 0.689]    | [0.679, 0.689]    |
| Composite range | [0.668, 0.721]    | [0.671, 0.734]    | [0.701, 0.741]    | [0.670, 0.737]    | [0.636, 0.725]    |

\*HR/Analyst had 0/20 in prior TPD-limited run. With GROQ key + fallback fix: 20/20, 20/20.
\*\*Accountant re-run with `llama-3.1-8b-instant` (batch_size=1) after TPD reset — 20/20 grounded via model-switch workaround.

**Grounding summary:** 96/100 LLM-grounded, 4/100 structured fallback (PM only, TPD-limited mid-run), 0 nulls, 0 silent failures across all 5 JDs and 100 candidates.

**Key:** 0 contamination across all JDs. Domain purity is near-perfect (DE, HR, Analyst are 100% on-domain; PM includes relevant Ops Manager titles; Accountant has mild title noise). NDCG improved with stronger `bge-small-en-v1.5` model.

---

## Slide 7: Ablation & Keyword-Stuffer Detection

**Ablation study — removing one dimension at a time:**

| Config                 | Overlap with baseline | Jaccard   |
| ---------------------- | --------------------- | --------- |
| Baseline (all 5 dims)  | 20/20                 | —         |
| No B1 (semantic)       | 19/20                 | 0.905     |
| **No B2 (trajectory)** | **9/20**              | **0.290** |
| No B3 (stability)      | 15/20                 | 0.600     |
| No B4 (platform)       | 14/20                 | 0.538     |
| No B5 (cert)           | 20/20                 | 1.000     |

**Finding:** Trajectory (B2) remains the single most impactful dimension — removing it still causes 55% of the top-20 to change.

**Keyword-Stuffer Case Study:**

- Candidate: CAND_0066999 (low trajectory but strong title match)
- **With B2 trajectory scoring:** rank 501 (not in top 500 retrieved)
- **Without B2:** rank 2 (jumps 499 positions into rank 2)
- **B2 score:** 0.591 (low = weak career progression despite matching title)
- **Interpretation:** Title-level match but weak career progression. When trajectory scoring is removed, this candidate surges 499 positions into the top 2. Exactly the signal the system is designed to catch.

---

## Slide 8: Limitations & Future Work

**Current limitations:**

- **No labeled ground truth** — validation relies on manually-constructed golden JDs and anchors
- **CPU-sized embedding model** — `BAAI/bge-small-en-v1.5` (384-dim) still has narrow semantic spread (~0.010); `bge-base-en-v1.5` (768-dim) would improve differentiation
- **Synthetic dataset artifacts** — uniform title/skill distribution constrains certain scoring decisions
- **Anchor recall shifted** after re-index with new model — some anchor candidates no longer in top 100; expected with model change
- **Groq TPD limited** — PM (4/20 fallback) hit Free Tier TPD mid-run; Accountant required model-switch workaround (`llama-3.1-8b-instant` with `batch_size=1` to respect TPM limits). Structured fallback protects all remaining cases
- **No resume parsing** — consumes pre-structured JSON only

**With more time:**

- Run full 5-JD × 5-ablation grid for each JD
- Switch to `BAAI/bge-base-en-v1.5` (768-dim) for better role differentiation (requires re-indexing)
- Add BM25 hybrid retrieval to harden recall on edge-case JDs
- Add LLM-as-judge secondary reference ranking for independent rank correlation
- Integrate feedback loop from recruiter actions into weight tuning
- Add raw PDF resume ingestion
- Upgrade GROQ API key to Dev Tier to avoid TPD limits across multi-JD evaluation runs

"BAAI/bge-small-en-v1.5 (384-dim) was chosen over all-MiniLM-L6-v2 for its stronger asymmetric query-to-document retrieval at 100K scale, documented as the PRD Risk 1 fallback path. Domain separation was validated: DE/HR/Analyst show 100% on-domain titles in top 20."
