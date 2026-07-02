### Session Anchored Summary

Last updated: 2026-07-02 — Full 5-JD evaluation after re-index + fallback grounding fix

#### What's been built
- **IndoRuns**: A candidate-to-JD matching engine using SentenceTransformers + ANN retrieval + multi-faceted scoring (semantic, trajectory, stability, platform, cert) + LLM-grounded explanations.
- **Pipeline**: `index` → build doc embeddings + FAISS index + feature store. `query` → JD parse → ANN retrieval → score → rank → explain.
- **Database**: PostgreSQL `candidates` (100K rows), `work_experiences` (row-per-role), `skills`, `certifications`.

#### Pipeline state
| Stage | Unit tests | smoke-test | full-run |
|-------|-----------|------------|----------|
| Indexing | 152/152 pass | 5 records: 8s, 100% load, 4 artifacts | **100,000 records: ~3h 16m** (100% loaded, 0 skipped, `bge-small-en-v1.5`) |
| JD parsing | 36/36 pass | – | ✅ (Groq-powered, ~1.5-2s) |
| Retrieval | 15/15 pass | – | ANN recalls 500 candidates (0.1s) |
| Scoring | 203/203 pass | – | **1.2s** (was 116s — 100× speedup from skill-vector caching) |
| Ranking + explanation | included above | – | Top-20 output with Groq explanations (~60-80s with TPD retries) |
| **Total** | **203/203 pass** | – | **~3h 16m index + ~80-140s query** |

#### Key design decisions
1. `BAAI/bge-small-en-v1.5` (384-dim) — stronger asymmetric retrieval than `all-MiniLM-L6-v2`, chosen for domain separation at scale.
2. Vector store: FAISS IndexFlatIP, 500-overfetch. Profile store: in-memory dict.
3. Feature store extracts latest title, seniority, role category, years_exp, promotion_rate, co_tenures from work_experiences.
4. Composite score: `0.35×semantic + 0.25×trajectory + 0.15×stability + 0.20×platform + 0.05×cert`.

#### Three bugs fixed

| Bug | What changed | Impact |
|-----|-------------|--------|
| **BUG 1 — JD embedding input** | Added `build_jd_query_doc()` to `candidate_doc_builder.py`; query now builds structured doc from `jd_intent` fields | JD vector encodes structured query (skills+domain+soft skills) |
| **BUG 2 — Per-candidate skill re-encoding** | Pre-encode JD skill vectors once; pass cached `skill_vectors` into `semantic_score()` | Scoring: **116s → 1.2s** (100× faster) |
| **BUG 3 — Fallback grounding** | Added structured fallback explanation when LLM fails (TPD/429); fallback uses actual candidate data (years, skills) | **Grounding: 20/20 for DE, HR, Analyst** (was 16, 0, 0). PM 16/20, Accountant 0/20 (TPD exhausted) — fallback protects all |

#### Final 5-JD Evaluation Results (2026-07-02)

| Metric | DE | HR | Analyst | PM | Accountant |
|--------|----|----|---------|----|------------|
| Grounding (prev) | 16/20 | 0/20\* | 0/20\* | 0/20\* | 12/20\*\* |
| **Grounding (now)** | **20/20** | **20/20** | **20/20** | **16/20** | **20/20\*\*** |
| Strong matches | 9 | 13 | 20 | 13 | — |
| Moderate matches | 11 | 7 | 0 | 3 | — |
| Fallback | 0 | 0 | 0 | 4 | 0 |
| Semantic range | [0.657, 0.668] | [0.663, 0.676] | [0.692, 0.698] | [0.677, 0.689] | [0.679, 0.689] |
| Composite range | [0.668, 0.721] | [0.671, 0.734] | [0.701, 0.741] | [0.670, 0.737] | [0.636, 0.725] |

\*HR/Analyst/PM had 0/20 due to Groq TPD in prior run; now 20/20, 20/20, 16/20.
\*\*Accountant re-run with `llama-3.1-8b-instant` (batch_size=1) after TPD reset — 20/20 grounded.

**Overall: 96/100 LLM-grounded, 4/100 structured fallback (PM only), 0 nulls.**

#### Outputs
- `data/outputs/de/` `hr/` `analyst/` `pm/` `accountant/` — each contains `ranked_output.csv` and `ranked_output.json`
- `app/streamlit_app.py` — Streamlit demo app (works with any output directory)

#### What's next / open issues
- [ ] Upgrade GROQ key to Dev Tier to avoid TPD limits across 5 JD runs
- [ ] Semantic spread (0.006-0.013) still very narrow — consider `bge-base-en-v1.5` (768-dim) for better differentiation
- [ ] (Minor) trait-scoring is a stub — always returns 1.0
- [ ] (Minor) profile_store dict lacks career_history — explanations for skills/certs limited
