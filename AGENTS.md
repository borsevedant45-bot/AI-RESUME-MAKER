### Session Anchored Summary

Last updated: after BUG-1/2/3 fixes and re-run with matching model

#### What's been built
- **IndoRuns**: A candidate-to-JD matching engine using SentenceTransformers + ANN retrieval + multi-faceted scoring (semantic, trajectory, stability, platform, cert) + LLM-grounded explanations.
- **Pipeline**: `index` → build doc embeddings + FAISS index + feature store. `query` → JD parse → ANN retrieval → score → rank → explain.
- **Database**: PostgreSQL `candidates` (100K rows), `work_experiences` (row-per-role), `skills`, `certifications`.

#### Pipeline state
| Stage | Unit tests | smoke-test | full-run |
|-------|-----------|------------|----------|
| Indexing | 152/152 pass | 5 records: 8s, 100% load, 4 artifacts | **100,000 records: 41m 31s** (100% loaded, 0 skipped) |
| JD parsing | 36/36 pass | – | ✅ (Groq-powered, ~4s) |
| Retrieval | 15/15 pass | – | ANN recalls 500 candidates (0.1s) |
| Scoring | 203/203 pass | – | **1.2s** (was 116s — 100× speedup from skill-vector caching) |
| Ranking + explanation | included above | – | Top-20 output with Groq explanations (58s) |
| **Total** | **203/203 pass** | – | **41m index + 80s query** |

#### Key design decisions
1. `all-MiniLM-L6-v2` (384-dim) chosen for CPU practicality — 100K docs in 40m.
2. Vector store: FAISS IndexFlatIP, 500-overfetch. Profile store: in-memory dict.
3. 3-level threading in indexing: work-stealing pool (doc building) + per-core workers + embedding pool. `_MAX_WORKERS = 5`.
4. Feature store extracts latest title, seniority, role category, years_exp, promotion_rate, co_tenures from work_experiences.
5. Composite score: `0.35×semantic + 0.25×trajectory + 0.15×stability + 0.20×platform + 0.05×cert`.

#### Three bugs fixed

| Bug | What changed | Impact |
|-----|-------------|--------|
| **BUG 1 — JD embedding input** | Added `build_jd_query_doc()` to `candidate_doc_builder.py`; query now builds structured doc from `jd_intent` fields (incl. soft skills) instead of raw JD text | JD vector now encodes structured query (skills+domain+soft skills) |
| **BUG 2 — Per-candidate skill re-encoding** | Pre-encode JD skill vectors once before scoring loop; pass `skill_vectors` dict into `semantic_score()`; use cached vectors when available | Scoring: **116s → 1.2s** (100× faster) |
| **BUG 3 — Composite weights** | Already correct at 0.35/0.25/0.15/0.20/0.05 per spec. Fixed `settings.yaml` model from `bge-small-en-v1.5` → `all-MiniLM-L6-v2` to match index | ANN retrieval now compares vectors in same embedding space |

#### Top-20 results after fixes (Senior Data Engineer JD)

| Metric | Before | After |
|--------|--------|-------|
| Semantic range | [0.493, 0.506] (spread 0.013) | [0.635, 0.661] (spread 0.026) |
| Domain mismatches | 6 PMs, 4 HRs, 2 Marketing | **0** — all 20 are DE/SE/DS/AI |
| Top candidate | Project Manager (0.678) | Sr Software Eng SQL/Spark/Cloud (0.730) |
| Scoring time | 116s | 1.2s |

#### Why model mismatch caused the original problem
`settings.yaml` had `model_name: "BAAI/bge-small-en-v1.5"` but the 100K index was built with `all-MiniLM-L6-v2`. ANN was comparing vectors from different embedding spaces → effectively random retrieval. Fixing this to `all-MiniLM-L6-v2` immediately eliminated all domain mismatches.

#### Outputs
- `data/outputs/ranked_output.json` — full 20 results with per-slot scores, explanations, and grounding validation

#### What's next / open issues
- [ ] Semantic spread (0.026) still narrow — consider switching to a model that better differentiates engineering roles (e.g., `BAAI/bge-base-en-v1.5` with 768-dim, but requires re-indexing)
- [ ] Evaluate quality/coverage of top-20 against a human baseline
- [ ] (Minor) trait-scoring is a stub — always returns 1.0
- [ ] (Minor) profile_store dict lacks career_history — explanations for skills/certs limited
