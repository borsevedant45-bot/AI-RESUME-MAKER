# Redrob Intelligent Candidate Discovery & Ranking Engine

> Replace keyword-matching ATS logic with contextual, explainable candidate ranking across 100,000 profiles.

[![Status](https://img.shields.io/badge/status-hackathon%20submission-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 1. What This Is

Traditional ATS platforms rank candidates with Boolean keyword matching. This engine replaces that filter with a **hybrid retrieve → score → explain pipeline**:

1. **JD Parse** — an LLM call extracts structured intent (seniority, must-have v. nice-to-have skills, domain, implicit soft skills) from free-text JD
2. **ANN Retrieval** — `all-MiniLM-L6-v2` embeddings + FAISS narrow 100,000 candidates to 500
3. **Multi-faceted Scoring** — five deterministic pure functions (semantic, trajectory, stability, platform, cert) score reproducibly in 1.2s
4. **Ranker** — composite = `B1×0.35 + B2×0.25 + B3×0.15 + B4×0.20 + B5×0.05`
5. **Explanation Generator** — second, tightly-grounded LLM call writes fact-checked justifications for each of the final top 20

The LLM does the two things LLMs are good at (understanding messy language once) and pure functions do the thing they're good at (scoring 500 candidates the same way every time).

---

## 2. Architecture

```
═══ OFFLINE — INDEXING PIPELINE ═══

  100,000 candidate profiles (JSONL)
    │
    ├──► Candidate Document Builder ──► all-MiniLM-L6-v2 ──► candidate_vectors.npy (100K × 384)
    │                                                              │
    │                                                              ▼
    │                                                       FAISS IndexFlatIP (persisted)
    │
    └──► Structured Feature Extractor ──► candidate_features.parquet
          (trajectory / stability /            (joined on candidate_id)
           platform / cert sub-features)

═══ ONLINE — QUERY PIPELINE ═══

  Recruiter pastes a JD
    │
    ▼
  1. JD Parser          (1 LLM call)              → jd_intent.json
    │
    ▼
  2. JD Embedder        (same model as index)
    │
    ▼
  3. ANN Retrieval      FAISS top-500
    │
    ▼
  4. Multi-Signal Scoring  (deterministic, 1.2s)
     B1 Semantic Skill Match        (0.35)
     B2 Trajectory & Seniority Fit  (0.25)
     B3 Career Stability            (0.15)
     B4 Platform Activity & Intent  (0.20)
     B5 Certification Bonus        (+0.05, capped)
    │
    ▼
  5. Ranker              → top 20 by composite_score
    │
    ▼
  6. Explanation Generator (batched LLM calls)
    │
    ▼
  7. Output Writer       ranked_output.csv / ranked_output.json
```

---

## 3. Installation & Setup

**Requirements:** Python 3.10+, ~4 GB free disk (for embeddings + index), a Groq API key.

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-org>/redrob-ranking-engine.git
cd redrob-ranking-engine

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

```bash
# 4. Set your API key (never commit this — read from environment only)
export GROQ_API_KEY=gsk_xxxxxxxxxxxx

# 5. Place the dataset at: data/raw/candidates.jsonl
```

All tunable parameters live in `config/settings.yaml` — nothing is hardcoded.

---

## 4. Usage

### Index (run once, ~41 min for 100K profiles on a laptop CPU)

```bash
python main.py index --data data/raw/candidates.jsonl --processed data/processed/
```

### Query (per JD, target < 60s)

```bash
python main.py query --jd data/jd.txt --processed data/processed/ --output data/outputs/
```

### Combined first-time run

```bash
python main.py run --data data/raw/candidates.jsonl --jd data/jd.txt --output data/outputs/
```

### Inspect output

```bash
cat data/outputs/jd_intent.json
python -c "import pandas as pd; df = pd.read_csv('data/outputs/ranked_output.csv'); print(df.head(5).to_string())"
```

---

## 5. Scoring Methodology

| Sub-score | Weight | What it measures |
|-----------|--------|------------------|
| **B1 — Semantic Skill Match** | 0.35 | Embedding similarity + JD skill coverage, with semantic equivalence |
| **B2 — Trajectory & Seniority Fit** | 0.25 | Seniority alignment + promotion history + stretch readiness |
| **B3 — Career Stability** | 0.15 | Average tenure, job-hopping pattern, education-tier tiebreaker |
| **B4 — Platform Activity & Intent** | 0.20 | Job-seeking signals, recruiter demand, hire reliability |
| **B5 — Certification Bonus** | +0.05 (cap 0.10) | Relevance- and recency-weighted certification boost |

`composite_score = B1×0.35 + B2×0.25 + B3×0.15 + B4×0.20 + B5×0.05`

**Explainability is narration, not re-judgment.** The explanation generator is handed already-computed sub-scores and exact evidence fields; a post-generation grounding check confirms each explanation cites at least one real data point before it's accepted.

---

## 6. Repository Structure

```
redrob-ranking-engine/
├── config/settings.yaml
├── src/
│   ├── data_loader/
│   ├── jd_parser/
│   ├── embedder/
│   ├── feature_extractor/
│   ├── retriever/
│   ├── scorer/              # b1_semantic.py … b5_cert.py + composite.py
│   ├── ranker/
│   ├── explainer/
│   ├── output_writer/
│   └── pipeline/
├── tests/                   # 203/203 unit tests
├── scripts/                 # Evaluation & ablation tools
│   ├── evaluate.py
│   ├── ablation.py
│   ├── check_explanations.py
│   └── find_anchors.py
├── data/{raw,processed,outputs}/
├── docs/
├── main.py
└── requirements.txt
```

---

## 7. Evaluation Summary

### All 5 Golden JDs

| JD | Sem Range | Comp Range | Fit@20 | Contam. | Grounding |
|----|-----------|------------|--------|---------|-----------|
| Senior Data Engineer | [0.645, 0.666] | [0.697, 0.732] | 0/3 | 0 | **16/20** |
| HR Manager | [0.600, 0.620] | [0.689, 0.736] | 0/3 | 0 | 0/20\* |
| Mid Data Analyst | [0.619, 0.640] | [0.682, 0.711] | 0/3 | 0 | 0/20\* |
| Program Lead (implicit) | [0.572, 0.590] | [0.672, 0.740] | 0/3 | 0 | 0/20\* |
| Senior Accountant | [0.610, 0.624] | [0.639, 0.722] | 0/3 | 0 | **12/20\*\*** |

\*HR/Analyst/PM hit Groq TPD limit — structured fallback data used.
\*\*Accountant used llama-3.1-8b-instant fallback model; 12/20 grounded.

### Indexing Performance

| Metric | Value |
|--------|-------|
| Records | 100,000, 41m 31s, 0 skipped |
| Model | `all-MiniLM-L6-v2` (384-dim) |
| Artifacts | 4 files, ~300 MB |

### Scoring Time

| Before (per-candidate re-encode) | After (skill-vector caching) | Speedup |
|:--:|:--:|:--:|
| 116s | **1.2s** | **100×** |

### Ablation — Removing B2 (trajectory) causes the most disruption

| Config | Overlap | Jaccard |
|--------|---------|---------|
| Baseline | 20/20 | — |
| No B1 (semantic) | 19/20 | 0.905 |
| **No B2 (trajectory)** | **9/20** | **0.290** |
| No B3 (stability) | 9/20 | 0.290 |
| No B4 (platform) | 10/20 | 0.333 |
| No B5 (cert) | 19/20 | 0.905 |

### Keyword-Stuffer Caught

CAND_0026806 (Software Engineer, 6.6+ yrs): with B2 ranking = rank **501**; without B2 = rank **6**. Low trajectory score (0.599) suppresses a candidate with title-level match but weak career progression — exactly the signal the system is designed to catch.

---

## 8. Known Limitations

- **No labeled ground truth** — "ranking quality" validated against manually-constructed golden JDs and anchor candidates, not a benchmark score
- **Embedding model is CPU-sized** — `all-MiniLM-L6-v2` (384-dim) chosen for laptop-CPU feasibility; may benefit from a higher-dim model or BM25 hybrid retrieval
- **Synthetic dataset artifacts** — uniform title/skill/geography distribution
- **No resume parsing** — consumes pre-structured JSON profiles only
- **No feedback loop** — recruiter actions aren't fed into weight tuning

---

## 9. Credits & License

Built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

JD understanding and candidate explanation generation powered by **Groq (llama-3.3-70b-versatile)**. Candidate embeddings via **sentence-transformers** (`all-MiniLM-L6-v2`). Vector search via **FAISS**.

Licensed under the [MIT License](LICENSE).
