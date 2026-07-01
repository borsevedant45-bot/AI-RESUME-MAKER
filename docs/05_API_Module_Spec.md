# API & Code Structure Specification

## Redrob Intelligent Candidate Discovery & Ranking Engine

**Document:** `docs/05_API_Module_Spec.md`
**Version:** 1.0
**Inputs:** `01_PRD.md`, `02_Data_Understanding_Report.md`, `03_System_Architecture_Document.md`, `04_Ranking_Scoring_Explainability_Methodology.md`
**Status:** Approved — implementation blueprint for build phase

---

## Table of Contents

1. [Repository Folder Structure](#1-repository-folder-structure)
2. [Core Data Objects](#2-core-data-objects)
3. [Module Specifications & Function Signatures](#3-module-specifications--function-signatures)
4. [Configuration Management](#4-configuration-management)
5. [End-to-End CLI Usage](#5-end-to-end-cli-usage)
6. [Logging & Error-Handling Conventions](#6-logging--error-handling-conventions)

---

## 1. Repository Folder Structure

```
redrob-ranking-engine/
│
├── config/
│   └── settings.yaml                  # All tunable params: model names, weights, thresholds
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_loader/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Reads raw JSONL; yields CandidateProfile objects
│   │   └── validator.py               # Per-record schema validation; emits warnings on bad records
│   │
│   ├── jd_parser/
│   │   ├── __init__.py
│   │   ├── parser.py                  # LLM call → JDIntent; includes retry + post-parse validation
│   │   └── prompt_templates.py        # System and user prompt strings (no logic, just strings)
│   │
│   ├── embedder/
│   │   ├── __init__.py
│   │   ├── candidate_doc_builder.py   # Builds the concatenated text doc per candidate
│   │   ├── embedder.py                # SentenceTransformer wrapper; batch + single encode
│   │   └── index_builder.py           # Builds + persists FAISS index and candidate_vectors.npy
│   │
│   ├── feature_extractor/
│   │   ├── __init__.py
│   │   ├── skill_features.py          # skill_strength_score per skill; returns {skill: score} dict
│   │   ├── trajectory_features.py     # latest_seniority, promotion_rate, trajectory_score_base
│   │   ├── stability_features.py      # avg_tenure_months, job_hopping_flag
│   │   └── platform_features.py       # active_intent_score, hire_reliability_score pre-computation
│   │
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── retriever.py               # FAISS ANN search; returns top-N candidate_ids + distances
│   │   └── hard_filter.py             # Binary salary/location filters; only when JD states constraints
│   │
│   ├── scorer/
│   │   ├── __init__.py
│   │   ├── b1_semantic.py             # semantic_score() — embedding sim + skill coverage
│   │   ├── b2_trajectory.py           # trajectory_score() — seniority fit + stretch readiness
│   │   ├── b3_stability.py            # stability_score() — tenure, hopping flag, edu tier
│   │   ├── b4_platform.py             # platform_score() — intent + reliability + github
│   │   ├── b5_cert.py                 # cert_bonus() — relevance × recency, capped at 0.10
│   │   └── composite.py               # composite_score() — weighted sum of B1–B5
│   │
│   ├── ranker/
│   │   ├── __init__.py
│   │   └── ranker.py                  # Sorts by composite score; applies tiebreaker; selects top-N
│   │
│   ├── explainer/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py          # Builds per-candidate explanation prompts from sub-scores + fields
│   │   ├── explainer.py               # Batched LLM calls; returns list[CandidateExplanation]
│   │   └── grounding_validator.py     # Post-gen check: explanation must cite ≥1 real datum
│   │
│   ├── output_writer/
│   │   ├── __init__.py
│   │   └── writer.py                  # Writes ranked_output.csv and ranked_output.json
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── indexing_pipeline.py       # Offline: load → embed → extract features → build index
│       └── query_pipeline.py          # Online: parse JD → retrieve → score → rank → explain → write
│
├── tests/
│   ├── unit/
│   │   ├── test_jd_parser.py          # Golden JD inputs → expected JDIntent field values
│   │   ├── test_skill_features.py     # Known skill records → expected skill_strength_score
│   │   ├── test_trajectory.py         # Constructed career histories → expected trajectory_score
│   │   ├── test_stability.py          # Tenure arrays → expected stability_score + hopping_flag
│   │   ├── test_platform.py           # Redrob signal dicts → expected platform_score
│   │   ├── test_b1_semantic.py        # Mock vectors + skill records → expected semantic_score
│   │   └── test_composite.py          # Known B1–B5 values → expected composite_score
│   │
│   └── integration/
│       ├── test_full_pipeline.py      # Golden JD + small candidate sample → top-3 expected
│       └── test_explanation_grounding.py  # Explanations must pass grounding validator
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset EDA: distributions, flag validation
│   ├── 02_embedding_recall_check.ipynb # Validation 1: ANN recall rate on test JDs
│   ├── 03_score_distribution_check.ipynb # Validations 2–5: feature distribution plots
│   └── 04_golden_jd_ranking.ipynb     # Validation 6: end-to-end on golden test cases
│
├── data/
│   ├── raw/
│   │   └── candidates.jsonl           # Source dataset (100K records, not committed to git)
│   ├── processed/
│   │   ├── candidate_vectors.npy      # Pre-computed embeddings (100K × 384)
│   │   ├── candidate_index.faiss      # FAISS flat index (persisted after indexing run)
│   │   └── candidate_features.parquet # Structured feature rows (100K rows, one per candidate)
│   └── outputs/
│       ├── jd_intent.json             # Parsed JD intent (written per query run)
│       └── ranked_output.csv          # Final top-20 ranked results (written per query run)
│
├── main.py                            # CLI entrypoint: dispatches to index or query sub-command
├── requirements.txt
└── README.md
```

---

## 2. Core Data Objects

All inter-module data is typed using Python `dataclasses` (or `pydantic` models if validation strictness is preferred). Serialization to/from JSON/Parquet uses the field names below as the canonical schema.

### 2.1 `JDIntent`

Produced by `jd_parser`. Consumed by `embedder`, `retriever`, `scorer` (all five), `explainer`.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class WorkContext:
    work_mode: Optional[str]                 # "remote" | "hybrid" | "onsite" | "flexible" | None
    location_required: Optional[str]         # city/region string, or None
    location_is_hard_requirement: bool       # False if JD doesn't state a hard constraint
    salary_min_lpa: Optional[int]            # None if not stated
    salary_max_lpa: Optional[int]            # None if not stated

@dataclass
class JDIntent:
    seniority_level: float                   # 0.2 | 0.5 | 0.75 | 1.0
    seniority_evidence: str                  # Exact phrase from JD that justified inference
    must_have_skills: list[str]              # Non-negotiable skill names
    nice_to_have_skills: list[str]           # Preferred but not gating
    core_problems_to_solve: str              # 2–4 sentence domain/problem summary
    implicit_soft_skills: list[str]          # e.g. ["technical ownership", "mentorship"]
    domain_tags: list[str]                   # From controlled vocab; 1–3 tags
    requires_technical_github_signals: bool  # True for eng/data/devops JDs
    work_context: WorkContext
    salary_stated: bool                      # True only if JD explicitly names a salary range
```

### 2.2 `CandidateProfile`

Produced by `data_loader`. Consumed by `embedder`, `feature_extractor`.

```python
@dataclass
class SkillRecord:
    name: str
    proficiency: str          # "beginner" | "intermediate" | "advanced" | "expert"
    endorsements: int
    duration_years: float

@dataclass
class RoleRecord:
    company: str
    title: str
    start_date: str           # ISO date string "YYYY-MM"
    end_date: Optional[str]   # None if current role
    duration_months: int
    industry: str
    company_size: str
    description: str
    is_current_role: bool     # Derived: end_date is None

@dataclass
class EducationRecord:
    degree: str
    field_of_study: str
    institution_tier: str     # "tier_1" | "tier_2" | "tier_3" | "tier_4"
    graduation_year: Optional[int]

@dataclass
class CertRecord:
    name: str
    issuer: str
    issue_year: int

@dataclass
class RedrobSignals:
    profile_completeness_score: float
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    profile_views_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hrs: float
    search_appearances_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    github_activity_score: float
    open_to_work: bool
    willing_to_relocate: bool
    email_verified: bool
    phone_verified: bool
    linkedin_connected: bool
    work_mode_preference: str
    expected_salary_min: int
    expected_salary_max: int

@dataclass
class CandidateProfile:
    candidate_id: str
    experience_years: float
    country: str
    industry: str
    current_title: str
    current_company: str
    company_size: str
    location: str
    skills: list[SkillRecord]
    career_history: list[RoleRecord]
    education: list[EducationRecord]
    certifications: list[CertRecord]
    redrob_signals: RedrobSignals
```

### 2.3 `CandidateFeatureRow`

Produced by `feature_extractor` + `embedder` (offline). Consumed by `scorer`, `explainer`. Persisted as one row in `candidate_features.parquet`.

```python
@dataclass
class CandidateFeatureRow:
    candidate_id: str

    # --- Offline-computed, JD-independent features ---
    # Embedding
    embedding_index: int                     # Row index in candidate_vectors.npy

    # Trajectory (B2)
    latest_seniority: float                  # 0.2 | 0.5 | 0.75 | 1.0
    promotion_rate: float                    # [0, 1]
    experience_years: float

    # Stability (B3)
    avg_tenure_months: float
    job_hopping_flag: int                    # 0 or 1
    institution_tier: str                    # Best tier across education entries

    # Platform (B4 — pre-computable sub-components)
    active_intent_score: float               # [0, 1]; computed from redrob_signals
    hire_reliability_score: float            # [0, 1]; computed from redrob_signals
    github_activity_score: float             # Raw scalar from redrob_signals
    endorsements_received: int               # Raw count from redrob_signals

    # Passthrough signals for explanation and hard filters
    open_to_work: bool
    willing_to_relocate: bool
    work_mode_preference: str
    notice_period_days: int
    expected_salary_min: int
    expected_salary_max: int
    location: str

    # Skill records (serialized as JSON blob in parquet)
    skill_strength_scores: dict[str, float]  # {skill_name_lower: strength_score}

    # Cert records (serialized as JSON blob)
    cert_records: list[dict]                 # [{"name": str, "issue_year": int}]

    # Profile quality flag
    thin_profile: bool                       # True if candidate_doc < 50 chars after construction
```

### 2.4 `ScoreBreakdown`

Produced by `scorer`. Consumed by `ranker`, `explainer`, `output_writer`.

```python
@dataclass
class ScoreBreakdown:
    candidate_id: str
    semantic_score: float      # B1 [0, 1]
    trajectory_score: float    # B2 [0, 1]
    stability_score: float     # B3 [0, 1]
    platform_score: float      # B4 [0, 1]
    cert_bonus: float          # B5 [0, 0.10]
    composite_score: float     # Weighted sum [0, ~1.0]
```

### 2.5 `CandidateExplanation`

Produced by `explainer`. Consumed by `output_writer`.

```python
@dataclass
class CandidateExplanation:
    candidate_id: str
    match_summary: str         # 1 sentence
    skill_alignment: str       # 2–3 sentences
    seniority_assessment: str  # 1–2 sentences
    trajectory_signal: str     # 1–2 sentences
    platform_summary: str      # 1–2 sentences
    flags: str                 # "No flags" or specific recruiter concerns
    grounding_validated: bool  # True if post-gen check passed; False = fallback used
```

### 2.6 `RankedResult`

The final output object per top-20 candidate. Produced by `ranker` + `explainer`. Written by `output_writer`.

```python
@dataclass
class RankedResult:
    rank: int
    candidate_id: str
    composite_score: float
    semantic_score: float
    trajectory_score: float
    stability_score: float
    platform_score: float
    cert_bonus: float
    explanation: CandidateExplanation
```

---

## 3. Module Specifications & Function Signatures

### 3.1 `src/data_loader/loader.py`

**Purpose:** Stream raw JSONL into typed `CandidateProfile` objects. Handles malformed records gracefully without aborting the run.

```python
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

def load_candidates(jsonl_path: Path) -> Iterator[CandidateProfile]:
    """
    Lazily yields CandidateProfile objects from a JSONL file.
    Skips and logs malformed records rather than raising.

    Args:
        jsonl_path: Path to the candidates.jsonl source file.

    Yields:
        CandidateProfile — one per valid line.

    Side effects:
        Logs a WARNING per skipped record including line number and error type.
        Logs a final INFO summary: total lines, valid profiles, skipped count.
    """
    ...

def load_candidates_batch(
    jsonl_path: Path,
    batch_size: int = 1000
) -> Iterator[list[CandidateProfile]]:
    """
    Yields lists of CandidateProfile objects in batches.
    Used by the indexing pipeline for memory-efficient processing.
    """
    ...
```

---

### 3.2 `src/data_loader/validator.py`

```python
def validate_candidate(raw: dict) -> tuple[bool, str]:
    """
    Checks that a raw candidate dict has the minimum required fields.
    Returns (True, "") if valid; (False, reason_string) if not.

    Required fields: candidate_id, experience_years, skills (non-empty list),
    career_history (non-empty list), redrob_signals.
    """
    ...
```

---

### 3.3 `src/jd_parser/parser.py`

**Purpose:** Single LLM call converts free-text JD into a validated `JDIntent`. The only open-ended LLM reasoning step in the pipeline.

```python
import groq
from src.config import Settings

def parse_job_description(
    jd_text: str,
    client: groq.Groq,
    settings: Settings,
) -> JDIntent:
    """
    Calls Groq llama-3.3-70b-versatile once with a structured JSON extraction prompt.
    Validates the response against JDIntent schema.
    Retries once on validation failure with a corrective prompt.

    Args:
        jd_text:  Free-text job description string from recruiter.
        client:   Groq client (injected for testability).
        settings: Config object carrying model name, max_tokens, etc.

    Returns:
        JDIntent — validated structured intent object.

    Raises:
        JDParseError: If both the initial call and retry return invalid JSON
                      or fail schema validation.

    Side effects:
        Writes jd_intent.json to settings.output_dir.
        Logs INFO with token usage and parse latency.
    """
    ...

def _validate_jd_intent(raw: dict) -> tuple[bool, str]:
    """
    Validates the raw LLM response dict against JDIntent schema rules:
    - All required fields present with correct types.
    - seniority_level is in {0.2, 0.5, 0.75, 1.0} (rounds if needed).
    - must_have_skills is non-empty.
    Returns (True, "") or (False, error_message).
    """
    ...
```

---

### 3.4 `src/embedder/candidate_doc_builder.py`

**Purpose:** Converts a `CandidateProfile` into a single rich text string for embedding. Handles thin profile fallback.

```python
def build_candidate_doc(profile: CandidateProfile) -> tuple[str, bool]:
    """
    Constructs the candidate embedding document from skills, career history,
    certifications, and education field of study.
    Most-recent roles appear first (recency bias in concatenation).
    Excludes: candidate_id, location, company names, raw numeric signals.

    Args:
        profile: CandidateProfile object.

    Returns:
        Tuple of (doc_string, thin_profile_flag).
        thin_profile_flag is True if the resulting doc is under 50 characters
        after stripping whitespace (triggers a semantic_score cap in scorer).

    Notes:
        For roles where description is under 20 characters, the fallback
        uses "title in industry" to preserve at least domain signal.
    """
    ...
```

---

### 3.5 `src/embedder/index_builder.py`

**Purpose:** Offline pipeline step. Encodes all candidates and builds the persisted FAISS index and feature parquet.

```python
from pathlib import Path
import numpy as np
import faiss

def build_candidate_index(
    jsonl_path: Path,
    output_dir: Path,
    settings: Settings,
) -> None:
    """
    Full offline indexing pipeline:
    1. Load all candidate profiles.
    2. Build candidate documents and encode in batches.
    3. Build FAISS flat index over normalized vectors.
    4. Extract all structured features per candidate.
    5. Persist: candidate_vectors.npy, candidate_index.faiss,
                candidate_features.parquet, candidate_id_map.json.

    Args:
        jsonl_path:  Path to candidates.jsonl.
        output_dir:  Directory to write all four output artifacts.
        settings:    Config with embedding model name, batch size, etc.

    Side effects:
        Writes four files to output_dir.
        Logs INFO progress every 10,000 records.
        Logs final INFO: total indexed, skipped, wall-clock time.

    Raises:
        IndexBuildError: If fewer than 90% of records indexed successfully.
    """
    ...

def load_index(processed_dir: Path) -> tuple[faiss.Index, np.ndarray, dict]:
    """
    Loads the pre-built FAISS index, candidate vectors, and candidate_id → row_index map.

    Returns:
        (faiss_index, candidate_vectors_matrix, id_to_index_map)
    """
    ...
```

---

### 3.6 `src/embedder/embedder.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def get_model(model_name: str) -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model.
    Called once at startup; returned model is reused across all encode calls.
    """
    ...

def encode_batch(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Batch encodes a list of texts with L2 normalization.
    Returns float32 array of shape (len(texts), embedding_dim).
    """
    ...

def encode_single(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Encodes one text string and returns a normalized 1D vector.
    Used for JD embedding at query time and per-skill semantic fallback in B1.
    """
    ...
```

---

### 3.7 `src/feature_extractor/skill_features.py`

```python
def compute_skill_strength(skill: SkillRecord) -> float:
    """
    Computes the weighted skill strength score for one skill entry.
    Formula: proficiency_weight*0.5 + min(duration/5,1)*0.35 + min(endorsements/50,1)*0.15
    Returns float in [0, 1].
    """
    ...

def build_skill_strength_map(skills: list[SkillRecord]) -> dict[str, float]:
    """
    Returns {skill_name.lower(): strength_score} for all skills in a profile.
    """
    ...
```

---

### 3.8 `src/feature_extractor/trajectory_features.py`

```python
SENIORITY_MAP: dict[str, float]  # Module-level constant; loaded from settings

def map_title_to_seniority(title: str) -> float:
    """
    Scans a job title string for seniority keywords and returns a level float.
    Defaults to 0.5 (mid) if no keyword matches.
    """
    ...

def detect_promotions(career_history: list[RoleRecord]) -> float:
    """
    Groups roles by company, sorts by start_date, detects seniority increases.
    Returns promotion_rate = detected_promotions / eligible_companies.
    Returns 0.0 if no company has multiple roles.
    """
    ...

def compute_trajectory_base(profile: CandidateProfile) -> dict:
    """
    Computes the JD-independent trajectory components for the feature store.
    Returns dict with keys: latest_seniority, promotion_rate.
    (JD-dependent seniority_fit_score is computed in b2_trajectory.py at query time.)
    """
    ...
```

---

### 3.9 `src/feature_extractor/stability_features.py`

```python
def compute_avg_tenure(career_history: list[RoleRecord]) -> float:
    """Returns mean duration_months across all roles. Returns 0.0 for empty history."""
    ...

def detect_job_hopping(career_history: list[RoleRecord]) -> int:
    """
    Returns 1 if 3+ consecutive roles each had duration_months < 12, else 0.
    Consecutive is defined by chronological sort on start_date.
    """
    ...

def best_institution_tier(education: list[EducationRecord]) -> str:
    """Returns the highest (lowest-number) tier across all education entries."""
    ...
```

---

### 3.10 `src/feature_extractor/platform_features.py`

```python
def compute_active_intent_score(signals: RedrobSignals) -> float:
    """
    Weighted combination of open_to_work (0.4 if False, 1.0 if True),
    applications_30d, profile_completeness, search_appearances.
    Returns float in [0, 1].
    """
    ...

def compute_hire_reliability_score(signals: RedrobSignals) -> float:
    """
    Weighted combination of interview_completion_rate, offer_acceptance_rate,
    avg_response_time (inverted), email_verified, phone_verified.
    Returns float in [0, 1].
    """
    ...
```

---

### 3.11 `src/retriever/retriever.py`

```python
import faiss
import numpy as np

def retrieve_top_n(
    jd_vector: np.ndarray,
    index: faiss.Index,
    id_to_index_map: dict[str, int],
    top_n: int = 500,
) -> list[tuple[str, float]]:
    """
    Runs ANN search and returns ordered list of (candidate_id, cosine_similarity).
    Highest similarity first.

    Args:
        jd_vector:      Normalized 1D query vector (same dim as index).
        index:          Pre-built FAISS flat index.
        id_to_index_map: Maps candidate_id to row index in vectors matrix.
        top_n:          Shortlist size (default 500).

    Returns:
        List of (candidate_id, similarity_score) tuples, length top_n.
    """
    ...
```

---

### 3.12 `src/retriever/hard_filter.py`

```python
def apply_hard_filters(
    shortlist: list[str],
    feature_store: dict[str, CandidateFeatureRow],
    jd_intent: JDIntent,
) -> list[str]:
    """
    Applies binary compatibility filters ONLY when JDIntent explicitly states
    a hard constraint (salary_stated=True and salary_max_lpa is set, or
    location_is_hard_requirement=True).

    Never filters on: open_to_work, notice_period, work_mode, education tier.

    Args:
        shortlist:      candidate_ids from ANN retrieval.
        feature_store:  In-memory dict of candidate features keyed by candidate_id.
        jd_intent:      Parsed JD intent object.

    Returns:
        Filtered list of candidate_ids.

    Side effects:
        Logs INFO with count removed by each filter type.
    """
    ...
```

---

### 3.13 `src/scorer/b1_semantic.py`

```python
def semantic_score(
    candidate_vector: np.ndarray,
    jd_vector: np.ndarray,
    skill_strength_scores: dict[str, float],
    jd_intent: JDIntent,
    model: SentenceTransformer,
    thin_profile: bool = False,
) -> float:
    """
    Computes B1 semantic score as:
        embed_sim_norm * 0.60 + coverage_score * 0.40

    Embed sim: cosine similarity (dot product of normalized vectors), rescaled to [0, 1].
    Coverage: weighted mean of skill_strength for JD's must-have and nice-to-have skills,
              with semantic fallback (discounted 0.6×) for skills not directly listed.

    Caps score at 0.55 if thin_profile=True.

    Returns float in [0, 1].
    """
    ...
```

---

### 3.14 `src/scorer/b2_trajectory.py`

```python
def trajectory_score(
    feature_row: CandidateFeatureRow,
    jd_intent: JDIntent,
) -> float:
    """
    Computes B2 trajectory score as:
        seniority_fit * 0.60 + trajectory_momentum * 0.40

    Seniority fit: penalizes mismatch symmetrically in both directions.
    Stretch readiness override: mid (0.5) vs senior (0.75) JD with
        promotion_rate >= 0.5 and experience_years >= 5 → fit = 0.75.
    Trajectory momentum: promotion_rate*0.5 + exp_years_norm*0.3 + latest_seniority*0.2.

    Returns float in [0, 1].
    """
    ...
```

---

### 3.15 `src/scorer/b3_stability.py`

```python
def stability_score(feature_row: CandidateFeatureRow) -> float:
    """
    Computes B3 stability score as:
        min(avg_tenure_months / 36, 1) - hopping_penalty + edu_bonus

    hopping_penalty = 0.30 if job_hopping_flag else 0.0
    edu_bonus: tier_1=0.05, tier_2=0.03, tier_3=0.01, tier_4=0.0

    Returns float clamped to [0, 1].
    """
    ...
```

---

### 3.16 `src/scorer/b4_platform.py`

```python
def platform_score(
    feature_row: CandidateFeatureRow,
    jd_intent: JDIntent,
) -> float:
    """
    Computes B4 platform score.

    If jd_intent.requires_technical_github_signals:
        active_intent*0.40 + hire_reliability*0.35 + tech_engagement*0.25
    Else:
        active_intent*0.55 + hire_reliability*0.45

    tech_engagement = github_norm*0.60 + endorse_norm*0.40

    active_intent_score and hire_reliability_score are pre-computed in
    CandidateFeatureRow (JD-independent).

    Returns float in [0, 1].
    """
    ...
```

---

### 3.17 `src/scorer/b5_cert.py`

```python
def cert_bonus(
    feature_row: CandidateFeatureRow,
    jd_vector: np.ndarray,
    model: SentenceTransformer,
    current_year: int = 2026,
) -> float:
    """
    For each cert in feature_row.cert_records:
        relevance_norm = rescaled cosine_sim(cert_embedding, jd_vector)
        recency_weight = max(0.5, 1.0 - (current_year - issue_year) * 0.10)
        contribution = relevance_norm * recency_weight

    Returns min(max(contributions), 0.10).
    Returns 0.0 if no certs.
    """
    ...
```

---

### 3.18 `src/scorer/composite.py`

```python
def composite_score(scores: ScoreBreakdown) -> float:
    """
    Computes final composite score:
        b1*0.35 + b2*0.25 + b3*0.15 + b4*0.20 + b5*0.05

    Weights are read from the scores object's source settings at call time.
    Returns float (practical range 0.0–0.955 given B5 cap).
    """
    ...

def score_candidate(
    candidate_id: str,
    feature_row: CandidateFeatureRow,
    candidate_vector: np.ndarray,
    jd_vector: np.ndarray,
    jd_intent: JDIntent,
    model: SentenceTransformer,
    settings: Settings,
) -> ScoreBreakdown:
    """
    Convenience wrapper that computes B1–B5 and composite in one call.
    This is the primary entry point used by the query pipeline's scoring loop.

    Returns ScoreBreakdown with all sub-scores and composite populated.
    """
    ...
```

---

### 3.19 `src/ranker/ranker.py`

```python
def rank_candidates(
    score_breakdowns: list[ScoreBreakdown],
    top_n: int = 20,
) -> list[ScoreBreakdown]:
    """
    Sorts candidates by composite_score descending.
    Tiebreaker: higher platform_score wins among candidates within 0.001
    of each other on composite.

    Returns the top_n ScoreBreakdown objects with rank order implied by list position.
    """
    ...
```

---

### 3.20 `src/explainer/explainer.py`

```python
def generate_explanations(
    top_candidates: list[ScoreBreakdown],
    feature_store: dict[str, CandidateFeatureRow],
    profile_store: dict[str, CandidateProfile],
    jd_intent: JDIntent,
    client: groq.Groq,
    settings: Settings,
) -> list[CandidateExplanation]:
    """
    Batches top-N candidates (default 4–5 per call) and calls Groq llama-3.3-70b-versatile
    to generate structured explanations grounded in computed sub-scores and
    actual candidate field values.

    Each explanation is post-validated by grounding_validator.
    Failed validations trigger one re-generation attempt, then fall back
    to a structured field summary so no candidate's slot is silently dropped.

    Args:
        top_candidates:  Ranked list of ScoreBreakdown objects (top 20).
        feature_store:   Feature rows keyed by candidate_id.
        profile_store:   Original profiles keyed by candidate_id (for raw field values).
        jd_intent:       Parsed JD intent for context injection into prompts.
        client:          Groq client.
        settings:        Config with batch size, max_tokens, model name.

    Returns:
        List of CandidateExplanation objects, one per top candidate.
        Preserves input order.

    Side effects:
        Logs INFO per batch: candidate IDs, token usage, latency.
        Logs WARNING for each explanation that failed grounding and used fallback.
    """
    ...
```

---

### 3.21 `src/explainer/grounding_validator.py`

```python
def validate_grounding(
    explanation: CandidateExplanation,
    feature_row: CandidateFeatureRow,
    profile: CandidateProfile,
) -> bool:
    """
    Verifies the explanation references at least one concrete datum from the
    candidate's actual profile. Accepted evidence types: skill names, company names,
    role titles, tenure numbers, cert names, or platform metric values.

    Returns True if grounded; False if generic/hallucinated.
    """
    ...

def build_fallback_explanation(
    score: ScoreBreakdown,
    feature_row: CandidateFeatureRow,
    profile: CandidateProfile,
    jd_intent: JDIntent,
) -> CandidateExplanation:
    """
    Template-populated fallback when LLM explanation fails grounding twice.
    Uses actual field values directly — never invents. Sets grounding_validated=False.
    Used only as a last resort; the pipeline never silently drops a top-20 slot.
    """
    ...
```

---

### 3.22 `src/output_writer/writer.py`

```python
import csv
import json
from pathlib import Path

def write_ranked_output(
    ranked_results: list[RankedResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Writes the final deliverable in both CSV and JSON formats.

    CSV columns (in order):
        rank, candidate_id, composite_score, semantic_score, trajectory_score,
        stability_score, platform_score, cert_bonus,
        match_summary, skill_alignment, seniority_assessment,
        trajectory_signal, platform_summary, flags, grounding_validated

    JSON: list of RankedResult objects serialized with dataclasses.asdict().

    Returns:
        Tuple of (csv_path, json_path).

    Side effects:
        Logs INFO with output file paths and row count written.
    """
    ...
```

---

### 3.23 `src/pipeline/indexing_pipeline.py`

```python
def run_indexing(
    jsonl_path: Path,
    processed_dir: Path,
    settings: Settings,
) -> None:
    """
    Offline pipeline. Runs once (or when dataset changes).
    Steps:
        1. Load candidates from JSONL.
        2. Build candidate documents and extract structured features.
        3. Batch encode documents → candidate_vectors.npy.
        4. Build FAISS flat index → candidate_index.faiss.
        5. Write candidate_features.parquet.
        6. Write candidate_id_map.json (candidate_id → row_index).

    Idempotent: re-running overwrites existing processed files.
    """
    ...
```

---

### 3.24 `src/pipeline/query_pipeline.py`

```python
def run_query(
    jd_text: str,
    processed_dir: Path,
    output_dir: Path,
    settings: Settings,
    groq_client: groq.Groq,
) -> list[RankedResult]:
    """
    Online query pipeline. Runs once per JD.
    Steps:
        1. parse_job_description → JDIntent (1 LLM call).
        2. Encode JDIntent as query vector.
        3. ANN retrieval → shortlist of ~500 candidate_ids.
        4. Hard filters (applied only if JDIntent requires them).
        5. Load feature rows from parquet for shortlisted candidates.
        6. Score all shortlisted candidates (B1–B5 + composite).
        7. Rank + select top 20.
        8. Generate explanations for top 20 (batched LLM calls).
        9. Write ranked_output.csv and ranked_output.json.

    Returns list of RankedResult for the top 20.

    Logs INFO timing for each stage. Total latency target: < 60 seconds.
    """
    ...
```

---

## 4. Configuration Management

All tunable parameters are kept in `config/settings.yaml` and loaded into a typed `Settings` dataclass at startup. Nothing that a researcher or judge might want to tweak is hardcoded inside a module.

### 4.1 `config/settings.yaml`

```yaml
# ── Model configuration ──────────────────────────────────────────────────────
embedding:
  model_name: "BAAI/bge-small-en-v1.5"
  fallback_model_name: "all-MiniLM-L6-v2"
  batch_size: 256
  embedding_dim: 384

llm:
  model: "llama-3.3-70b-versatile"
  max_tokens: 4000
  explanation_batch_size: 4 # candidates per explanation API call

# ── Retrieval configuration ───────────────────────────────────────────────────
retrieval:
  top_n_shortlist: 500 # ANN retrieval shortlist size
  top_n_output: 20 # Final ranked output size

# ── Scoring weights (must sum to 1.0 across B1–B4; B5 is additive) ───────────
scoring_weights:
  semantic: 0.35 # B1
  trajectory: 0.25 # B2
  stability: 0.15 # B3
  platform: 0.20 # B4
  cert_bonus_multiplier: 0.05 # B5 effective weight in composite formula

# ── Sub-score formula parameters ─────────────────────────────────────────────
skill_strength:
  proficiency_weight: 0.50
  duration_weight: 0.35
  endorsement_weight: 0.15
  max_duration_years: 5.0 # Normalization ceiling for duration
  max_endorsements: 50 # Normalization ceiling for endorsements

trajectory:
  # Seniority level constants — do not change without updating prompt templates
  seniority_levels:
    intern: 0.1
    trainee: 0.1
    junior: 0.2
    associate: 0.2
    entry: 0.2
    graduate: 0.2
    mid: 0.5 # Default: no keyword present
    senior: 0.75
    lead: 0.75
    manager: 0.75
    principal: 1.0
    staff: 1.0
    director: 1.0
    vp: 1.0
    head: 1.0
    chief: 1.0
  stretch_readiness:
    min_promotion_rate: 0.5
    min_experience_years: 5.0
    fit_override_value: 0.75

stability:
  strong_tenure_months: 36 # avg tenure above this → tenure_norm = 1.0
  hopping_penalty: 0.30
  consecutive_short_tenure_threshold_months: 12
  consecutive_short_tenure_count: 3
  edu_bonus:
    tier_1: 0.05
    tier_2: 0.03
    tier_3: 0.01
    tier_4: 0.00

platform:
  passive_open_to_work_score: 0.40 # Weight for open_to_work=False (not 0)
  max_applications_norm: 10
  max_search_appearances_norm: 200
  max_response_time_norm_hrs: 200
  github_activity_max: 96.9
  max_endorsements_norm: 100

cert_bonus:
  max_bonus: 0.10
  recency_decay_per_year: 0.10
  recency_floor: 0.50

# ── Embedding similarity thresholds ──────────────────────────────────────────
thresholds:
  semantic_fallback_discount: 0.60 # Discount for semantic-fallback skill matching
  domain_relevance_min_cosine: 0.60 # Minimum cosine to count a role as domain-relevant (B2)
  thin_profile_char_limit: 50 # Candidate doc shorter than this → thin_profile=True
  thin_profile_description_min: 20 # Role description shorter than this → use title fallback
  thin_profile_semantic_cap: 0.55 # Max semantic_score for thin profiles

# ── Ranking ───────────────────────────────────────────────────────────────────
ranking:
  tiebreaker_composite_tolerance: 0.001 # Scores within this range → tiebreak on platform_score

# ── Paths (overridden by CLI args) ────────────────────────────────────────────
paths:
  raw_data: "data/raw/candidates.jsonl"
  processed_dir: "data/processed"
  output_dir: "data/outputs"

# ── Logging ───────────────────────────────────────────────────────────────────
logging:
  level: "INFO" # DEBUG | INFO | WARNING | ERROR
  log_file: "logs/pipeline.log" # null to disable file logging
```

### 4.2 `Settings` Dataclass

```python
# src/config.py
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # Populated from settings.yaml; accessed as settings.scoring_weights.semantic
    # ... (mirrors the yaml structure above as nested dataclasses)

    @classmethod
    def from_yaml(cls, path: Path = Path("config/settings.yaml")) -> "Settings":
        """Loads and validates settings from YAML. Raises on missing required fields."""
        ...

    def override(self, **kwargs) -> "Settings":
        """
        Returns a new Settings with specific fields overridden.
        Used by tests to inject custom weights without touching the yaml file.
        """
        ...
```

The `Settings` object is instantiated once at startup and injected into every module that needs it. No module reads `settings.yaml` directly — they receive the `Settings` object via their function signatures. This makes unit tests trivially able to override weights without file I/O.

---

## 5. End-to-End CLI Usage

### 5.1 `main.py` CLI Interface

```
Usage:
  python main.py index  [--data PATH] [--processed PATH] [--config PATH]
  python main.py query  [--jd PATH]   [--processed PATH] [--output PATH] [--config PATH]
  python main.py run    [--data PATH] [--jd PATH] [--output PATH] [--config PATH]

Commands:
  index    Run the offline indexing pipeline (embed + feature extract + build FAISS index).
           Must complete before any query runs. Re-run only if the candidate dataset changes.

  query    Run the online query pipeline given a pre-built index.
           Parses the JD, retrieves, scores, ranks, explains, and writes output.

  run      Convenience alias: runs index then query in sequence.
           Useful for first-time setup or hackathon demo.

Options:
  --data PATH        Path to candidates.jsonl           [default: data/raw/candidates.jsonl]
  --jd PATH          Path to a plain-text JD file       [default: data/jd.txt]
  --processed PATH   Directory for index artifacts      [default: data/processed/]
  --output PATH      Directory for ranked output files  [default: data/outputs/]
  --config PATH      Path to settings.yaml              [default: config/settings.yaml]
```

### 5.2 Full Demo Walkthrough

**Step 0 — Install dependencies (once)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:

```
groq>=1.4.0
sentence-transformers>=2.6.0
faiss-cpu>=1.8.0
pandas>=2.1.0
pyarrow>=14.0.0
numpy>=1.26.0
pyyaml>=6.0.1
```

**Step 1 — Set environment variable**

```bash
export GROQ_API_KEY= gsk_placeholder_key_abc123
```

**Step 2 — Build the candidate index (runs once, ~10–20 min for 100K on CPU)**

```bash
python main.py index \
  --data data/raw/candidates.jsonl \
  --processed data/processed/
```

Expected console output:

```
[INFO] Loading candidates from data/raw/candidates.jsonl
[INFO] 100000 valid profiles loaded, 0 skipped
[INFO] Building candidate documents...
[INFO] Encoding batch 1/391 (256 docs)...
...
[INFO] Encoding complete: 100000 vectors (384-dim) in 847s
[INFO] Building FAISS flat index...
[INFO] Index built: 100000 vectors, exact L2, normalized cosine
[INFO] Extracting structured features for all candidates...
[INFO] Features written to data/processed/candidate_features.parquet
[INFO] Index written to data/processed/candidate_index.faiss
[INFO] Vectors written to data/processed/candidate_vectors.npy
[INFO] ID map written to data/processed/candidate_id_map.json
[INFO] Indexing complete. Total wall-clock time: 913s
```

**Step 3 — Write your JD to a text file**

```bash
cat > data/jd.txt << 'EOF'
We are looking for a Senior Data Engineer to join our Data Platform team...
[paste full JD text here]
EOF
```

**Step 4 — Run the query pipeline**

```bash
python main.py query \
  --jd data/jd.txt \
  --processed data/processed/ \
  --output data/outputs/
```

Expected console output:

```
[INFO] Parsing job description...
[INFO] JD parsed in 2.1s (input tokens: 487, output tokens: 312)
[INFO] JDIntent written to data/outputs/jd_intent.json
[INFO] Encoding JD query vector...
[INFO] Running ANN retrieval (top 500)...
[INFO] Retrieved 500 candidates in 0.3s
[INFO] Applying hard filters: salary_stated=False, location_hard=False → 0 removed
[INFO] Scoring 500 candidates on B1–B5...
[INFO] Scoring complete in 8.4s
[INFO] Ranked 500 candidates; selecting top 20
[INFO] Generating explanations for top 20 (batch size 4)...
[INFO] Batch 1/5 (candidates 1–4) complete in 6.2s
[INFO] Batch 2/5 (candidates 5–8) complete in 5.9s
[INFO] Batch 3/5 (candidates 9–12) complete in 6.1s
[INFO] Batch 4/5 (candidates 13–16) complete in 5.8s
[INFO] Batch 5/5 (candidates 17–20) complete in 6.3s
[INFO] All 20 explanations passed grounding validation
[INFO] Results written to data/outputs/ranked_output.csv
[INFO] Results written to data/outputs/ranked_output.json
[INFO] Total query pipeline time: 43.1s
```

**Step 5 — Inspect outputs**

```bash
# Review the parsed JD intent (confirm system understood the JD correctly)
cat data/outputs/jd_intent.json

# Preview the top 5 results
python -c "
import pandas as pd
df = pd.read_csv('data/outputs/ranked_output.csv')
print(df[['rank','candidate_id','composite_score','semantic_score',
          'trajectory_score','platform_score','match_summary']].head(5).to_string())
"
```

**Step 6 (optional) — Run with custom weights for tuning**

Override scoring weights without editing `settings.yaml` using environment variable overrides (handled by `Settings.from_env_overrides()`):

```bash
SCORE_WEIGHT_SEMANTIC=0.40 SCORE_WEIGHT_TRAJECTORY=0.20 \
python main.py query --jd data/jd.txt --processed data/processed/ --output data/outputs/
```

---

## 6. Logging & Error-Handling Conventions

### 6.1 Logging Setup

All modules use Python's standard `logging` library. A single logger per module is created at module load time:

```python
# At the top of every src/ module:
import logging
logger = logging.getLogger(__name__)
# Module name becomes the logger name: e.g. "src.scorer.b1_semantic"
```

The root logger is configured once in `main.py` before any module is imported:

```python
import logging
import sys
from pathlib import Path

def configure_logging(settings: Settings) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if settings.logging.log_file:
        Path(settings.logging.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.logging.log_file))

    logging.basicConfig(
        level=settings.logging.level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
        force=True,
    )
```

**Log level conventions:**

| Level     | When to use                                                                                                                                   |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEBUG`   | Per-candidate score values during development; individual skill strength computations                                                         |
| `INFO`    | Pipeline stage start/end with timing; batch progress every 10K records; file writes; API call summaries (token counts, latency)               |
| `WARNING` | Skipped/malformed candidate records; failed grounding validation (with fallback used); JD parse retry triggered; index recall check below 90% |
| `ERROR`   | API call failure after retries; file write failure; unrecoverable schema violations                                                           |

### 6.2 Error-Handling Strategy

**Principle: partial failures should degrade gracefully, never abort silently.**

#### Malformed Candidate Records (Data Loader)

```python
# In load_candidates():
for line_number, line in enumerate(f, start=1):
    try:
        raw = json.loads(line)
        is_valid, reason = validate_candidate(raw)
        if not is_valid:
            logger.warning(
                "Skipping candidate at line %d: %s", line_number, reason
            )
            skip_count += 1
            continue
        yield _parse_profile(raw)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON at line %d: %s", line_number, str(e))
        skip_count += 1
```

At end of file: `logger.info("Loaded %d profiles, skipped %d records", valid_count, skip_count)`.

If `skip_count / total > 0.10` (more than 10% of records invalid), raise `DataLoadError` — this threshold indicates a data format problem, not isolated bad records.

#### JD Parse Failure

```python
# In parse_job_description():
try:
    raw_response = _call_llm(prompt, client, settings)
    is_valid, error = _validate_jd_intent(raw_response)
    if not is_valid:
        logger.warning("JD parse failed validation (%s). Retrying with corrective prompt.", error)
        raw_response = _call_llm(
            _build_corrective_prompt(prompt, raw_response, error),
            client, settings
        )
        is_valid, error = _validate_jd_intent(raw_response)
        if not is_valid:
            raise JDParseError(f"JD parse failed after retry: {error}")
    return _build_jd_intent(raw_response)
except Groq.APIError as e:
    raise JDParseError(f"Groq API error during JD parsing: {e}") from e
```

#### Scoring Failures (Individual Candidates)

```python
# In the scoring loop in query_pipeline.py:
scored = []
for cand_id in shortlist:
    try:
        breakdown = score_candidate(cand_id, ...)
        scored.append(breakdown)
    except Exception as e:
        logger.warning(
            "Scoring failed for candidate %s: %s. Skipping.", cand_id, str(e)
        )
        # Individual candidate failure never stops the loop
```

If fewer than `top_n_output` (20) candidates score successfully, the pipeline logs an `ERROR` and writes however many did succeed, never returning a zero-result file.

#### Explanation Generation Failures

```python
# In generate_explanations():
explanations = []
for batch in batches:
    try:
        batch_results = _call_explanation_llm(batch, jd_intent, client, settings)
        for result in batch_results:
            if validate_grounding(result, ...):
                explanations.append(result)
            else:
                logger.warning("Grounding failed for %s; retrying.", result.candidate_id)
                retry_result = _call_explanation_llm_single(result.candidate_id, ...)
                if validate_grounding(retry_result, ...):
                    explanations.append(retry_result)
                else:
                    logger.warning("Grounding retry failed for %s; using fallback.", result.candidate_id)
                    explanations.append(build_fallback_explanation(...))
    except groq.APIError as e:
        logger.error("Explanation API call failed for batch: %s. Using fallbacks.", str(e))
        for cand in batch:
            explanations.append(build_fallback_explanation(cand, ...))
```

This guarantees the output always has 20 rows — even if some explanations are fallback-template rather than LLM-generated.

#### FAISS Index Missing

```python
# In load_index():
if not (processed_dir / "candidate_index.faiss").exists():
    raise IndexNotFoundError(
        f"No FAISS index found at {processed_dir}. "
        "Run `python main.py index` first to build the candidate index."
    )
```

### 6.3 Custom Exceptions

```python
# src/exceptions.py

class RedrobPipelineError(Exception):
    """Base exception for all pipeline errors."""

class DataLoadError(RedrobPipelineError):
    """Raised when >10% of candidate records fail validation."""

class JDParseError(RedrobPipelineError):
    """Raised when JD parsing fails after retry."""

class IndexBuildError(RedrobPipelineError):
    """Raised when <90% of candidates are successfully indexed."""

class IndexNotFoundError(RedrobPipelineError):
    """Raised when the query pipeline cannot find a pre-built index."""

class ScoringError(RedrobPipelineError):
    """Raised when the shortlist produces fewer than top_n_output scored candidates."""
```

All exceptions inherit from `RedrobPipelineError` so callers can catch the base class for generic handling while still differentiating specific failure types when needed.

---

_Document owner: Engineering lead_
_Last updated: June 2026_
_Status: Approved — ready for implementation_
