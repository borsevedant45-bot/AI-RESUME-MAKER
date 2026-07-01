# Ranking, Scoring & Explainability Methodology

## Redrob Intelligent Candidate Discovery & Ranking Engine

**Document:** `docs/04_Ranking_Scoring_Explainability_Methodology.md`
**Version:** 1.0
**Inputs:** `01_PRD.md`, `02_Data_Understanding_Report.md`, `03_System_Architecture_Document.md`
**Status:** Approved — primary algorithmic reference for build phase

---

## Table of Contents

1. [JD Understanding — Structured Intent Extraction](#1-jd-understanding--structured-intent-extraction)
2. [Candidate Representation — Profile-to-Vector Conversion](#2-candidate-representation--profile-to-vector-conversion)
3. [Multi-Factor Scoring — Formulas, Weights, and Rationale](#3-multi-factor-scoring--formulas-weights-and-rationale)
4. [Retrieval & Ranking — From 100K to the Final 20](#4-retrieval--ranking--from-100k-to-the-final-20)
5. [Explainability Generation — Grounded Candidate Justifications](#5-explainability-generation--grounded-candidate-justifications)
6. [Edge Cases — Thin Profiles, Keyword-Stuffers, and Passive Candidates](#6-edge-cases--thin-profiles-keyword-stuffers-and-passive-candidates)
7. [End-to-End Worked Example](#7-end-to-end-worked-example)

---

## 1. JD Understanding — Structured Intent Extraction

### 1.1 The Problem with Raw JD Text

A raw Job Description is written for _humans_ — it mixes formal requirements with aspirational language, buries the most important needs inside paragraphs of company boilerplate, and expresses seniority expectations implicitly rather than explicitly ("own the entire data infrastructure" says senior+ without using the word). Feeding this raw text directly into an embedding model and calling it "JD understanding" is a category error: the embedding will match surface language, not intent.

The JD parser converts unstructured prose into a structured intent object that every downstream module can consume uniformly. This is the system's single point of open-ended LLM reasoning — it happens once per query, costs one API call, and its output becomes the fixed, inspectable anchor for all subsequent deterministic computation.

### 1.2 Extraction Strategy — Structured LLM Prompting

The parser calls llama-3.3-70b-versatile with a carefully constrained prompt that requires a JSON response conforming to a fixed schema. It does not ask the model to "summarize the JD" or "list the skills" — it asks the model to populate named fields in a defined structure, which dramatically reduces the surface area for hallucination and output variance.

**Extraction dimensions:**

| Field                               | What the LLM Must Determine                                                                                                                               | Example                                                                                                    |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `seniority_level`                   | Float 0–1 on the `junior(0.2) → mid(0.5) → senior(0.75) → staff_plus(1.0)` scale                                                                          | "Lead a team of 4 engineers" → 0.75                                                                        |
| `seniority_evidence`                | The exact phrase from the JD that justified the seniority inference                                                                                       | "lead a team of 4 engineers"                                                                               |
| `must_have_skills`                  | Array of strings: skills/tools/platforms with no wiggle room                                                                                              | `["Python", "Kubernetes", "AWS"]`                                                                          |
| `nice_to_have_skills`               | Array of strings: mentioned but not gating                                                                                                                | `["Rust", "GCP"]`                                                                                          |
| `core_problems_to_solve`            | 2–4 sentence description of the actual work — what problem space this role operates in                                                                    | "Build and operate real-time ML inference pipelines on AWS…"                                               |
| `implicit_soft_skills`              | Named soft skills extracted from behavioral language, not explicit statements                                                                             | "own the roadmap" → `ownership`; "drive alignment across PMs and engineers" → `cross-functional influence` |
| `domain_tags`                       | 1–3 tags from a controlled vocabulary (e.g., `data-engineering`, `frontend`, `hr`, `finance`, `devops`)                                                   | `["data-engineering", "mlops"]`                                                                            |
| `requires_technical_github_signals` | Boolean: is this a software/engineering/data role where GitHub activity is a meaningful signal?                                                           | `true` for ML Eng; `false` for HR Manager                                                                  |
| `work_context`                      | Object capturing any explicitly stated work-mode (`remote`/`hybrid`/`onsite`), salary range (if stated in LPA), location requirement (if stated and hard) | `{"work_mode": "hybrid", "salary_max_lpa": 30}`                                                            |
| `salary_stated`                     | Boolean: whether the JD explicitly states a salary range                                                                                                  | `false`                                                                                                    |

### 1.3 The Extraction Prompt

```
System:
You are a senior technical recruiter with 10 years of experience parsing Job Descriptions.
Your task is to extract structured information from a Job Description and return it as
valid JSON only — no preamble, no explanation, no markdown fences.
Return exactly the JSON schema defined below.

Schema:
{
  "seniority_level": <float 0.0–1.0: 0.2=junior, 0.5=mid, 0.75=senior, 1.0=staff_plus>,
  "seniority_evidence": <string: the exact JD phrase that determined seniority>,
  "must_have_skills": [<string>, ...],
  "nice_to_have_skills": [<string>, ...],
  "core_problems_to_solve": <string: 2–4 sentence summary of actual work and domain>,
  "implicit_soft_skills": [<string>, ...],
  "domain_tags": [<string from: data-engineering, frontend, backend, devops, mlops,
                              finance, hr, operations, content, sales, mechanical,
                              civil, project-management>],
  "requires_technical_github_signals": <boolean>,
  "work_context": {
    "work_mode": <"remote"|"hybrid"|"onsite"|"flexible"|null>,
    "location_required": <string|null>,
    "location_is_hard_requirement": <boolean>,
    "salary_min_lpa": <int|null>,
    "salary_max_lpa": <int|null>
  },
  "salary_stated": <boolean>
}

Rules:
- Extract only what is in the JD. Do not invent requirements.
- For seniority, infer from behavioral language if the level is not stated explicitly:
  "lead a team" → senior; "own the product roadmap" → senior;
  "report to the CTO" alone does not imply seniority.
- For must-have vs. nice-to-have: if the JD says "required", "must have", "essential",
  or lists skills without qualification in a "requirements" section → must_have.
  If the JD says "preferred", "nice to have", "bonus", "plus" → nice_to_have.
  If ambiguous, classify as must_have and let the scorer handle weighting.
- For implicit soft skills, look for ownership language ("own", "drive", "accountable"),
  leadership language ("lead", "manage", "mentor"), and collaboration language
  ("partner", "align", "influence").

User:
Parse the following Job Description and return the JSON object:

{JD_TEXT}
```

### 1.4 Post-Parse Validation

Before `jd_intent.json` is consumed by the rest of the pipeline, a lightweight validation function checks:

- All required fields are present and typed correctly.
- `seniority_level` is in the allowed set `{0.2, 0.5, 0.75, 1.0}` — if the LLM returned 0.6 (not in the scale), round to nearest.
- `must_have_skills` is non-empty (a JD with no must-haves is almost certainly a parse failure).
- If validation fails, the parser retries once with a corrective prompt that includes the failed output and the specific error.

The validated `jd_intent.json` is written to disk and logged. A recruiter (or a judge) can read it directly to confirm the system understood the JD correctly before any candidate ranking occurs.

---

## 2. Candidate Representation — Profile-to-Vector Conversion

### 2.1 Two Parallel Representations Per Candidate

Each candidate is represented in two forms that live in separate stores, both keyed by `candidate_id`:

1. **A dense embedding vector** — captures semantic meaning of the professional narrative; used for ANN retrieval in the first filtering stage.
2. **A structured feature row** — numeric and categorical signals used by the deterministic scoring formulas (B1–B5); used in the full scoring stage.

Both are pre-computed once at index-build time (offline). The online query pipeline never re-reads raw JSONL for 100K candidates.

### 2.2 Building the Candidate Document (for Embedding)

A single "candidate document" string is constructed per profile by concatenating fields in a defined order. The concatenation template is chosen to give the embedding model the richest possible signal without padding it with low-signal fields (location, company names) that add noise.

```python
def build_candidate_doc(profile: dict) -> str:
    # 1. Skills — proficiency-annotated for richer semantic signal
    skill_parts = [
        f"{s['name']} ({s['proficiency']})"
        for s in profile.get("skills", [])
    ]
    skills_str = "Skills: " + ", ".join(skill_parts) if skill_parts else ""

    # 2. Career history — title + industry context + description for each role,
    #    most recent roles first (recency bias in concatenation)
    role_parts = []
    for role in sorted(profile.get("career_history", []),
                       key=lambda r: r.get("start_date", ""), reverse=True):
        role_str = (
            f"{role.get('title', '')} in {role.get('industry', '')}: "
            f"{role.get('description', '').strip()}"
        )
        role_parts.append(role_str)
    career_str = "Career: " + " | ".join(role_parts) if role_parts else ""

    # 3. Certifications — name only; brief but semantically informative
    cert_parts = [c["name"] for c in profile.get("certifications", [])]
    certs_str = "Certifications: " + ", ".join(cert_parts) if cert_parts else ""

    # 4. Education field of study — adds domain signal (e.g., "Data Science")
    edu_parts = [
        e.get("field_of_study", "")
        for e in profile.get("education", [])
        if e.get("field_of_study")
    ]
    edu_str = "Education: " + ", ".join(edu_parts) if edu_parts else ""

    return " | ".join(filter(None, [skills_str, career_str, certs_str, edu_str]))
```

**What is deliberately excluded from the embedding:** `candidate_id`, `country`, `location`, `current_company` (fictional names add noise), raw numeric signals from `redrob_signals` (these go into structured features), and `languages` (zero variance in this dataset).

**Handling thin profiles (Flag 7 mitigation):** If `career_history` descriptions are empty or under 20 characters after stripping whitespace, the builder falls back to concatenating `title + industry` for that role to preserve at least the role-level domain signal. Profiles that remain under 50 characters total after construction are flagged as `thin_profile=True` in the feature store — this flag is used in edge-case handling (see §6).

### 2.3 Embedding Computation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Batch encode all 100K documents
docs = [build_candidate_doc(p) for p in profiles]
embeddings = model.encode(docs, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True)  # normalize for cosine similarity
# Shape: (100000, 384)

# Persist to disk
import numpy as np
np.save("candidate_vectors.npy", embeddings)
```

Normalization at encode time means dot product and cosine similarity are equivalent — this simplifies the FAISS index configuration (L2 on normalized vectors equals cosine similarity).

### 2.4 Building the Structured Feature Row (for Scoring)

In parallel with embedding, the offline indexer computes every numeric feature defined in the Data Understanding Report §3. The result is a Parquet file with one row per candidate and one column per feature.

**Features computed offline (JD-independent):**

| Feature                  | Type      | Note                                                       |
| ------------------------ | --------- | ---------------------------------------------------------- |
| `avg_tenure_months`      | Float     | Mean of `career_history[].duration_months`                 |
| `job_hopping_flag`       | Int (0/1) | ≥3 consecutive roles < 12 months                           |
| `promotion_rate`         | Float     | Detected promotions / eligible companies                   |
| `latest_seniority`       | Float     | Seniority keyword map on most recent title                 |
| `trajectory_score`       | Float     | Composite (see §3.2) — JD-independent component            |
| `active_intent_score`    | Float     | From `redrob_signals` (see §3.4)                           |
| `hire_reliability_score` | Float     | From `redrob_signals` (see §3.4)                           |
| `github_activity_score`  | Float     | Raw signal from `redrob_signals`                           |
| `endorsements_received`  | Int       | Raw signal from `redrob_signals`                           |
| `skill_records`          | JSON blob | Dict of `{skill_name: skill_strength_score}` per candidate |
| `cert_records`           | JSON blob | List of `{name, issue_year}` per candidate                 |
| `open_to_work`           | Bool      | Raw flag                                                   |
| `willing_to_relocate`    | Bool      | Raw flag                                                   |
| `work_mode_preference`   | Str       | Raw categorical                                            |
| `notice_period_days`     | Int       | Raw signal                                                 |
| `expected_salary_min`    | Int       | LPA                                                        |
| `expected_salary_max`    | Int       | LPA                                                        |
| `thin_profile`           | Bool      | Set by build_candidate_doc fallback                        |
| `institution_tier`       | Str       | Best tier across all education entries                     |

**Features computed online (JD-dependent):**

The following cannot be pre-computed because they depend on the JD intent object:

- `jd_skill_coverage_score` — which JD skills a candidate covers (requires the JD skill list)
- `seniority_fit_score` — gap between `latest_seniority` and `jd_seniority` (requires the JD seniority)
- `domain_experience_years` — years in JD-relevant domains (requires the JD embedding)
- `semantic_score` (B1) — cosine similarity between candidate vector and JD vector
- `cert_bonus` (B5) — relevance of certs to this specific JD

These are computed per-candidate in the scoring stage after ANN retrieval.

---

## 3. Multi-Factor Scoring — Formulas, Weights, and Rationale

The scoring engine applies five independent scoring functions (B1–B5) to each candidate in the ~500-candidate shortlist. Every function is a pure function of the candidate's feature row and the `jd_intent` object — no global state, no cross-candidate comparisons. This is what makes the pipeline reproducible and independently testable.

### 3.1 B1 — Semantic Score (Weight: 0.35)

The semantic score combines two signals: a vector-level similarity score that captures overall professional relevance, and a skill-level coverage score that explicitly checks whether the JD's required skills appear (or their semantic equivalents appear) in the candidate's profile.

```python
def semantic_score(
    candidate_vector: np.ndarray,   # pre-computed, normalized
    jd_vector: np.ndarray,          # computed at query time
    skill_records: dict,            # {skill_name: strength_score}
    jd_intent: dict
) -> float:

    # Component A: Overall embedding cosine similarity
    # (dot product of normalized vectors = cosine similarity)
    embed_sim = float(np.dot(candidate_vector, jd_vector))
    # Range: [-1, 1] but practically [0.2, 0.95] for relevant candidates
    embed_sim_norm = (embed_sim + 1) / 2  # rescale to [0, 1]

    # Component B: JD skill coverage
    must_have = jd_intent["must_have_skills"]
    nice_to_have = jd_intent["nice_to_have_skills"]

    must_have_scores = []
    for skill_name in must_have:
        # Direct match first
        direct = skill_records.get(skill_name.lower(), None)
        if direct is not None:
            must_have_scores.append(direct)
        else:
            # Semantic fallback: cosine similarity between skill term
            # embedding and candidate vector — allows "Terraform" to
            # partially match a candidate with "Infrastructure as Code"
            skill_vec = model.encode(skill_name, normalize_embeddings=True)
            sem_fallback = float(np.dot(candidate_vector, skill_vec))
            must_have_scores.append(max(0, (sem_fallback + 1) / 2 * 0.6))
            # Semantic fallback is discounted (0.6x) — partial credit only

    # Nice-to-have skills contribute at 40% of the weight of must-haves
    nice_scores = []
    for skill_name in nice_to_have:
        direct = skill_records.get(skill_name.lower(), None)
        if direct is not None:
            nice_scores.append(direct * 0.4)
        else:
            skill_vec = model.encode(skill_name, normalize_embeddings=True)
            sem_fallback = float(np.dot(candidate_vector, skill_vec))
            nice_scores.append(max(0, (sem_fallback + 1) / 2 * 0.6 * 0.4))

    all_skill_scores = must_have_scores + nice_scores
    coverage_score = np.mean(all_skill_scores) if all_skill_scores else embed_sim_norm

    # Final B1: embed_sim dominates (60%), skill coverage adds specificity (40%)
    b1 = embed_sim_norm * 0.60 + coverage_score * 0.40
    return float(np.clip(b1, 0, 1))
```

**Why 0.35 weight:** Semantic fit is the system's _core differentiator_ over keyword ATS. It is the one score that can detect "AWS + GCP + Terraform" → "Cloud Expert" and "Built scalable web applications using React" → "React expertise." Giving it the highest weight (0.35) ensures that a candidate who genuinely matches the JD's domain rises above one who has ticked adjacent boxes in adjacent domains.

**Why 60/40 embed vs. coverage split:** Pure embedding similarity is a blunt instrument — it rewards candidates who are broadly in the right neighborhood but may not have the specific must-have skills. The coverage sub-score adds targeted precision. The 60/40 split gives the broad neighborhood signal primacy (it sees the full profile, not just skills) while letting the coverage check penalize gaps in specific required tools.

---

### 3.2 B2 — Trajectory Score (Weight: 0.25)

The trajectory score evaluates seniority fit and career growth signal simultaneously. The key design goal is "stretch readiness" detection: a mid-level engineer with consistent promotions should rank comparably for a senior role to a senior engineer with a flat career arc.

```python
# --- Seniority keyword map (used during offline feature extraction) ---
SENIORITY_MAP = {
    "intern": 0.1, "trainee": 0.1, "junior": 0.2, "associate": 0.2,
    "entry": 0.2, "graduate": 0.2,
    # mid-level: no qualifier → 0.5
    "senior": 0.75, "lead": 0.75, "principal": 1.0, "staff": 1.0,
    "manager": 0.75, "director": 1.0, "vp": 1.0, "head": 1.0,
    "chief": 1.0, "cto": 1.0, "ceo": 1.0
}

def latest_seniority(career_history: list) -> float:
    """Return seniority level from the most recent role title."""
    if not career_history:
        return 0.5  # default to mid if no history
    latest_title = sorted(career_history,
                          key=lambda r: r.get("start_date", ""),
                          reverse=True)[0].get("title", "").lower()
    for keyword, level in SENIORITY_MAP.items():
        if keyword in latest_title:
            return level
    return 0.5  # no seniority keyword → treat as mid-level

def promotion_rate(career_history: list) -> float:
    """Fraction of companies where a title seniority increase was detected."""
    companies = {}
    for role in career_history:
        co = role.get("company", "unknown")
        companies.setdefault(co, []).append(role)

    promotions = 0
    eligible = 0
    for co, roles in companies.items():
        if len(roles) < 2:
            continue
        eligible += 1
        sorted_roles = sorted(roles, key=lambda r: r.get("start_date", ""))
        levels = [latest_seniority([r]) for r in sorted_roles]
        if levels[-1] > levels[0]:
            promotions += 1

    return promotions / eligible if eligible > 0 else 0.0


def trajectory_score(candidate_features: dict, jd_intent: dict) -> float:
    cand_seniority = candidate_features["latest_seniority"]
    jd_seniority = jd_intent["seniority_level"]
    promo_rate = candidate_features["promotion_rate"]
    exp_years = candidate_features.get("experience_years", 5.0)
    traj = candidate_features.get("trajectory_score_base", None)

    # -- Seniority fit: penalizes both under- AND over-qualification --
    gap = abs(cand_seniority - jd_seniority)
    if gap == 0.0:
        seniority_fit = 1.0
    elif gap <= 0.25:
        seniority_fit = 0.80
    elif gap <= 0.50:
        seniority_fit = 0.50
    else:
        seniority_fit = max(0.2, 1.0 - gap * 2.0)

    # -- Stretch readiness override --
    # A mid-level (0.5) candidate applying for senior (0.75):
    # if they have promotions and solid experience, treat as near-match (0.75)
    if (cand_seniority == 0.5 and jd_seniority == 0.75
            and promo_rate >= 0.5 and exp_years >= 5):
        seniority_fit = 0.75  # stretch bonus: one sub-level penalty, not two

    # -- Trajectory momentum --
    # Captures upward velocity regardless of JD gap
    traj_momentum = (
        promo_rate * 0.50 +
        min(exp_years / 10.0, 1.0) * 0.30 +
        cand_seniority * 0.20   # absolute level as mild signal
    )

    # -- Final B2: seniority fit is primary (60%), momentum secondary (40%) --
    b2 = seniority_fit * 0.60 + traj_momentum * 0.40
    return float(np.clip(b2, 0, 1))
```

**Why 0.25 weight:** Career trajectory is the PRD's key innovation — detecting stretch readiness requires meaningful weight. It is lower than semantic fit (0.35) because a candidate who is a domain mismatch cannot be rescued by good trajectory, but higher than stability (0.15) because being ready for the role is more important than not having job-hopped.

**Overqualification penalty:** The symmetric `abs(gap)` penalty applies equally to candidates who are too senior. A VP applying for a mid-level role is a poor pipeline fit because they are unlikely to accept the offer, and even if they do, they will churn quickly. The trajectory score penalizes this rather than rewarding it.

---

### 3.3 B3 — Stability Score (Weight: 0.15)

```python
def stability_score(candidate_features: dict) -> float:
    avg_tenure = candidate_features["avg_tenure_months"]
    hopping_flag = candidate_features["job_hopping_flag"]  # 0 or 1
    edu_tier = candidate_features.get("institution_tier", "tier_3")

    # Tenure norm: 36 months (3 years) is the "strong" threshold
    # Below 12 months average is weak; 24+ is neutral; 36+ is strong
    tenure_norm = min(avg_tenure / 36.0, 1.0)

    # Job hopping: -0.30 penalty if ≥3 consecutive roles < 12 months
    hopping_penalty = 0.30 if hopping_flag else 0.0

    # Education tier: marginal signal, never a gate
    edu_bonus_map = {"tier_1": 0.05, "tier_2": 0.03, "tier_3": 0.01, "tier_4": 0.0}
    edu_bonus = edu_bonus_map.get(edu_tier, 0.0)

    b3 = min(max(tenure_norm - hopping_penalty + edu_bonus, 0.0), 1.0)
    return float(b3)
```

**Why 0.15 weight:** Stability is a genuine hiring signal — a chronically job-hopping candidate is a poor investment — but it should not gate a high-performing candidate with one anomalous short stint (a layoff, a startup that folded). At 0.15, it can influence rankings meaningfully but cannot override strong semantic and trajectory scores.

---

### 3.4 B4 — Platform Score (Weight: 0.20)

Platform signals are this dataset's unique differentiator over a traditional ATS. They answer a question no resume can: "Is this person actually reachable, serious about a move, and likely to complete the hiring funnel?"

```python
def platform_score(candidate_features: dict, jd_intent: dict) -> float:
    rs = candidate_features["redrob_signals"]  # convenience alias

    # -- Active intent: how actively is this candidate seeking? --
    open_to_work_score = 1.0 if rs["open_to_work"] else 0.4
    # Passive candidates score 0.4 (not 0.0) — they are still viable targets
    apps_norm = min(rs["applications_submitted_30d"] / 10.0, 1.0)
    completeness_norm = rs["profile_completeness_score"] / 100.0
    search_norm = min(rs["search_appearances_30d"] / 200.0, 1.0)

    active_intent = (
        open_to_work_score * 0.35 +
        apps_norm          * 0.25 +
        completeness_norm  * 0.20 +
        search_norm        * 0.20
    )

    # -- Hire reliability: will the process actually close? --
    response_speed = 1.0 - min(rs["avg_response_time_hrs"] / 200.0, 1.0)
    verif_score = (0.5 if rs["email_verified"] else 0.0 +
                   0.5 if rs["phone_verified"] else 0.0)

    hire_reliability = (
        rs["interview_completion_rate"] * 0.40 +
        rs["offer_acceptance_rate"]     * 0.30 +
        response_speed                  * 0.20 +
        verif_score                     * 0.10
    )

    # -- Technical engagement: only for engineering/data/devops JDs --
    requires_github = jd_intent.get("requires_technical_github_signals", False)
    if requires_github:
        github_norm = rs["github_activity_score"] / 96.9
        endorse_norm = min(rs["endorsements_received"] / 100.0, 1.0)
        tech_engagement = github_norm * 0.60 + endorse_norm * 0.40

        b4 = (
            active_intent    * 0.40 +
            hire_reliability * 0.35 +
            tech_engagement  * 0.25
        )
    else:
        b4 = (
            active_intent    * 0.55 +
            hire_reliability * 0.45
        )

    return float(np.clip(b4, 0, 1))
```

**Why 0.20 weight:** Platform signals are ranked second among the "soft" dimensions because they reflect a _real behavioral reality_ absent from any resume-only ATS. A semantically strong candidate who takes 200 hours to respond and has a 30% offer acceptance rate is a genuinely worse pipeline investment than one with a 2-hour response time and a 90% acceptance rate. However, at 0.20, platform signals cannot rescue a domain-mismatched candidate or override strong semantic and trajectory scores. They are the tiebreaker layer for candidates who are otherwise comparable on substance.

**The passive candidate treatment:** `open_to_work = False` scores 0.4 (not 0.0) on the intent sub-component. This is deliberate: 64.7% of candidates are passive, and the best candidates disproportionately tend not to advertise availability. A passive candidate with 200+ search appearances and a high hiring manager response rate is a high-value candidate regardless of their `open_to_work` flag.

---

### 3.5 B5 — Certification Bonus (Weight: 0.05, additive, capped at 0.10)

```python
def cert_bonus(candidate_features: dict, jd_vector: np.ndarray,
               jd_intent: dict, current_year: int = 2026) -> float:
    contributions = []
    for cert in candidate_features.get("cert_records", []):
        cert_vec = model.encode(cert["name"], normalize_embeddings=True)
        relevance = float(np.dot(cert_vec, jd_vector))
        relevance_norm = (relevance + 1) / 2  # rescale to [0, 1]

        # Recency decay: 10% per year, floor at 0.5
        years_old = current_year - cert.get("issue_year", current_year)
        recency_weight = max(0.5, 1.0 - years_old * 0.10)

        contributions.append(relevance_norm * recency_weight)

    # Take the single best cert (max), cap additive contribution at 0.10
    return float(min(max(contributions, default=0.0), 0.10))
```

**Why 0.05 weight with a 0.10 cap:** Certifications are meaningful for roles like Cloud Architect (AWS cert), Project Manager (PMP), or Finance (CFA) — but they should never be the deciding factor over five years of hands-on experience. At 0.05 weight with a 0.10 absolute cap, the maximum possible cert bonus adds 0.05 to a composite score (0.10 \* 0.05 weight), which is a meaningful tiebreaker but cannot elevate a weak candidate above a strong one.

---

### 3.6 Composite Score Formula

```python
def composite_score(b1, b2, b3, b4, b5) -> float:
    """
    Weighted sum of all five scoring dimensions.
    Maximum possible score: 0.35 + 0.25 + 0.15 + 0.20 + (0.10 * 0.05) = 0.955
    Practical range for strong candidates: 0.55 – 0.85
    """
    return (
        b1 * 0.35 +   # Semantic Skill Match — primary differentiator
        b2 * 0.25 +   # Career Trajectory & Seniority Fit
        b3 * 0.15 +   # Career Stability
        b4 * 0.20 +   # Platform Activity & Intent Signals
        b5            # Certifications: additive bonus at face value (not multiplied by 0.05)
        # Note: b5 is the raw cert bonus value (already capped at 0.10),
        # multiplied by the effective weight at composite assembly:
        # final = b1*0.35 + b2*0.25 + b3*0.15 + b4*0.20 + b5*0.05
    )
    # Explicit:
    return b1*0.35 + b2*0.25 + b3*0.15 + b4*0.20 + b5*0.05
```

**Weight rationale summary:**

| Dimension     | Weight | Primary Rationale                                                        |
| ------------- | ------ | ------------------------------------------------------------------------ |
| B1 Semantic   | 0.35   | Replaces keyword matching; the core system value proposition             |
| B2 Trajectory | 0.25   | Captures stretch readiness and seniority fit — PRD's key innovation      |
| B4 Platform   | 0.20   | This dataset's unique advantage over traditional ATS; behavioral reality |
| B3 Stability  | 0.15   | Real signal but should not gate high performers                          |
| B5 Certs      | 0.05   | Tiebreaker, not a primary ranking factor                                 |

---

## 4. Retrieval & Ranking — From 100K to the Final 20

The pipeline uses a progressively narrowing funnel. Each stage is cheaper to run than the previous one, but more expensive in LLM cost — the architecture ensures that expensive operations see only a tiny fraction of the full candidate pool.

```
100,000 candidates
     │
     ▼  [ANN Retrieval — FAISS cosine similarity]
   ~500 candidates (recall stage — optimized for not losing good candidates)
     │
     ▼  [Hard Filters — applied only if JD states explicit hard requirements]
   ~400–500 candidates (approximately; depends on JD constraints)
     │
     ▼  [B1–B5 Multi-Signal Scoring — deterministic formulas]
   ~500 candidates with composite scores
     │
     ▼  [Ranker — sort by composite_score, apply tiebreaker]
    Top 20 candidates
     │
     ▼  [Explanation Generator — LLM, batched]
    Top 20 with explanations
```

### 4.1 Stage 1 — ANN Retrieval (100K → ~500)

**Query vector construction:**

The JD is embedded using the same model used to build the candidate index, but the input is the structured `jd_intent.json` object serialized to a text representation — not the raw JD prose. This produces a cleaner, less noisy query vector because boilerplate is stripped.

```python
def build_jd_query_doc(jd_intent: dict) -> str:
    """Serialize jd_intent to a searchable text form."""
    parts = [
        "Core problems: " + jd_intent["core_problems_to_solve"],
        "Required skills: " + ", ".join(jd_intent["must_have_skills"]),
        "Nice to have: " + ", ".join(jd_intent["nice_to_have_skills"]),
        "Soft skills: " + ", ".join(jd_intent["implicit_soft_skills"]),
        "Domain: " + ", ".join(jd_intent["domain_tags"]),
    ]
    return " | ".join(parts)

jd_doc = build_jd_query_doc(jd_intent)
jd_vector = model.encode(jd_doc, normalize_embeddings=True)
```

**FAISS retrieval:**

```python
import faiss

# Load the pre-built index (built once at indexing time)
index = faiss.read_index("candidate_index.faiss")

# Retrieve top 500 by cosine similarity
k = 500
distances, indices = index.search(jd_vector.reshape(1, -1), k)
# indices shape: (1, 500) — row indices into the candidate matrix
shortlisted_ids = [candidate_ids[i] for i in indices[0]]
```

**Why 500:** The shortlist size balances recall (not losing strong candidates) against scoring cost. At 500, the scoring engine computes 500 × 5 scoring functions — all vectorized arithmetic, no API calls. Running scoring on 200 would be slightly faster but risks dropping borderline-strong candidates who score slightly lower on semantics but very high on trajectory or platform signals. 500 gives enough headroom for the multi-dimensional rescoring to reorder effectively.

**Recall validation:** Before deploying, the ANN recall check (Data Understanding Report §5, Validation 1) must pass: embed 3 test JDs, confirm ≥90% of manually identified strong matches appear in the top-500. If not, switch to `bge-base-en-v1.5` or add a BM25 hybrid pass.

### 4.2 Stage 2 — Hard Filters (applied only when JD states explicit constraints)

Hard filters are applied as binary compatibility checks — a candidate either passes or does not. They are never continuous scoring dimensions. They are applied _only_ when the JD's `work_context` block explicitly states a constraint.

```python
def apply_hard_filters(shortlist: list, candidate_features: dict,
                        jd_intent: dict) -> list:
    ctx = jd_intent.get("work_context", {})
    filtered = []
    for cand_id in shortlist:
        f = candidate_features[cand_id]

        # Salary compatibility — only if JD states a ceiling
        if ctx.get("salary_max_lpa") and jd_intent["salary_stated"]:
            if f["expected_salary_min"] > ctx["salary_max_lpa"]:
                continue  # candidate's floor exceeds JD's ceiling

        # Location / relocation — only if JD states a hard location requirement
        if ctx.get("location_is_hard_requirement", False):
            jd_location = ctx.get("location_required", "")
            cand_location = f.get("location", "")
            # Pass if same city, or if candidate is willing to relocate
            if (jd_location not in cand_location
                    and not f.get("willing_to_relocate", False)):
                continue

        filtered.append(cand_id)
    return filtered
```

**What is NOT a hard filter:**

- `open_to_work` status — passive candidates are scored, never removed
- `notice_period_days` — flagged in explanation, not a filter
- `work_mode_preference` — soft alignment check in B4, not a filter
- Education tier — never a filter at any stage

### 4.3 Stage 3 — Multi-Signal Scoring (All ~500 candidates)

All five scoring functions are applied to every candidate in the shortlist. The scoring loop is vectorized where possible (B1 cosine similarity is a matrix multiply) and runs in serial for the formula-based dimensions (B2–B5 are lightweight arithmetic on the feature row).

```python
results = []
for cand_id in shortlist:
    f = load_features(cand_id)          # from parquet; fast columnar load
    v = candidate_vectors[cand_id]       # from .npy; fast array index

    b1 = semantic_score(v, jd_vector, f["skill_records"], jd_intent)
    b2 = trajectory_score(f, jd_intent)
    b3 = stability_score(f)
    b4 = platform_score(f, jd_intent)
    b5 = cert_bonus(f, jd_vector, jd_intent)
    comp = b1*0.35 + b2*0.25 + b3*0.15 + b4*0.20 + b5*0.05

    results.append({
        "candidate_id": cand_id,
        "composite_score": comp,
        "semantic_score": b1,
        "trajectory_score": b2,
        "stability_score": b3,
        "platform_score": b4,
        "cert_bonus": b5,
    })
```

### 4.4 Stage 4 — Ranking and Tiebreaker

```python
# Sort by composite score descending
ranked = sorted(results, key=lambda r: r["composite_score"], reverse=True)

# Tiebreaker: higher platform_score wins among candidates with identical
# composite (within 0.001) — reflects a more reliably hireable candidate
def tiebreaker_key(r):
    return (round(r["composite_score"], 3), r["platform_score"])

ranked = sorted(results, key=tiebreaker_key, reverse=True)

# Select top 20
top_20 = ranked[:20]
for rank, r in enumerate(top_20, start=1):
    r["rank"] = rank
```

**Tiebreaker rationale:** Among candidates who score nearly identically on composite (within rounding precision), the one with higher platform signals is genuinely the better practical choice — they are more likely to respond, complete the process, and accept an offer. This is the clearest decision criterion that doesn't require additional LLM reasoning.

---

## 5. Explainability Generation — Grounded Candidate Justifications

### 5.1 Design Principle: Narration, Not Re-Judgment

The explanation generator does not ask an LLM to evaluate a candidate. It asks it to narrate an already-completed, auditable computation. This distinction is critical:

- **Re-judgment (wrong):** "Here is this candidate's profile. How well do they match this job?"
  → The LLM produces its own independent assessment, which may contradict the scores and cannot be grounded in computed evidence.

- **Narration (correct):** "This candidate scored 0.82 on semantic fit because they have AWS (expert, 4 years), Terraform (advanced, 2 years), and Kubernetes (intermediate, 1 year), which match the JD's must-have skills. Write a recruiter-facing explanation citing these specific facts."
  → The LLM converts structured evidence into readable prose, without inventing anything.

### 5.2 Explanation Prompt Structure

Each candidate's prompt is built from the computed sub-scores and the specific source fields that produced them — not from the raw JSONL profile.

```python
def build_explanation_prompt(candidate: dict, jd_intent: dict) -> str:
    f = candidate["features"]
    scores = candidate["scores"]

    # Pull the specific evidence for each sub-score
    matched_skills = [
        f"{s} (strength: {candidate['skill_records'].get(s, 0):.2f})"
        for s in jd_intent["must_have_skills"]
        if candidate["skill_records"].get(s.lower(), 0) > 0.3
    ]
    career_summary = "; ".join([
        f"{r['title']} at {r['company']} ({r['duration_months']} months)"
        for r in sorted(f["career_history"],
                        key=lambda r: r["start_date"], reverse=True)[:3]
    ])
    promotions_evidence = (
        f"Detected {int(f['promotion_rate'] * 10)}/10 eligible companies with promotion"
        if f['promotion_rate'] > 0 else "No internal promotions detected"
    )
    platform_summary = (
        f"Open to work: {f['open_to_work']}; "
        f"Applications last 30d: {f['applications_submitted_30d']}; "
        f"Interview completion rate: {f['interview_completion_rate']:.0%}; "
        f"Offer acceptance rate: {f['offer_acceptance_rate']:.0%}; "
        f"Avg response time: {f['avg_response_time_hrs']:.0f}hrs; "
        f"Notice period: {f['notice_period_days']} days"
    )
    cert_evidence = (
        ", ".join([c["name"] for c in f.get("cert_records", [])[:3]])
        or "None held"
    )
    flags = []
    if f["job_hopping_flag"]:
        flags.append("3+ consecutive roles under 12 months — validate tenure intent in interview")
    if f["notice_period_days"] > 90:
        flags.append(f"Notice period is {f['notice_period_days']} days — plan timeline accordingly")
    if not f["open_to_work"] and f["applications_submitted_30d"] == 0:
        flags.append("Passive candidate — outreach required; no recent application activity")

    prompt = f"""
You are writing a candidate justification for a recruiter.
Your job is to convert the provided scores and evidence into a clear, specific,
recruiter-facing explanation. DO NOT invent information. Only use the evidence provided.
Use past-tense professional language. Do not use bullet points — write in paragraph form.

JD CONTEXT:
- Role seniority: {jd_intent['seniority_level']} ({jd_intent.get('seniority_evidence','')})
- Must-have skills: {', '.join(jd_intent['must_have_skills'])}
- Core problems to solve: {jd_intent['core_problems_to_solve']}

CANDIDATE SCORES:
- Composite: {scores['composite_score']:.3f}
- Semantic fit (B1): {scores['semantic_score']:.3f}
- Trajectory fit (B2): {scores['trajectory_score']:.3f}
- Stability (B3): {scores['stability_score']:.3f}
- Platform signals (B4): {scores['platform_score']:.3f}
- Cert bonus (B5): {scores['cert_bonus']:.3f}

EVIDENCE FOR EACH SCORE:
Semantic fit evidence — matched JD skills: {', '.join(matched_skills) or 'Weak direct skill match; similarity is through career context'}
Career timeline (3 most recent roles): {career_summary}
Trajectory evidence: {promotions_evidence}; Latest seniority level: {f['latest_seniority']}; Total experience: {f['experience_years']:.1f} years
Platform evidence: {platform_summary}
Certifications: {cert_evidence}
Flags to surface: {'; '.join(flags) if flags else 'None'}

Write the explanation in EXACTLY this structure:

1. MATCH SUMMARY (1 sentence): Start with "This candidate is a [strong/moderate/cautious] match for the [role name from JD] based on..."

2. SKILL ALIGNMENT (2-3 sentences): Explain which specific skills matched the JD's requirements, at what proficiency and duration, and call out any semantic equivalences (e.g., "Terraform and GCP map to the JD's 'cloud infrastructure' requirement").

3. SENIORITY ASSESSMENT (1-2 sentences): Explain why this candidate's seniority level aligns (or represents a calculated stretch) with the JD's requirement, citing the latest title and trajectory evidence.

4. TRAJECTORY SIGNAL (1-2 sentences): What does their career arc reveal? Cite promotions, increasing scope, or domain focus using the career timeline evidence.

5. PLATFORM SIGNAL SUMMARY (1-2 sentences): Summarize the B4 evidence — intent, responsiveness, reliability — in plain recruiter language.

6. FLAGS (if any): Concerns the recruiter should probe in screening. Write "No flags" if none.
"""
    return prompt
```

### 5.3 Batching Strategy

To avoid 20 individual API calls, candidates are batched 4–5 per call. Each candidate's evidence block is clearly delimited (`===CANDIDATE 1===`, etc.). The model is instructed to return a JSON array where each element contains `candidate_id` and the six explanation sections.

```python
def generate_explanations_batch(candidates_batch: list, jd_intent: dict) -> list:
    combined_prompt = f"""
You will write candidate justifications for {len(candidates_batch)} candidates.
For each, use ONLY the evidence provided. Return a JSON array with one object per candidate:
[{{"candidate_id": "...", "match_summary": "...", "skill_alignment": "...",
   "seniority_assessment": "...", "trajectory_signal": "...",
   "platform_summary": "...", "flags": "..."}}]

{chr(10).join([
    f"===CANDIDATE {i+1} (ID: {c['candidate_id']})===\n" +
    build_explanation_prompt(c, jd_intent)
    for i, c in enumerate(candidates_batch)
])}
"""
    # Single API call covers 4-5 candidates
    response = groq_client.messages.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4000,
        messages=[{"role": "user", "content": combined_prompt}]
    )
    raw = response.content[0].text
    return json.loads(raw)   # parse the JSON array
```

### 5.4 Post-Generation Grounding Validation

Before accepting any explanation, a validation check confirms it contains at least one literal data point from the candidate's actual profile:

```python
def validate_explanation_grounding(explanation: dict, candidate: dict) -> bool:
    """
    Check that the explanation references at least one concrete candidate datum.
    Accepted evidence types: a skill name, a company name, a title, a tenure number,
    a certification name, or a platform metric value.
    """
    full_text = " ".join(explanation.values()).lower()
    grounding_candidates = (
        [s.lower() for s in candidate["skill_records"].keys()] +
        [r["company"].lower() for r in candidate["features"]["career_history"]] +
        [r["title"].lower() for r in candidate["features"]["career_history"]] +
        [c["name"].lower() for c in candidate["features"].get("cert_records", [])] +
        [str(candidate["features"]["notice_period_days"]),
         str(candidate["features"]["avg_tenure_months"])]
    )
    return any(datum in full_text for datum in grounding_candidates)
```

If an explanation fails grounding validation, it is re-generated once with a stronger instruction ("You MUST mention the candidate's specific skills by name in your response"). If it fails a second time, the output falls back to a structured field summary (a template populated with the actual feature values) so the pipeline never silently drops a top-20 candidate's explanation.

### 5.5 Sample Output Format

```json
{
  "rank": 1,
  "candidate_id": "cand_00041872",
  "composite_score": 0.821,
  "semantic_score": 0.847,
  "trajectory_score": 0.763,
  "stability_score": 0.722,
  "platform_score": 0.811,
  "cert_bonus": 0.088,
  "explanation": {
    "match_summary": "This candidate is a strong match for the Senior Cloud Infrastructure Engineer role based on 6 years of hands-on AWS and Terraform experience combined with a proven promotion trajectory and high platform engagement.",
    "skill_alignment": "The candidate lists AWS (expert, 4 years) and Terraform (advanced, 2 years) as primary skills, directly covering the JD's must-haves. GCP (intermediate, 1 year) provides additional coverage for the JD's multi-cloud context. Kubernetes (intermediate, 18 months) maps to the JD's container orchestration requirement. The only notable gap is the JD's preference for Rust, which does not appear in this candidate's profile.",
    "seniority_assessment": "The candidate's current title of 'Senior Cloud Engineer' at Acme Corp aligns exactly with the JD's senior-level requirement. With 8.5 years of total experience and a promotion from Cloud Engineer to Senior Cloud Engineer within the same company, they are not a stretch hire — they are a direct match.",
    "trajectory_signal": "The candidate shows a consistent upward arc: Cloud Support Associate (18 months) → Cloud Engineer (30 months) → Senior Cloud Engineer (current, 36 months). The internal promotion at Acme Corp and increasing scope of responsibilities (from support tickets to designing multi-region architectures) indicate a candidate at the right point in their growth curve for this role.",
    "platform_summary": "The candidate is actively open to work and has submitted 8 applications in the last 30 days, suggesting genuine availability. Their interview completion rate of 88% and offer acceptance rate of 72% indicate a reliable hiring pipeline, and their average response time of 12 hours is well below the dataset median of 133 hours.",
    "flags": "Notice period is 120 days — if the role has a start-date urgency, confirm flexibility early in the screening call."
  }
}
```

---

## 6. Edge Cases — Thin Profiles, Keyword-Stuffers, and Passive Candidates

### 6.1 Thin Profiles

**Definition:** A candidate whose `build_candidate_doc` output is under 50 characters, or whose career history descriptions are all under 20 characters each.

**What goes wrong without handling:** The embedding vector for a thin profile is poorly differentiated — it clusters near the centroid of the embedding space because there's no rich text to push it toward a specific domain. This means it achieves mediocre but non-zero cosine similarity with _every_ JD, landing in the shortlist for many queries even though it's a genuinely weak match.

**Handling strategy:**

1. **At index time:** Flag `thin_profile = True` in the feature store.
2. **At ANN retrieval:** Thin-profile candidates are retrieved normally (no special treatment — we don't want to suppress them if they genuinely match).
3. **At scoring:** `thin_profile = True` applies a `semantic_score` cap of 0.55. A thin profile cannot achieve a high semantic score because the embedding lacks sufficient signal to justify it — the cap reflects this epistemic uncertainty.
4. **At explanation generation:** The explanation prompt includes an instruction: "This candidate has a thin profile (sparse career history text). Surface this as a flag: the recruiter should conduct a structured interview to surface experience not captured in the profile."

```python
# In semantic_score():
if candidate_features.get("thin_profile", False):
    b1 = min(b1, 0.55)  # cap semantic score for thin profiles
```

### 6.2 Keyword-Stuffers (the Original ATS Gaming Problem)

**The failure mode:** A candidate who repeats "Senior Frontend Engineer" and "React, TypeScript, Node.js" across every section of their profile — not because they have deep expertise, but because they know how to game keyword systems. The core problem statement describes exactly this: they rank above better candidates in keyword-based ATS, but their actual experience is shallow.

**How this system prevents it:**

1. **The `skill_strength_score` formula neutralizes raw keyword presence.** A candidate who lists "React (expert, 6 months, 0 endorsements)" scores `0.25 * 1.0 + 0.35 * 0.10 + 0.15 * 0.0 = 0.285` — equivalent to a beginner. Keyword stuffing gains nothing if duration and endorsements are low.

2. **The embedding operates on _meaning_, not _frequency_.** Repeating "React" five times in a candidate document does not increase the cosine similarity with the JD vector — transformer embeddings are not TF-IDF. The embedding reflects domain expertise across the full career narrative, not the presence of specific tokens.

3. **The trajectory score penalizes empty seniority claims.** A candidate who titles every role "Senior Frontend Engineer" with no promotions detected and only 2 years of total experience will have `latest_seniority = 0.75` (senior keyword present) but `promotion_rate = 0.0` and `experience_years = 2.0`. The trajectory composite becomes:

   ```
   seniority_fit (vs. senior JD) = 1.0  (level matches)
   traj_momentum = 0.0*0.50 + min(2/10, 1)*0.30 + 0.75*0.20 = 0 + 0.06 + 0.15 = 0.21
   b2 = 1.0*0.60 + 0.21*0.40 = 0.60 + 0.084 = 0.684
   ```

   vs. a genuine senior with 6 years and one promotion:

   ```
   seniority_fit = 1.0
   traj_momentum = 1.0*0.50 + 0.6*0.30 + 0.75*0.20 = 0.50 + 0.18 + 0.15 = 0.83
   b2 = 1.0*0.60 + 0.83*0.40 = 0.60 + 0.332 = 0.932
   ```

   The genuine senior scores 0.93 vs. the keyword-stuffer's 0.68 on trajectory alone — a difference large enough (0.25 × 0.25 = 0.06 composite gap) to matter at the margin between ranked 15th and ranked 25th.

4. **Platform signals further differentiate substance from resume inflation.** A candidate who inflates titles typically has average-to-low endorsements, no GitHub activity, and their `recruiter_response_rate` (the rate at which _recruiters_ engage back after seeing them in search) will be low — because human recruiters who have seen their profile didn't click. This directly penalizes them in B4.

**Explicit detection flag (optional, post-hoc):**

```python
def keyword_stuffing_risk(candidate: dict, jd_intent: dict) -> bool:
    """
    Heuristic flag for review: candidate has the right title keywords but
    low skill_strength_scores for the JD's must-haves.
    """
    must_haves = jd_intent["must_have_skills"]
    if not must_haves:
        return False
    skill_strengths = [
        candidate["skill_records"].get(s.lower(), 0)
        for s in must_haves
    ]
    avg_strength = np.mean(skill_strengths) if skill_strengths else 0
    title_match = any(
        kw.lower() in candidate["features"].get("current_title", "").lower()
        for kw in must_haves
    )
    return title_match and avg_strength < 0.35
```

If this flag fires for a top-20 candidate, the explanation prompt is instructed to note: _"Title keywords match the JD, but skill depth signals (proficiency, tenure, endorsements) are below average for the matched skills — validate in technical screening."_

### 6.3 Passive High-Intent Candidates

**The tension:** `open_to_work = False` for 64.7% of the dataset. Some of the best candidates in a real talent pool are passive — they're currently employed, performing well, and not actively searching. Penalizing them heavily for passivity would replicate the worst behavior of traditional ATS (which only shows you the actively applying pool, not the best available talent).

**Resolution:** The platform score formula applies a 0.4 weight (not 0.0) to the `open_to_work = False` signal. The remaining three platform sub-components (`applications_submitted_30d`, `profile_completeness_score`, `search_appearances_30d`) can fully compensate. A passive candidate who appears in 400+ recruiter searches per month and has a 95% offer acceptance rate scores:

```
active_intent = 0.4*0.35 + 0*0.25 + 0.9*0.20 + min(400/200, 1)*0.20
              = 0.14 + 0 + 0.18 + 0.20 = 0.52

hire_reliability (assuming strong signals): 0.80

platform_score (non-technical JD) = 0.52*0.55 + 0.80*0.45 = 0.286 + 0.360 = 0.646
```

Compared to an active but mediocre candidate:

```
active_intent = 1.0*0.35 + 0.5*0.25 + 0.5*0.20 + 0.2*0.20
              = 0.35 + 0.125 + 0.10 + 0.04 = 0.615

hire_reliability (assuming average signals): 0.55

platform_score = 0.615*0.55 + 0.55*0.45 = 0.338 + 0.248 = 0.586
```

The passive high-signal candidate (0.646) outscores the active mediocre candidate (0.586) on platform score. If they also dominate on semantic fit and trajectory, they will rank above the active candidate — which is the correct behavior.

**In the explanation:** Passive candidates are flagged not as a penalty but as an operational note: _"This is a passive candidate — outreach required. Their platform profile shows strong recruiter interest (X search appearances in 30 days) and reliable follow-through (Y% offer acceptance rate)."_

---

## 7. End-to-End Worked Example

### 7.1 Sample Job Description

```
We are looking for a Senior Data Engineer to join our Data Platform team.
You will be responsible for building and maintaining real-time data pipelines
that serve our ML models and downstream analytics teams. You will own the
end-to-end architecture of our Kafka-based event streaming infrastructure
and lead the migration of our on-premise data warehouse to GCP BigQuery.

Must have:
- 5+ years of data engineering experience
- Strong expertise in Apache Kafka and event-driven architecture
- Hands-on experience with GCP (BigQuery, Dataflow, Pub/Sub)
- Proficiency in Python and SQL
- Experience with orchestration tools (Apache Airflow)

Nice to have:
- Experience with dbt
- Familiarity with Terraform for infrastructure provisioning

You will mentor junior engineers, participate in architecture reviews, and
collaborate cross-functionally with the ML and Analytics teams.
```

### 7.2 JD Parser Output — `jd_intent.json`

```json
{
  "seniority_level": 0.75,
  "seniority_evidence": "lead the migration... mentor junior engineers... own the end-to-end architecture",
  "must_have_skills": [
    "Apache Kafka",
    "GCP",
    "BigQuery",
    "Python",
    "SQL",
    "Apache Airflow"
  ],
  "nice_to_have_skills": ["dbt", "Terraform"],
  "core_problems_to_solve": "Build and maintain real-time data pipelines serving ML models and analytics. Own Kafka-based event streaming architecture. Lead migration of on-premise warehouse to GCP BigQuery. Mentor junior engineers and lead architecture reviews.",
  "implicit_soft_skills": [
    "technical ownership",
    "cross-functional collaboration",
    "mentorship"
  ],
  "domain_tags": ["data-engineering"],
  "requires_technical_github_signals": true,
  "work_context": {
    "work_mode": null,
    "location_required": null,
    "location_is_hard_requirement": false,
    "salary_min_lpa": null,
    "salary_max_lpa": null
  },
  "salary_stated": false
}
```

### 7.3 Sample Candidate Profile (Key Fields)

**Candidate ID:** `cand_00072341`

**Skills:**

- Apache Kafka: advanced, 3.5 years, 18 endorsements
- Python: expert, 6 years, 41 endorsements
- SQL: expert, 6 years, 35 endorsements
- BigQuery: advanced, 2 years, 12 endorsements
- Apache Airflow: intermediate, 1.5 years, 6 endorsements
- GCP: intermediate, 2 years, 8 endorsements
- dbt: beginner, 0.5 years, 1 endorsement
- Spark: advanced, 3 years, 22 endorsements

**Career history:**

1. Junior Data Analyst, IT Services Corp — 18 months
2. Data Engineer, Initech — 28 months (same company, promoted from junior analyst internally 12 months in)
3. Senior Data Engineer, Globex Inc — 31 months (current)

**Total experience:** 6.4 years

**Redrob signals:**

- `open_to_work`: False
- `applications_submitted_30d`: 2
- `profile_completeness_score`: 82%
- `search_appearances_30d`: 340
- `interview_completion_rate`: 0.91
- `offer_acceptance_rate`: 0.78
- `avg_response_time_hrs`: 9
- `email_verified`: True, `phone_verified`: True
- `github_activity_score`: 61.3
- `endorsements_received`: 143
- `notice_period_days`: 90

**Certifications:** Google Cloud Professional Data Engineer (issued 2023)

**Education:** B.Tech, Computer Science, tier_2 institution

### 7.4 Step-by-Step Score Computation

#### B1 — Semantic Score

**Component A: Embedding similarity**

The candidate document concatenates their Kafka + Python + SQL + BigQuery + Airflow + GCP skills with career descriptions referencing "real-time data pipelines", "Kafka consumer groups", "BigQuery partitioned tables", "Airflow DAGs." The JD query document references "Kafka event streaming", "GCP BigQuery", "real-time pipelines."

The two documents occupy the same dense region of embedding space. Assume cosine similarity = 0.78 (strong overlap).

```
embed_sim_norm = (0.78 + 1) / 2 = 0.89
```

**Component B: Skill coverage**

| JD Must-Have   | Candidate Strength Score                                                       | Notes                          |
| -------------- | ------------------------------------------------------------------------------ | ------------------------------ |
| Apache Kafka   | `0.75*0.5 + min(3.5/5,1)*0.35 + min(18/50,1)*0.15 = 0.375+0.245+0.054 = 0.674` | Advanced, 3.5yr, 18 endorse    |
| GCP            | `0.5*0.5 + min(2/5,1)*0.35 + min(8/50,1)*0.15 = 0.25+0.14+0.024 = 0.414`       | Intermediate, 2yr, 8 endorse   |
| BigQuery       | `0.75*0.5 + min(2/5,1)*0.35 + min(12/50,1)*0.15 = 0.375+0.14+0.036 = 0.551`    | Advanced, 2yr, 12 endorse      |
| Python         | `1.0*0.5 + min(6/5,1)*0.35 + min(41/50,1)*0.15 = 0.50+0.35+0.123 = 0.973`      | Expert, 6yr, 41 endorse        |
| SQL            | `1.0*0.5 + min(6/5,1)*0.35 + min(35/50,1)*0.15 = 0.50+0.35+0.105 = 0.955`      | Expert, 6yr, 35 endorse        |
| Apache Airflow | `0.5*0.5 + min(1.5/5,1)*0.35 + min(6/50,1)*0.15 = 0.25+0.105+0.018 = 0.373`    | Intermediate, 1.5yr, 6 endorse |

Average must-have coverage: `(0.674+0.414+0.551+0.973+0.955+0.373) / 6 = 3.940 / 6 = 0.657`

Nice-to-have (dbt): `0.25*0.5 + 0.35*0.1 + 0.15*0.02 = 0.125+0.035+0.003 = 0.163` → `× 0.40 = 0.065`
Nice-to-have (Terraform): not in profile → semantic fallback ≈ 0.35 (adjacent to infrastructure tools) → `× 0.60 × 0.40 = 0.084`

Combined coverage: `(0.657*6 + 0.065 + 0.084) / (6 + 0.4 + 0.4) = (3.940+0.149) / 6.8 = 0.601`

Wait — simplified to mean of all contributions:

```
all_skill_scores = [0.674, 0.414, 0.551, 0.973, 0.955, 0.373, 0.065, 0.084]
coverage_score = mean = 4.089 / 8 = 0.511
```

**B1 final:**

```
b1 = 0.89 * 0.60 + 0.511 * 0.40
   = 0.534 + 0.204
   = 0.738
```

#### B2 — Trajectory Score

**Seniority of latest role:** "Senior Data Engineer" → `latest_seniority = 0.75`

**Seniority fit vs. JD (jd_seniority = 0.75):**

```
gap = |0.75 - 0.75| = 0.0 → seniority_fit = 1.0
```

**Promotion rate:**

- At IT Services Corp: Junior Data Analyst → (promoted to) Data Engineer within Initech? Wait — the career history shows the promotion happened at Initech (not IT Services Corp). Initech has 2 roles (Analyst and then promoted to Data Engineer). One company, one promotion detected.
- Globex Inc: one role → not eligible.

```
promotion_rate = 1 detected / 1 eligible = 1.0
```

**Trajectory momentum:**

```
traj_momentum = 1.0*0.50 + min(6.4/10, 1)*0.30 + 0.75*0.20
              = 0.50 + 0.192 + 0.15
              = 0.842
```

**B2 final:**

```
b2 = 1.0 * 0.60 + 0.842 * 0.40
   = 0.60 + 0.337
   = 0.937
```

#### B3 — Stability Score

**Average tenure:**

```
avg_tenure = (18 + 28 + 31) / 3 = 77/3 = 25.7 months
```

(Note: the 28 months at Initech covers both the analyst period and the post-promotion period — the promotion is within the same company/role record.)

**Job hopping flag:** No consecutive roles under 12 months → `job_hopping_flag = 0`

**Education tier:** tier_2 → `edu_bonus = 0.03`

**B3:**

```
tenure_norm = min(25.7 / 36.0, 1.0) = 0.714
b3 = min(0.714 - 0 + 0.03, 1.0) = 0.744
```

#### B4 — Platform Score

**Active intent:**

```
open_to_work_score = 0.4  (passive)
apps_norm = min(2/10, 1) = 0.20
completeness_norm = 82/100 = 0.82
search_norm = min(340/200, 1) = 1.0

active_intent = 0.4*0.35 + 0.20*0.25 + 0.82*0.20 + 1.0*0.20
              = 0.14 + 0.05 + 0.164 + 0.20
              = 0.554
```

**Hire reliability:**

```
response_speed = 1.0 - min(9/200, 1) = 1.0 - 0.045 = 0.955
verif_score = 0.5 + 0.5 = 1.0

hire_reliability = 0.91*0.40 + 0.78*0.30 + 0.955*0.20 + 1.0*0.10
                 = 0.364 + 0.234 + 0.191 + 0.10
                 = 0.889
```

**Technical engagement** (required = True for data-engineering JD):

```
github_norm = 61.3 / 96.9 = 0.633
endorse_norm = min(143/100, 1) = 1.0

tech_engagement = 0.633*0.60 + 1.0*0.40 = 0.380 + 0.40 = 0.780
```

**B4 final:**

```
b4 = 0.554*0.40 + 0.889*0.35 + 0.780*0.25
   = 0.222 + 0.311 + 0.195
   = 0.728
```

#### B5 — Certification Bonus

Google Cloud Professional Data Engineer (issued 2023):

- Cosine similarity between "Google Cloud Professional Data Engineer" and JD vector (which is heavily GCP/data-engineering): assume `cosine_sim = 0.84`
- `relevance_norm = (0.84 + 1) / 2 = 0.92`
- `years_old = 2026 - 2023 = 3`
- `recency_weight = max(0.5, 1.0 - 3*0.10) = max(0.5, 0.70) = 0.70`
- `cert_contribution = 0.92 * 0.70 = 0.644`
- `b5 = min(0.644, 0.10) = 0.10` ← hits the cap

#### Composite Score

```
composite = 0.738*0.35 + 0.937*0.25 + 0.744*0.15 + 0.728*0.20 + 0.10*0.05
          = 0.258  + 0.234  + 0.112  + 0.146  + 0.005
          = 0.755
```

### 7.5 Explanation Generated

The following is the output from the explanation generator for `cand_00072341`:

---

**Rank #1 | Candidate ID: cand_00072341 | Composite Score: 0.755**

**Match Summary:** This candidate is a strong match for the Senior Data Engineer role based on 6.4 years of data engineering experience spanning Python, SQL, Apache Kafka, BigQuery, and Airflow, combined with a demonstrated promotion trajectory and a Google Cloud Professional Data Engineer certification directly relevant to the GCP migration objective.

**Skill Alignment:** The candidate covers five of the six JD must-have skills with meaningful depth. Python (expert, 6 years, 41 endorsements) and SQL (expert, 6 years, 35 endorsements) are their strongest signals, reflecting the deep proficiency this role requires for complex pipeline logic. Apache Kafka (advanced, 3.5 years) directly addresses the JD's core Kafka event-streaming architecture requirement. BigQuery (advanced, 2 years) and Apache Airflow (intermediate, 1.5 years) round out the must-have list, with GCP (intermediate, 2 years) providing the broader cloud context. The one meaningful gap is that Airflow experience is at the intermediate level with a relatively short tenure — this should be probed in a technical screen. dbt is listed as a beginner skill with minimal endorsement, which addresses the nice-to-have partially but should not be treated as a strength.

**Seniority Assessment:** The candidate's current title of Senior Data Engineer at Globex Inc is an exact level match for the JD's senior requirement (confirmed by both the title keyword and their 6.4 years of total experience). This is not a stretch hire — they are currently performing the role this JD describes.

**Trajectory Signal:** The career arc shows clear upward progression: Junior Data Analyst (18 months) → Data Engineer via internal promotion at Initech (28 months total) → Senior Data Engineer at Globex Inc (31 months, current). The internal promotion at Initech, moving from analyst to data engineer within the same organization, demonstrates that they were trusted with increasing responsibility rather than title-chasing through job changes. The move to Globex Inc represents a deliberate step into a senior role.

**Platform Signal Summary:** This candidate is currently passive (not flagged as open to work) but shows strong recruiter-side demand — appearing in 340 recruiter searches in the last 30 days and submitting 2 applications. Their process reliability is high: 91% interview completion rate and 78% offer acceptance rate. Critically, their average response time of 9 hours is well below the platform median, indicating they are responsive when contacted despite their passive status. 143 endorsements received and a GitHub activity score of 61.3 further support their technical credibility for this engineering role.

**Flags:** Notice period is 90 days — standard for the Indian market but worth confirming flexibility if the team has a start-date target. This is a passive candidate; outreach is required — they are not actively applying at volume. Given their Airflow proficiency is self-reported at intermediate with limited endorsements, a focused technical question about complex Airflow DAG patterns (dynamic task mapping, cross-DAG dependencies) is recommended in the screening call.

---

_Document owner: ML Engineering lead_
_Last updated: June 2026_
_Status: Approved — primary algorithmic reference for implementation_
