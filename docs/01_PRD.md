# Product Requirements Document

## Redrob Intelligent Candidate Discovery & Ranking Engine

**Version:** 1.0  
**Challenge:** Redrob Hackathon — ₹50 Lakh+ Prize Pool  
**Dataset:** 100,000 candidate profiles (JSONL, synthetic/simulated)  
**Deliverable Scope:** GitHub repo + methodology PDF + ranked output file

---

## 1. Problem Statement

Traditional ATS systems fail not because they process too little data, but because they process it the wrong way: Boolean keyword matching treats a resume as a bag of tokens, not a story of capability. The consequence is a **false negative problem at scale** — a candidate who "built highly scalable web applications using React at Meta" scores lower than one who keyword-stuffed "Senior Frontend Engineer" five times, because the ATS cannot distinguish accomplishment from repetition. The deeper cost is systemic: career trajectory, implicit seniority signals, and behavioral intent (is this person actually looking?) are invisible to keyword filters, so the best candidates are systematically buried while the most ATS-optimized resumes surface. This engine replaces that filter with contextual semantic understanding.

---

## 2. Primary User Persona — The Technical Recruiter

**Name:** Priya Mehta, 4 years as a senior tech recruiter at a mid-to-large IT services firm  
**Tools today:** LinkedIn Recruiter + internal ATS (Workday/Naukri-based)  
**Mandate:** Fill 8–12 open roles per month; hiring managers escalate if top-of-funnel quality is poor

### Current Workflow Pain Points

| Pain Point                                                           | Root Cause                                                  | Impact                                                                   |
| -------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| Spends 3–5 hours per JD manually scanning 200+ ATS shortlists        | Keyword filters return high-volume, low-signal results      | Cognitive overload; she pattern-matches on company names, not actual fit |
| Can't assess "readiness" from a resume alone                         | ATS gives no view of career trajectory or growth signals    | Relies on gut feel; misses high-potential mid-level candidates           |
| Shortlist she sends to hiring manager gets rejected ~40% of the time | Mismatch between JD intent and keyword-matched profile      | Credibility hit; adds a re-screening loop                                |
| Passive top-tier candidates never rise to the surface                | No behavioral signal integration; platform activity ignored | Best candidates are never contacted                                      |
| Can't justify why candidate X was picked over candidate Y            | Black-box ATS scoring                                       | Fails to build hiring manager trust in the process                       |

**What Priya actually needs:** Paste a JD, get back 20 candidates she can defend to a hiring manager in under 10 minutes.

---

## 3. Product Goals — Mapped to Judging Criteria

The three judging dimensions directly map to three product goals:

### Goal 1 — Ranking Quality (Judge Criterion 1)

**"Are the top 20 candidates genuinely the best matches out of 100,000?"**

- The system must move beyond surface-level skill overlap and reason about semantic equivalence (e.g., "AWS, GCP, infrastructure" → "Cloud Expert"), career trajectory (mid-level ready for senior), and platform engagement signals (active vs. dormant candidates).
- Success metric: The top 20 must be defensible to a domain expert who reads the JD and all 100K profiles. No obvious mismatches in the top 10.

### Goal 2 — Methodology Clarity (Judge Criterion 2)

**"Is the code clean, modular, and production-ready? Is the technical choice well-reasoned?"**

- Architecture must be decomposed into clearly named, independently testable modules: JD parser → embedding engine → signal integrator → scorer → ranker → explainer.
- Every major technical choice (embedding model, weighting scheme, scoring formula) must be documented in the methodology PDF with a rationale that a non-ML engineer can follow.
- Success metric: A new engineer can extend one module without touching the others.

### Goal 3 — Explainability (Judge Criterion 3)

**"How good is the AI at explaining its decisions for shortlisted candidates?"**

- For every candidate in the top 20, the system must produce a human-readable justification covering: semantic skill match, seniority alignment, career trajectory signal, and platform/behavioral signal.
- Explanations must be specific and grounded in the candidate's actual data — not templates. "Candidate has 6 years of experience in cloud infrastructure (AWS, GCP), matches the JD's 'Cloud Expert' requirement semantically" beats "Candidate has relevant skills."
- Success metric: A recruiter can forward the explanation to a hiring manager verbatim without editing.

---

## 4. Core Capabilities

### A. Deep JD Understanding

**Input:** Free-text Job Description pasted by recruiter  
**Output:** Structured intent object used downstream

The JD parser must extract:

- **Seniority level** — explicit ("Senior") or implicit (e.g., "lead a team of engineers" → leadership expectation, maps to senior+)
- **Core technical requirements** — must-have skills, tools, platforms, languages (with distinction between hard requirements and nice-to-haves)
- **Domain intent** — what problem space is this role operating in? (e.g., "build data pipelines" → data engineering, not just "Python")
- **Implicit soft skill requirements** — "own the roadmap" → product ownership; "collaborate with cross-functional stakeholders" → communication/influence
- **Work context signals** — industry, company size hints, work mode (remote/onsite/hybrid if stated), or salary range if stated

**Implementation note:** This is an LLM extraction step. The output is a structured JSON object, not a keyword list. Fields map 1:1 to scoring dimensions in module B.

---

### B. Contextual Relevance & Signal Integration

**Input:** JD intent object + 100K candidate profiles  
**Output:** Scored candidate list with per-dimension sub-scores

Five scoring dimensions, each weighted independently:

#### B1 — Semantic Skill Match

Map candidate skills + career history text to JD requirements using embedding similarity. Must handle:

- Synonym/equivalence resolution: "Terraform, AWS, GCP" → "Cloud Expert"; "Vue.js, React, Angular" → "Frontend Framework"
- Proficiency weighting: `advanced/expert` proficiency for a required skill > `beginner` for the same skill
- Duration weighting: a skill with 4 years of experience outweighs one listed with 6 months

**Dataset fields used:** `skills[].name`, `skills[].proficiency`, `skills[].duration`, `career_history[].description`

#### B2 — Career Trajectory & Seniority Fit

Assess whether the candidate is a genuine match for the JD's seniority level — including candidates who are one step below but showing strong upward trajectory.

- Compute years of total experience and relevant-domain experience separately
- Identify promotions within career history (title progression within same company)
- Detect "stretch readiness": a strong mid-level with consistent promotion cadence is a valid match for a senior JD
- Penalize seniority mismatch in both directions (overqualified candidates are poor pipeline fits too)

**Dataset fields used:** `profile.experience_years`, `career_history[].title`, `career_history[].duration_months`, `career_history[].company`

#### B3 — Career Stability

Assess whether this candidate is a realistic hire and long-term fit.

- Compute average tenure per role; flag candidates with ≥3 consecutive short tenures (<12 months each) as potential flight risks
- Distinguish voluntary vs. layoff patterns is not possible with this dataset — treat as neutral
- Education tier weighting (optional, light): tier_1/tier_2 institutions get a marginal signal bump, not a hard filter

**Dataset fields used:** `career_history[].duration_months`, `education[].institution_tier`

#### B4 — Platform Activity & Intent Signals (Redrob Signals)

This is the dataset's most differentiated feature set versus a typical ATS. Use it.

- **Active intent signals:** `open_to_work` (binary flag), `applications_submitted_30d`, `profile_completeness_score`
- **Market validation signals:** `saved_by_recruiters_30d`, `search_appearances_30d`, `recruiter_response_rate`
- **Reliability signals:** `interview_completion_rate`, `offer_acceptance_rate`, `verified_email`, `verified_phone`
- **Technical credibility signal (for tech roles):** `github_activity_score`
- **Logistical fit:** `notice_period_days`, `willing_to_relocate`, `work_mode_preference`, `expected_salary_min/max`

Apply a composite "hire-ability" sub-score from these fields. A semantically matched candidate with zero platform activity and 150-day notice period is a worse practical choice than a slightly lower semantic match who is open-to-work and responds in 24 hours.

**Dataset fields used:** All fields under `redrob_signals`

#### B5 — Certifications Bonus

For roles where certifications matter (cloud, project management, finance), award a multiplicative bonus when a candidate holds a directly relevant certification.

- Map certification names to JD domain: "AWS Certified Cloud Practitioner" → cloud roles; "Scrum Master" → agile project roles
- Treat as a boost, not a gate — a candidate without a cert but with 5 years of relevant experience outranks one with only the cert

**Dataset fields used:** `certifications[].name`, `certifications[].issuer`

---

### C. Trustworthy Shortlisting & Explainability

**Input:** Scored candidate list  
**Output:** Top 20 ranked candidates with structured explanations

For each of the top 20 candidates, generate a justification that includes:

1. **Match summary (1 sentence):** "This candidate is a strong match for the Senior Cloud Architect role based on 8 years of AWS/GCP infrastructure experience and a consistent promotion trajectory."
2. **Skill alignment detail:** Which specific skills matched, how, and at what proficiency — with any semantic mappings called out explicitly.
3. **Seniority assessment:** Why this candidate's level aligns (or represents a calculated stretch) with the JD's requirement.
4. **Trajectory signal:** What the career history reveals about growth direction — promotions, increasing scope, specialization.
5. **Platform signal summary:** Engagement level, response rate, notice period, logistical fit (relocation, work mode).
6. **Flags (if any):** Any concerns the recruiter should validate in the interview (e.g., recent short tenures, lower platform activity despite semantic fit).

Explanations are generated by an LLM prompted with the candidate's data and the JD intent object. They are not templated string substitutions.

---

## 5. End-to-End User Flow

```
[Recruiter] Pastes JD into system
       ↓
[JD Parser Module] Extracts structured intent object
  → seniority level, technical requirements, soft skill signals, domain
       ↓
[Embedding Engine] Generates vector representation of JD intent
       ↓
[Candidate Retrieval] ANN search over pre-computed candidate embeddings
  → Retrieves top-N candidates (e.g., 500) for full scoring
  (Pre-indexing of 100K candidates runs once at startup)
       ↓
[Signal Integration Scorer] Computes 5 sub-scores per candidate
  → Semantic Skill Match (B1)
  → Seniority/Trajectory Fit (B2)
  → Career Stability (B3)
  → Platform Activity & Intent (B4)
  → Certifications Bonus (B5)
  → Weighted composite score
       ↓
[Ranker] Sorts by composite score → selects top 20
       ↓
[Explainability Module] Generates per-candidate human-readable justification
       ↓
[Output] Ranked CSV/JSON (candidate_id, rank, composite_score, sub_scores, explanation)
         + Console/UI display for recruiter review
```

**Latency target:** For the hackathon, full pipeline runtime for a new JD query (post-indexing) should complete in under 60 seconds on a standard laptop CPU. Pre-indexing of 100K candidates is a one-time setup step.

---

## 6. Non-Goals & Scope Boundaries

This is a **hackathon submission on a static dataset**, not a production ATS integration. The following are explicitly out of scope:

| Out of Scope                                                  | Rationale                                                 |
| ------------------------------------------------------------- | --------------------------------------------------------- |
| Real-time candidate database updates                          | Dataset is static; no ingestion pipeline needed           |
| Multi-recruiter collaboration or role-based access            | Single-user CLI/script interaction is sufficient          |
| Resume parsing from raw PDF/DOCX files                        | Dataset provides pre-structured JSON profiles             |
| Integration with existing ATS (Workday, Greenhouse, etc.)     | Not required by the challenge brief                       |
| Candidate-facing features (application portal, notifications) | Recruiter-side tool only                                  |
| Bias auditing or fairness certification                       | Important in production; out of scope for hackathon demo  |
| Model fine-tuning on domain-specific recruitment data         | Use pre-trained embedding models off the shelf            |
| A/B testing or feedback loop from recruiter actions           | No mechanism for this in a static challenge               |
| Salary negotiation or offer management features               | Not in the problem scope                                  |
| Multi-language resume support                                 | Dataset is English/Hindi, but all text fields are English |

---

## 7. Constraints & Assumptions

### Dataset Constraints

| Constraint                                                           | Implication                                                                                                                     |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 100% of candidates speak English and Hindi                           | No multilingual NLP complexity; process all text as English                                                                     |
| Only 35.3% are actively open-to-work                                 | Platform activity signals matter; passive candidates can still be strong matches — weight intent signals but don't gate on them |
| Salary data is available (min/max LPA)                               | Use for JD-to-candidate salary range compatibility check if the JD specifies a range; otherwise exclude                         |
| 72% have verified email, 61.8% verified phone                        | Use verification status as a reliability signal in B4, not a hard filter                                                        |
| Skills are self-reported (133 unique skills, avg ~9.6 per candidate) | Self-reported skills inflate proficiency claims; weight endorsed skills and skills with long duration more heavily              |
| Career history descriptions are free-text, variable quality          | Embedding over description text is essential; keyword matching over these will fail                                             |
| GitHub Activity Score exists but is a single scalar (0–96.9)         | Treat as a proxy for technical engagement; meaningful primarily for software/engineering JDs                                    |
| Institution tier is highly skewed (tier_3/4 = 75% of candidates)     | Do not use education tier as a primary filter; use as a marginal tiebreaker signal only                                         |
| 10 main job titles are near-uniformly distributed                    | The dataset is not skewed toward any one role type; the scoring system must generalize across all 10 title domains              |

### Technical Assumptions

- Embedding model: Use `sentence-transformers` (e.g., `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`) for fast, CPU-compatible inference on 100K profiles
- ANN index: Use `FAISS` (flat L2 or IVF) for sub-second retrieval over 100K vectors
- LLM for JD parsing and explanation generation: Use Groq API (llama-3.3-70b-versatile) via API calls; not deployed locally
- All processing: Python 3.10+, modular package structure
- Output format: CSV with columns `[rank, candidate_id, composite_score, semantic_score, trajectory_score, stability_score, platform_score, cert_bonus, explanation]`

---

## 8. Open Risks & Questions to Resolve Before Building

### Risk 1 — Embedding Model Recall at Semantic Matching

**Risk:** A lightweight embedding model (`all-MiniLM-L6-v2`) may not reliably map "AWS + GCP + Terraform" to "Cloud Expert" in a JD. If recall at the ANN retrieval stage is poor, the best candidates won't even reach the full scoring step.  
**Resolution path:** Run a quick benchmark: embed 5 test JDs, retrieve top-500, manually check if obvious matches are present. If recall is poor, try `bge-base-en-v1.5` or add a BM25 sparse retrieval layer as a hybrid fallback.

### Risk 2 — Sub-Score Weighting Is Arbitrary Without Validation

**Risk:** The weights assigned to B1–B5 are currently judgment calls. A wrong weighting scheme could cause the platform activity score to dominate over semantic fit for a passive but highly qualified candidate.  
**Resolution path:** Define a set of 5–10 "golden test cases" (manually constructed JDs with known correct top-3 answers) and tune weights against these before final submission.

### Risk 3 — LLM Explanation Quality Is Inconsistent

**Risk:** LLM-generated explanations may be generic, hallucinate details not in the candidate's profile, or vary wildly in length and quality.  
**Resolution path:** Use a tightly constrained prompt template that requires the LLM to cite specific fields from the candidate JSON. Add a post-processing validation step that checks the explanation contains at least one candidate-specific data point (name, skill, company, or tenure) before accepting it.

### Risk 4 — Career History Description Quality Is Uneven

**Risk:** The free-text career descriptions in the dataset are synthetic and may not contain rich semantic content for embedding. If descriptions are thin, B1 semantic scoring degrades.  
**Resolution path:** Inspect 20–30 random career history descriptions before finalizing the embedding strategy. If descriptions are sparse, construct a richer "candidate document" by concatenating title + company + skills + description into a single embedding target.

### Risk 5 — Runtime Exceeds Acceptable Demo Threshold

**Risk:** Embedding 100K profiles at JD-query time is too slow for a live demo. Pre-indexing is essential but adds setup complexity.  
**Resolution path:** Pre-compute and persist candidate embeddings to disk (`.npy` file) as part of an indexing script. Query time should be: embed JD (< 1s) → ANN search (< 1s) → score top-500 (< 10s) → LLM explain top-20 (< 30s). Total: under 60s.

### Open Questions

1. **Does the challenge provide a specific test JD, or must we supply our own?** → Clarify before final submission. If a test JD is provided, optimize the pipeline against it.
2. **Is the ranked output judged on a hidden test JD or the JD we demonstrate?** → This affects whether generalization or peak performance matters more.
3. **Is there a required CSV column format for the output file?** → The brief says "format specified by the challenge" — locate the exact spec.
4. **What is the definition of "top 20" — strict top-20, or top-20 with ties broken by some tiebreaker?** → Define tiebreaker rule (e.g., higher platform activity wins ties) before submission.

---

_Document owner: Hackathon team lead_  
_Last updated: June 2026_  
_Status: Approved for development_
