# Data Understanding & Feature Engineering Report
## Redrob Intelligent Candidate Discovery & Ranking Engine

**Document:** `docs/02_Data_Understanding_Report.md`  
**Version:** 1.0  
**Dataset:** 100,000 candidate profiles (JSONL, synthetic/simulated)  
**Status:** Pre-build analysis — feeds directly into scoring module design

---

## Table of Contents

1. [Data Dictionary](#1-data-dictionary)
2. [Data Quality Flags & Risk Assessment](#2-data-quality-flags--risk-assessment)
3. [Feature Engineering Plan](#3-feature-engineering-plan)
4. [Embedding vs. Structured Feature Allocation](#4-embedding-vs-structured-feature-allocation)
5. [Feature Validation Strategy](#5-feature-validation-strategy)

---

## 1. Data Dictionary

The dataset contains 100,000 JSONL records. Each record has six top-level sections. Below is a field-by-field inventory grounded in the statistics from the challenge brief.

---

### Section 1 — `profile` (Basic Candidate Info)

| Field | Type | Expected Range / Values | Notes |
|---|---|---|---|
| `candidate_id` | String | Unique identifier per record | Primary key; never use as a feature |
| `experience_years` | Float | 1.0 – 16.9 (avg: 7.2) | Continuous; core seniority signal |
| `country` | Categorical | India (75.1%), USA (10%), Australia / Canada / UK (~2.5% each) | Highly India-skewed |
| `industry` | Categorical | IT Services (30%), Software (22%), Manufacturing (22%), others | Top 3 cover ~74% of dataset |
| `current_title` | Categorical | 10 titles (~5.7–5.8% each): Business Analyst, HR Manager, Mechanical Engineer, Accountant, Project Manager, Customer Support, Operations Manager, Content Writer, Sales Executive, Civil Engineer | Near-uniform distribution |
| `current_company` | Categorical | ~10 mock companies (~7.4–7.6% each): Infosys, Wayne Enterprises, Wipro, Initech, Pied Piper, Globex Inc, Acme Corp, Dunder Mifflin, TCS, Hooli | Largely fictional; low signal value |
| `company_size` | Categorical | 40.5% at "10001+" employees; other tiers unspecified in stats | Right-skewed toward very large orgs |
| `location` | Categorical | ~10 Indian cities (~4.2% each): Bhubaneswar, Noida, Hyderabad, Jaipur, Bangalore, Kolkata, Indore, Pune, Chennai, Delhi | Near-uniform across cities |
| `expected_salary_min` | Integer (LPA) | ~12.2 LPA average | Continuous; use for JD salary alignment |
| `expected_salary_max` | Integer (LPA) | ~19.8 LPA average | Continuous; use for JD salary alignment |
| `open_to_work` | Boolean | True: 35.3%, False: 64.7% | Binary intent signal |
| `willing_to_relocate` | Boolean | True: 28.8%, False: 71.2% | Binary geo-flexibility signal |
| `work_mode_preference` | Categorical | ~25% each: hybrid, onsite, flexible, remote | Uniformly distributed; match to JD preference |
| `email_verified` | Boolean | True: 72.0% | Reliability signal, not a filter |
| `phone_verified` | Boolean | True: 61.8% | Reliability signal, not a filter |
| `linkedin_connected` | Boolean | True: 36.0% | Platform connectivity signal |
| `notice_period_days` | Integer | 0 – 150 (avg: 87 days) | Hire-timeline compatibility |

---

### Section 2 — `career_history[]` (Array of Role Objects)

Average 3 roles per candidate; range 1–9.

| Field | Type | Expected Range / Values | Notes |
|---|---|---|---|
| `company` | String | Free-text company name | Often fictional/mock in this dataset |
| `title` | String | Free-text job title | Variable; critical for seniority inference |
| `start_date` | Date | Inferred from durations | Use to compute tenure and recency |
| `end_date` | Date / Null | Null if current role | Null signals current position |
| `duration_months` | Integer | Derivable from dates | Tenure per role; key for stability scoring |
| `industry` | Categorical | Same distribution as profile industry | Role-level industry for domain-specific relevance |
| `company_size` | Categorical | Same distribution as profile company_size | Role-level org context |
| `description` | String | Free-text; variable length and quality | Primary semantic embedding target |

**Computed at read time:**
- `is_current_role`: `end_date` is null → True
- `role_recency_months`: months since end_date (or 0 if current)

---

### Section 3 — `education[]` (Array of Education Objects)

Average 1.4 education entries per candidate.

| Field | Type | Expected Range / Values | Notes |
|---|---|---|---|
| `degree` | Categorical | M.E. (17.6K), M.S. (17.6K), M.Sc (17.6K), M.Tech (17.5K), Ph.D (17.5K), B.Tech (17.5K), B.E. (17.3K), B.Sc (17.2K) | Near-perfectly uniform across 8 degree types |
| `field_of_study` | Categorical | IT, Data Science, ML, Computer Engineering, AI, CS (~12K each); Statistics, Chemical Eng, Electronics, Physics (~6.7K each) | Two tiers of frequency |
| `institution_tier` | Categorical | tier_1 (4.9%), tier_2 (19.9%), tier_3 (38.1%), tier_4 (37.1%) | Heavily skewed to lower tiers; tier_1 is rare |
| `graduation_year` | Integer | Derivable from experience + current year | Useful for experience timeline validation |

---

### Section 4 — `skills[]` (Array of Skill Objects)

960,302 total entries; 133 unique skills; average ~9.6 skills per candidate.

| Field | Type | Expected Range / Values | Notes |
|---|---|---|---|
| `name` | Categorical | 133 unique values | Skill identity; basis for semantic grouping |
| `proficiency` | Categorical | beginner / intermediate / advanced / expert | Self-reported; treat with skepticism |
| `endorsements` | Integer | 0 – 242 (dataset avg: ~30 endorsements received per candidate — note this is profile-level, not per-skill) | Social validation proxy |
| `duration_years` | Float | Not explicitly stated; inferred from career history | Years of active use of the skill |

**Most common skills (each at ~12% prevalence ~12,000+ candidates):**
HTML, Databricks, Redux, Terraform, Angular, Figma, Salesforce CRM, Vue.js, Sales, Accounting, Agile, Kafka, Excel, BigQuery, CI/CD, Project Management, Airflow, AWS, Flask, Scrum

---

### Section 5 — `redrob_signals` (Platform Behavioral Data)

All signals are at the candidate level (not per-role).

| Field | Type | Range | Avg | Notes |
|---|---|---|---|---|
| `profile_completeness_score` | Float (%) | 25.0 – 99.9 | 56.8% | How fully the candidate filled their profile |
| `connection_count` | Integer | 10 – 1,898 | 346 | Network size proxy |
| `endorsements_received` | Integer | 0 – 242 | 30 | Total endorsements across all skills |
| `notice_period_days` | Integer | 0 – 150 | 87 | Days until candidate can join |
| `profile_views_30d` | Integer | 0 – 374 | 48 | Recruiter interest in the candidate |
| `applications_submitted_30d` | Integer | 0 – 24 | 5.4 | Active job-seeking behavior |
| `recruiter_response_rate` | Float (0–1) | 0.02 – 0.95 | 0.40 | Rate at which recruiters engage back |
| `avg_response_time_hrs` | Float | 2.1 – 280 | 133 | Candidate responsiveness when contacted |
| `search_appearances_30d` | Integer | 0 – 1,490 | 118 | How often candidate appears in recruiter searches |
| `saved_by_recruiters_30d` | Integer | 0 – 80 | 7.7 | Passive demand signal |
| `interview_completion_rate` | Float (0–1) | 0.30 – 1.0 | 0.60 | Reliability in follow-through |
| `offer_acceptance_rate` | Float (0–1) | 0.15 – 0.93 | 0.50 | Conversion reliability |
| `github_activity_score` | Float | 0 – 96.9 | 29.0 | Technical open-source engagement proxy |
| `open_to_work` | Boolean | — | 35.3% True | Active intent declaration |
| `willing_to_relocate` | Boolean | — | 28.8% True | Geo-flexibility |
| `email_verified` | Boolean | — | 72.0% True | Data reliability signal |
| `phone_verified` | Boolean | — | 61.8% True | Data reliability signal |
| `linkedin_connected` | Boolean | — | 36.0% True | Platform integration depth |
| `work_mode_preference` | Categorical | hybrid / onsite / flexible / remote | ~25% each | Workplace fit signal |
| `expected_salary_min` | Integer (LPA) | — | ~12.2 | Budget compatibility lower bound |
| `expected_salary_max` | Integer (LPA) | — | ~19.8 | Budget compatibility upper bound |

---

### Section 6 — `certifications[]` and `languages[]`

| Field | Type | Expected Range / Values | Notes |
|---|---|---|---|
| `certifications[].name` | String | e.g., "AWS Certified Cloud Practitioner", "Scrum Master" | 25% of candidates hold at least one cert |
| `certifications[].issuer` | String | Free-text | Useful for cert prestige weighting |
| `certifications[].issue_date` | Date | — | Recency of cert; older certs decay in weight |
| `languages[].name` | Categorical | English (100%), Hindi (100%) | Zero discriminatory power in this dataset |
| `languages[].proficiency` | Categorical | Native/Fluent/etc. | Also zero variance since all have English |

---

## 2. Data Quality Flags & Risk Assessment

This section identifies patterns in the dataset that could distort model behavior if not handled explicitly. Each flag includes a modeling implication.

---

### Flag 1 — Near-Uniform Job Title Distribution ⚠️ HIGH RISK

**Observation:** The 10 main job titles each account for 5.7–5.8% of the 100K dataset. This is statistically impossible in a real talent pool — IT job titles dominate real recruitment databases, not a uniform spread across civil engineers, accountants, and content writers.

**What it implies for ranking:** The dataset is designed to test generalization, not to simulate a real company's ATS. The scoring pipeline must work equally well for a "Senior Data Engineer" JD and a "Civil Engineer" JD, without any implicit bias toward IT roles. Do not use `current_title` as a hard filter — it will perform artificially well on this synthetic dataset and fail on any real one.

**Mitigation:** Rely on semantic matching of the full career history text rather than title-to-title string matching.

---

### Flag 2 — Uniformly Distributed Skill Prevalence ⚠️ HIGH RISK

**Observation:** The top 20 skills each appear in approximately 12,000+ candidates (~12% prevalence each). This is also artificial — in a real dataset, AWS would appear far more frequently than Redux or Figma in a tech-focused pool.

**What it implies for ranking:** Term-frequency weighting (TF-IDF style) will produce misleading results. A skill appearing in 12% of candidates should theoretically have high discriminatory power (it's specific), but every skill has the same 12% prevalence, so IDF scores will be nearly flat. Standard TF-IDF matching cannot differentiate candidates based on this dataset.

**Mitigation:** Rely on proficiency level, endorsement count, and duration of experience to differentiate candidates with the same listed skill, rather than the rarity of the skill itself.

---

### Flag 3 — Self-Reported Skills Are Inflated ⚠️ MEDIUM RISK

**Observation:** Skills are self-reported with proficiency levels ranging from beginner to expert. There is no external validation of these claims other than `endorsements_received` (profile-level average of 30) and `github_activity_score`.

**What it implies for ranking:** A candidate listing "AWS — Expert — 6 months" is over-representing their capability. The system must weight skill claims by a combination of (a) proficiency level, (b) endorsement count, and (c) duration of experience before trusting them at face value.

**Mitigation:** Compute a `skill_strength_score` per skill (see Section 3). Treat `endorsements_received` as a weak-but-real validation signal. For technical skills, let `github_activity_score` serve as an independent cross-check.

---

### Flag 4 — The 35% Open-to-Work Split Means Most Strong Candidates Are Passive ⚠️ MEDIUM RISK

**Observation:** Only 35.3% of candidates are actively flagged as open to work. If the engine uses `open_to_work` as a gating filter or gives it heavy weight, it will exclude 64.7% of the candidate pool, which likely includes highly qualified passive candidates.

**What it implies for ranking:** Do not hard-filter on `open_to_work`. Use it as one positive intent signal among several in the B4 platform score. A passive candidate with exceptional semantic fit should rank above an active but mediocre candidate.

**Mitigation:** `open_to_work` contributes a bonus (e.g., +10–15% to the platform activity sub-score), not a prerequisite. Pair it with `applications_submitted_30d` and `profile_completeness_score` for a fuller intent picture.

---

### Flag 5 — Education Tier Distribution Is Heavily Skewed Toward Tier 3/4 ⚠️ LOW-MEDIUM RISK

**Observation:** tier_3 (38.1%) + tier_4 (37.1%) = 75.2% of candidates. Only 4.9% attended tier_1 institutions. The degree type distribution is near-perfectly uniform across 8 degree types.

**What it implies for ranking:** Using institution tier as a significant scoring weight will demote 75% of the pool for a dataset attribute that is likely a placeholder. The degree uniformity (17.2K–17.6K per type) is also artificial.

**Mitigation:** Use institution tier as a marginal tiebreaker only (e.g., +2–3% to overall score for tier_1/2). Never use it as a primary filter. Do not include degree type in the core scoring formula.

---

### Flag 6 — Company Names Are Mostly Fictional ⚠️ LOW RISK

**Observation:** ~70–75% of company names are mock entities (Wayne Enterprises, Pied Piper, Hooli, Globex Inc, Dunder Mifflin). These cannot be used for prestige-based company scoring (e.g., FAANG bonuses).

**What it implies for ranking:** Do not implement a "company prestige" scoring dimension. It will be meaningless for this dataset and will not generalize to real deployments anyway.

**Mitigation:** Exclude company name from scoring entirely. Use company size from `career_history[].company_size` as a structural proxy (large-org experience is a valid signal regardless of the company's name).

---

### Flag 7 — Career History Description Quality Is Unknown ⚠️ HIGH RISK

**Observation:** Career history descriptions are free-text with "variable quality" (per PRD Risk 4). In a synthetic dataset, these may be short, formulaic, or semantically thin.

**What it implies for ranking:** Semantic embedding of career descriptions is the core of the B1 scoring dimension. If descriptions are sparse ("Managed team," "Wrote reports"), the embedding space will compress and lose discriminatory power.

**Mitigation (per PRD):** Inspect 20–30 random career descriptions before finalizing the embedding strategy. If sparse, construct a richer "candidate document" by concatenating: `skills[].name + proficiency` + `career_history[].title` + `career_history[].description` into a single embedding input. Do not embed description alone.

---

### Flag 8 — GitHub Activity Score Is a Single Scalar ⚠️ LOW RISK

**Observation:** `github_activity_score` ranges 0–96.9 with an average of 29.0. It is a platform-derived scalar with no breakdown (commits, repos, stars, etc.).

**What it implies for ranking:** It is a meaningful signal for engineering/technical JDs only. For non-technical roles (HR Manager, Content Writer, Accountant), a high GitHub score is irrelevant and should not boost or penalize the candidate.

**Mitigation:** Gate `github_activity_score` usage on JD domain. In the JD parser output, include a `requires_technical_github_signals` boolean. Only include GitHub score in B4 computation when this flag is True.

---

### Flag 9 — Uniform Geographic Distribution ⚠️ LOW RISK

**Observation:** Candidates are spread equally across ~10 Indian cities (~4.2% each). This is not realistic but also not problematic — it simply means location-based filtering has low discriminatory power.

**What it implies for ranking:** Do not build a location-proximity scoring dimension for this dataset. It will compress to near-zero variance. If the JD specifies a location requirement, apply a binary compatibility check using `willing_to_relocate` rather than a continuous proximity score.

---

## 3. Feature Engineering Plan

This section defines all derived features for the three matching dimensions in the PRD: Semantic Fit (B1), Career Trajectory (B2), and Stability & Activity (B3+B4). Each feature is specified with its formula, input fields, expected range, and rationale.

---

### Dimension 1 — Semantic Fit Features (B1)

#### Feature 1.1 — `candidate_embedding_vector`

**What it is:** A dense vector representation of the candidate's full professional text profile.

**Input fields:**
- `career_history[].title` (all roles, concatenated)
- `career_history[].description` (all roles, concatenated)
- `skills[].name` (all skills, listed with proficiency level as context)
- `certifications[].name` (if present)

**Construction:**
```
candidate_doc = (
  "Skills: " + ", ".join(f"{s.name} ({s.proficiency})" for s in skills) +
  " | Career: " + " | ".join(f"{r.title} at {r.industry}: {r.description}" for r in career_history) +
  " | Certs: " + ", ".join(c.name for c in certifications)
)
embedding = SentenceTransformer.encode(candidate_doc)
```

**Output:** Dense vector (e.g., 384 dimensions for `all-MiniLM-L6-v2`). Pre-computed once and stored to disk.

**Rationale:** Captures semantic meaning that keyword matching misses. "AWS, GCP, Terraform" maps close to "Cloud Expert" in embedding space. Concatenation ensures the embedding reflects the full professional identity, not just one field.

---

#### Feature 1.2 — `skill_strength_score` (per skill, and aggregated)

**What it is:** A weighted score per listed skill that adjusts the nominal skill entry by quality signals.

**Formula:**
```
proficiency_weight = {beginner: 0.25, intermediate: 0.5, advanced: 0.75, expert: 1.0}

skill_strength(s) = (
  proficiency_weight[s.proficiency] * 0.5 +
  min(s.duration_years / 5.0, 1.0) * 0.35 +
  min(s.endorsements / 50, 1.0) * 0.15
)
```

**Output:** Float in [0, 1] per skill. Aggregated as a dict `{skill_name: strength_score}` per candidate.

**Rationale:** Deflates inflated self-reported skills. A candidate with "AWS — Expert — 6 months — 0 endorsements" scores much lower than "AWS — Advanced — 3 years — 20 endorsements." The 0.5/0.35/0.15 split prioritizes proficiency and duration over endorsements.

---

#### Feature 1.3 — `jd_skill_coverage_score`

**What it is:** Fraction of JD-required skills a candidate demonstrably covers, weighted by skill strength.

**Formula:**
```
jd_skills = [skills extracted from JD intent object with semantic expansion]
coverage_score = mean(skill_strength.get(jd_skill, semantic_fallback_score) for jd_skill in jd_skills)
```

Where `semantic_fallback_score` is the cosine similarity between the candidate embedding and the JD skill term embedding, allowing synonymous skills to count even if not explicitly listed.

**Output:** Float in [0, 1].

---

### Dimension 2 — Career Trajectory Features (B2)

#### Feature 2.1 — `trajectory_score`

**What it is:** A composite signal of whether the candidate's career shows upward progression in seniority and scope.

**Input fields:** `career_history[].title`, `career_history[].company`, `career_history[].duration_months`

**Sub-components:**

**a. Promotion Detection:**
```
for each company where candidate held ≥2 roles:
    sort roles by start_date
    if later_role.title contains higher seniority keyword (Senior, Lead, Principal, Manager, Director, VP, Head)
    and earlier_role.title does not → count as detected_promotion
promotion_rate = detected_promotions / total_companies_with_multiple_roles
```

**b. Seniority Level Mapping:**
```
seniority_keywords = {
  junior: 0.2,   # Junior, Associate, Intern, Trainee
  mid: 0.5,      # (no qualifier)
  senior: 0.75,  # Senior, Lead
  staff_plus: 1.0  # Principal, Staff, Manager, Director, VP, Head
}
latest_seniority = map_title(career_history[-1].title)  # most recent role
```

**c. Trajectory Score:**
```
trajectory_score = (
  latest_seniority * 0.5 +
  promotion_rate * 0.3 +
  min(profile.experience_years / 10.0, 1.0) * 0.2
)
```

**Output:** Float in [0, 1]. A strong mid-level with promotions scores ~0.6–0.7, comparable to a senior with no promotions. This enables "stretch readiness" detection.

---

#### Feature 2.2 — `seniority_fit_score`

**What it is:** Measures alignment between the candidate's actual seniority level and the JD's required seniority level. Penalizes both underqualification and overqualification.

**Formula:**
```
jd_seniority = parsed from JD intent object (0.0–1.0 scale)
candidate_seniority = latest_seniority from Feature 2.1

seniority_gap = abs(candidate_seniority - jd_seniority)

if seniority_gap == 0: fit = 1.0
elif seniority_gap == 0.25: fit = 0.8  # one level off
elif seniority_gap == 0.5: fit = 0.5   # two levels off
else: fit = max(0.2, 1.0 - seniority_gap * 2)
```

**Special case — Stretch Readiness:** If `candidate_seniority == mid (0.5)` and `jd_seniority == senior (0.75)` and `trajectory_score > 0.65`, apply a stretch bonus: `fit = 0.75` instead of 0.8 (treat as near-match, not one full level down).

**Output:** Float in [0, 1].

---

#### Feature 2.3 — `domain_experience_years`

**What it is:** Total years of experience in roles that semantically match the JD's domain, as opposed to total career experience.

**Formula:**
```
for each role in career_history:
    domain_sim = cosine_similarity(embed(role.title + " " + role.description), jd_embedding)
    if domain_sim > 0.6:
        domain_experience_months += role.duration_months

domain_experience_years = domain_experience_months / 12.0
```

**Output:** Float (years). A PM with 8 years experience but only 2 in the relevant domain scores lower than one with 4 focused years.

---

### Dimension 3 — Stability Features (B3)

#### Feature 3.1 — `avg_tenure_months`

**What it is:** Mean duration across all career roles.

**Formula:**
```
avg_tenure = mean(r.duration_months for r in career_history)
```

**Output:** Float (months). Dataset average tenure derivable from 3 roles per candidate and ~7.2 years total experience ≈ 28.8 months average per role.

---

#### Feature 3.2 — `job_hopping_flag`

**What it is:** Binary flag indicating concerning short-tenure pattern.

**Formula:**
```
short_tenure_roles = [r for r in career_history if r.duration_months < 12]
consecutive_short = max_consecutive_run(short_tenure_roles in timeline order)
job_hopping_flag = 1 if consecutive_short >= 3 else 0
```

**Output:** Binary (0/1). Signals potential flight risk.

---

#### Feature 3.3 — `stability_score`

**What it is:** Composite career stability signal.

**Formula:**
```
# Normalize avg_tenure: 24 months is neutral, >36 is strong, <12 is weak
tenure_norm = min(avg_tenure_months / 36.0, 1.0)

# Penalize job hopping
hopping_penalty = 0.3 if job_hopping_flag else 0.0

# Light education tier bonus
edu_tier_bonus = {tier_1: 0.05, tier_2: 0.03, tier_3: 0.01, tier_4: 0.0}
max_edu_bonus = max(edu_tier_bonus.get(e.institution_tier, 0) for e in education)

stability_score = min(tenure_norm - hopping_penalty + max_edu_bonus, 1.0)
```

**Output:** Float clamped to [0, 1].

---

### Dimension 4 — Platform Activity & Intent Features (B4)

#### Feature 4.1 — `active_intent_score`

**What it is:** How actively and seriously this candidate is pursuing new opportunities right now.

**Formula:**
```
active_intent_score = (
  (1.0 if open_to_work else 0.4) * 0.35 +                          # 0.4 if passive (not 0 — passive candidates are still viable)
  min(applications_submitted_30d / 10.0, 1.0) * 0.25 +             # max at 10 apps/month
  min(profile_completeness_score / 100.0, 1.0) * 0.20 +            # reward completeness
  min(search_appearances_30d / 200.0, 1.0) * 0.20                   # recruiter demand signal
)
```

**Output:** Float in [0, 1]. A passive candidate who appears 200 times in recruiter searches still scores ~0.6.

---

#### Feature 4.2 — `hire_reliability_score`

**What it is:** How likely the candidate is to complete the hiring process and accept an offer.

**Formula:**
```
hire_reliability_score = (
  interview_completion_rate * 0.40 +
  offer_acceptance_rate * 0.30 +
  (1.0 - min(avg_response_time_hrs / 200.0, 1.0)) * 0.20 +    # faster response is better
  (0.5 if email_verified else 0.0 + 0.5 if phone_verified else 0.0) * 0.10
)
```

**Output:** Float in [0, 1].

---

#### Feature 4.3 — `technical_engagement_score`

**What it is:** Platform signals of technical credibility, only applied when JD domain is technical.

**Formula (applied conditionally):**
```
if jd_requires_technical_signals:
    technical_engagement_score = (
      min(github_activity_score / 96.9, 1.0) * 0.60 +
      min(endorsements_received / 100.0, 1.0) * 0.40
    )
else:
    technical_engagement_score = None  # excluded from scoring
```

**Output:** Float in [0, 1] or None.

---

#### Feature 4.4 — `platform_score` (B4 composite)

**What it is:** The final B4 sub-score.

**Formula:**
```
if technical_engagement_score is not None:
    platform_score = (
      active_intent_score * 0.40 +
      hire_reliability_score * 0.35 +
      technical_engagement_score * 0.25
    )
else:
    platform_score = (
      active_intent_score * 0.55 +
      hire_reliability_score * 0.45
    )
```

**Output:** Float in [0, 1].

---

### Dimension 5 — Certifications Bonus (B5)

#### Feature 5.1 — `cert_bonus`

**What it is:** A small additive bonus for relevant, recent certifications.

**Formula:**
```
for cert in certifications:
    relevance = cosine_similarity(embed(cert.name), jd_embedding)
    recency_weight = max(0.5, 1.0 - (current_year - cert.issue_year) * 0.1)  # 10% decay per year, floor 0.5
    cert_contributions.append(relevance * recency_weight)

cert_bonus = min(max(cert_contributions, default=0), 0.1)  # capped at 0.10 additive
```

**Output:** Float in [0, 0.10]. Capped to prevent certifications from dominating the composite score.

---

### Composite Score Formula

```
composite_score = (
  semantic_score   * 0.35 +   # B1 — embedding similarity + skill coverage
  trajectory_score * 0.25 +   # B2 — seniority fit + career trajectory
  stability_score  * 0.15 +   # B3 — tenure, hopping, education
  platform_score   * 0.20 +   # B4 — intent + reliability + github
  cert_bonus       * 0.05     # B5 — additive cert relevance
)
```

**Weight rationale:**
- Semantic skill match (0.35) is the primary differentiator; it replaces keyword matching.
- Trajectory (0.25) is the PRD's key innovation — detecting stretch readiness requires meaningful weight.
- Platform (0.20) is this dataset's unique differentiator vs. traditional ATS; weighted higher than stability.
- Stability (0.15) matters but should not gate high-performing candidates with one short stint.
- Certifications (0.05) are a bonus, not a core ranking factor.

---

## 4. Embedding vs. Structured Feature Allocation

A clear boundary between what goes into the embedding vs. what stays structured prevents double-counting and model confusion.

### Use Semantic Embedding For

| Field / Content | Reason |
|---|---|
| `career_history[].description` (all roles concatenated) | Rich free text; full of domain intent, impact language, and implicit skill signals |
| `career_history[].title` (all roles) | Job titles carry domain and seniority signals that embedding handles better than string matching |
| `skills[].name + proficiency` (as a formatted string) | Enables synonym resolution ("Terraform" ↔ "Infrastructure as Code") |
| `certifications[].name` | Short but semantically meaningful for relevance matching |
| **JD full text** | Embed once per query to produce the JD vector for ANN retrieval |

### Use Structured / Numeric Features For

| Field | Reason |
|---|---|
| `profile.experience_years` | Continuous number; better as a direct comparison, not embedded |
| `career_history[].duration_months` | Arithmetic (avg, max, consecutive patterns) not semantic |
| `redrob_signals.*` (all numeric signals) | Scalar comparisons, ratio computations; embedding them adds no value |
| `skills[].proficiency` | Ordinal categorical; use as a multiplier weight, not embedded text |
| `skills[].endorsements` | Integer count; direct normalization and weighting |
| `profile.open_to_work`, `email_verified`, etc. | Binary flags; direct use in formula |
| `education[].institution_tier` | Ordinal categorical tiebreaker |
| `expected_salary_min` / `expected_salary_max` | Numeric range comparison with JD budget |

### Hybrid Fields (Both)

| Field | How to Split |
|---|---|
| `career_history[].title` | Embed as text for domain/seniority semantic matching; also run through `seniority_keyword_map` for structured seniority score |
| `industry` | Embed the domain description for semantic matching; also use as a categorical filter if JD specifies an industry hard requirement |

---

## 5. Feature Validation Strategy

Before trusting engineered features in the ranking pipeline, validate each against "golden test cases" — 5–10 manually constructed JDs with known correct top-3 candidates.

### Validation 1 — Embedding Recall Check (for `candidate_embedding_vector`)

**Test:** Embed 3 test JDs (one engineering, one non-technical, one mixed). Run ANN retrieval to get top-500 candidates. Manually verify: are candidates who obviously match the JD domain present in the top-500?

**Pass criterion:** 90%+ of manually identified strong matches appear in the top-500. If not, switch to a stronger embedding model (`bge-base-en-v1.5`) or add BM25 hybrid retrieval.

---

### Validation 2 — Skill Strength Distribution Check (for `skill_strength_score`)

**Test:** Compute `skill_strength_score` for all 100K candidates across their top skill. Plot the distribution.

**Pass criterion:** The distribution should NOT be uniform. If it is, the weighting formula is not differentiating — likely because endorsements and duration data are also synthetically uniform. In that case, increase proficiency weight to 0.70 and reduce endorsement weight to 0.05.

---

### Validation 3 — Trajectory Score Sanity Check (for `trajectory_score`)

**Test:** Sample 50 candidates with clearly senior titles (Director, VP, Head) and 50 with junior titles. Compute `trajectory_score` for each group.

**Pass criterion:** Senior-titled group should have mean `trajectory_score` ≥ 0.65; junior group ≤ 0.40. If overlap is high, the seniority keyword map needs expansion.

---

### Validation 4 — Job Hopping Flag Rate (for `job_hopping_flag`)

**Test:** Compute the flag across all 100K candidates. What fraction gets flagged?

**Pass criterion:** Expect 5–15% flagging rate. If > 30%, the synthetic dataset may have unrealistically many short-tenure roles — loosen the threshold to 4+ consecutive short tenures. If < 2%, the data is too uniform and the flag is useless.

---

### Validation 5 — Platform Score Distribution (for `platform_score`)

**Test:** Plot `platform_score` distribution across all 100K candidates.

**Pass criterion:** Should show a roughly normal distribution centered around 0.45–0.55 (given dataset averages). A bimodal or near-uniform distribution suggests formula weights need rebalancing. In particular, verify that passive candidates (`open_to_work=False`) are NOT systematically scoring below 0.3 (that would effectively gate them out).

---

### Validation 6 — End-to-End Ranking Sanity (Full Pipeline)

**Test:** Run the full pipeline on a "golden JD" where you have manually identified 3 candidates from the dataset who are clearly the best matches (by reading their profiles).

**Pass criterion:** All 3 appear in the top 20 of the ranked output. If not, identify which sub-score failed to elevate them and adjust weights accordingly.

---

### Validation 7 — Explanation Grounding Check (for LLM explanations)

**Test:** Generate explanations for the top 10 candidates from the golden JD test. For each explanation, verify: does it mention at least one specific data point from the candidate's actual profile (a specific skill name, company, title, or tenure figure)?

**Pass criterion:** 10/10 explanations must contain at least one candidate-specific datum. If any are generic, tighten the prompt template to require explicit field citations.

---

## Appendix — Feature Summary Table

| Feature | Module | Type | Range | Primary Input Fields |
|---|---|---|---|---|
| `candidate_embedding_vector` | B1 | Dense vector | 384-dim | career descriptions, titles, skills, certs |
| `skill_strength_score` | B1 | Float per skill | [0, 1] | skills.proficiency, skills.duration, endorsements |
| `jd_skill_coverage_score` | B1 | Float | [0, 1] | skill_strength_scores vs. JD skills |
| `trajectory_score` | B2 | Float | [0, 1] | career titles, promotion detection, experience_years |
| `seniority_fit_score` | B2 | Float | [0, 1] | trajectory_score vs. JD seniority |
| `domain_experience_years` | B2 | Float | 0–17 | career roles vs. JD embedding similarity |
| `avg_tenure_months` | B3 | Float | 1–200 | career_history.duration_months |
| `job_hopping_flag` | B3 | Binary | 0/1 | career_history.duration_months (consecutive) |
| `stability_score` | B3 | Float | [0, 1] | avg_tenure, hopping_flag, institution_tier |
| `active_intent_score` | B4 | Float | [0, 1] | open_to_work, applications_30d, completeness, search_appearances |
| `hire_reliability_score` | B4 | Float | [0, 1] | interview_completion_rate, offer_acceptance_rate, response_time, verifications |
| `technical_engagement_score` | B4 | Float / None | [0, 1] | github_activity_score, endorsements_received |
| `platform_score` | B4 | Float | [0, 1] | active_intent + hire_reliability + technical_engagement |
| `cert_bonus` | B5 | Float | [0, 0.10] | certifications.name vs. JD embedding, cert recency |
| `composite_score` | Final | Float | [0, 1.10] | Weighted sum of B1–B5 |

---

*Report owner: Data/ML lead*  
*Last updated: June 2026*  
*Status: Approved — feeds into module specification for scoring pipeline*
