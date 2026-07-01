from dataclasses import dataclass, field
from typing import Optional


# ── JD Parser Models ────────────────────────────────────────────────────────────

@dataclass
class WorkContext:
    work_mode: Optional[str] = None
    location_required: Optional[str] = None
    location_is_hard_requirement: bool = False
    salary_min_lpa: Optional[int] = None
    salary_max_lpa: Optional[int] = None


@dataclass
class JDIntent:
    seniority_level: float = 0.5
    seniority_evidence: str = ""
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    core_problems_to_solve: str = ""
    implicit_soft_skills: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    requires_technical_github_signals: bool = False
    work_context: WorkContext = field(default_factory=WorkContext)
    salary_stated: bool = False


# ── Candidate Profile Models ────────────────────────────────────────────────────

@dataclass
class SkillRecord:
    name: str = ""
    proficiency: str = "beginner"
    endorsements: int = 0
    duration_years: float = 0.0


@dataclass
class RoleRecord:
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: Optional[str] = None
    duration_months: int = 0
    industry: str = ""
    company_size: str = ""
    description: str = ""
    is_current_role: bool = False


@dataclass
class EducationRecord:
    degree: str = ""
    field_of_study: str = ""
    institution_tier: str = "tier_3"
    graduation_year: Optional[int] = None


@dataclass
class CertRecord:
    name: str = ""
    issuer: str = ""
    issue_year: int = 0


@dataclass
class RedrobSignals:
    profile_completeness_score: float = 0.0
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    profile_views_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hrs: float = 0.0
    search_appearances_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = 0.0
    github_activity_score: float = -1.0
    open_to_work: bool = False
    willing_to_relocate: bool = False
    email_verified: bool = False
    phone_verified: bool = False
    linkedin_connected: bool = False
    work_mode_preference: str = "hybrid"
    expected_salary_min: int = 0
    expected_salary_max: int = 0


@dataclass
class CandidateProfile:
    candidate_id: str = ""
    experience_years: float = 0.0
    country: str = ""
    industry: str = ""
    current_title: str = ""
    current_company: str = ""
    company_size: str = ""
    location: str = ""
    skills: list[SkillRecord] = field(default_factory=list)
    career_history: list[RoleRecord] = field(default_factory=list)
    education: list[EducationRecord] = field(default_factory=list)
    certifications: list[CertRecord] = field(default_factory=list)
    redrob_signals: RedrobSignals = field(default_factory=RedrobSignals)


# ── Feature Store Model ─────────────────────────────────────────────────────────

@dataclass
class CandidateFeatureRow:
    candidate_id: str = ""
    embedding_index: int = -1
    current_title: str = ""
    latest_seniority: float = 0.5
    promotion_rate: float = 0.0
    experience_years: float = 0.0
    avg_tenure_months: float = 0.0
    job_hopping_flag: int = 0
    institution_tier: str = "tier_3"
    active_intent_score: float = 0.0
    hire_reliability_score: float = 0.0
    github_activity_score: float = -1.0
    endorsements_received: int = 0
    open_to_work: bool = False
    willing_to_relocate: bool = False
    work_mode_preference: str = "hybrid"
    notice_period_days: int = 0
    expected_salary_min: int = 0
    expected_salary_max: int = 0
    location: str = ""
    skill_strength_scores: dict[str, float] = field(default_factory=dict)
    cert_records: list[dict] = field(default_factory=list)
    thin_profile: bool = False


# ── Scoring Models ──────────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    candidate_id: str = ""
    semantic_score: float = 0.0
    trajectory_score: float = 0.0
    stability_score: float = 0.0
    platform_score: float = 0.0
    cert_bonus: float = 0.0
    composite_score: float = 0.0


# ── Explainability Models ───────────────────────────────────────────────────────

@dataclass
class CandidateExplanation:
    candidate_id: str = ""
    match_summary: str = ""
    skill_alignment: str = ""
    seniority_assessment: str = ""
    trajectory_signal: str = ""
    platform_summary: str = ""
    flags: str = ""
    grounding_validated: bool = False


@dataclass
class RankedResult:
    rank: int = 0
    candidate_id: str = ""
    composite_score: float = 0.0
    semantic_score: float = 0.0
    trajectory_score: float = 0.0
    stability_score: float = 0.0
    platform_score: float = 0.0
    cert_bonus: float = 0.0
    explanation: CandidateExplanation = field(default_factory=CandidateExplanation)
