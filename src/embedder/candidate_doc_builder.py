import logging
from typing import Any

from src.models import CandidateProfile

logger = logging.getLogger(__name__)


def build_jd_query_doc(jd_intent: dict[str, Any]) -> str:
    parts = [
        "Core problems: " + jd_intent.get("core_problems_to_solve", ""),
        "Required skills: " + ", ".join(jd_intent.get("must_have_skills", [])),
        "Nice to have: " + ", ".join(jd_intent.get("nice_to_have_skills", [])),
        "Soft skills: " + ", ".join(jd_intent.get("implicit_soft_skills", [])),
        "Domain: " + ", ".join(jd_intent.get("domain_tags", [])),
    ]
    return " | ".join(p for p in parts if p.split(": ", 1)[1])


def build_candidate_doc(profile: CandidateProfile) -> tuple[str, bool]:
    """
    Constructs the candidate embedding document from skills, career history,
    certifications, and education field of study.
    Most-recent roles appear first (recency bias in concatenation).
    Excludes: candidate_id, location, company names, raw numeric signals.

    Returns:
        Tuple of (doc_string, thin_profile_flag).
    """
    skill_parts = [
        f"{s.name} ({s.proficiency})"
        for s in profile.skills
    ]
    skills_str = "Skills: " + ", ".join(skill_parts) if skill_parts else ""

    sorted_roles = sorted(
        profile.career_history,
        key=lambda r: r.start_date,
        reverse=True,
    )
    role_parts = []
    for role in sorted_roles:
        desc = (role.description or "").strip()
        if desc and len(desc) >= 20:
            role_str = f"{role.title} in {role.industry}: {desc}"
        else:
            role_str = f"{role.title} in {role.industry}"
        role_parts.append(role_str)
    career_str = "Career: " + " | ".join(role_parts) if role_parts else ""

    cert_parts = [c.name for c in profile.certifications]
    certs_str = "Certifications: " + ", ".join(cert_parts) if cert_parts else ""

    edu_parts = [
        e.field_of_study
        for e in profile.education
        if e.field_of_study
    ]
    edu_str = "Education: " + ", ".join(edu_parts) if edu_parts else ""

    doc = " | ".join(filter(None, [skills_str, career_str, certs_str, edu_str]))
    thin = len(doc.strip()) < 50

    if thin:
        logger.debug("Thin profile detected for candidate %s (doc length: %d)", profile.candidate_id, len(doc))

    return doc, thin
