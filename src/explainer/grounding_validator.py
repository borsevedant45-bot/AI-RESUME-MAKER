import logging
from src.models import CandidateExplanation, CandidateFeatureRow

logger = logging.getLogger(__name__)


def validate_grounding(
    explanation: CandidateExplanation,
    feature_row: CandidateFeatureRow,
    career_history: list[dict] | None = None,
) -> bool:
    """
    Verifies the explanation references at least one concrete datum from the
    candidate's actual profile: skill names, company names, role titles,
    cert names, tenure numbers, or platform metric values.
    """
    full_text = " ".join([
        explanation.match_summary,
        explanation.skill_alignment,
        explanation.seniority_assessment,
        explanation.trajectory_signal,
        explanation.platform_summary,
        explanation.flags,
    ]).lower()

    grounding_candidates: list[str] = (
        list(feature_row.skill_strength_scores.keys()) +
        [str(feature_row.notice_period_days),
         str(int(feature_row.avg_tenure_months)),
         str(feature_row.experience_years)]
    )

    history = career_history or []
    for r in history:
        if r.get("company"):
            grounding_candidates.append(r["company"].lower())
        if r.get("title"):
            grounding_candidates.append(r["title"].lower())

    for c in feature_row.cert_records:
        if c.get("name"):
            grounding_candidates.append(c["name"].lower())

    result = any(datum in full_text for datum in grounding_candidates if datum)
    if not result:
        logger.debug("Grounding failed for %s. Candidates checked: %s", feature_row.candidate_id, grounding_candidates[:10])
    return result


def build_fallback_explanation(
    explanation: CandidateExplanation,
    feature_row: CandidateFeatureRow,
) -> CandidateExplanation:
    """Template-populated fallback when LLM explanation fails grounding."""
    top_skills = sorted(
        feature_row.skill_strength_scores.items(),
        key=lambda x: -x[1],
    )[:5]
    skills_str = ", ".join(f"{s[0]} ({s[1]:.2f})" for s in top_skills) if top_skills else "Not specified"

    flags = []
    if feature_row.job_hopping_flag:
        flags.append("Job hopping risk")
    if feature_row.notice_period_days > 90:
        flags.append(f"Long notice period ({feature_row.notice_period_days} days)")

    explanation.match_summary = (
        f"Candidate scored {feature_row.candidate_id} based on "
        f"{feature_row.experience_years:.1f} years experience with skills: {skills_str}."
    )
    explanation.skill_alignment = f"Key skills present: {skills_str}"
    explanation.seniority_assessment = f"Seniority level: {feature_row.latest_seniority:.2f}"
    explanation.trajectory_signal = (
        f"Promotion rate: {feature_row.promotion_rate:.2f}; "
        f"Average tenure: {feature_row.avg_tenure_months:.1f} months"
    )
    explanation.platform_summary = (
        f"Active intent score: {feature_row.active_intent_score:.2f}; "
        f"Hire reliability: {feature_row.hire_reliability_score:.2f}"
    )
    explanation.flags = "; ".join(flags) if flags else "No flags"
    explanation.grounding_validated = False
    return explanation
