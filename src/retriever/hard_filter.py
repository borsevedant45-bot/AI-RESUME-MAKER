import logging
from typing import Any

from src.models import JDIntent, CandidateFeatureRow

logger = logging.getLogger(__name__)


def apply_hard_filters(
    shortlist: list[str],
    feature_store: dict[str, CandidateFeatureRow],
    jd_intent: JDIntent,
) -> list[str]:
    """
    Applies binary compatibility filters ONLY when JDIntent explicitly states
    a hard constraint (salary_stated and salary_max_lpa, or location_is_hard_requirement).
    Never filters on: open_to_work, notice_period, work_mode, education tier.
    """
    ctx = jd_intent.work_context
    filtered = []
    removed_salary = 0
    removed_location = 0

    for cand_id in shortlist:
        f = feature_store.get(cand_id)
        if f is None:
            continue

        if ctx.salary_max_lpa is not None and jd_intent.salary_stated:
            if f.expected_salary_min > ctx.salary_max_lpa:
                removed_salary += 1
                continue

        if ctx.location_is_hard_requirement and ctx.location_required:
            jd_loc = ctx.location_required.lower()
            cand_loc = f.location.lower()
            if jd_loc not in cand_loc and not f.willing_to_relocate:
                removed_location += 1
                continue

        filtered.append(cand_id)

    if removed_salary:
        logger.info("Hard filter (salary) removed %d candidates", removed_salary)
    if removed_location:
        logger.info("Hard filter (location) removed %d candidates", removed_location)

    return filtered

DOMAIN_TITLE_KEYWORDS = {
    "hr": ["hr manager"],
    "finance": ["accountant"],
    "data-engineering": ["data engineer", "business analyst", "data analyst", "analytics engineer"],
    "project-management": ["project manager", "operations manager"],
    "mlops": ["ml engineer", "data scientist"],
    "frontend": ["frontend engineer"],
    "sales": ["sales executive"],
    "content": ["content writer"],
    "mechanical": ["mechanical engineer"],
    "civil": ["civil engineer"],
}

def apply_domain_title_filter(
    shortlist: list[str],
    feature_store: dict[str, CandidateFeatureRow],
    jd_intent: JDIntent | dict[str, Any],
) -> list[str]:
    """
    Post-retrieval filter: keeps only candidates whose current_title
    is broadly compatible with the JD domain tags.
    Falls back to full shortlist if fewer than 100 candidates pass.
    """
    domain_tags = (
        jd_intent.domain_tags if isinstance(jd_intent, JDIntent)
        else jd_intent.get("domain_tags", [])
    )
    if not domain_tags:
        return shortlist

    title_keywords = []
    for tag in domain_tags:
        title_keywords.extend(DOMAIN_TITLE_KEYWORDS.get(tag, []))

    if not title_keywords:
        return shortlist

    filtered = []
    for cid in shortlist:
        fr = feature_store.get(cid)
        if fr is None:
            continue
        title_text = (fr.current_title or "").lower()
        if any(kw in title_text for kw in title_keywords):
            filtered.append(cid)

    if len(filtered) < 50:
        return shortlist

    return filtered