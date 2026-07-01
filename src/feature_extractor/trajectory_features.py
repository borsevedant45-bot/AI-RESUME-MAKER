import logging
from collections import defaultdict

from src.models import CandidateProfile, RoleRecord

logger = logging.getLogger(__name__)

SENIORITY_MAP = {
    "intern": 0.1, "trainee": 0.1,
    "junior": 0.2, "associate": 0.2, "entry": 0.2, "graduate": 0.2,
    "senior": 0.75, "lead": 0.75,
    "principal": 1.0, "staff": 1.0,
    "manager": 0.75,
    "director": 1.0, "vp": 1.0, "head": 1.0,
    "chief": 1.0, "cto": 1.0, "ceo": 1.0,
}


def map_title_to_seniority(title: str) -> float:
    """Scans a job title for seniority keywords and returns a level float."""
    title_lower = title.lower()
    for keyword, level in SENIORITY_MAP.items():
        if keyword in title_lower:
            return level
    return 0.5


def detect_promotions(career_history: list[RoleRecord]) -> float:
    """
    Groups roles by company, sorts by start_date, detects seniority increases.
    Returns promotion_rate = detected_promotions / eligible_companies.
    """
    companies = defaultdict(list)
    for role in career_history:
        companies[role.company].append(role)

    promotions = 0
    eligible = 0

    for co, roles in companies.items():
        if len(roles) < 2:
            continue
        eligible += 1
        sorted_roles = sorted(roles, key=lambda r: r.start_date)
        levels = [map_title_to_seniority(r.title) for r in sorted_roles]
        if levels[-1] > levels[0]:
            promotions += 1

    return promotions / eligible if eligible > 0 else 0.0


def compute_trajectory_base(profile: CandidateProfile) -> dict:
    """Returns dict with keys: latest_seniority, promotion_rate."""
    if not profile.career_history:
        return {"latest_seniority": 0.5, "promotion_rate": 0.0}

    latest_title = sorted(profile.career_history, key=lambda r: r.start_date, reverse=True)[0].title
    return {
        "latest_seniority": map_title_to_seniority(latest_title),
        "promotion_rate": detect_promotions(profile.career_history),
    }
