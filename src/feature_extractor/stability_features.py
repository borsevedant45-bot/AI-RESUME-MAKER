from src.models import RoleRecord, EducationRecord

TIER_ORDER = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4}


def compute_avg_tenure(career_history: list[RoleRecord]) -> float:
    """Returns mean duration_months across all roles. Returns 0.0 for empty history."""
    if not career_history:
        return 0.0
    return sum(r.duration_months for r in career_history) / len(career_history)


def detect_job_hopping(career_history: list[RoleRecord]) -> int:
    """
    Returns 1 if 3+ consecutive roles each had duration_months < 12, else 0.
    Consecutive is defined by chronological sort on start_date.
    """
    sorted_roles = sorted(career_history, key=lambda r: r.start_date)
    max_consecutive = 0
    current_run = 0

    for role in sorted_roles:
        if role.duration_months < 12:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 0

    return 1 if max_consecutive >= 3 else 0


def best_institution_tier(education: list[EducationRecord]) -> str:
    """Returns the highest (lowest-number) tier across all education entries."""
    if not education:
        return "tier_4"
    best = min(
        (e.institution_tier for e in education),
        key=lambda t: TIER_ORDER.get(t, 4),
    )
    return best
