def validate_candidate(raw: dict) -> tuple[bool, str]:
    """
    Checks that a raw candidate dict has the minimum required fields:
    - candidate_id (non-empty)
    - profile.years_of_experience (non-negative number)
    - skills (non-empty list)
    - career_history (non-empty list)
    - redrob_signals (present)
    """
    if not raw.get("candidate_id"):
        return False, "missing candidate_id"

    profile = raw.get("profile", {})
    if not profile:
        return False, "missing profile block"

    yoe = profile.get("years_of_experience")
    if yoe is None or not isinstance(yoe, (int, float)) or yoe < 0:
        return False, "invalid or missing years_of_experience"

    if not raw.get("skills"):
        return False, "skills list is empty or missing"

    if not raw.get("career_history"):
        return False, "career_history list is empty or missing"

    if "redrob_signals" not in raw or raw["redrob_signals"] is None:
        return False, "redrob_signals block is missing"

    return True, ""
