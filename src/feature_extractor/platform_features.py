from src.models import RedrobSignals


def compute_active_intent_score(signals: RedrobSignals) -> float:
    """
    Weighted combination of open_to_work (0.4 if False, 1.0 if True),
    applications_30d, profile_completeness, search_appearances.
    """
    open_score = 1.0 if signals.open_to_work else 0.4
    apps_norm = min(signals.applications_submitted_30d / 10.0, 1.0)
    completeness_norm = signals.profile_completeness_score / 100.0
    search_norm = min(signals.search_appearances_30d / 200.0, 1.0)

    return round(
        open_score * 0.35 +
        apps_norm * 0.25 +
        completeness_norm * 0.20 +
        search_norm * 0.20,
        4,
    )


def compute_hire_reliability_score(signals: RedrobSignals) -> float:
    """
    Weighted combination of interview_completion_rate, offer_acceptance_rate,
    avg_response_time (inverted), email_verified, phone_verified.
    """
    response_speed = 1.0 - min(signals.avg_response_time_hrs / 200.0, 1.0)
    verif_score = (0.5 if signals.email_verified else 0.0) + (0.5 if signals.phone_verified else 0.0)

    oar = signals.offer_acceptance_rate
    if oar < 0:
        oar = 0.5  # default if no history

    return round(
        signals.interview_completion_rate * 0.40 +
        oar * 0.30 +
        response_speed * 0.20 +
        verif_score * 0.10,
        4,
    )
