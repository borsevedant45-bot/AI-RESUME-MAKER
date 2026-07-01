from src.models import CandidateFeatureRow, JDIntent


def platform_score(
    feature_row: CandidateFeatureRow,
    jd_intent: JDIntent,
    github_activity_max: float = 96.9,
    max_endorsements_norm: int = 100,
) -> float:
    """
    Computes B4 platform score.

    If jd_intent.requires_technical_github_signals:
        active_intent*0.40 + hire_reliability*0.35 + tech_engagement*0.25
    Else:
        active_intent*0.55 + hire_reliability*0.45
    """
    active_intent = feature_row.active_intent_score
    hire_reliability = feature_row.hire_reliability_score

    if jd_intent.requires_technical_github_signals:
        github_norm = max(feature_row.github_activity_score, 0) / github_activity_max
        endorse_norm = min(feature_row.endorsements_received / max_endorsements_norm, 1.0)
        tech_engagement = github_norm * 0.60 + endorse_norm * 0.40

        b4 = (
            active_intent * 0.40 +
            hire_reliability * 0.35 +
            tech_engagement * 0.25
        )
    else:
        b4 = (
            active_intent * 0.55 +
            hire_reliability * 0.45
        )

    return float(max(0.0, min(b4, 1.0)))
