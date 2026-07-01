import logging

import numpy as np

from src.models import CandidateFeatureRow, JDIntent

logger = logging.getLogger(__name__)


def trajectory_score(
    feature_row: CandidateFeatureRow,
    jd_intent: JDIntent,
    min_promotion_rate: float = 0.5,
    min_experience_years: float = 5.0,
    fit_override_value: float = 0.75,
) -> float:
    """
    Computes B2 trajectory score as:
        seniority_fit * 0.60 + trajectory_momentum * 0.40
    """
    cand_seniority = feature_row.latest_seniority
    jd_seniority = jd_intent.seniority_level
    promo_rate = feature_row.promotion_rate
    exp_years = feature_row.experience_years

    # Seniority fit with symmetrical penalty
    gap = abs(cand_seniority - jd_seniority)
    if gap == 0.0:
        seniority_fit = 1.0
    elif gap <= 0.25:
        seniority_fit = 0.80
    elif gap <= 0.50:
        seniority_fit = 0.50
    else:
        seniority_fit = max(0.2, 1.0 - gap * 2.0)

    # Stretch readiness override
    if (cand_seniority == 0.5 and jd_seniority == 0.75
            and promo_rate >= min_promotion_rate and exp_years >= min_experience_years):
        seniority_fit = fit_override_value

    # Trajectory momentum
    traj_momentum = (
        promo_rate * 0.50 +
        min(exp_years / 10.0, 1.0) * 0.30 +
        cand_seniority * 0.20
    )

    b2 = seniority_fit * 0.60 + traj_momentum * 0.40
    return float(np.clip(b2, 0, 1))
