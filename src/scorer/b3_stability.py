from src.models import CandidateFeatureRow

EDU_BONUS_MAP = {
    "tier_1": 0.05,
    "tier_2": 0.03,
    "tier_3": 0.01,
    "tier_4": 0.00,
}


def stability_score(
    feature_row: CandidateFeatureRow,
    strong_tenure_months: int = 36,
    hopping_penalty: float = 0.30,
) -> float:
    """
    Computes B3 stability score as:
        min(avg_tenure_months / strong_tenure_months, 1) - hopping_penalty + edu_bonus
    """
    tenure_norm = min(feature_row.avg_tenure_months / strong_tenure_months, 1.0)
    penalty = hopping_penalty if feature_row.job_hopping_flag else 0.0
    edu_bonus = EDU_BONUS_MAP.get(feature_row.institution_tier, 0.0)

    b3 = min(max(tenure_norm - penalty + edu_bonus, 0.0), 1.0)
    return float(b3)
