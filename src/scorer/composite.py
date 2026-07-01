import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Settings
from src.models import CandidateFeatureRow, JDIntent, ScoreBreakdown
from src.scorer.b1_semantic import semantic_score
from src.scorer.b2_trajectory import trajectory_score
from src.scorer.b3_stability import stability_score
from src.scorer.b4_platform import platform_score
from src.scorer.b5_cert import cert_bonus

logger = logging.getLogger(__name__)


def composite_score(scores: ScoreBreakdown) -> float:
    """
    Computes final composite score:
        b1*0.35 + b2*0.25 + b3*0.15 + b4*0.20 + b5*0.05
    """
    return (
        scores.semantic_score * 0.35 +
        scores.trajectory_score * 0.25 +
        scores.stability_score * 0.15 +
        scores.platform_score * 0.20 +
        scores.cert_bonus * 0.05
    )


def score_candidate(
    candidate_id: str,
    feature_row: CandidateFeatureRow,
    candidate_vector: np.ndarray,
    jd_vector: np.ndarray,
    jd_intent: JDIntent,
    model: SentenceTransformer,
    settings: Settings,
    skill_vectors: dict[str, np.ndarray] | None = None,
) -> ScoreBreakdown:
    """
    Convenience wrapper that computes B1-B5 and composite in one call.
    """
    b1 = semantic_score(
        candidate_vector=candidate_vector,
        jd_vector=jd_vector,
        skill_strength_scores=feature_row.skill_strength_scores,
        jd_intent=jd_intent,
        model=model,
        thin_profile=feature_row.thin_profile,
        semantic_fallback_discount=settings.thresholds.semantic_fallback_discount,
        skill_vectors=skill_vectors,
    )
    b2 = trajectory_score(
        feature_row=feature_row,
        jd_intent=jd_intent,
        min_promotion_rate=settings.trajectory.stretch_readiness.min_promotion_rate,
        min_experience_years=settings.trajectory.stretch_readiness.min_experience_years,
        fit_override_value=settings.trajectory.stretch_readiness.fit_override_value,
    )
    b3 = stability_score(
        feature_row=feature_row,
        strong_tenure_months=settings.stability.strong_tenure_months,
        hopping_penalty=settings.stability.hopping_penalty,
    )
    b4 = platform_score(
        feature_row=feature_row,
        jd_intent=jd_intent,
        github_activity_max=settings.platform.github_activity_max,
        max_endorsements_norm=settings.platform.max_endorsements_norm,
    )
    b5 = cert_bonus(
        feature_row=feature_row,
        jd_vector=jd_vector,
        model=model,
        current_year=2026,
        max_bonus=settings.cert_bonus.max_bonus,
    )

    sb = ScoreBreakdown(
        candidate_id=candidate_id,
        semantic_score=b1,
        trajectory_score=b2,
        stability_score=b3,
        platform_score=b4,
        cert_bonus=b5,
        composite_score=0.0,
    )
    sb.composite_score = composite_score(sb)
    return sb
