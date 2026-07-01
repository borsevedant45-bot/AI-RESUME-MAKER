import logging
from src.models import ScoreBreakdown

logger = logging.getLogger(__name__)


def rank_candidates(
    score_breakdowns: list[ScoreBreakdown],
    top_n: int = 20,
    tiebreaker_tolerance: float = 0.001,
) -> list[ScoreBreakdown]:
    """
    Sorts candidates by composite_score descending.
    Tiebreaker: higher platform_score wins among candidates within tolerance
    of each other on composite.
    """
    def sort_key(sb: ScoreBreakdown) -> tuple:
        return (round(sb.composite_score, 3), sb.platform_score)

    ranked = sorted(score_breakdowns, key=sort_key, reverse=True)

    logger.info("Ranked %d candidates; selected top %d", len(score_breakdowns), top_n)
    return ranked[:top_n]
