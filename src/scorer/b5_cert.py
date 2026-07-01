import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import CandidateFeatureRow

logger = logging.getLogger(__name__)


def cert_bonus(
    feature_row: CandidateFeatureRow,
    jd_vector: np.ndarray,
    model: SentenceTransformer,
    current_year: int = 2026,
    max_bonus: float = 0.10,
    recency_decay: float = 0.10,
    recency_floor: float = 0.50,
) -> float:
    """
    For each cert:
        relevance_norm = rescaled cosine_sim(cert_embedding, jd_vector)
        recency_weight = max(floor, 1.0 - (current_year - issue_year) * decay)
        contribution = relevance_norm * recency_weight
    Returns min(max(contributions), max_bonus). Returns 0.0 if no certs.
    """
    if not feature_row.cert_records:
        return 0.0

    contributions = []
    for cert in feature_row.cert_records:
        cert_vec = model.encode(cert["name"], normalize_embeddings=True)
        relevance = float(np.dot(cert_vec, jd_vector))
        relevance_norm = (relevance + 1) / 2.0

        years_old = current_year - cert.get("issue_year", current_year)
        recency_weight = max(recency_floor, 1.0 - years_old * recency_decay)

        contributions.append(relevance_norm * recency_weight)

    return float(min(max(contributions, default=0.0), max_bonus))
