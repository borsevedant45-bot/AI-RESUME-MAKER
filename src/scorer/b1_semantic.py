import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import JDIntent

logger = logging.getLogger(__name__)


def semantic_score(
    candidate_vector: np.ndarray,
    jd_vector: np.ndarray,
    skill_strength_scores: dict[str, float],
    jd_intent: JDIntent,
    model: SentenceTransformer,
    thin_profile: bool = False,
    semantic_fallback_discount: float = 0.60,
    skill_vectors: dict[str, np.ndarray] | None = None,
) -> float:
    """
    Computes B1 semantic score as:
        embed_sim_norm * 0.60 + coverage_score * 0.40

    Caps score at 0.55 if thin_profile=True.
    """
    embed_sim = float(np.dot(candidate_vector, jd_vector))
    embed_sim_norm = (embed_sim + 1) / 2.0

    must_have = jd_intent.must_have_skills
    nice_to_have = jd_intent.nice_to_have_skills

    must_have_scores = []
    for skill_name in must_have:
        key = skill_name.lower()
        direct = skill_strength_scores.get(key)
        if direct is not None:
            must_have_scores.append(direct)
        elif skill_vectors is not None and key in skill_vectors:
            skill_vec = skill_vectors[key]
            sem_fallback = float(np.dot(candidate_vector, skill_vec))
            must_have_scores.append(max(0, (sem_fallback + 1) / 2.0 * semantic_fallback_discount))
        else:
            skill_vec = model.encode(skill_name, normalize_embeddings=True)
            sem_fallback = float(np.dot(candidate_vector, skill_vec))
            must_have_scores.append(max(0, (sem_fallback + 1) / 2.0 * semantic_fallback_discount))

    nice_scores = []
    for skill_name in nice_to_have:
        key = skill_name.lower()
        direct = skill_strength_scores.get(key)
        if direct is not None:
            nice_scores.append(direct * 0.4)
        elif skill_vectors is not None and key in skill_vectors:
            skill_vec = skill_vectors[key]
            sem_fallback = float(np.dot(candidate_vector, skill_vec))
            nice_scores.append(max(0, (sem_fallback + 1) / 2.0 * semantic_fallback_discount * 0.4))
        else:
            skill_vec = model.encode(skill_name, normalize_embeddings=True)
            sem_fallback = float(np.dot(candidate_vector, skill_vec))
            nice_scores.append(max(0, (sem_fallback + 1) / 2.0 * semantic_fallback_discount * 0.4))

    all_scores = must_have_scores + nice_scores
    coverage_score = float(np.mean(all_scores)) if all_scores else embed_sim_norm

    b1 = embed_sim_norm * 0.60 + coverage_score * 0.40

    if thin_profile:
        b1 = min(b1, 0.55)

    return float(np.clip(b1, 0, 1))
