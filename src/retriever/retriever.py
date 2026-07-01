import logging
import faiss
import numpy as np

logger = logging.getLogger(__name__)


def retrieve_top_n(
    jd_vector: np.ndarray,
    index: faiss.Index,
    id_to_index_map: dict[str, int],
    top_n: int = 500,
) -> list[tuple[str, float]]:
    """
    Runs ANN search and returns ordered list of (candidate_id, cosine_similarity).
    Highest similarity first.
    """
    index_to_id = {v: k for k, v in id_to_index_map.items()}
    distances, indices = index.search(jd_vector.reshape(1, -1), top_n)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        candidate_id = index_to_id.get(int(idx), str(idx))
        cosine_sim = 1.0 - distances[0][i] / 2.0
        results.append((candidate_id, float(cosine_sim)))

    logger.info("Retrieved %d candidates in top-%d search", len(results), top_n)
    return results
