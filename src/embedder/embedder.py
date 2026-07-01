import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_batch(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 256,
) -> np.ndarray:
    """Batch encodes a list of texts with L2 normalization."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def encode_single(text: str, model: SentenceTransformer) -> np.ndarray:
    """Encodes one text string and returns a normalized 1D vector."""
    return model.encode(text, normalize_embeddings=True).astype(np.float32)
