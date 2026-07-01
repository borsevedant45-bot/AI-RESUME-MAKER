import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.embedder.embedder import get_model, encode_batch, encode_single


class FakeSentenceTransformer:
    """Minimal fake that returns identity-ish vectors for any input."""

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if kwargs.get("normalize_embeddings", False):
                vec = vec / np.linalg.norm(vec)
            return vec
        results = np.random.rand(len(sentences), 3).astype(np.float32)
        if kwargs.get("normalize_embeddings", False):
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            results = results / norms
        return results


class TestGetModel:
    def test_returns_model_instance(self):
        with patch("src.embedder.embedder.SentenceTransformer", return_value=FakeSentenceTransformer()):
            model = get_model("test-model")
            assert model is not None

    def test_caches_model(self):
        with patch("src.embedder.embedder.SentenceTransformer", return_value=FakeSentenceTransformer()) as mock:
            model1 = get_model("test-cache-model")
            model2 = get_model("test-cache-model")
            assert model1 is model2
            mock.assert_called_once()

    def test_different_models_not_cached_together(self):
        with patch("src.embedder.embedder.SentenceTransformer") as mock:
            mock.side_effect = lambda name: FakeSentenceTransformer()
            m1 = get_model("model-a")
            m2 = get_model("model-b")
            assert m1 is not m2
            assert mock.call_count == 2


class TestEncodeBatch:
    def test_returns_float32_array(self):
        model = FakeSentenceTransformer()
        texts = ["hello world", "test sentence"]
        result = encode_batch(texts, model, batch_size=2)
        assert result.dtype == np.float32
        assert result.shape == (2, 3)

    def test_normalizes_embeddings(self):
        model = FakeSentenceTransformer()
        texts = ["hello world"]
        result = encode_batch(texts, model, batch_size=1)
        # Check L2 norm is ~1
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_list(self):
        model = FakeSentenceTransformer()
        result = encode_batch([], model)
        assert result.shape[0] == 0


class TestEncodeSingle:
    def test_returns_1d_array(self):
        model = FakeSentenceTransformer()
        result = encode_single("test query", model)
        assert result.ndim == 1

    def test_normalized(self):
        model = FakeSentenceTransformer()
        result = encode_single("test query", model)
        norm = np.linalg.norm(result)
        assert np.allclose(norm, 1.0, atol=1e-5)

    def test_float32_type(self):
        model = FakeSentenceTransformer()
        result = encode_single("test query", model)
        assert result.dtype == np.float32
