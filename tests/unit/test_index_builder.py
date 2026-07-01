import json
import logging
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import faiss
import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.exceptions import IndexBuildError
from src.embedder.index_builder import build_candidate_index, load_index


# ---- Helpers -----------------------------------------------------------

SAMPLE_CANDIDATES = [
    {
        "candidate_id": "cand_001",
        "profile": {"years_of_experience": 5.0, "current_industry": "Tech"},
        "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 48}],
        "career_history": [{"company": "Acme", "title": "Senior Engineer", "start_date": "2020-01", "industry": "Tech", "description": "Built platform"}],
        "education": [{"degree": "B.Tech", "field_of_study": "CS", "tier": "tier_1"}],
        "certifications": [{"name": "AWS Certified", "issuer": "AWS", "year": 2022}],
        "redrob_signals": {"open_to_work_flag": True, "applications_submitted_30d": 5, "email_verified": True, "phone_verified": True},
    },
    {
        "candidate_id": "cand_002",
        "profile": {"years_of_experience": 3.0, "current_industry": "Finance"},
        "skills": [{"name": "Java", "proficiency": "advanced", "endorsements": 10, "duration_months": 24}],
        "career_history": [{"company": "Beta", "title": "Developer", "start_date": "2021-06", "industry": "Finance", "description": "Built backend services"}],
        "education": [],
        "certifications": [],
        "redrob_signals": {},
    },
]

SETTINGS = Settings()


def _make_jsonl(tmp_path, candidates=None):
    """Write sample candidates to a temp JSONL file and return the path."""
    candidates = candidates or SAMPLE_CANDIDATES
    path = tmp_path / "candidates.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    return path


# ---- build_candidate_index --------------------------------------------


class TestBuildCandidateIndex:
    def test_successful_build(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        jsonl_path = _make_jsonl(tmp_path)
        out_dir = tmp_path / "index"

        with patch("src.embedder.index_builder.get_model") as mock_get_model:
            mock_model = MagicMock()
            fake_vectors = np.random.rand(2, 384).astype(np.float32)
            fake_vectors = fake_vectors / np.linalg.norm(fake_vectors, axis=1, keepdims=True)
            mock_model.encode.return_value = fake_vectors
            mock_get_model.return_value = mock_model

            build_candidate_index(jsonl_path, out_dir, SETTINGS)

        # Check all 4 artifacts were written
        assert (out_dir / "candidate_vectors.npy").exists()
        assert (out_dir / "candidate_index.faiss").exists()
        assert (out_dir / "candidate_features.parquet").exists()
        assert (out_dir / "candidate_id_map.json").exists()

        # Verify content of artifacts
        vectors = np.load(str(out_dir / "candidate_vectors.npy"))
        assert vectors.shape == (2, 384)

        with open(out_dir / "candidate_id_map.json") as f:
            id_map = json.load(f)
        assert "cand_001" in id_map
        assert "cand_002" in id_map

        df = pd.read_parquet(str(out_dir / "candidate_features.parquet"))
        assert len(df) == 2
        assert list(df.columns) is not None

        # FAISS index should be IP type
        index = faiss.read_index(str(out_dir / "candidate_index.faiss"))
        assert index.ntotal == 2
        assert isinstance(index, faiss.IndexFlatIP)

    def test_raises_on_empty_profiles(self, tmp_path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        with pytest.raises(IndexBuildError, match="No valid"):
            build_candidate_index(jsonl_path, tmp_path, SETTINGS)

    def test_raises_on_low_success_rate(self, tmp_path):
        jsonl_path = _make_jsonl(tmp_path)
        out_dir = tmp_path / "failing"

        with (
            patch("src.embedder.index_builder.get_model") as mock_get_model,
            patch("src.embedder.index_builder.build_candidate_doc") as mock_build_doc,
        ):
            mock_model = MagicMock()
            fake_vectors = np.random.rand(1, 384).astype(np.float32)
            mock_model.encode.return_value = fake_vectors
            mock_get_model.return_value = mock_model
            # Make the second candidate fail by raising an exception
            mock_build_doc.side_effect = [
                ("doc for cand_001", False),
                Exception("Simulated failure"),
            ]

            with pytest.raises(IndexBuildError, match=">= 90%"):
                build_candidate_index(jsonl_path, out_dir, SETTINGS)

    def test_logs_progress_every_10000(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        # Create 25000 candidates
        many_candidates = []
        for i in range(25000):
            many_candidates.append({
                "candidate_id": f"cand_{i:05d}",
                "profile": {"years_of_experience": 1.0},
                "skills": [{"name": "X", "proficiency": "beginner"}],
                "career_history": [{"company": "C", "title": "E", "start_date": "2020-01", "industry": "T", "description": "Some description text"}],
                "education": [],
                "certifications": [],
                "redrob_signals": {},
            })
        jsonl_path = _make_jsonl(tmp_path, many_candidates)
        out_dir = tmp_path / "progress_test"

        with patch("src.embedder.index_builder.get_model") as mock_get_model:
            mock_model = MagicMock()
            fake_vectors = np.random.rand(25000, 384).astype(np.float32)
            mock_model.encode.return_value = fake_vectors
            mock_get_model.return_value = mock_model

            build_candidate_index(jsonl_path, out_dir, SETTINGS)

        progress_messages = [
            m for m in caplog.messages
            if "Progress:" in m
        ]
        assert len(progress_messages) == 2  # at 10000 and 20000
        assert any("10000" in m for m in progress_messages)
        assert any("20000" in m for m in progress_messages)

    def test_uses_flatip_index(self, tmp_path):
        jsonl_path = _make_jsonl(tmp_path)
        out_dir = tmp_path / "ip_test"

        with patch("src.embedder.index_builder.get_model") as mock_get_model:
            mock_model = MagicMock()
            fake_vectors = np.random.rand(2, 384).astype(np.float32)
            mock_model.encode.return_value = fake_vectors
            mock_get_model.return_value = mock_model

            build_candidate_index(jsonl_path, out_dir, SETTINGS)

        index = faiss.read_index(str(out_dir / "candidate_index.faiss"))
        assert isinstance(index, faiss.IndexFlatIP)
        assert index.is_trained
        assert index.ntotal == 2


# ---- load_index --------------------------------------------------------


class TestLoadIndex:
    def test_loads_all_artifacts(self, tmp_path):
        # First build a real index
        jsonl_path = _make_jsonl(tmp_path)
        out_dir = tmp_path / "index_for_load"

        with patch("src.embedder.index_builder.get_model") as mock_get_model:
            mock_model = MagicMock()
            fake_vectors = np.random.rand(2, 384).astype(np.float32)
            fake_vectors = fake_vectors / np.linalg.norm(fake_vectors, axis=1, keepdims=True)
            mock_model.encode.return_value = fake_vectors
            mock_get_model.return_value = mock_model

            build_candidate_index(jsonl_path, out_dir, SETTINGS)

        # Now load it back
        index, vectors, id_map, feature_df = load_index(out_dir)

        assert isinstance(index, faiss.IndexFlatIP)
        assert index.ntotal == 2

        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (2, 384)

        assert isinstance(id_map, dict)
        assert id_map == {"cand_001": 0, "cand_002": 1}

        assert isinstance(feature_df, pd.DataFrame)
        assert len(feature_df) == 2
        assert "candidate_id" in feature_df.columns
        assert feature_df.iloc[0]["candidate_id"] == "cand_001"

    def test_load_index_raises_on_missing_artifacts(self, tmp_path):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_index(tmp_path)
