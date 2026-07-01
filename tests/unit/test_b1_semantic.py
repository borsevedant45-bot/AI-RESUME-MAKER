import numpy as np
import pytest
from src.models import JDIntent
from src.scorer.b1_semantic import semantic_score


class FakeModel:
    """Returns fixed vectors; used for semantic fallback in B1."""

    def encode(self, text: str, **kwargs):
        if "Terraform" in text:
            vec = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        elif "dbt" in text:
            vec = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        else:
            vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            vec = vec / np.linalg.norm(vec)
        return vec


class TestSemanticScore:
    def test_embed_sim_norm(self):
        """embed_sim_norm = (dot + 1) / 2 for normalized vectors."""
        cv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jv = np.array([0.8, 0.6, 0.0], dtype=np.float32)  # dot=0.8
        jv = jv / np.linalg.norm(jv)
        intent = JDIntent(must_have_skills=[], nice_to_have_skills=[])
        result = semantic_score(cv, jv, {}, intent, FakeModel())
        # embed_sim_norm = (0.8 + 1)/2 = 0.9
        # coverage_score falls back to embed_sim_norm when must/nice empty
        # b1 = 0.9*0.6 + 0.9*0.4 = 0.9
        assert result == pytest.approx(0.9, abs=0.01)

    def test_coverage_must_have_direct_matches(self):
        cv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # jv normalized -> same as cv
        intent = JDIntent(must_have_skills=["Python", "SQL"])
        skill_scores = {"python": 0.9, "sql": 0.8}
        result = semantic_score(cv, jv, skill_scores, intent, FakeModel())
        # embed_sim = 1.0, embed_sim_norm = 1.0
        # must_have scores = [0.9, 0.8], nice = []
        # coverage = mean([0.9, 0.8]) = 0.85
        # b1 = 1.0*0.6 + 0.85*0.4 = 0.6 + 0.34 = 0.94
        assert result == pytest.approx(0.94, abs=0.01)

    def test_nice_to_have_at_40pct_weight(self):
        cv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        intent = JDIntent(nice_to_have_skills=["dbt"])
        skill_scores = {"dbt": 0.7}
        result = semantic_score(cv, jv, skill_scores, intent, FakeModel())
        # nice score = 0.7 * 0.4 = 0.28
        # coverage = mean([0.28]) = 0.28
        # b1 = 1.0*0.6 + 0.28*0.4 = 0.6 + 0.112 = 0.712
        assert result == pytest.approx(0.712, abs=0.01)

    def test_semantic_fallback_discount(self):
        cv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        intent = JDIntent(must_have_skills=["Terraform"])
        # Terraform not in skill_scores -> use semantic fallback
        # FakeModel encodes "Terraform" -> vec [0.3, 0.4, 0.5] normalized
        # dot(cv, skill_vec) -> depends on the actual normalized vec
        result = semantic_score(cv, jv, {}, intent, FakeModel())
        # embed_sim_norm = 1.0
        # skill_vec = [0.3, 0.4, 0.5] / norm = [0.3, 0.4, 0.5]/0.707
        # normalize: norm = sqrt(0.09+0.16+0.25) = sqrt(0.5) = 0.7071
        # normalized = [0.424, 0.566, 0.707]
        # dot(cv, normalized) = 1.0*0.424 = 0.424
        # sem_fallback = (0.424+1)/2 * 0.6 = 0.712*0.6 = 0.427
        # coverage = mean([0.427]) = 0.427
        # b1 = 1.0*0.6 + 0.427*0.4 = 0.6 + 0.171 = 0.771
        assert result == pytest.approx(0.771, abs=0.01)

    def test_thin_profile_cap(self):
        cv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        intent = JDIntent(must_have_skills=[], nice_to_have_skills=[])
        result = semantic_score(cv, jv, {}, intent, FakeModel(), thin_profile=True)
        # b1 would be 0.9, capped at 0.55
        assert result == pytest.approx(0.55, abs=0.01)

    def test_clipped_to_01(self):
        cv = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        intent = JDIntent(must_have_skills=[], nice_to_have_skills=[])
        result = semantic_score(cv, jv, {}, intent, FakeModel())
        # dot = -1.0, embed_sim_norm = 0.0
        # b1 = 0.0*0.6 + 0.0*0.4 = 0.0
        assert 0.0 <= result <= 1.0

    def test_worked_example(self):
        """Replicates section 7.4 B1 computation from the methodology doc."""
        cv = np.array([0.78, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # embed_sim = 0.78, embed_sim_norm = 0.89

        intent = JDIntent(
            must_have_skills=["Apache Kafka", "GCP", "BigQuery", "Python", "SQL", "Apache Airflow"],
            nice_to_have_skills=["dbt", "Terraform"],
        )
        skill_scores = {
            "apache kafka": 0.674,
            "gcp": 0.414,
            "bigquery": 0.551,
            "python": 0.973,
            "sql": 0.955,
            "apache airflow": 0.373,
            "dbt": 0.163,
        }
        result = semantic_score(cv, jv, skill_scores, intent, FakeModel())
        # doc says b1 = 0.738
        assert result == pytest.approx(0.738, abs=0.02)
