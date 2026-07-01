import numpy as np
import pytest
from src.config import Settings
from src.models import CandidateFeatureRow, JDIntent, ScoreBreakdown
from src.scorer.composite import composite_score, score_candidate
from src.scorer.b1_semantic import semantic_score
from src.scorer.b2_trajectory import trajectory_score
from src.scorer.b3_stability import stability_score
from src.scorer.b4_platform import platform_score
from src.scorer.b5_cert import cert_bonus


class FakeModel:
    def encode(self, text, **kwargs):
        import numpy as np
        vec = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            vec = vec / np.linalg.norm(vec)
        return vec


class TestCompositeScore:
    def test_composite_formula(self):
        sb = ScoreBreakdown(
            candidate_id="cand_001",
            semantic_score=0.738,
            trajectory_score=0.937,
            stability_score=0.744,
            platform_score=0.728,
            cert_bonus=0.10,
        )
        result = composite_score(sb)
        # composite = 0.738*0.35 + 0.937*0.25 + 0.744*0.15 + 0.728*0.20 + 0.10*0.05
        #           = 0.2583 + 0.23425 + 0.1116 + 0.1456 + 0.005
        #           = 0.75475
        assert result == pytest.approx(0.755, abs=0.001)

    def test_zeros(self):
        sb = ScoreBreakdown(candidate_id="cand_001")
        result = composite_score(sb)
        assert result == 0.0

    def test_perfect_scores(self):
        sb = ScoreBreakdown(
            candidate_id="cand_001",
            semantic_score=1.0,
            trajectory_score=1.0,
            stability_score=1.0,
            platform_score=1.0,
            cert_bonus=0.10,
        )
        result = composite_score(sb)
        # 1.0*0.35 + 1.0*0.25 + 1.0*0.15 + 1.0*0.20 + 0.10*0.05
        # = 0.35 + 0.25 + 0.15 + 0.20 + 0.005 = 0.955
        assert result == pytest.approx(0.955, abs=0.001)

    def test_worked_example_composite(self):
        """Replicates section 7.4 composite score computation."""
        sb = ScoreBreakdown(
            candidate_id="cand_00072341",
            semantic_score=0.738,
            trajectory_score=0.937,
            stability_score=0.744,
            platform_score=0.728,
            cert_bonus=0.10,
        )
        result = composite_score(sb)
        # doc says composite = 0.755
        assert result == pytest.approx(0.755, abs=0.005)


class TestScoreCandidate:
    def test_score_candidate_returns_breakdown(self):
        settings = Settings()
        fr = CandidateFeatureRow(
            candidate_id="cand_001",
            latest_seniority=0.75,
            promotion_rate=1.0,
            experience_years=6.4,
            avg_tenure_months=25.7,
            job_hopping_flag=0,
            institution_tier="tier_2",
            active_intent_score=0.554,
            hire_reliability_score=0.889,
            github_activity_score=61.3,
            endorsements_received=143,
            skill_strength_scores={},
            cert_records=[
                {"name": "Google Cloud Professional Data Engineer", "issue_year": 2023},
            ],
            thin_profile=False,
        )
        cv = np.array([0.78, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jd = JDIntent(
            seniority_level=0.75,
            must_have_skills=["Apache Kafka", "GCP", "BigQuery", "Python", "SQL", "Apache Airflow"],
            nice_to_have_skills=["dbt", "Terraform"],
            requires_technical_github_signals=True,
        )

        result = score_candidate("cand_001", fr, cv, jv, jd, FakeModel(), settings)

        assert isinstance(result, ScoreBreakdown)
        assert result.candidate_id == "cand_001"
        assert result.semantic_score >= 0
        assert result.trajectory_score >= 0
        assert result.stability_score >= 0
        assert result.platform_score >= 0
        assert result.cert_bonus >= 0
        assert result.composite_score > 0

    def test_worked_example_full_pipeline(self):
        """End-to-end check: all B1-B5 scores + composite from section 7.4."""
        settings = Settings()
        fr = CandidateFeatureRow(
            candidate_id="cand_00072341",
            latest_seniority=0.75,
            promotion_rate=1.0,
            experience_years=6.4,
            avg_tenure_months=25.7,
            job_hopping_flag=0,
            institution_tier="tier_2",
            active_intent_score=0.554,
            hire_reliability_score=0.889,
            github_activity_score=61.3,
            endorsements_received=143,
            skill_strength_scores={
                "apache kafka": 0.674, "gcp": 0.414, "bigquery": 0.551,
                "python": 0.973, "sql": 0.955, "apache airflow": 0.373,
                "dbt": 0.163,
            },
            cert_records=[
                {"name": "Google Cloud Professional Data Engineer", "issue_year": 2023},
            ],
            thin_profile=False,
        )
        cv = np.array([0.78, 0.0, 0.0], dtype=np.float32)
        jv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        jd = JDIntent(
            seniority_level=0.75,
            must_have_skills=["Apache Kafka", "GCP", "BigQuery", "Python", "SQL", "Apache Airflow"],
            nice_to_have_skills=["dbt", "Terraform"],
            requires_technical_github_signals=True,
        )

        result = score_candidate("cand_00072341", fr, cv, jv, jd, FakeModel(), settings)

        # Assert all 5 sub-scores within tolerance of documented values
        assert result.semantic_score == pytest.approx(0.738, abs=0.02)
        assert result.trajectory_score == pytest.approx(0.937, abs=0.01)
        assert result.stability_score == pytest.approx(0.744, abs=0.01)
        assert result.platform_score == pytest.approx(0.728, abs=0.01)
        assert result.cert_bonus == pytest.approx(0.10, abs=0.01)
        assert result.composite_score == pytest.approx(0.755, abs=0.01)
