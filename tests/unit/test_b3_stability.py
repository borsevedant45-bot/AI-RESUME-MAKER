import pytest
from src.models import CandidateFeatureRow
from src.scorer.b3_stability import stability_score


def _feature_row(**kwargs):
    defaults = dict(
        candidate_id="cand_001", avg_tenure_months=0.0, job_hopping_flag=0,
        institution_tier="tier_3",
    )
    defaults.update(kwargs)
    return CandidateFeatureRow(**defaults)


class TestStabilityScore:
    def test_zero_tenure(self):
        fr = _feature_row(avg_tenure_months=0.0)
        result = stability_score(fr)
        # tenure_norm = 0, no hopping, edu = tier_3 → 0.01
        # b3 = max(min(0 - 0 + 0.01, 1), 0) = 0.01
        assert result == pytest.approx(0.01, abs=0.001)

    def test_strong_tenure(self):
        fr = _feature_row(avg_tenure_months=48.0)
        result = stability_score(fr)
        # tenure_norm = min(48/36,1) = 1.0
        # b3 = min(max(1.0 + 0.01, 0), 1) = 1.0
        assert result == pytest.approx(1.0, abs=0.001)

    def test_job_hopping_penalty(self):
        fr = _feature_row(avg_tenure_months=36.0, job_hopping_flag=1)
        result = stability_score(fr)
        # tenure_norm = 1.0, penalty = 0.30, edu = tier_3 → 0.01
        # b3 = min(max(1.0 - 0.30 + 0.01, 0), 1) = 0.71
        assert result == pytest.approx(0.71, abs=0.001)

    def test_tier_1_bonus(self):
        fr = _feature_row(avg_tenure_months=36.0, institution_tier="tier_1")
        result = stability_score(fr)
        # tenure_norm = 1.0, no hopping, edu = tier_1 → 0.05
        # b3 = min(max(1.0 + 0.05, 0), 1) = 1.0 (capped)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_hopping_and_bad_tier(self):
        fr = _feature_row(avg_tenure_months=12.0, job_hopping_flag=1, institution_tier="tier_4")
        result = stability_score(fr)
        # tenure_norm = min(12/36, 1) = 0.333
        # hopping_penalty = 0.30, edu_bonus = 0.0
        # b3 = min(max(0.333 - 0.30 + 0.0, 0), 1) = 0.033
        assert result == pytest.approx(0.033, abs=0.001)

    def test_clamped_below_zero(self):
        fr = _feature_row(avg_tenure_months=5.0, job_hopping_flag=1, institution_tier="tier_4")
        result = stability_score(fr)
        # tenure_norm = 5/36 = 0.139, penalty = 0.30, edu = 0.0
        # b3 = max(0.139 - 0.30, 0) = 0.0
        assert result == pytest.approx(0.0, abs=0.001)

    def test_worked_example(self):
        """Replicates section 7.4 B3 computation."""
        fr = _feature_row(
            avg_tenure_months=25.7, job_hopping_flag=0, institution_tier="tier_2",
        )
        result = stability_score(fr)
        # doc says b3 = 0.744
        assert result == pytest.approx(0.744, abs=0.01)
