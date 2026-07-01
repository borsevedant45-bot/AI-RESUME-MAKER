import pytest
from src.models import CandidateFeatureRow, JDIntent
from src.scorer.b2_trajectory import trajectory_score


def _feature_row(**kwargs):
    defaults = dict(
        candidate_id="cand_001", latest_seniority=0.5, promotion_rate=0.0,
        experience_years=0.0,
    )
    defaults.update(kwargs)
    return CandidateFeatureRow(**defaults)


class TestTrajectoryScore:
    def test_exact_seniority_match(self):
        fr = _feature_row(latest_seniority=0.75, promotion_rate=0.5, experience_years=6.0)
        jd = JDIntent(seniority_level=0.75)
        result = trajectory_score(fr, jd)
        # gap=0.0 → seniority_fit=1.0
        # traj_momentum = 0.5*0.5 + min(6/10,1)*0.3 + 0.75*0.2 = 0.25+0.18+0.15=0.58
        # b2 = 1.0*0.6 + 0.58*0.4 = 0.6+0.232 = 0.832
        assert result == pytest.approx(0.832, abs=0.01)

    def test_gap_0_25(self):
        fr = _feature_row(latest_seniority=0.5, promotion_rate=0.0, experience_years=3.0)
        jd = JDIntent(seniority_level=0.75)
        result = trajectory_score(fr, jd)
        # gap=0.25 → seniority_fit=0.80
        # traj_momentum = 0.0*0.5 + min(3/10,1)*0.3 + 0.5*0.2 = 0+0.09+0.1=0.19
        # b2 = 0.80*0.6 + 0.19*0.4 = 0.48+0.076 = 0.556
        assert result == pytest.approx(0.556, abs=0.01)

    def test_gap_0_50(self):
        fr = _feature_row(latest_seniority=0.5)
        jd = JDIntent(seniority_level=1.0)
        result = trajectory_score(fr, jd)
        # gap=0.50 → seniority_fit=0.50
        # b2 = 0.50*0.6 + ...*0.4 >= 0.30
        assert result > 0.30

    def test_gap_large(self):
        fr = _feature_row(latest_seniority=0.2)
        jd = JDIntent(seniority_level=1.0)
        result = trajectory_score(fr, jd)
        # gap=0.80 → seniority_fit = max(0.2, 1-0.8*2) = max(0.2, -0.6) = 0.2
        assert result > 0.12  # lower bound

    def test_stretch_readiness_override(self):
        fr = _feature_row(latest_seniority=0.5, promotion_rate=0.6, experience_years=5.5)
        jd = JDIntent(seniority_level=0.75)
        result = trajectory_score(fr, jd)
        # stretch override: seniority_fit = 0.75 (instead of 0.80)
        # traj_momentum = 0.6*0.5 + min(5.5/10,1)*0.3 + 0.5*0.2 = 0.30+0.165+0.10=0.565
        # b2 = 0.75*0.6 + 0.565*0.4 = 0.45 + 0.226 = 0.676
        assert result == pytest.approx(0.676, abs=0.01)

    def test_stretch_no_override_low_promo(self):
        fr = _feature_row(latest_seniority=0.5, promotion_rate=0.3, experience_years=5.5)
        jd = JDIntent(seniority_level=0.75)
        result = trajectory_score(fr, jd)
        # gap=0.25 → seniority_fit=0.80 (no override, promo_rate < 0.5)
        # traj_momentum = 0.3*0.5 + 0.3*0.55 + 0.5*0.2 = 0.15+0.165+0.10=0.415
        # b2 = 0.80*0.6 + 0.415*0.4 = 0.48 + 0.166 = 0.646
        assert result == pytest.approx(0.646, abs=0.01)

    def test_worked_example(self):
        """Replicates section 7.4 B2 computation."""
        fr = _feature_row(
            latest_seniority=0.75, promotion_rate=1.0, experience_years=6.4,
        )
        jd = JDIntent(seniority_level=0.75)
        result = trajectory_score(fr, jd)
        # doc says b2 = 0.937
        assert result == pytest.approx(0.937, abs=0.01)
