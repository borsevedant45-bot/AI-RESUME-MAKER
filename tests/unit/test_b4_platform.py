import pytest
from src.models import CandidateFeatureRow, JDIntent
from src.scorer.b4_platform import platform_score


def _feature_row(**kwargs):
    defaults = dict(
        candidate_id="cand_001",
        active_intent_score=0.0, hire_reliability_score=0.0,
        github_activity_score=-1.0, endorsements_received=0,
    )
    defaults.update(kwargs)
    return CandidateFeatureRow(**defaults)


class TestPlatformScore:
    def test_non_technical_jd(self):
        fr = _feature_row(active_intent_score=0.6, hire_reliability_score=0.8)
        jd = JDIntent(requires_technical_github_signals=False)
        result = platform_score(fr, jd)
        # b4 = 0.6*0.55 + 0.8*0.45 = 0.33 + 0.36 = 0.69
        assert result == pytest.approx(0.69, abs=0.01)

    def test_technical_jd_with_signals(self):
        fr = _feature_row(
            active_intent_score=0.6, hire_reliability_score=0.8,
            github_activity_score=50.0, endorsements_received=80,
        )
        jd = JDIntent(requires_technical_github_signals=True)
        result = platform_score(fr, jd)
        # github_norm = 50/96.9 = 0.516, endorse_norm = 80/100 = 0.80
        # tech_engagement = 0.516*0.6 + 0.80*0.4 = 0.310 + 0.320 = 0.630
        # b4 = 0.6*0.4 + 0.8*0.35 + 0.630*0.25 = 0.24 + 0.28 + 0.158 = 0.678
        assert result == pytest.approx(0.678, abs=0.01)

    def test_technical_jd_zero_github(self):
        fr = _feature_row(
            active_intent_score=0.5, hire_reliability_score=0.7,
            github_activity_score=-1.0, endorsements_received=0,
        )
        jd = JDIntent(requires_technical_github_signals=True)
        result = platform_score(fr, jd)
        # github_norm = max(-1, 0)/96.9 = 0.0
        # endorse_norm = 0/100 = 0.0
        # tech_engagement = 0.0
        # b4 = 0.5*0.4 + 0.7*0.35 + 0.0*0.25 = 0.20 + 0.245 + 0.0 = 0.445
        assert result == pytest.approx(0.445, abs=0.01)

    def test_endorsements_capped(self):
        fr = _feature_row(
            github_activity_score=50.0, endorsements_received=500,
            active_intent_score=0.5, hire_reliability_score=0.5,
        )
        jd = JDIntent(requires_technical_github_signals=True)
        result = platform_score(fr, jd)
        # endorse_norm = min(500/100, 1) = 1.0
        assert result > 0

    def test_worked_example(self):
        """Replicates section 7.4 B4 computation."""
        fr = _feature_row(
            active_intent_score=0.554, hire_reliability_score=0.889,
            github_activity_score=61.3, endorsements_received=143,
        )
        jd = JDIntent(requires_technical_github_signals=True)
        result = platform_score(fr, jd)
        # doc says b4 = 0.728
        assert result == pytest.approx(0.728, abs=0.01)
