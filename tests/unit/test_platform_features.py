import pytest
from src.feature_extractor.platform_features import (
    compute_active_intent_score, compute_hire_reliability_score
)
from src.models import RedrobSignals


def _signals(**kwargs):
    defaults = dict(
        profile_completeness_score=0.0, connection_count=0, endorsements_received=0,
        notice_period_days=0, profile_views_30d=0, applications_submitted_30d=0,
        recruiter_response_rate=0.0, avg_response_time_hrs=0.0, search_appearances_30d=0,
        saved_by_recruiters_30d=0, interview_completion_rate=0.0, offer_acceptance_rate=-1.0,
        github_activity_score=-1.0, open_to_work=False, willing_to_relocate=False,
        email_verified=False, phone_verified=False, linkedin_connected=False,
        work_mode_preference="hybrid", expected_salary_min=0, expected_salary_max=0,
    )
    defaults.update(kwargs)
    return RedrobSignals(**defaults)


class TestComputeActiveIntentScore:
    def test_open_to_work_fully_active(self):
        sig = _signals(open_to_work=True, applications_submitted_30d=10,
                       profile_completeness_score=100, search_appearances_30d=200)
        score = compute_active_intent_score(sig)
        expected = 1.0 * 0.35 + 1.0 * 0.25 + 1.0 * 0.20 + 1.0 * 0.20
        assert score == pytest.approx(expected, abs=1e-4)

    def test_not_open_to_work(self):
        sig = _signals(open_to_work=False)
        score = compute_active_intent_score(sig)
        assert pytest.approx(score, abs=1e-4) == 0.4 * 0.35

    def test_applications_capped_at_10(self):
        sig = _signals(applications_submitted_30d=100)
        score = compute_active_intent_score(sig)
        app_term = 1.0 * 0.25
        assert pytest.approx(score, abs=1e-4) == 0.4 * 0.35 + app_term

    def test_search_appearances_capped(self):
        sig = _signals(search_appearances_30d=500)
        score = compute_active_intent_score(sig)
        search_term = 1.0 * 0.20
        assert pytest.approx(score, abs=1e-4) == 0.4 * 0.35 + search_term


class TestComputeHireReliabilityScore:
    def test_ideal_candidate(self):
        sig = _signals(
            interview_completion_rate=1.0, offer_acceptance_rate=1.0,
            avg_response_time_hrs=0, email_verified=True, phone_verified=True,
        )
        score = compute_hire_reliability_score(sig)
        expected = 1.0 * 0.40 + 1.0 * 0.30 + 1.0 * 0.20 + 1.0 * 0.10
        assert score == pytest.approx(expected, abs=1e-4)

    def test_no_offer_history(self):
        sig = _signals(offer_acceptance_rate=-1.0)
        score = compute_hire_reliability_score(sig)
        assert pytest.approx(score, abs=1e-4) == 0.0 * 0.40 + 0.5 * 0.30 + 1.0 * 0.20 + 0.0 * 0.10

    def test_max_response_time(self):
        sig = _signals(avg_response_time_hrs=200)
        score = compute_hire_reliability_score(sig)
        assert pytest.approx(score, abs=1e-4) == 0.0 * 0.40 + 0.5 * 0.30 + 0.0 * 0.20 + 0.0 * 0.10

    def test_response_speed_scaling(self):
        sig = _signals(avg_response_time_hrs=100)
        score = compute_hire_reliability_score(sig)
        response_speed = 1.0 - 100.0 / 200.0
        assert pytest.approx(score, abs=1e-4) == 0.0 * 0.40 + 0.5 * 0.30 + response_speed * 0.20 + 0.0 * 0.10

    def test_half_verification(self):
        sig = _signals(email_verified=True, phone_verified=False)
        score = compute_hire_reliability_score(sig)
        assert pytest.approx(score, abs=1e-4) == 0.0 * 0.40 + 0.5 * 0.30 + 1.0 * 0.20 + 0.5 * 0.10
