import json
import pytest
from src.explainer.grounding_validator import validate_grounding, build_fallback_explanation
from src.models import CandidateExplanation, CandidateFeatureRow


def _explanation(**kwargs):
    defaults = dict(
        candidate_id="cand_001", match_summary="", skill_alignment="",
        seniority_assessment="", trajectory_signal="", platform_summary="",
        flags="", grounding_validated=False,
    )
    defaults.update(kwargs)
    return CandidateExplanation(**defaults)


def _feature_row(**kwargs):
    defaults = dict(
        candidate_id="cand_001",
        skill_strength_scores={"python": 0.9, "kafka": 0.7},
        notice_period_days=60, avg_tenure_months=30.0,
        experience_years=6.0,
        job_hopping_flag=0, latest_seniority=0.75,
        promotion_rate=0.5, active_intent_score=0.5,
        hire_reliability_score=0.8, open_to_work=True,
        institution_tier="tier_2", endorsements_received=50,
        github_activity_score=40.0, willing_to_relocate=False,
        work_mode_preference="hybrid", expected_salary_min=0,
        expected_salary_max=0, location="",
        cert_records=[{"name": "AWS Certified", "issue_year": 2023}],
        thin_profile=False,
    )
    defaults.update(kwargs)
    return CandidateFeatureRow(**defaults)


class TestValidateGrounding:
    def test_finds_skill_name(self):
        exp = _explanation(match_summary="This candidate knows Python and Kafka well")
        fr = _feature_row()
        assert validate_grounding(exp, fr) is True

    def test_finds_company_name(self):
        exp = _explanation(match_summary="Worked at AcmeCorp for 3 years")
        fr = _feature_row()
        result = validate_grounding(exp, fr, career_history=[{"company": "AcmeCorp", "title": "Engineer"}])
        assert result is True

    def test_finds_role_title(self):
        exp = _explanation(skill_alignment="As a Senior Engineer they built systems")
        fr = _feature_row()
        result = validate_grounding(exp, fr, career_history=[{"company": "Acme", "title": "Senior Engineer"}])
        assert result is True

    def test_finds_tenure_number(self):
        exp = _explanation(trajectory_signal="Average tenure of 30 months shows stability")
        fr = _feature_row(avg_tenure_months=30.0)
        assert validate_grounding(exp, fr) is True

    def test_finds_notice_period(self):
        exp = _explanation(flags="Notice period is 60 days")
        fr = _feature_row(notice_period_days=60)
        assert validate_grounding(exp, fr) is True

    def test_finds_cert_name(self):
        exp = _explanation(skill_alignment="Holds an AWS Certified credential")
        fr = _feature_row()
        assert validate_grounding(exp, fr) is True

    def test_fails_on_generic_text(self):
        exp = _explanation(match_summary="This candidate is a strong match for the role")
        fr = _feature_row(skill_strength_scores={})
        assert validate_grounding(exp, fr) is False

    def test_case_insensitive(self):
        exp = _explanation(match_summary="This candidate knows PYTHON")
        fr = _feature_row(skill_strength_scores={"python": 0.9})
        assert validate_grounding(exp, fr) is True


class TestBuildFallbackExplanation:
    def test_populates_all_fields(self):
        exp = _explanation(candidate_id="cand_001")
        fr = _feature_row(
            skill_strength_scores={"python": 0.9, "kafka": 0.7},
        )
        result = build_fallback_explanation(exp, fr)

        assert result.grounding_validated is False
        assert "python (0.90)" in result.match_summary
        assert "kafka (0.70)" in result.match_summary
        assert result.flags == "No flags"

    def test_includes_nonzero_flags(self):
        exp = _explanation(candidate_id="cand_001")
        fr = _feature_row(
            job_hopping_flag=1, notice_period_days=120,
        )
        result = build_fallback_explanation(exp, fr)
        assert "Job hopping" in result.flags
        assert "Long notice" in result.flags
        assert result.grounding_validated is False
