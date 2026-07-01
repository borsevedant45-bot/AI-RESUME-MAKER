import json
from unittest.mock import MagicMock, patch

import pytest
from src.config import Settings
from src.explainer.explainer import generate_explanations, _call_batch_llm
from src.models import (
    JDIntent, CandidateFeatureRow, CandidateExplanation, ScoreBreakdown,
)


SETTINGS = Settings()


def _feature_row(candidate_id="cand_001", **kwargs):
    defaults = dict(
        candidate_id=candidate_id,
        skill_strength_scores={"python": 0.9, "kafka": 0.7},
        notice_period_days=60, avg_tenure_months=30.0,
        experience_years=6.0, job_hopping_flag=0, latest_seniority=0.75,
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


def _score_breakdown(candidate_id="cand_001", composite=0.78):
    return ScoreBreakdown(
        candidate_id=candidate_id, semantic_score=0.8, trajectory_score=0.9,
        stability_score=0.7, platform_score=0.8, cert_bonus=0.05,
        composite_score=composite,
    )


def _fake_response(content: str, prompt_tokens: int = 200, completion_tokens: int = 150):
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


JD_INTENT = JDIntent(
    seniority_level=0.75,
    seniority_evidence="lead the team",
    must_have_skills=["Python", "Kafka"],
    nice_to_have_skills=["GCP"],
    core_problems_to_solve="Build data pipelines",
    implicit_soft_skills=["leadership"],
    domain_tags=["data-engineering"],
    requires_technical_github_signals=True,
)


class TestGenerateExplanations:
    def test_returns_explanations_for_batch(self):
        client = MagicMock()
        valid_json = json.dumps([
            {"candidate_id": "cand_001", "match_summary": "Strong match with Python",
             "skill_alignment": "Python (0.90)", "seniority_assessment": "Senior level",
             "trajectory_signal": "Promoted once", "platform_summary": "Active",
             "flags": "No flags"},
        ])
        client.chat.completions.create.return_value = _fake_response(valid_json)

        top = [_score_breakdown("cand_001")]
        store = {"cand_001": _feature_row("cand_001")}
        profile_store = {"cand_001": {"career_history": [{"company": "Acme", "title": "Senior Engineer", "duration_months": 36}]}}

        results = generate_explanations(top, store, profile_store, JD_INTENT, client, SETTINGS)

        assert len(results) == 1
        assert results[0].candidate_id == "cand_001"
        assert results[0].grounding_validated is True

    def test_falls_back_on_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")

        top = [_score_breakdown("cand_001")]
        store = {"cand_001": _feature_row("cand_001")}
        profile_store = {"cand_001": {}}

        results = generate_explanations(top, store, profile_store, JD_INTENT, client, SETTINGS)

        assert len(results) == 1
        assert results[0].candidate_id == "cand_001"
        assert results[0].grounding_validated is False

    def test_retries_on_grounding_failure(self):
        client = MagicMock()
        # First response: generic text that won't ground
        generic_response = json.dumps([
            {"candidate_id": "cand_001", "match_summary": "Generic summary",
             "skill_alignment": "Generic", "seniority_assessment": "Generic",
             "trajectory_signal": "Generic", "platform_summary": "Generic",
             "flags": "No flags"},
        ])
        # Second response: specific text that grounds
        specific_response = json.dumps([
            {"candidate_id": "cand_001", "match_summary": "Strong match using Python",
             "skill_alignment": "Python (0.90)", "seniority_assessment": "Senior",
             "trajectory_signal": "Promoted", "platform_summary": "Active",
             "flags": "No flags"},
        ])
        client.chat.completions.create.side_effect = [
            _fake_response(generic_response),
            _fake_response(specific_response),
        ]

        top = [_score_breakdown("cand_001")]
        store = {"cand_001": _feature_row("cand_001")}
        profile_store = {"cand_001": {}}

        results = generate_explanations(top, store, profile_store, JD_INTENT, client, SETTINGS)

        assert len(results) == 1
        assert results[0].grounding_validated is True
        assert client.chat.completions.create.call_count == 2

    def test_batches_multiple_candidates(self):
        client = MagicMock()
        valid_json = json.dumps([
            {"candidate_id": "cand_001", "match_summary": "Python (0.90)",
             "skill_alignment": "", "seniority_assessment": "",
             "trajectory_signal": "", "platform_summary": "", "flags": ""},
            {"candidate_id": "cand_002", "match_summary": "Kafka (0.70)",
             "skill_alignment": "", "seniority_assessment": "",
             "trajectory_signal": "", "platform_summary": "", "flags": ""},
        ])
        client.chat.completions.create.return_value = _fake_response(valid_json)

        top = [_score_breakdown("cand_001"), _score_breakdown("cand_002", composite=0.72)]
        store = {"cand_001": _feature_row("cand_001"), "cand_002": _feature_row("cand_002")}
        profile_store = {"cand_001": {}, "cand_002": {}}

        results = generate_explanations(top, store, profile_store, JD_INTENT, client, SETTINGS)

        assert len(results) == 2
        assert results[0].grounding_validated is True
        assert results[1].grounding_validated is True

    def test_logs_token_usage(self, caplog):
        caplog.set_level("INFO")
        client = MagicMock()
        valid_json = json.dumps([
            {"candidate_id": "cand_001", "match_summary": "Python (0.90)",
             "skill_alignment": "", "seniority_assessment": "",
             "trajectory_signal": "", "platform_summary": "", "flags": ""},
        ])
        client.chat.completions.create.return_value = _fake_response(valid_json, prompt_tokens=250, completion_tokens=80)

        top = [_score_breakdown("cand_001")]
        store = {"cand_001": _feature_row("cand_001")}
        profile_store = {"cand_001": {}}

        generate_explanations(top, store, profile_store, JD_INTENT, client, SETTINGS)

        assert any("input_tokens=250" in m for m in caplog.messages)
        assert any("output_tokens=80" in m for m in caplog.messages)


class TestCallBatchLLM:
    def test_parses_json_array(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response(
            json.dumps([{"candidate_id": "cand_001"}])
        )
        result = _call_batch_llm("prompt", client, SETTINGS)
        assert result == [{"candidate_id": "cand_001"}]

    def test_parses_single_object(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response(
            json.dumps({"candidate_id": "cand_001"})
        )
        result = _call_batch_llm("prompt", client, SETTINGS)
        assert result == [{"candidate_id": "cand_001"}]

    def test_parses_candidates_wrapper(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response(
            json.dumps({"candidates": [{"candidate_id": "cand_001"}]})
        )
        result = _call_batch_llm("prompt", client, SETTINGS)
        assert result == [{"candidate_id": "cand_001"}]

    def test_returns_none_on_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("fail")
        result = _call_batch_llm("prompt", client, SETTINGS)
        assert result is None
