import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.exceptions import JDParseError
from src.jd_parser.parser import parse_job_description, _validate_jd_intent, _build_jd_intent
from src.jd_parser.prompt_templates import (
    JD_PARSE_SYSTEM_PROMPT, JD_PARSE_USER_PROMPT,
    CORRECTIVE_SYSTEM_PROMPT, CORRECTIVE_USER_PROMPT,
)
from src.models import JDIntent
from src.config import Settings


SETTINGS = Settings()


def _fake_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestPromptTemplates:
    def test_system_prompt_contains_schema(self):
        assert "seniority_level" in JD_PARSE_SYSTEM_PROMPT
        assert "must_have_skills" in JD_PARSE_SYSTEM_PROMPT
        assert "json_object" not in JD_PARSE_SYSTEM_PROMPT  # instructs valid JSON

    def test_user_prompt_formats(self):
        result = JD_PARSE_USER_PROMPT.format(jd_text="Hiring a Senior Engineer")
        assert "Hiring a Senior Engineer" in result

    def test_corrective_prompt_contains_error(self):
        result = CORRECTIVE_USER_PROMPT.format(
            error="must_have_skills is empty",
            invalid_output='{"seniority_level": 0.5}',
            jd_text="Hiring an engineer",
        )
        assert "must_have_skills is empty" in result
        assert '"seniority_level": 0.5' in result
        assert "Hiring an engineer" in result


class TestParseJobDescription:
    def test_successful_parse(self, tmp_path):
        client = MagicMock()
        valid_response = {
            "seniority_level": 0.75,
            "seniority_evidence": "lead the team",
            "must_have_skills": ["Python", "Kafka"],
            "nice_to_have_skills": ["GCP"],
            "core_problems_to_solve": "Build data pipelines",
            "implicit_soft_skills": ["leadership"],
            "domain_tags": ["data-engineering"],
            "requires_technical_github_signals": True,
            "work_context": {"work_mode": None, "location_required": None, "location_is_hard_requirement": False, "salary_min_lpa": None, "salary_max_lpa": None},
            "salary_stated": False,
        }
        client.chat.completions.create.return_value = _fake_response(json.dumps(valid_response))

        result = parse_job_description("Hiring a Senior Engineer", client, SETTINGS, output_dir=tmp_path)

        assert isinstance(result, JDIntent)
        assert result.seniority_level == 0.75
        assert result.must_have_skills == ["Python", "Kafka"]
        assert result.requires_technical_github_signals is True

        # Check output file written
        intent_file = tmp_path / "jd_intent.json"
        assert intent_file.exists()

    def test_retry_on_validation_failure(self):
        client = MagicMock()
        # First response: invalid (missing must_have_skills)
        invalid = {
            "seniority_level": 0.75,
            "must_have_skills": [],
            "core_problems_to_solve": "Build pipelines",
            "domain_tags": ["data-engineering"],
        }
        # Second response: valid
        valid = dict(invalid)
        valid["must_have_skills"] = ["Python"]

        client.chat.completions.create.side_effect = [
            _fake_response(json.dumps(invalid)),
            _fake_response(json.dumps(valid)),
        ]

        result = parse_job_description("Hiring", client, SETTINGS)
        assert result.must_have_skills == ["Python"]
        assert client.chat.completions.create.call_count == 2

    def test_raise_after_two_failures(self):
        client = MagicMock()
        invalid = {
            "seniority_level": 0.75,
            "must_have_skills": [],
            "core_problems_to_solve": "",
            "domain_tags": [],
        }
        client.chat.completions.create.return_value = _fake_response(json.dumps(invalid))

        with pytest.raises(JDParseError, match="failed after retry"):
            parse_job_description("Hiring", client, SETTINGS)

    def test_rounds_seniority_level(self):
        client = MagicMock()
        response = {
            "seniority_level": 0.6,  # not in allowed set
            "seniority_evidence": "",
            "must_have_skills": ["Python"],
            "nice_to_have_skills": [],
            "core_problems_to_solve": "Build stuff",
            "implicit_soft_skills": [],
            "domain_tags": ["backend"],
            "requires_technical_github_signals": False,
            "work_context": {},
            "salary_stated": False,
        }
        client.chat.completions.create.return_value = _fake_response(json.dumps(response))

        result = parse_job_description("Hiring", client, SETTINGS)
        # 0.6 rounds to nearest allowed: 0.5 (distance 0.1) vs 0.75 (distance 0.15) -> 0.5
        assert result.seniority_level == 0.5

    def test_logs_token_usage(self, caplog):
        caplog.set_level("INFO")
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response(
            json.dumps({"seniority_level": 0.5, "seniority_evidence": "", "must_have_skills": ["Python"], "nice_to_have_skills": [], "core_problems_to_solve": "Build", "implicit_soft_skills": [], "domain_tags": ["backend"], "requires_technical_github_signals": False, "work_context": {}, "salary_stated": False}),
            prompt_tokens=120, completion_tokens=60,
        )

        parse_job_description("Hiring", client, SETTINGS)
        assert any("input_tokens=120" in m for m in caplog.messages)
        assert any("output_tokens=60" in m for m in caplog.messages)


class TestValidateJDIntent:
    def test_missing_required_field(self):
        ok, err = _validate_jd_intent({"seniority_level": 0.5})
        assert ok is False
        assert "missing" in err

    def test_empty_must_have_skills(self):
        ok, err = _validate_jd_intent({
            "seniority_level": 0.5,
            "must_have_skills": [],
            "core_problems_to_solve": "X",
            "domain_tags": ["backend"],
        })
        assert ok is False
        assert "empty" in err

    def test_valid(self):
        ok, err = _validate_jd_intent({
            "seniority_level": 0.5,
            "must_have_skills": ["Python"],
            "core_problems_to_solve": "X",
            "domain_tags": ["backend"],
        })
        assert ok is True
        assert err == ""


class TestBuildJDIntent:
    def test_builds_with_defaults(self):
        intent = _build_jd_intent({})
        assert isinstance(intent, JDIntent)
        assert intent.seniority_level == 0.5
        assert intent.must_have_skills == []

    def test_builds_work_context(self):
        raw = {
            "work_context": {
                "work_mode": "remote",
                "location_required": "Bangalore",
                "location_is_hard_requirement": True,
                "salary_min_lpa": 20,
                "salary_max_lpa": 30,
            },
        }
        intent = _build_jd_intent(raw)
        assert intent.work_context.work_mode == "remote"
        assert intent.work_context.salary_min_lpa == 20
        assert intent.work_context.location_is_hard_requirement is True
