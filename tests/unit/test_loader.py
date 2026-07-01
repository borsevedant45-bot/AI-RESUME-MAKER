import json
import pytest
from pathlib import Path
from src.data_loader.loader import load_candidates, load_candidates_batch, _parse_profile
from src.exceptions import DataLoadError
from src.models import CandidateProfile, SkillRecord, RoleRecord, EducationRecord, CertRecord


class TestParseProfile:
    def test_minimal_profile(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 3.0, "country": "IN", "current_industry": "Tech"},
            "skills": [],
            "career_history": [],
            "education": [],
            "certifications": [],
            "redrob_signals": {},
        }
        profile = _parse_profile(raw)
        assert isinstance(profile, CandidateProfile)
        assert profile.candidate_id == "cand_001"
        assert profile.experience_years == 3.0

    def test_skill_parsing(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 5.0},
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
                {"name": "AWS", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
            ],
            "career_history": [],
            "education": [],
            "certifications": [],
            "redrob_signals": {},
        }
        profile = _parse_profile(raw)
        assert len(profile.skills) == 2
        assert profile.skills[0].name == "Python"
        assert profile.skills[0].proficiency == "expert"
        assert profile.skills[0].duration_years == 4.0
        assert profile.skills[1].name == "AWS"

    def test_career_history_parsing(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 5.0},
            "skills": [],
            "career_history": [
                {"company": "Acme", "title": "Engineer", "start_date": "2020-01", "end_date": None, "duration_months": 36, "industry": "Tech"},
                {"company": "Beta", "title": "Intern", "start_date": "2019-01", "end_date": "2019-12", "duration_months": 12, "industry": "Tech"},
            ],
            "education": [],
            "certifications": [],
            "redrob_signals": {},
        }
        profile = _parse_profile(raw)
        assert len(profile.career_history) == 2
        assert profile.career_history[0].company == "Acme"
        assert profile.career_history[0].is_current_role is True
        assert profile.career_history[1].is_current_role is False

    def test_education_parsing(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 5.0},
            "skills": [],
            "career_history": [],
            "education": [
                {"degree": "B.Tech", "field_of_study": "CS", "tier": "tier_1", "end_year": 2018},
            ],
            "certifications": [],
            "redrob_signals": {},
        }
        profile = _parse_profile(raw)
        assert len(profile.education) == 1
        assert profile.education[0].degree == "B.Tech"
        assert profile.education[0].institution_tier == "tier_1"

    def test_cert_parsing(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 5.0},
            "skills": [],
            "career_history": [],
            "education": [],
            "certifications": [{"name": "AWS SA Pro", "issuer": "AWS", "year": 2022}],
            "redrob_signals": {},
        }
        profile = _parse_profile(raw)
        assert len(profile.certifications) == 1
        assert profile.certifications[0].name == "AWS SA Pro"

    def test_redrob_signals_parsing(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 5.0},
            "skills": [],
            "career_history": [],
            "education": [],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 85.0,
                "connection_count": 500,
                "open_to_work_flag": True,
                "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            },
        }
        profile = _parse_profile(raw)
        signals = profile.redrob_signals
        assert signals.profile_completeness_score == 85.0
        assert signals.connection_count == 500
        assert signals.open_to_work is True
        assert signals.expected_salary_min == 20
        assert signals.expected_salary_max == 30


class TestLoadCandidates:
    def test_load_valid_jsonl(self, tmp_path):
        data = [
            {"candidate_id": "c_1", "profile": {"years_of_experience": 5}, "skills": [{"name": "Py"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}},
            {"candidate_id": "c_2", "profile": {"years_of_experience": 3}, "skills": [{"name": "JS"}], "career_history": [{"company": "B"}], "education": [], "certifications": [], "redrob_signals": {}},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        profiles = list(load_candidates(path))
        assert len(profiles) == 2
        assert profiles[0].candidate_id == "c_1"
        assert profiles[1].candidate_id == "c_2"

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"candidate_id": "c_1", "profile": {"years_of_experience": 5}, "skills": [{"name": "Py"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}\n')
            f.write("\n")
            f.write('{"candidate_id": "c_2", "profile": {"years_of_experience": 3}, "skills": [{"name": "JS"}], "career_history": [{"company": "B"}], "education": [], "certifications": [], "redrob_signals": {}}\n')

        profiles = list(load_candidates(path))
        assert len(profiles) == 2

    def test_skips_malformed_json(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"candidate_id": "c_1", "profile": {"years_of_experience": 5}, "skills": [{"name": "Py"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}\n')
            f.write("not valid json\n")
            f.write('{"candidate_id": "c_2", "profile": {"years_of_experience": 3}, "skills": [{"name": "JS"}], "career_history": [{"company": "B"}], "education": [], "certifications": [], "redrob_signals": {}}\n')
            # extra valid records to keep skip rate under 10%
            for i in range(10):
                f.write('{"candidate_id": "c_%d", "profile": {"years_of_experience": 1}, "skills": [{"name": "X"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}\n' % (i + 10))

        profiles = list(load_candidates(path))
        assert len(profiles) == 12  # 12 valid + 1 skipped

    def test_skips_missing_fields(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"candidate_id": "c_1", "profile": {"years_of_experience": 5}, "skills": [{"name": "Py"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}\n')
            f.write('not valid json either\n')
            # extra valid records to keep skip rate under 10%
            for i in range(10):
                f.write('{"candidate_id": "c_%d", "profile": {"years_of_experience": 1}, "skills": [{"name": "X"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}\n' % (i + 2))

        profiles = list(load_candidates(path))
        assert len(profiles) == 11  # 11 valid + 1 skipped

    def test_raises_on_high_skip_rate(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            # invalid JSON causes parser to skip
            for _ in range(5):
                f.write("{{{bad json}}}\n")

        with pytest.raises(DataLoadError, match="exceeds 10%"):
            list(load_candidates(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        profiles = list(load_candidates(path))
        assert profiles == []


class TestLoadCandidatesBatch:
    def test_batches_correctly(self, tmp_path):
        data_lines = []
        for i in range(25):
            data_lines.append(json.dumps({"candidate_id": f"c_{i}", "profile": {"years_of_experience": 5}, "skills": [{"name": "Py"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}))
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(data_lines))

        batches = list(load_candidates_batch(path, batch_size=10))
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    def test_single_batch(self, tmp_path):
        data_lines = []
        for i in range(3):
            data_lines.append(json.dumps({"candidate_id": f"c_{i}", "profile": {"years_of_experience": 5}, "skills": [{"name": "Py"}], "career_history": [{"company": "C"}], "education": [], "certifications": [], "redrob_signals": {}}))
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(data_lines))

        batches = list(load_candidates_batch(path, batch_size=10))
        assert len(batches) == 1
        assert len(batches[0]) == 3
