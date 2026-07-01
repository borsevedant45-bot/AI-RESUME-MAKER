import pytest
from src.data_loader.validator import validate_candidate


class TestValidateCandidate:
    def test_valid_candidate(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 5.0},
            "skills": [{"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 36}],
            "career_history": [{"company": "Acme", "title": "Engineer", "start_date": "2020-01-01"}],
            "redrob_signals": {},
        }
        ok, msg = validate_candidate(raw)
        assert ok is True
        assert msg == ""

    def test_missing_candidate_id(self):
        ok, msg = validate_candidate({"profile": {}, "skills": [], "career_history": [], "redrob_signals": {}})
        assert ok is False
        assert "candidate_id" in msg

    def test_empty_candidate_id(self):
        ok, msg = validate_candidate({"candidate_id": "", "profile": {}, "skills": [], "career_history": [], "redrob_signals": {}})
        assert ok is False
        assert "candidate_id" in msg

    def test_missing_profile_block(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {}, "skills": [{"name": "X"}], "career_history": [{"company": "Co"}], "redrob_signals": {}})
        assert ok is False
        assert "profile" in msg

    def test_missing_years_of_experience(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {"country": "IN"}, "skills": [{"name": "X"}], "career_history": [{"company": "Co"}], "redrob_signals": {}})
        assert ok is False
        assert "years_of_experience" in msg

    def test_negative_years_of_experience(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {"years_of_experience": -1}, "skills": [{"name": "X"}], "career_history": [{"company": "Co"}], "redrob_signals": {}})
        assert ok is False
        assert "years_of_experience" in msg

    def test_non_numeric_years_of_experience(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {"years_of_experience": "five"}, "skills": [{"name": "X"}], "career_history": [{"company": "Co"}], "redrob_signals": {}})
        assert ok is False
        assert "years_of_experience" in msg

    def test_zero_years_of_experience_is_valid(self):
        raw = {
            "candidate_id": "cand_001",
            "profile": {"years_of_experience": 0},
            "skills": [{"name": "Python"}],
            "career_history": [{"company": "Acme", "title": "Intern"}],
            "redrob_signals": {},
        }
        ok, msg = validate_candidate(raw)
        assert ok is True

    def test_missing_skills(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {"years_of_experience": 5}, "skills": [], "career_history": [{"company": "Co"}], "redrob_signals": {}})
        assert ok is False
        assert "skills" in msg

    def test_missing_career_history(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {"years_of_experience": 5}, "skills": [{"name": "X"}], "career_history": [], "redrob_signals": {}})
        assert ok is False
        assert "career_history" in msg

    def test_missing_redrob_signals(self):
        ok, msg = validate_candidate({"candidate_id": "cand_001", "profile": {"years_of_experience": 5}, "skills": [{"name": "X"}], "career_history": [{"company": "Co"}]})
        assert ok is False
        assert "redrob_signals" in msg

    def test_empty_dict(self):
        ok, msg = validate_candidate({})
        assert ok is False
        assert "candidate_id" in msg
