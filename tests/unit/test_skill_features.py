import pytest
from src.feature_extractor.skill_features import compute_skill_strength, build_skill_strength_map
from src.models import SkillRecord


class TestComputeSkillStrength:
    def _skill(self, name="Python", proficiency="beginner", endorsements=0, duration_years=0.0):
        return SkillRecord(name=name, proficiency=proficiency, endorsements=endorsements, duration_years=duration_years)

    def test_beginner_zero_duration_zero_endorse(self):
        s = self._skill(proficiency="beginner")
        score = compute_skill_strength(s)
        assert score == 0.25 * 0.50 + 0.0 * 0.35 + 0.0 * 0.15  # 0.125

    def test_expert_five_years_50_endorse(self):
        s = self._skill(proficiency="expert", duration_years=5.0, endorsements=50)
        score = compute_skill_strength(s)
        expected = 1.0 * 0.50 + 1.0 * 0.35 + 1.0 * 0.15
        assert score == pytest.approx(expected, abs=1e-4)

    def test_expert_ten_years_capped(self):
        s = self._skill(proficiency="expert", duration_years=10.0, endorsements=100)
        score = compute_skill_strength(s)
        expected = 1.0 * 0.50 + 1.0 * 0.35 + 1.0 * 0.15
        assert score == pytest.approx(expected, abs=1e-4)

    def test_intermediate_partial(self):
        s = self._skill(proficiency="intermediate", duration_years=2.0, endorsements=10)
        score = compute_skill_strength(s)
        expected = 0.5 * 0.50 + (2.0 / 5.0) * 0.35 + (10.0 / 50.0) * 0.15
        assert score == pytest.approx(expected, abs=1e-4)

    def test_advanced_three_years_25_endorse(self):
        s = self._skill(proficiency="advanced", duration_years=3.0, endorsements=25)
        score = compute_skill_strength(s)
        expected = 0.75 * 0.50 + (3.0 / 5.0) * 0.35 + (25.0 / 50.0) * 0.15
        assert score == pytest.approx(expected, abs=1e-4)

    def test_score_capped_at_one(self):
        s = self._skill(proficiency="expert", duration_years=100, endorsements=1000)
        score = compute_skill_strength(s)
        assert score == 1.0

    def test_rounding_to_four_decimals(self):
        s = self._skill(proficiency="advanced", duration_years=1.0, endorsements=3)
        score = compute_skill_strength(s)
        assert len(str(score).split(".")[1]) <= 4

    def test_unknown_proficiency_defaults_to_beginner(self):
        s = self._skill(proficiency="grandmaster", duration_years=2.0, endorsements=5)
        score = compute_skill_strength(s)
        expected = 0.25 * 0.50 + (2.0 / 5.0) * 0.35 + (5.0 / 50.0) * 0.15
        assert score == pytest.approx(expected, abs=1e-4)


class TestBuildSkillStrengthMap:
    def test_empty_list(self):
        assert build_skill_strength_map([]) == {}

    def test_single_skill(self):
        skills = [SkillRecord(name="Python", proficiency="expert", endorsements=50, duration_years=5.0)]
        result = build_skill_strength_map(skills)
        assert "python" in result
        expected = 1.0 * 0.50 + 1.0 * 0.35 + 1.0 * 0.15
        assert result["python"] == pytest.approx(expected, abs=1e-4)

    def test_multiple_skills(self):
        skills = [
            SkillRecord(name="Python", proficiency="expert", endorsements=50, duration_years=5.0),
            SkillRecord(name="AWS", proficiency="advanced", endorsements=20, duration_years=3.0),
        ]
        result = build_skill_strength_map(skills)
        assert len(result) == 2
        assert "python" in result
        assert "aws" in result

    def test_keys_are_lowercased(self):
        skills = [SkillRecord(name="Kubernetes", proficiency="intermediate")]
        result = build_skill_strength_map(skills)
        assert "Kubernetes" not in result
        assert "kubernetes" in result
