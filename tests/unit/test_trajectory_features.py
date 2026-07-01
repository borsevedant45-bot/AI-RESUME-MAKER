import pytest
from src.feature_extractor.trajectory_features import (
    map_title_to_seniority, detect_promotions, compute_trajectory_base
)
from src.models import CandidateProfile, RoleRecord


class TestMapTitleToSeniority:
    def test_intern(self):
        assert map_title_to_seniority("Software Engineering Intern") == 0.1

    def test_junior(self):
        assert map_title_to_seniority("Junior Developer") == 0.2

    def test_associate(self):
        assert map_title_to_seniority("Associate Analyst") == 0.2

    def test_entry(self):
        assert map_title_to_seniority("Entry Level Engineer") == 0.2

    def test_graduate(self):
        assert map_title_to_seniority("Graduate Engineer") == 0.2

    def test_senior(self):
        assert map_title_to_seniority("Senior Engineer") == 0.75

    def test_lead(self):
        assert map_title_to_seniority("Tech Lead") == 0.75

    def test_manager(self):
        assert map_title_to_seniority("Engineering Manager") == 0.75

    def test_principal(self):
        assert map_title_to_seniority("Principal Architect") == 1.0

    def test_staff(self):
        assert map_title_to_seniority("Staff Engineer") == 1.0

    def test_director(self):
        assert map_title_to_seniority("Director of Engineering") == 1.0

    def test_vp(self):
        assert map_title_to_seniority("VP of Technology") == 1.0

    def test_head(self):
        assert map_title_to_seniority("Head of Product") == 1.0

    def test_chief(self):
        assert map_title_to_seniority("Chief Architect") == 1.0

    def test_cto(self):
        assert map_title_to_seniority("CTO") == 1.0

    def test_ceo(self):
        assert map_title_to_seniority("CEO") == 1.0

    def test_mid_default(self):
        assert map_title_to_seniority("Software Developer") == 0.5

    def test_case_insensitive(self):
        assert map_title_to_seniority("SENIOR ENGINEER") == 0.75

    def test_trainee(self):
        assert map_title_to_seniority("Trainee Engineer") == 0.1


class TestDetectPromotions:
    def _role(self, company, title, start_date, duration_months=12):
        return RoleRecord(
            company=company, title=title, start_date=start_date,
            duration_months=duration_months,
        )

    def test_no_roles(self):
        assert detect_promotions([]) == 0.0

    def test_single_role(self):
        roles = [self._role("Acme", "Engineer", "2020-01")]
        assert detect_promotions(roles) == 0.0

    def test_promotion_within_company(self):
        roles = [
            self._role("Acme", "Junior Engineer", "2020-01"),
            self._role("Acme", "Senior Engineer", "2022-01"),
        ]
        assert detect_promotions(roles) == 1.0

    def test_no_promotion(self):
        roles = [
            self._role("Acme", "Engineer", "2020-01"),
            self._role("Acme", "Engineer", "2022-01"),
        ]
        assert detect_promotions(roles) == 0.0

    def test_multiple_companies_some_promotions(self):
        roles = [
            self._role("Acme", "Associate", "2018-01"),
            self._role("Acme", "Senior", "2020-01"),
            self._role("Beta", "Engineer", "2020-06"),
            self._role("Beta", "Director", "2022-01"),
        ]
        assert detect_promotions(roles) == 1.0

    def test_multiple_companies_no_promotions(self):
        roles = [
            self._role("Acme", "Engineer", "2018-01"),
            self._role("Acme", "Engineer", "2020-01"),
            self._role("Beta", "Engineer", "2020-06"),
        ]
        assert detect_promotions(roles) == 0.0

    def test_single_role_per_company_skipped(self):
        roles = [
            self._role("Acme", "Engineer", "2020-01"),
            self._role("Beta", "Engineer", "2021-01"),
            self._role("Gamma", "Senior", "2022-01"),
        ]
        assert detect_promotions(roles) == 0.0


class TestComputeTrajectoryBase:
    def _profile(self, roles=None):
        return CandidateProfile(
            candidate_id="cand_001",
            career_history=roles or [],
        )

    def test_empty_career(self):
        result = compute_trajectory_base(self._profile([]))
        assert result == {"latest_seniority": 0.5, "promotion_rate": 0.0}

    def test_latest_seniority(self):
        roles = [
            RoleRecord(company="Acme", title="Engineer", start_date="2020-01"),
            RoleRecord(company="Beta", title="Senior Engineer", start_date="2022-01"),
        ]
        result = compute_trajectory_base(self._profile(roles))
        assert result["latest_seniority"] == 0.75

    def test_promotion_rate_included(self):
        roles = [
            RoleRecord(company="Acme", title="Junior", start_date="2018-01"),
            RoleRecord(company="Acme", title="Senior", start_date="2020-01"),
        ]
        result = compute_trajectory_base(self._profile(roles))
        assert result["promotion_rate"] == 1.0
