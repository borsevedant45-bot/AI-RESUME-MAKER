import pytest
from src.feature_extractor.stability_features import (
    compute_avg_tenure, detect_job_hopping, best_institution_tier
)
from src.models import RoleRecord, EducationRecord


class TestComputeAvgTenure:
    def _role(self, duration_months):
        return RoleRecord(
            company="Acme", title="Engineer", start_date="2020-01",
            duration_months=duration_months,
        )

    def test_empty_history(self):
        assert compute_avg_tenure([]) == 0.0

    def test_single_role(self):
        assert compute_avg_tenure([self._role(24)]) == 24.0

    def test_multiple_roles(self):
        roles = [self._role(12), self._role(24), self._role(36)]
        assert compute_avg_tenure(roles) == 24.0

    def test_mixed_durations(self):
        roles = [self._role(10), self._role(20), self._role(30)]
        assert compute_avg_tenure(roles) == 20.0


class TestDetectJobHopping:
    def _role(self, duration_months, start_date="2020-01"):
        return RoleRecord(
            company="Acme", title="Engineer", start_date=start_date,
            duration_months=duration_months,
        )

    def test_empty_history(self):
        assert detect_job_hopping([]) == 0

    def test_no_short_roles(self):
        roles = [self._role(24), self._role(36), self._role(48)]
        assert detect_job_hopping(roles) == 0

    def test_one_short_role(self):
        roles = [self._role(6), self._role(24), self._role(36)]
        assert detect_job_hopping(roles) == 0

    def test_two_short_roles(self):
        roles = [self._role(6), self._role(8), self._role(24)]
        assert detect_job_hopping(roles) == 0

    def test_three_consecutive_short_roles(self):
        roles = [self._role(6), self._role(8), self._role(10)]
        assert detect_job_hopping(roles) == 1

    def test_four_consecutive_short_roles(self):
        roles = [self._role(6), self._role(8), self._role(10), self._role(4)]
        assert detect_job_hopping(roles) == 1

    def test_three_short_not_consecutive(self):
        roles = [
            self._role(6, "2020-01"),
            self._role(24, "2021-01"),
            self._role(8, "2023-01"),
            self._role(10, "2024-01"),
            self._role(24, "2025-01"),
        ]
        assert detect_job_hopping(roles) == 0

    def test_three_short_at_end(self):
        roles = [
            self._role(24, "2018-01"),
            self._role(6, "2020-01"),
            self._role(8, "2021-01"),
            self._role(10, "2022-01"),
        ]
        assert detect_job_hopping(roles) == 1

    def test_hopping_with_gap(self):
        roles = [
            self._role(6, "2020-01"),
            self._role(6, "2021-01"),
            self._role(24, "2022-01"),
            self._role(6, "2024-01"),
            self._role(6, "2025-01"),
            self._role(6, "2026-01"),
        ]
        assert detect_job_hopping(roles) == 1


class TestBestInstitutionTier:
    def _edu(self, tier):
        return EducationRecord(degree="B.Tech", institution_tier=tier)

    def test_empty_list(self):
        assert best_institution_tier([]) == "tier_4"

    def test_single_tier_1(self):
        assert best_institution_tier([self._edu("tier_1")]) == "tier_1"

    def test_single_tier_3(self):
        assert best_institution_tier([self._edu("tier_3")]) == "tier_3"

    def test_best_is_tier_1_among_mixed(self):
        edus = [self._edu("tier_3"), self._edu("tier_1"), self._edu("tier_2")]
        assert best_institution_tier(edus) == "tier_1"

    def test_all_tier_4(self):
        edus = [self._edu("tier_4"), self._edu("tier_4")]
        assert best_institution_tier(edus) == "tier_4"
