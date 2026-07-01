import pytest
from src.models import ScoreBreakdown
from src.ranker.ranker import rank_candidates


def _sb(candidate_id, composite, platform):
    return ScoreBreakdown(
        candidate_id=candidate_id,
        composite_score=composite,
        platform_score=platform,
        semantic_score=0.5,
        trajectory_score=0.5,
        stability_score=0.5,
        cert_bonus=0.0,
    )


class TestRankCandidates:
    def test_sorts_by_composite_descending(self):
        sb1 = _sb("a", 0.8, 0.5)
        sb2 = _sb("b", 0.9, 0.5)
        sb3 = _sb("c", 0.7, 0.5)
        result = rank_candidates([sb1, sb2, sb3], top_n=3)
        assert [r.candidate_id for r in result] == ["b", "a", "c"]

    def test_tiebreaker_by_platform_score(self):
        sb1 = _sb("a", 0.800, 0.7)
        sb2 = _sb("b", 0.801, 0.3)
        result = rank_candidates([sb1, sb2], top_n=2)
        # b has higher composite (0.801 > 0.800) → b first
        assert result[0].candidate_id == "b"

    def test_tiebreaker_within_tolerance(self):
        sb1 = _sb("a", 0.8004, 0.7)  # rounds to 0.800
        sb2 = _sb("b", 0.8002, 0.9)  # rounds to 0.800
        result = rank_candidates([sb1, sb2], top_n=2)
        # Both round to 0.800 → tiebreaker: higher platform_score (b: 0.9 > a: 0.7)
        assert result[0].candidate_id == "b"

    def test_top_n_selection(self):
        sbs = [_sb(f"c_{i}", 0.9 - i * 0.01, 0.5) for i in range(10)]
        result = rank_candidates(sbs, top_n=3)
        assert len(result) == 3

    def test_top_n_larger_than_list(self):
        sbs = [_sb("a", 0.8, 0.5), _sb("b", 0.7, 0.5)]
        result = rank_candidates(sbs, top_n=10)
        assert len(result) == 2
