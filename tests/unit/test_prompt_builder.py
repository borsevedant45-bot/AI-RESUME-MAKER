from src.explainer.prompt_builder import build_explanation_prompt
from src.models import JDIntent, CandidateFeatureRow, ScoreBreakdown


class TestBuildExplanationPrompt:
    def test_contains_all_sections(self):
        jd = JDIntent(
            seniority_level=0.75,
            seniority_evidence="lead the team",
            must_have_skills=["Python", "Kafka"],
            core_problems_to_solve="Build data pipelines",
        )
        fr = CandidateFeatureRow(
            candidate_id="cand_001",
            skill_strength_scores={"python": 0.9, "kafka": 0.7},
            latest_seniority=0.75,
            promotion_rate=0.5,
            experience_years=6.0,
            avg_tenure_months=30.0,
            job_hopping_flag=0,
            notice_period_days=60,
            open_to_work=False,
            active_intent_score=0.5,
            hire_reliability_score=0.8,
            cert_records=[{"name": "AWS Certified", "issue_year": 2023}],
        )
        sb = ScoreBreakdown(
            candidate_id="cand_001",
            semantic_score=0.8, trajectory_score=0.9, stability_score=0.7,
            platform_score=0.8, cert_bonus=0.05, composite_score=0.78,
        )

        prompt = build_explanation_prompt("cand_001", sb, fr, jd)

        assert "MATCH SUMMARY" in prompt
        assert "SKILL ALIGNMENT" in prompt
        assert "SENIORITY ASSESSMENT" in prompt
        assert "TRAJECTORY SIGNAL" in prompt
        assert "PLATFORM SIGNAL SUMMARY" in prompt
        assert "FLAGS" in prompt

        # Contains scores
        assert "0.780" in prompt  # composite
        assert "0.800" in prompt  # semantic

        # Contains evidence
        assert "0.75" in prompt  # latest_seniority
        assert "5/10" in prompt  # promotion rate displayed as detected X/10
        assert "6.0" in prompt  # experience_years

    def test_includes_matched_skills(self):
        jd = JDIntent(must_have_skills=["Python", "Rust", "Kafka"])
        fr = CandidateFeatureRow(
            candidate_id="cand_001",
            skill_strength_scores={"python": 0.95, "kafka": 0.6, "rust": 0.2},
        )
        sb = ScoreBreakdown(candidate_id="cand_001")
        prompt = build_explanation_prompt("cand_001", sb, fr, jd)

        # Python (0.95) and Kafka (0.6) are > 0.3 threshold
        assert "python (strength: 0.95)" in prompt or "Python (strength: 0.95)" in prompt
        # Rust (0.2) is below threshold, should not appear in matched skills
        assert "rust (strength: 0.20)" not in prompt.lower()

    def test_flags_job_hopping(self):
        jd = JDIntent(must_have_skills=["Python"])
        fr = CandidateFeatureRow(
            candidate_id="cand_001",
            skill_strength_scores={"python": 0.5},
            job_hopping_flag=1,
            notice_period_days=100,
            open_to_work=False,
        )
        sb = ScoreBreakdown(candidate_id="cand_001")
        prompt = build_explanation_prompt("cand_001", sb, fr, jd)

        assert "consecutive roles" in prompt
        assert "Notice period" in prompt
        assert "Passive candidate" in prompt

    def test_no_flags(self):
        jd = JDIntent(must_have_skills=["Python"])
        fr = CandidateFeatureRow(
            candidate_id="cand_001",
            skill_strength_scores={"python": 0.5},
            job_hopping_flag=0,
            notice_period_days=30,
            open_to_work=True,
        )
        sb = ScoreBreakdown(candidate_id="cand_001")
        prompt = build_explanation_prompt("cand_001", sb, fr, jd)

        assert "Flags to surface: None" in prompt
