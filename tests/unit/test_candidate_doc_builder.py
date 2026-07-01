import pytest
from src.embedder.candidate_doc_builder import build_candidate_doc
from src.models import (
    CandidateProfile, SkillRecord, RoleRecord, EducationRecord, CertRecord,
)


def _profile(
    skills=None, roles=None, education=None, certs=None,
    candidate_id="cand_001", experience_years=5.0,
):
    return CandidateProfile(
        candidate_id=candidate_id,
        experience_years=experience_years,
        skills=skills or [],
        career_history=roles or [],
        education=education or [],
        certifications=certs or [],
    )


class TestBuildCandidateDoc:
    def test_empty_profile(self):
        doc, thin = build_candidate_doc(_profile())
        assert doc == ""
        assert thin is True

    def test_skills_section(self):
        skills = [SkillRecord(name="Python", proficiency="expert")]
        doc, thin = build_candidate_doc(_profile(skills=skills))
        assert "Skills:" in doc
        assert "Python (expert)" in doc
        assert thin is True  # only 1 skill, likely < 50 chars

    def test_career_history_most_recent_first(self):
        roles = [
            RoleRecord(
                company="Acme", title="Senior Engineer",
                start_date="2022-01", industry="Tech",
                description="Led platform team",
            ),
            RoleRecord(
                company="Beta", title="Junior Engineer",
                start_date="2020-01", industry="Tech",
                description="Built features",
            ),
        ]
        doc, thin = build_candidate_doc(_profile(roles=roles))
        parts = doc.split("Career: ")[1].split(" | ")
        # Most recent role should come first
        assert "Senior Engineer" in parts[0]
        assert "Junior Engineer" in parts[1]

    def test_career_description_long(self):
        roles = [
            RoleRecord(
                company="Acme", title="Engineer",
                start_date="2020-01", industry="Tech",
                description="Built and maintained the core platform infrastructure",
            ),
        ]
        doc, thin = build_candidate_doc(_profile(roles=roles))
        assert "Engineer in Tech: Built and maintained" in doc

    def test_career_description_short_fallback(self):
        roles = [
            RoleRecord(
                company="Acme", title="Engineer",
                start_date="2020-01", industry="Tech",
                description="Short",
            ),
        ]
        doc, thin = build_candidate_doc(_profile(roles=roles))
        # Should fall back to just "title in industry" without colon+description
        assert "Engineer in Tech" in doc
        assert "Engineer in Tech: Engineer in Tech" not in doc

    def test_career_empty_description(self):
        roles = [
            RoleRecord(
                company="Acme", title="Engineer",
                start_date="2020-01", industry="Tech",
                description="",
            ),
        ]
        doc, thin = build_candidate_doc(_profile(roles=roles))
        assert doc == "Career: Engineer in Tech"
        # No colon after "in Tech" (no description appended)
        assert doc.endswith("Engineer in Tech")

    def test_certifications_section(self):
        certs = [CertRecord(name="AWS Certified", issuer="AWS", issue_year=2022)]
        doc, thin = build_candidate_doc(_profile(certs=certs))
        assert "Certifications:" in doc
        assert "AWS Certified" in doc

    def test_education_field_of_study(self):
        edu = [
            EducationRecord(degree="B.Tech", field_of_study="Computer Science", institution_tier="tier_1"),
            EducationRecord(degree="M.Tech", field_of_study="", institution_tier="tier_1"),
        ]
        doc, thin = build_candidate_doc(_profile(education=edu))
        assert "Education:" in doc
        assert "Computer Science" in doc
        # Empty field_of_study should be excluded
        assert "Education: Computer Science" in doc

    def test_full_doc_joins_with_pipe(self):
        skills = [SkillRecord(name="Python", proficiency="expert")]
        roles = [
            RoleRecord(
                company="Acme", title="Engineer",
                start_date="2020-01", industry="Tech",
                description="Full desc that is long enough",
            ),
        ]
        certs = [CertRecord(name="AWS", issuer="AWS", issue_year=2023)]
        edu = [EducationRecord(degree="B.Tech", field_of_study="CS")]
        doc, thin = build_candidate_doc(_profile(skills=skills, roles=roles, certs=certs, education=edu))
        assert "Skills:" in doc
        assert "Career:" in doc
        assert "Certifications:" in doc
        assert "Education:" in doc
        assert " | " in doc
        assert thin is False

    def test_thin_profile_detection(self):
        # Very short doc
        doc, thin = build_candidate_doc(_profile())
        assert thin is True

    def test_non_thin_profile(self):
        skills = [SkillRecord(name="Python", proficiency="expert", endorsements=50, duration_years=5.0)]
        roles = [
            RoleRecord(
                company="Acme", title="Senior Engineer",
                start_date="2020-01", industry="Tech",
                description="Built and maintained core platform infrastructure for the team",
            ),
        ]
        doc, thin = build_candidate_doc(_profile(skills=skills, roles=roles))
        assert len(doc) >= 50
        assert thin is False

    def test_excludes_candidate_id(self):
        skills = [SkillRecord(name="Python", proficiency="expert")]
        doc, thin = build_candidate_doc(_profile(skills=skills, candidate_id="secret_123"))
        assert "secret_123" not in doc

    def test_excludes_location(self):
        profile = _profile()
        profile.location = "Bangalore, India"
        doc, thin = build_candidate_doc(profile)
        assert "Bangalore" not in doc
        assert "India" not in doc

    def test_excludes_company_name(self):
        roles = [
            RoleRecord(
                company="AcmeCorp", title="Engineer",
                start_date="2020-01", industry="Tech",
                description="Long enough description for test",
            ),
        ]
        doc, thin = build_candidate_doc(_profile(roles=roles))
        assert "AcmeCorp" not in doc
