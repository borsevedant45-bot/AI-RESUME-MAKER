import json
import logging
from pathlib import Path
from typing import Iterator

from src.exceptions import DataLoadError
from src.models import (
    CandidateProfile, SkillRecord, RoleRecord, EducationRecord,
    CertRecord, RedrobSignals,
)

logger = logging.getLogger(__name__)


def load_candidates(jsonl_path: Path) -> Iterator[CandidateProfile]:
    """
    Lazily yields CandidateProfile objects from a JSONL file.
    Skips and logs malformed records rather than raising.
    """
    valid_count = 0
    skip_count = 0
    total_lines = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                raw = json.loads(line)
                profile = _parse_profile(raw)
                valid_count += 1
                yield profile
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("Skipping candidate at line %d: %s", line_number, str(e))
                skip_count += 1

    logger.info("Loaded %d valid profiles, skipped %d records", valid_count, skip_count)
    if total_lines > 0 and skip_count / total_lines > 0.10:
        raise DataLoadError(
            f"Skipped {skip_count}/{total_lines} records ({skip_count/total_lines:.1%}) — exceeds 10% threshold"
        )


def load_candidates_batch(
    jsonl_path: Path,
    batch_size: int = 1000,
) -> Iterator[list[CandidateProfile]]:
    """
    Yields lists of CandidateProfile objects in batches.
    Used by the indexing pipeline for memory-efficient processing.
    """
    batch = []
    for profile in load_candidates(jsonl_path):
        batch.append(profile)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _parse_profile(raw: dict) -> CandidateProfile:
    profile_raw = raw.get("profile", {})
    signals_raw = raw.get("redrob_signals", {})

    skills = [
        SkillRecord(
            name=s.get("name", ""),
            proficiency=s.get("proficiency", "beginner"),
            endorsements=s.get("endorsements", 0),
            duration_years=s.get("duration_months", 0) / 12.0,
        )
        for s in raw.get("skills", [])
    ]

    career = [
        RoleRecord(
            company=r.get("company", ""),
            title=r.get("title", ""),
            start_date=r.get("start_date", ""),
            end_date=r.get("end_date"),
            duration_months=r.get("duration_months", 0),
            industry=r.get("industry", ""),
            company_size=r.get("company_size", ""),
            description=r.get("description", ""),
            is_current_role=r.get("end_date") is None,
        )
        for r in raw.get("career_history", [])
    ]

    education = [
        EducationRecord(
            degree=e.get("degree", ""),
            field_of_study=e.get("field_of_study", ""),
            institution_tier=e.get("tier", "tier_3"),
            graduation_year=e.get("end_year"),
        )
        for e in raw.get("education", [])
    ]

    certs = [
        CertRecord(
            name=c.get("name", ""),
            issuer=c.get("issuer", ""),
            issue_year=c.get("year", 0),
        )
        for c in raw.get("certifications", [])
    ]

    signals = RedrobSignals(
        profile_completeness_score=signals_raw.get("profile_completeness_score", 0.0),
        connection_count=signals_raw.get("connection_count", 0),
        endorsements_received=signals_raw.get("endorsements_received", 0),
        notice_period_days=signals_raw.get("notice_period_days", 0),
        profile_views_30d=signals_raw.get("profile_views_received_30d", 0),
        applications_submitted_30d=signals_raw.get("applications_submitted_30d", 0),
        recruiter_response_rate=signals_raw.get("recruiter_response_rate", 0.0),
        avg_response_time_hrs=signals_raw.get("avg_response_time_hours", 0.0),
        search_appearances_30d=signals_raw.get("search_appearance_30d", 0),
        saved_by_recruiters_30d=signals_raw.get("saved_by_recruiters_30d", 0),
        interview_completion_rate=signals_raw.get("interview_completion_rate", 0.0),
        offer_acceptance_rate=signals_raw.get("offer_acceptance_rate", -1.0),
        github_activity_score=signals_raw.get("github_activity_score", -1.0),
        open_to_work=signals_raw.get("open_to_work_flag", False),
        willing_to_relocate=signals_raw.get("willing_to_relocate", False),
        email_verified=signals_raw.get("verified_email", False),
        phone_verified=signals_raw.get("verified_phone", False),
        linkedin_connected=signals_raw.get("linkedin_connected", False),
        work_mode_preference=signals_raw.get("preferred_work_mode", "hybrid"),
        expected_salary_min=signals_raw.get("expected_salary_range_inr_lpa", {}).get("min", 0),
        expected_salary_max=signals_raw.get("expected_salary_range_inr_lpa", {}).get("max", 0),
    )

    return CandidateProfile(
        candidate_id=raw.get("candidate_id", ""),
        experience_years=profile_raw.get("years_of_experience", 0.0),
        country=profile_raw.get("country", ""),
        industry=profile_raw.get("current_industry", ""),
        current_title=profile_raw.get("current_title", ""),
        current_company=profile_raw.get("current_company", ""),
        company_size=profile_raw.get("current_company_size", ""),
        location=profile_raw.get("location", ""),
        skills=skills,
        career_history=career,
        education=education,
        certifications=certs,
        redrob_signals=signals,
    )
