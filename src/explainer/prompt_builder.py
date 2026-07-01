from src.models import JDIntent, CandidateFeatureRow, ScoreBreakdown


def build_explanation_prompt(
    candidate_id: str,
    scores: ScoreBreakdown,
    feature_row: CandidateFeatureRow,
    jd_intent: JDIntent,
    career_history: list[dict] | None = None,
) -> str:
    """Builds a per-candidate explanation prompt from computed scores and evidence.
    Matches the structure defined in doc 04 §5.2."""

    matched_skills = [
        f"{s} (strength: {feature_row.skill_strength_scores.get(s.lower(), 0):.2f})"
        for s in jd_intent.must_have_skills
        if feature_row.skill_strength_scores.get(s.lower(), 0) > 0.3
    ]

    career_lines = (career_history or [])[:3]
    career_summary = "; ".join([
        f"{r.get('title', '')} at {r.get('company', '')} ({r.get('duration_months', 0)} months)"
        for r in career_lines
    ]) if career_lines else "No career history available"

    promo_rate = feature_row.promotion_rate
    promotions_evidence = (
        f"Detected {int(promo_rate * 10)}/10 eligible companies with promotion"
        if promo_rate > 0 else "No internal promotions detected"
    )

    rs = feature_row
    platform_summary = (
        f"Open to work: {rs.open_to_work}; "
        f"Applications last 30d: {rs.active_intent_score * 10:.0f} (derived); "
        f"Notice period: {rs.notice_period_days} days"
    )

    cert_names = [c.get("name", "") for c in rs.cert_records[:3]]
    cert_evidence = ", ".join(cert_names) if cert_names else "None held"

    flags = []
    if rs.job_hopping_flag:
        flags.append("3+ consecutive roles under 12 months — validate tenure intent in interview")
    if rs.notice_period_days > 90:
        flags.append(f"Notice period is {rs.notice_period_days} days — plan timeline accordingly")
    if not rs.open_to_work:
        flags.append("Passive candidate — outreach required; no recent application activity")

    # Build skill records dict for grounding injection
    skill_records = feature_row.skill_strength_scores
    grounding_instruction = (
        "CRITICAL: You MUST mention at least one of these exact values in your response: "
        f"Skills: {', '.join(list(skill_records.keys())[:5])}. "
        f"Experience: {feature_row.experience_years:.1f} years. "
        f"Notice period: {feature_row.notice_period_days} days. "
        "If you do not cite at least one exact skill name or number, your response will be rejected."
    )

    # Explicit threshold logic to counter Qwen 7B hedging bias
    comp_score = scores.composite_score
    if comp_score >= 0.68:
        match_label = "strong"
    elif comp_score >= 0.55:
        match_label = "moderate"
    else:
        match_label = "cautious"

    prompt = f"""You are writing a candidate justification for a recruiter.
Your job is to convert the provided scores and evidence into a clear, specific,
recruiter-facing explanation. DO NOT invent information. Only use the evidence provided.
Use past-tense professional language. Do not use bullet points — write in paragraph form.

JD CONTEXT:
- Role seniority: {jd_intent.seniority_level} ({jd_intent.seniority_evidence})
- Must-have skills: {', '.join(jd_intent.must_have_skills)}
- Core problems to solve: {jd_intent.core_problems_to_solve}

CANDIDATE SCORES:
- Composite: {scores.composite_score:.3f}
- Semantic fit (B1): {scores.semantic_score:.3f}
- Trajectory fit (B2): {scores.trajectory_score:.3f}
- Stability (B3): {scores.stability_score:.3f}
- Platform signals (B4): {scores.platform_score:.3f}
- Cert bonus (B5): {scores.cert_bonus:.3f}

EVIDENCE FOR EACH SCORE:
Semantic fit evidence — matched JD skills: {', '.join(matched_skills) or 'Weak direct skill match; similarity is through career context'}
Career timeline (3 most recent roles): {career_summary}
Trajectory evidence: {promotions_evidence}; Latest seniority level: {feature_row.latest_seniority}; Total experience: {feature_row.experience_years:.1f} years
Platform evidence: {platform_summary}
Certifications: {cert_evidence}
Flags to surface: {'; '.join(flags) if flags else 'None'}

Write the explanation in EXACTLY this structure:

1. MATCH SUMMARY (1 sentence): Start with "This candidate is a {match_label} match for the [role name from JD] based on..."

2. SKILL ALIGNMENT (2-3 sentences): Explain which specific skills matched the JD's requirements, at what proficiency and duration, and call out any semantic equivalences (e.g., "Terraform and GCP map to the JD's 'cloud infrastructure' requirement").

3. SENIORITY ASSESSMENT (1-2 sentences): Explain why this candidate's seniority level aligns (or represents a calculated stretch) with the JD's requirement, citing the latest title and trajectory evidence.

4. TRAJECTORY SIGNAL (1-2 sentences): What does their career arc reveal? Cite promotions, increasing scope, or domain focus using the career timeline evidence.

5. PLATFORM SIGNAL SUMMARY (1-2 sentences): Summarize the B4 evidence — intent, responsiveness, reliability — in plain recruiter language.

6. FLAGS (if any): Concerns the recruiter should probe in screening. Write "No flags" if none.

{grounding_instruction}"""
    return prompt