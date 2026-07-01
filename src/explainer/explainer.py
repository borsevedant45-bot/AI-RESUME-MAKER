import json
import logging
import time
from typing import Any

from src.config import Settings
from src.models import JDIntent, CandidateFeatureRow, CandidateExplanation, ScoreBreakdown
from src.explainer.prompt_builder import build_explanation_prompt
from src.explainer.grounding_validator import validate_grounding, build_fallback_explanation

logger = logging.getLogger(__name__)

BATCH_SYSTEM_PROMPT = """\
You will write candidate justifications for a recruiter.
For each candidate, use ONLY the evidence provided.
Return a JSON array with one object per candidate:
[{"candidate_id": "...", "match_summary": "...", "skill_alignment": "...",
  "seniority_assessment": "...", "trajectory_signal": "...",
  "platform_summary": "...", "flags": "..."}]
"""

STRONGER_INSTRUCTION = """\
IMPORTANT: You MUST mention the candidate's specific skill names, role titles,
company names, or numerical metrics from the evidence in your response.
Do not write generic statements. Ground every claim in the actual data provided."""


def generate_explanations(
    top_candidates: list[ScoreBreakdown],
    feature_store: dict[str, CandidateFeatureRow],
    profile_store: dict[str, dict[str, Any]],
    jd_intent: JDIntent,
    client: Any,
    settings: Settings,
) -> list[CandidateExplanation]:
    """Batched LLM calls for top-N explanations with grounding validation and retry."""
    batch_size = settings.llm.explanation_batch_size
    explanations: list[CandidateExplanation] = []

    for batch_start in range(0, len(top_candidates), batch_size):
        batch = top_candidates[batch_start:batch_start + batch_size]
        batch_ids = [c.candidate_id for c in batch]
        logger.info("Generating explanations for batch: %s", batch_ids)

        expected_ids = {sb.candidate_id for sb in batch}
        covered_ids: set[str] = set()

        # Build combined batch prompt (skip candidates with no feature row)
        batch_prompt_parts = []
        for i, sb in enumerate(batch):
            fr = feature_store.get(sb.candidate_id)
            if fr is None:
                continue
            career_hist = (profile_store.get(sb.candidate_id) or {}).get("career_history", [])
            candidate_prompt = build_explanation_prompt(
                candidate_id=sb.candidate_id,
                scores=sb,
                feature_row=fr,
                jd_intent=jd_intent,
                career_history=career_hist,
            )
            batch_prompt_parts.append(
                f"===CANDIDATE {i + 1} (ID: {sb.candidate_id})===\n{candidate_prompt}"
            )

        combined_user = "\n\n".join(batch_prompt_parts)

        if combined_user.strip():
            parsed_list = _call_batch_llm(combined_user, client, settings)
            if parsed_list is None:
                parsed_list = _call_batch_llm(combined_user, client, settings, stronger=True)

            if parsed_list is not None:
                for entry in parsed_list:
                    cand_id = entry.get("candidate_id", "")
                    if cand_id not in expected_ids:
                        continue
                    covered_ids.add(cand_id)
                    fr = feature_store.get(cand_id)
                    career_hist = (profile_store.get(cand_id) or {}).get("career_history", [])

                    exp = CandidateExplanation(
                        candidate_id=cand_id,
                        match_summary=entry.get("match_summary") or "",
                        skill_alignment=entry.get("skill_alignment") or "",
                        seniority_assessment=entry.get("seniority_assessment") or "",
                        trajectory_signal=entry.get("trajectory_signal") or "",
                        platform_summary=entry.get("platform_summary") or "No platform signal data cited.",
                        flags=entry.get("flags") or "No flags",
                        grounding_validated=False,
                    )

                    if fr and validate_grounding(exp, fr, career_history=career_hist):
                        exp.grounding_validated = True
                    else:
                        logger.warning("Grounding failed for %s; retrying with stronger instruction", cand_id)
                        single_prompt = build_explanation_prompt(
                            candidate_id=cand_id, scores=_find_sb(batch, cand_id),
                            feature_row=fr, jd_intent=jd_intent, career_history=career_hist,
                        ) if fr else ""
                        if single_prompt:
                            retry_list = _call_batch_llm(single_prompt, client, settings, stronger=True)
                            if retry_list and len(retry_list) > 0:
                                retry_entry = retry_list[0]
                                exp = CandidateExplanation(
                                    candidate_id=cand_id,
                                    match_summary=retry_entry.get("match_summary") or "",
                                    skill_alignment=retry_entry.get("skill_alignment") or "",
                                    seniority_assessment=retry_entry.get("seniority_assessment") or "",
                                    trajectory_signal=retry_entry.get("trajectory_signal") or "",
                                    platform_summary=retry_entry.get("platform_summary") or "No platform signal data cited.",
                                    flags=retry_entry.get("flags") or "No flags",
                                    grounding_validated=False,
                                )
                                if fr and validate_grounding(exp, fr, career_history=career_hist):
                                    exp.grounding_validated = True
                                else:
                                    logger.warning("Grounding still failed for %s; using fallback", cand_id)
                                    exp = build_fallback_explanation(exp, fr) if fr else _empty_fallback(cand_id)
                            else:
                                exp = build_fallback_explanation(exp, fr) if fr else _empty_fallback(cand_id)
                        else:
                            exp = build_fallback_explanation(exp, fr) if fr else _empty_fallback(cand_id)

                    explanations.append(exp)

        # Fill fallback for any batch candidate that was not covered
        for sb in batch:
            if sb.candidate_id not in covered_ids:
                fr = feature_store.get(sb.candidate_id)
                exp = CandidateExplanation(candidate_id=sb.candidate_id)
                if fr:
                    exp = build_fallback_explanation(exp, fr)
                else:
                    exp = _empty_fallback(sb.candidate_id)
                explanations.append(exp)

    logger.info("Generated %d explanations", len(explanations))
    return explanations


def _empty_fallback(candidate_id: str) -> CandidateExplanation:
    """Minimal non-null explanation when no feature data is available."""
    return CandidateExplanation(
        candidate_id=candidate_id,
        match_summary=f"Candidate {candidate_id} ranked based on composite score {0:.3f}.",
        skill_alignment="No skill data available for detailed alignment.",
        seniority_assessment="Insufficient data for seniority assessment.",
        trajectory_signal="Insufficient data for trajectory assessment.",
        platform_summary="No platform data available.",
        flags="Insufficient data for flag detection.",
        grounding_validated=False,
    )


def _call_batch_llm(
    combined_prompt: str,
    client: Any,
    settings: Settings,
    stronger: bool = False,
) -> list[dict] | None:
    """Makes a single Groq call for explanation(s). Returns parsed list or None."""
    system_content = BATCH_SYSTEM_PROMPT
    if stronger:
        system_content = BATCH_SYSTEM_PROMPT + "\n\n" + STRONGER_INSTRUCTION

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=settings.llm.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": combined_prompt},
            ],
            max_tokens=settings.llm.max_tokens,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - start
        raw = response.choices[0].message.content
        usage = response.usage
        logger.info(
            "Explanation LLM call complete in %.2fs (input_tokens=%d, output_tokens=%d)",
            elapsed, usage.prompt_tokens, usage.completion_tokens,
        )

        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "candidates" in parsed:
            return parsed["candidates"]
        if isinstance(parsed, dict) and "candidate_id" in parsed:
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception as e:
        logger.error("Explanation LLM call failed: %s", str(e))
        return None


def _find_sb(batch: list[ScoreBreakdown], candidate_id: str) -> ScoreBreakdown:
    for sb in batch:
        if sb.candidate_id == candidate_id:
            return sb
    return ScoreBreakdown(candidate_id=candidate_id)
