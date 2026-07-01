import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import Settings
from src.exceptions import JDParseError
from src.models import JDIntent, WorkContext
from src.jd_parser.prompt_templates import (
    JD_PARSE_SYSTEM_PROMPT,
    JD_PARSE_USER_PROMPT,
    CORRECTIVE_SYSTEM_PROMPT,
    CORRECTIVE_USER_PROMPT,
)

logger = logging.getLogger(__name__)


def parse_job_description(
    jd_text: str,
    client: Any,
    settings: Settings,
    output_dir: Path | None = None,
) -> JDIntent:
    """
    Calls Groq llama-3.3-70b-versatile with JSON mode to extract structured
    JD intent. Validates, retries once on failure, writes jd_intent.json
    to output_dir if provided.
    """
    messages = [
        {"role": "system", "content": JD_PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": JD_PARSE_USER_PROMPT.format(jd_text=jd_text)},
    ]

    raw_response = _call_llm(messages, client, settings)
    is_valid, error = _validate_jd_intent(raw_response)

    if not is_valid:
        logger.warning("JD parse failed validation (%s). Retrying with corrective prompt.", error)
        corrective_messages = [
            {"role": "system", "content": CORRECTIVE_SYSTEM_PROMPT},
            {"role": "user", "content": CORRECTIVE_USER_PROMPT.format(
                error=error,
                invalid_output=json.dumps(raw_response, indent=2),
                jd_text=jd_text,
            )},
        ]
        raw_response = _call_llm(corrective_messages, client, settings)
        is_valid, error = _validate_jd_intent(raw_response)
        if not is_valid:
            raise JDParseError(f"JD parse failed after retry: {error}")

    intent = _build_jd_intent(raw_response)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "jd_intent.json", "w") as f:
            json.dump(raw_response, f, indent=2)
        logger.info("jd_intent.json written to %s", output_dir / "jd_intent.json")

    return intent


def _call_llm(messages: list, client: Any, settings: Settings) -> dict:
    start = time.time()
    response = client.chat.completions.create(
        model=settings.llm.model,
        messages=messages,
        max_tokens=settings.llm.max_tokens,
        response_format={"type": "json_object"},
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    usage = response.usage
    logger.info(
        "LLM call complete in %.2fs (input_tokens=%d, output_tokens=%d, total_tokens=%d)",
        elapsed,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens if hasattr(usage, "total_tokens") else usage.prompt_tokens + usage.completion_tokens,
    )
    return json.loads(content)


def _validate_jd_intent(raw: dict) -> tuple[bool, str]:
    required = ["seniority_level", "must_have_skills", "core_problems_to_solve", "domain_tags"]
    for field in required:
        if field not in raw:
            return False, f"missing required field: {field}"

    sl = raw.get("seniority_level")
    allowed = {0.2, 0.5, 0.75, 1.0}
    if sl not in allowed:
        nearest = min(allowed, key=lambda x: abs(x - sl))
        logger.info("Rounding seniority_level from %s to %s", sl, nearest)
        raw["seniority_level"] = nearest

    if not raw.get("must_have_skills"):
        return False, "must_have_skills is empty"

    return True, ""


def _build_jd_intent(raw: dict) -> JDIntent:
    wc = raw.get("work_context", {})
    return JDIntent(
        seniority_level=raw.get("seniority_level", 0.5),
        seniority_evidence=raw.get("seniority_evidence", ""),
        must_have_skills=raw.get("must_have_skills", []),
        nice_to_have_skills=raw.get("nice_to_have_skills", []),
        core_problems_to_solve=raw.get("core_problems_to_solve", ""),
        implicit_soft_skills=raw.get("implicit_soft_skills", []),
        domain_tags=raw.get("domain_tags", []),
        requires_technical_github_signals=raw.get("requires_technical_github_signals", False),
        work_context=WorkContext(
            work_mode=wc.get("work_mode"),
            location_required=wc.get("location_required"),
            location_is_hard_requirement=wc.get("location_is_hard_requirement", False),
            salary_min_lpa=wc.get("salary_min_lpa"),
            salary_max_lpa=wc.get("salary_max_lpa"),
        ),
        salary_stated=raw.get("salary_stated", False),
    )
