import logging
from src.models import SkillRecord

logger = logging.getLogger(__name__)

PROFICIENCY_WEIGHTS = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}


def compute_skill_strength(skill: SkillRecord) -> float:
    """
    Computes the weighted skill strength score for one skill entry.
    Formula: proficiency_weight*0.5 + min(duration/5,1)*0.35 + min(endorsements/50,1)*0.15
    """
    prof_weight = PROFICIENCY_WEIGHTS.get(skill.proficiency, 0.25)
    duration_norm = min(skill.duration_years / 5.0, 1.0)
    endorse_norm = min(skill.endorsements / 50, 1.0)

    score = (
        prof_weight * 0.50 +
        duration_norm * 0.35 +
        endorse_norm * 0.15
    )
    return round(min(score, 1.0), 4)


def build_skill_strength_map(skills: list[SkillRecord]) -> dict[str, float]:
    """Returns {skill_name.lower(): strength_score} for all skills in a profile."""
    return {s.name.lower(): compute_skill_strength(s) for s in skills}
