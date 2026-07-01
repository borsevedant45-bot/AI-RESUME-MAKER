import csv
import json
import logging
from pathlib import Path
from dataclasses import asdict

from src.models import RankedResult

logger = logging.getLogger(__name__)


def write_ranked_output(
    ranked_results: list[RankedResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Writes the final deliverable in both CSV and JSON formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ranked_output.csv"
    json_path = output_dir / "ranked_output.json"

    csv_columns = [
        "rank", "candidate_id", "composite_score", "semantic_score",
        "trajectory_score", "stability_score", "platform_score", "cert_bonus",
        "match_summary", "skill_alignment", "seniority_assessment",
        "trajectory_signal", "platform_summary", "flags", "grounding_validated",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for result in ranked_results:
            row = {
                "rank": result.rank,
                "candidate_id": result.candidate_id,
                "composite_score": result.composite_score,
                "semantic_score": result.semantic_score,
                "trajectory_score": result.trajectory_score,
                "stability_score": result.stability_score,
                "platform_score": result.platform_score,
                "cert_bonus": result.cert_bonus,
                "match_summary": result.explanation.match_summary,
                "skill_alignment": result.explanation.skill_alignment,
                "seniority_assessment": result.explanation.seniority_assessment,
                "trajectory_signal": result.explanation.trajectory_signal,
                "platform_summary": result.explanation.platform_summary,
                "flags": result.explanation.flags,
                "grounding_validated": result.explanation.grounding_validated,
            }
            writer.writerow(row)

    json_serializable = []
    for result in ranked_results:
        d = asdict(result)
        json_serializable.append(d)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_serializable, f, indent=2)

    logger.info("Output written to %s and %s", csv_path, json_path)
    return csv_path, json_path
