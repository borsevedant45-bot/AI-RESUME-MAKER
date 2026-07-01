import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from src.config import Settings
from src.jd_parser.parser import parse_job_description
from src.embedder.embedder import get_model, encode_single, encode_batch
from src.embedder.candidate_doc_builder import build_jd_query_doc
from src.embedder.index_builder import load_index
from src.exceptions import IndexNotFoundError
from src.retriever.retriever import retrieve_top_n
from src.retriever.hard_filter import apply_hard_filters, apply_domain_title_filter
from src.scorer.composite import score_candidate
from src.ranker.ranker import rank_candidates
from src.explainer.explainer import generate_explanations
from src.output_writer.writer import write_ranked_output
from src.models import CandidateFeatureRow, JDIntent, RankedResult, CandidateExplanation

logger = logging.getLogger(__name__)


def run_query(
    jd_text: str,
    processed_dir: Path,
    output_dir: Path,
    settings: Settings,
    llm_client: Any,
) -> list[RankedResult]:
    """Online query pipeline. Runs once per JD. Target <60s."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        raise IndexNotFoundError(f"Processed directory not found: {processed_dir}")

    artifacts = ["candidate_index.faiss", "candidate_vectors.npy",
                  "candidate_id_map.json", "candidate_features.parquet"]
    for art in artifacts:
        if not (processed_dir / art).exists():
            raise IndexNotFoundError(f"Index artifact missing: {processed_dir / art}")

    # 1. Parse JD
    t0 = time.time()
    jd_intent = parse_job_description(jd_text, llm_client, settings)
    with open(output_dir / "jd_intent.json", "w") as f:
        json.dump({
            "seniority_level": jd_intent.seniority_level,
            "seniority_evidence": jd_intent.seniority_evidence,
            "must_have_skills": jd_intent.must_have_skills,
            "nice_to_have_skills": jd_intent.nice_to_have_skills,
            "core_problems_to_solve": jd_intent.core_problems_to_solve,
            "implicit_soft_skills": jd_intent.implicit_soft_skills,
            "domain_tags": jd_intent.domain_tags,
            "requires_technical_github_signals": jd_intent.requires_technical_github_signals,
            "work_context": {
                "work_mode": jd_intent.work_context.work_mode,
                "location_required": jd_intent.work_context.location_required,
                "location_is_hard_requirement": jd_intent.work_context.location_is_hard_requirement,
                "salary_min_lpa": jd_intent.work_context.salary_min_lpa,
                "salary_max_lpa": jd_intent.work_context.salary_max_lpa,
            },
            "salary_stated": jd_intent.salary_stated,
        }, f, indent=2)
    logger.info("JD parsed in %.1fs", time.time() - t0)

    # 2. Encode JD intent as query vector
    t0 = time.time()
    model = get_model(settings.embedding.model_name)

    jd_doc = build_jd_query_doc(
        jd_intent.__dict__ if not isinstance(jd_intent, dict) else jd_intent
    )
    jd_vector = encode_single(jd_doc, model)
    logger.info("JD vector encoded in %.1fs (doc: %d chars)", time.time() - t0, len(jd_doc))

    # 3. Load index
    t0 = time.time()
    index, vectors, id_map, feature_df = load_index(processed_dir)
    logger.info("Index loaded in %.1fs", time.time() - t0)

    # 4. ANN retrieval
    t0 = time.time()
    shortlist = retrieve_top_n(jd_vector, index, id_map, settings.retrieval.top_n_shortlist)
    shortlist_ids = [c[0] for c in shortlist]
    logger.info("Retrieved %d candidates in %.1fs", len(shortlist_ids), time.time() - t0)

    # 5. Build feature store
    t0 = time.time()
    feature_store: dict[str, CandidateFeatureRow] = {}
    for _, row in feature_df.iterrows():
        fr = CandidateFeatureRow(
            candidate_id=row["candidate_id"],
            current_title=row.get("current_title", ""),
            latest_seniority=row.get("latest_seniority", 0.5),
            promotion_rate=row.get("promotion_rate", 0.0),
            experience_years=row.get("experience_years", 0.0),
            avg_tenure_months=row.get("avg_tenure_months", 0.0),
            job_hopping_flag=int(row.get("job_hopping_flag", 0)),
            institution_tier=row.get("institution_tier", "tier_3"),
            active_intent_score=row.get("active_intent_score", 0.0),
            hire_reliability_score=row.get("hire_reliability_score", 0.0),
            github_activity_score=row.get("github_activity_score", -1.0),
            endorsements_received=int(row.get("endorsements_received", 0)),
            open_to_work=bool(row.get("open_to_work", False)),
            willing_to_relocate=bool(row.get("willing_to_relocate", False)),
            work_mode_preference=row.get("work_mode_preference", "hybrid"),
            notice_period_days=int(row.get("notice_period_days", 0)),
            expected_salary_min=int(row.get("expected_salary_min", 0)),
            expected_salary_max=int(row.get("expected_salary_max", 0)),
            location=row.get("location", ""),
            skill_strength_scores=json.loads(row.get("skill_strength_scores", "{}")),
            cert_records=json.loads(row.get("cert_records", "[]")),
            thin_profile=bool(row.get("thin_profile", False)),
        )
        feature_store[row["candidate_id"]] = fr
    logger.info("Feature store loaded in %.1fs", time.time() - t0)

    id_to_idx = id_map

    # 6. Hard filters
    t0 = time.time()
    shortlist_ids = apply_hard_filters(shortlist_ids, feature_store, jd_intent)
    logger.info("Hard filters applied in %.1fs (%d remaining)", time.time() - t0, len(shortlist_ids))

    # 6b. Domain title filter (post-retrieval, post-hard-filters)
    t0 = time.time()
    shortlist_ids = apply_domain_title_filter(shortlist_ids, feature_store, jd_intent)
    logger.info("Domain title filter applied in %.1fs (%d remaining)", time.time() - t0, len(shortlist_ids))

    # 7. Pre-encode JD skill vectors once (avoids per-candidate re-encoding)
    t0 = time.time()
    all_skills = list(set(jd_intent.must_have_skills + jd_intent.nice_to_have_skills))
    if all_skills:
        skill_vecs_list = encode_batch(all_skills, model)
        skill_vectors = {skill.lower(): skill_vecs_list[i] for i, skill in enumerate(all_skills)}
    else:
        skill_vectors = {}
    logger.info("Pre-encoded %d skill vectors in %.1fs", len(all_skills), time.time() - t0)

    # 8. Score all candidates
    t0 = time.time()
    scored: list[ScoreBreakdown] = []
    for cand_id in shortlist_ids:
        fr = feature_store.get(cand_id)
        if fr is None:
            continue
        vec_idx = id_to_idx.get(cand_id)
        if vec_idx is None:
            continue
        cand_vec = vectors[int(vec_idx)]

        sb = score_candidate(
            candidate_id=cand_id,
            feature_row=fr,
            candidate_vector=cand_vec,
            jd_vector=jd_vector,
            jd_intent=jd_intent,
            model=model,
            settings=settings,
            skill_vectors=skill_vectors,
        )
        scored.append(sb)
    logger.info("Scored %d candidates in %.1fs", len(scored), time.time() - t0)

    # 9. Rank
    t0 = time.time()
    top_20 = rank_candidates(scored, settings.retrieval.top_n_output, settings.ranking.tiebreaker_composite_tolerance)
    logger.info("Ranking complete in %.1fs", time.time() - t0)

    # 10. Generate explanations (batched LLM calls)
    t0 = time.time()
    explanations = generate_explanations(
        top_20, feature_store, {}, jd_intent, llm_client, settings,
    )
    logger.info("Explanations generated in %.1fs", time.time() - t0)

    # 11. Assemble results
    ranked_results = []
    for i, sb in enumerate(top_20):
        exp = explanations[i] if i < len(explanations) else CandidateExplanation(candidate_id=sb.candidate_id)
        ranked_results.append(RankedResult(
            rank=i + 1,
            candidate_id=sb.candidate_id,
            composite_score=sb.composite_score,
            semantic_score=sb.semantic_score,
            trajectory_score=sb.trajectory_score,
            stability_score=sb.stability_score,
            platform_score=sb.platform_score,
            cert_bonus=sb.cert_bonus,
            explanation=exp,
        ))

    # 12. Write output
    csv_path, json_path = write_ranked_output(ranked_results, output_dir)

    total_time = time.time() - start_time
    logger.info("Total query pipeline time: %.1fs", total_time)
    return ranked_results
