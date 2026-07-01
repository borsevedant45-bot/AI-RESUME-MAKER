import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.config import Settings
from src.data_loader.loader import load_candidates
from src.embedder.candidate_doc_builder import build_candidate_doc
from src.embedder.embedder import get_model, encode_batch
from src.exceptions import IndexBuildError
from src.feature_extractor.skill_features import build_skill_strength_map
from src.feature_extractor.trajectory_features import compute_trajectory_base
from src.feature_extractor.stability_features import (
    compute_avg_tenure, detect_job_hopping, best_institution_tier,
)
from src.feature_extractor.platform_features import (
    compute_active_intent_score, compute_hire_reliability_score,
)

logger = logging.getLogger(__name__)


def build_candidate_index(
    jsonl_path: Path,
    output_dir: Path,
    settings: Settings,
) -> None:
    """
    Full offline indexing pipeline.

    1. Load all candidates from JSONL
    2. Build candidate docs + compute structured features
    3. Batch encode into (N, 384) normalized vectors
    4. Build FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity)
    5. Persist: candidate_vectors.npy, candidate_index.faiss,
       candidate_features.parquet, candidate_id_map.json

    Raises IndexBuildError if <90% of candidates are indexed successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # 1. Load all candidates
    logger.info("Loading candidates from %s", jsonl_path)
    all_profiles = list(load_candidates(jsonl_path))
    total_loaded = len(all_profiles)
    logger.info("Loaded %d profiles", total_loaded)

    if total_loaded == 0:
        raise IndexBuildError("No valid candidate profiles loaded")

    # 2. Build documents + extract features
    logger.info("Building candidate documents and extracting features...")
    docs = []
    feature_rows = []
    candidate_ids = []
    errors = 0

    for idx, profile in enumerate(all_profiles):
        if (idx + 1) % 10000 == 0:
            logger.info("Progress: processed %d / %d candidates", idx + 1, total_loaded)

        try:
            doc, thin = build_candidate_doc(profile)
            docs.append(doc)
            candidate_ids.append(profile.candidate_id)

            skill_map = build_skill_strength_map(profile.skills)
            traj = compute_trajectory_base(profile)
            avg_tenure = compute_avg_tenure(profile.career_history)
            hopping = detect_job_hopping(profile.career_history)
            tier = best_institution_tier(profile.education)
            intent = compute_active_intent_score(profile.redrob_signals)
            reliability = compute_hire_reliability_score(profile.redrob_signals)

            cert_list = [
                {"name": c.name, "issue_year": c.issue_year}
                for c in profile.certifications
            ]

            feature_rows.append({
                "candidate_id": profile.candidate_id,
                "current_title": profile.current_title,
                "latest_seniority": traj["latest_seniority"],
                "promotion_rate": traj["promotion_rate"],
                "experience_years": profile.experience_years,
                "avg_tenure_months": avg_tenure,
                "job_hopping_flag": hopping,
                "institution_tier": tier,
                "active_intent_score": intent,
                "hire_reliability_score": reliability,
                "github_activity_score": profile.redrob_signals.github_activity_score,
                "endorsements_received": profile.redrob_signals.endorsements_received,
                "open_to_work": profile.redrob_signals.open_to_work,
                "willing_to_relocate": profile.redrob_signals.willing_to_relocate,
                "work_mode_preference": profile.redrob_signals.work_mode_preference,
                "notice_period_days": profile.redrob_signals.notice_period_days,
                "expected_salary_min": profile.redrob_signals.expected_salary_min,
                "expected_salary_max": profile.redrob_signals.expected_salary_max,
                "location": profile.location,
                "skill_strength_scores": json.dumps(skill_map),
                "cert_records": json.dumps(cert_list),
                "thin_profile": thin,
            })
        except Exception as e:
            logger.warning("Failed to index candidate %s: %s", profile.candidate_id, str(e))
            errors += 1

    total_attempted = total_loaded
    total_indexed = len(docs)
    success_rate = total_indexed / total_attempted if total_attempted > 0 else 0.0

    logger.info(
        "Indexed %d / %d candidates (success rate: %.1f%%)",
        total_indexed, total_attempted, success_rate * 100,
    )

    if success_rate < 0.90:
        raise IndexBuildError(
            f"Indexed {total_indexed}/{total_attempted} candidates "
            f"({success_rate:.1%}) — must be >= 90%"
        )

    # 3. Batch encode
    logger.info("Encoding %d documents (batch_size=%d)...", len(docs), settings.embedding.batch_size)
    model = get_model(settings.embedding.model_name)
    vectors = encode_batch(docs, model, settings.embedding.batch_size)
    logger.info("Encoding complete: %d vectors (%d-dim)", vectors.shape[0], vectors.shape[1])

    # 4. Build FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity)
    logger.info("Building FAISS IndexFlatIP...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info("Index built: %d vectors, exact IP (cosine sim on normalized vecs)", index.ntotal)

    # 5. Persist artifacts
    np.save(str(output_dir / "candidate_vectors.npy"), vectors)
    faiss.write_index(index, str(output_dir / "candidate_index.faiss"))

    df = pd.DataFrame(feature_rows)
    df.to_parquet(str(output_dir / "candidate_features.parquet"), index=False)

    id_map = {cid: i for i, cid in enumerate(candidate_ids)}
    with open(output_dir / "candidate_id_map.json", "w") as f:
        json.dump(id_map, f)

    elapsed = time.time() - start_time
    logger.info("Indexing complete. Wall-clock time: %.1fs", elapsed)


def load_index(
    processed_dir: Path,
) -> tuple[faiss.Index, np.ndarray, dict[str, int], pd.DataFrame]:
    """Loads all four pre-built index artifacts."""
    index = faiss.read_index(str(processed_dir / "candidate_index.faiss"))
    vectors = np.load(str(processed_dir / "candidate_vectors.npy"))
    with open(processed_dir / "candidate_id_map.json", "r") as f:
        id_map = json.load(f)
    feature_df = pd.read_parquet(str(processed_dir / "candidate_features.parquet"))
    return index, vectors, id_map, feature_df
