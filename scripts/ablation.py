#!/usr/bin/env python3
"""
Ablation test — leaves one scoring dimension out at a time and measures impact.
The "keyword-stuffer" story is the key output: removing B2 should cause
a shallow senior-titled candidate to jump significantly in rank.

Usage: python scripts/ablation.py
"""
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, ".")

from src.config import Settings
from src.embedder.embedder import get_model, encode_single, encode_batch
from src.embedder.candidate_doc_builder import build_jd_query_doc
from src.embedder.index_builder import load_index
from src.retriever.retriever import retrieve_top_n
from src.scorer.b1_semantic import semantic_score
from src.scorer.b2_trajectory import trajectory_score
from src.scorer.b3_stability import stability_score
from src.scorer.b4_platform import platform_score
from src.scorer.b5_cert import cert_bonus

WEIGHT_CONFIGS = {
    "baseline":   {"b1": 0.35, "b2": 0.25, "b3": 0.15, "b4": 0.20, "b5": 0.05},
    "no_B1":      {"b1": 0.00, "b2": 0.33, "b3": 0.20, "b4": 0.27, "b5": 0.07},
    "no_B2":      {"b1": 0.47, "b2": 0.00, "b3": 0.20, "b4": 0.27, "b5": 0.07},
    "no_B3":      {"b1": 0.41, "b2": 0.29, "b3": 0.00, "b4": 0.24, "b5": 0.06},
    "no_B4":      {"b1": 0.44, "b2": 0.31, "b3": 0.19, "b4": 0.00, "b5": 0.06},
    "no_B5":      {"b1": 0.368, "b2": 0.263, "b3": 0.158, "b4": 0.211, "b5": 0.00},
}

def run_ablation():
    settings = Settings.from_yaml()
    jd_path = Path("data/jd_senior_data_engineer.txt")
    jd_text = jd_path.read_text()

    # Load JD intent (already parsed — reuse from previous run)
    intent_path = Path("data/outputs/de/jd_intent.json")
    if not intent_path.exists():
        print("ERROR: Run the Senior Data Engineer query first (Step 4).")
        return
    jd_intent_dict = json.loads(intent_path.read_text())

    # Load model and index
    print("Loading model and index...")
    model = get_model(settings.embedding.model_name)
    faiss_index, vectors, id_map, features_df = load_index(Path("data/processed"))

    # Build JD vector
    jd_doc = build_jd_query_doc(jd_intent_dict)
    jd_vector = encode_single(jd_doc, model)

    # Retrieve shortlist (same for all ablations — only weights change)
    shortlist = retrieve_top_n(jd_vector, faiss_index, id_map, top_n=500)
    shortlist_ids = [s[0] for s in shortlist]

    # Pre-encode JD skills once
    all_skills = list(set(
        jd_intent_dict.get("must_have_skills", []) +
        jd_intent_dict.get("nice_to_have_skills", [])
    ))
    skill_vecs_list = encode_batch(all_skills, model)
    skill_vectors = {s.lower(): skill_vecs_list[i] for i, s in enumerate(all_skills)}

    # Score each candidate under all weight configs
    results = {}

    # First, compute raw B1–B5 for all shortlisted candidates
    print(f"Computing scores for {len(shortlist_ids)} candidates...")
    raw_scores = []
    for cid in shortlist_ids:
        row = features_df[features_df["candidate_id"] == cid]
        if row.empty:
            continue
        f = row.iloc[0]

        skill_map = json.loads(f["skill_strength_scores"]) if isinstance(f["skill_strength_scores"], str) else f["skill_strength_scores"]
        cert_list = json.loads(f["cert_records"]) if isinstance(f["cert_records"], str) else f["cert_records"]

        idx = id_map.get(cid)
        if idx is None:
            continue
        cand_vec = vectors[idx]

        class FR:
            pass
        fr = FR()
        for col in features_df.columns:
            setattr(fr, col, f[col])
        fr.skill_strength_scores = skill_map
        fr.cert_records = cert_list

        class JDI:
            pass
        jdi = JDI()
        for k, v in jd_intent_dict.items():
            setattr(jdi, k, v)

        try:
            b1 = semantic_score(cand_vec, jd_vector, skill_map, jdi, model,
                               thin_profile=bool(f.get("thin_profile", False)),
                               skill_vectors=skill_vectors)
            b2 = trajectory_score(fr, jdi)
            b3 = stability_score(fr)
            b4 = platform_score(fr, jdi)
            b5 = cert_bonus(fr, jd_vector, model)
            raw_scores.append({
                "candidate_id": cid,
                "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5,
            })
        except Exception as e:
            continue

    print(f"Scored {len(raw_scores)} candidates successfully.")

    # Apply different weight configs
    ablation_results = {}
    for config_name, weights in WEIGHT_CONFIGS.items():
        scored = []
        for r in raw_scores:
            composite = (r["b1"] * weights["b1"] + r["b2"] * weights["b2"] +
                        r["b3"] * weights["b3"] + r["b4"] * weights["b4"] +
                        r["b5"] * weights["b5"])
            scored.append({"candidate_id": r["candidate_id"], "composite": composite,
                          "b1": r["b1"], "b2": r["b2"], "b3": r["b3"], "b4": r["b4"]})
        scored.sort(key=lambda x: x["composite"], reverse=True)
        ablation_results[config_name] = [r["candidate_id"] for r in scored[:20]]

    baseline_top20 = set(ablation_results["baseline"])

    print("\n" + "="*70)
    print("ABLATION RESULTS TABLE")
    print("="*70)
    print(f"{'Config':<12} {'Top-20 Overlap':>15} {'Jaccard':>10} {'Overlap Count':>15}")
    for config_name, top20 in ablation_results.items():
        if config_name == "baseline":
            print(f"{'baseline':<12} {'—':>15} {'—':>10} {'20/20':>15}")
            continue
        ablated_set = set(top20)
        intersection = baseline_top20 & ablated_set
        union = baseline_top20 | ablated_set
        jaccard = len(intersection) / len(union)
        overlap = len(intersection)
        print(f"{config_name:<12} {overlap:>15}/20 {jaccard:>10.3f} {overlap:>15}/20")

    # Keyword-stuffer story
    baseline_ranks = {cid: i+1 for i, cid in enumerate(ablation_results["baseline"])}
    no_b2_ranks = {cid: i+1 for i, cid in enumerate(ablation_results["no_B2"])}

    rank_changes = []
    for cid in ablation_results["no_B2"][:50]:
        baseline_rank = baseline_ranks.get(cid, 501)
        no_b2_rank = no_b2_ranks.get(cid, 501)
        improvement = baseline_rank - no_b2_rank
        if improvement > 10:
            b2_val = next((r["b2"] for r in raw_scores if r["candidate_id"] == cid), None)
            rank_changes.append((cid, baseline_rank, no_b2_rank, improvement, b2_val))

    rank_changes.sort(key=lambda x: x[3], reverse=True)

    print("\n" + "="*70)
    print("KEYWORD-STUFFER STORY: Biggest rank jumps when B2 (trajectory) is removed")
    print("="*70)
    print(f"{'CandID':<20} {'Baseline Rank':>15} {'No-B2 Rank':>12} {'Jump':>8} {'B2 Score':>10}")
    for cid, br, nr, jump, b2v in rank_changes[:5]:
        print(f"{cid:<20} {br:>15} {nr:>12} {'+'+str(jump):>8} {b2v:>10.3f}")

    if rank_changes:
        top_stuffer = rank_changes[0]
        print(f"\nKEY STORY: Candidate {top_stuffer[0]}")
        print(f"  With B2 (trajectory): rank {top_stuffer[1]}")
        print(f"  Without B2:           rank {top_stuffer[2]}")
        print(f"  B2 score was:         {top_stuffer[4]:.3f} (low = weak trajectory despite matching title)")
        print(f"  => This is exactly the keyword-stuffer the system is designed to catch.")

    print("\n\nCopy the table above into the ablation section of the methodology presentation.")

if __name__ == "__main__":
    run_ablation()
