#!/usr/bin/env python3
"""
Evaluation script — computes all metrics for the methodology presentation.
Usage: python scripts/evaluate.py [--anchors scripts/anchors_verified.json]
"""
import json
import argparse  # Added for dynamic terminal arguments
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUTS = {
    "de":         ("data/outputs/de/ranked_output.csv",         "Senior Data Engineer"),
    "hr":         ("data/outputs/hr/ranked_output.csv",         "HR Manager"),
    "analyst":    ("data/outputs/analyst/ranked_output.csv",    "Mid Data Analyst"),
    "pm":         ("data/outputs/pm/ranked_output.csv",         "Project Manager (Implicit)"),
    "accountant": ("data/outputs/accountant/ranked_output.csv", "Senior Accountant"),
}

def load_anchors(anchors_path):
    path = Path(anchors_path)
    if not path.exists():
        print(f"WARNING: {anchors_path} not found. Skipping anchor checks.")
        return {}
    return json.loads(path.read_text())

def load_ranked(csv_path):
    if not Path(csv_path).exists():
        return None
    return pd.read_csv(csv_path)

def compute_ndcg(ranked_ids, relevance_map, k=20):
    def dcg(ids):
        return sum(
            relevance_map.get(cid, 0) / np.log2(i + 2)
            for i, cid in enumerate(ids[:k])
        )
    actual = dcg(ranked_ids)
    ideal_order = sorted(relevance_map, key=relevance_map.get, reverse=True)
    ideal = dcg(ideal_order)
    return actual / ideal if ideal > 0 else 0.0

def evaluate_all():
    # Setup argument parsing to dynamically accept path flags
    parser = argparse.ArgumentParser(description="Evaluate output metrics against anchors.")
    parser.add_argument("--anchors", type=str, default="scripts/anchors.json", help="Path to anchors JSON file")
    args = parser.parse_args()

    # Pass the argument path to the anchor loader
    anchors = load_anchors(args.anchors)
    print(f"Loaded evaluation anchors from: {args.anchors}")

    print("\n" + "="*80)
    print("REDROB EVALUATION RESULTS")
    print("="*80)

    summary_rows = []

    for key, (csv_path, label) in OUTPUTS.items():
        print(f"\n{'-'*60}")
        print(f"JD: {label}")
        print(f"{'-'*60}")

        df = load_ranked(csv_path)
        if df is None:
            print(f"   [SKIP] Output not found: {csv_path}")
            continue

        top20_ids = df["candidate_id"].tolist()[:20]
        all_ids = df["candidate_id"].tolist()

        # Title distribution
        print(f"\nTitle distribution in top 20:")
        if "current_title" in df.columns:
            print(df.head(20)["current_title"].value_counts().to_string())

        # Score ranges
        print(f"\nScore ranges (top 20):")
        for col in ["composite_score", "semantic_score", "trajectory_score", "platform_score"]:
            if col in df.columns:
                vals = df.head(20)[col]
                print(f"   {col:20s}: [{vals.min():.3f}, {vals.max():.3f}]  spread={vals.max()-vals.min():.3f}")

        # Anchor checks
        anchor_data = anchors.get(key, {})
        fit_anchors = anchor_data.get("fit_anchors", [])
        nonfit_anchors = anchor_data.get("nonfit_anchors", [])

        if fit_anchors:
            fit_in_top20 = [a for a in fit_anchors if a in top20_ids]
            fit_in_top100 = [a for a in fit_anchors if a in all_ids[:100]]
            print(f"\nPrecision@20  : {len(fit_in_top20)}/{len(fit_anchors)} fit anchors in top 20")
            print(f"Recall@100    : {len(fit_in_top100)}/{len(fit_anchors)} fit anchors in top 100")
            for a in fit_anchors:
                rank = top20_ids.index(a) + 1 if a in top20_ids else (
                    all_ids.index(a) + 1 if a in all_ids else "NOT FOUND"
                )
                print(f"   {a}: rank {rank}")

        if nonfit_anchors:
            nonfit_in_top100 = [a for a in nonfit_anchors if a in all_ids[:100]]
            print(f"\nContamination@100: {len(nonfit_in_top100)}/{len(nonfit_anchors)} non-fit anchors in top 100")
            if nonfit_in_top100:
                print(f"   WARNING — non-fit anchors appeared: {nonfit_in_top100}")
            else:
                print(f"   PASS — no non-fit anchors in top 100")

        # NDCG (if anchors available)
        if fit_anchors:
            relevance_map = {a: 3 for a in fit_anchors}
            relevance_map.update({a: 0 for a in nonfit_anchors})
            ndcg = compute_ndcg(top20_ids, relevance_map, k=20)
            print(f"\nNDCG@20 (anchor-graded): {ndcg:.3f}")

        # Hard filter check for accountant JD
        if key == "accountant":
            total_candidates = len(df)
            print(f"\nHard filter check: {total_candidates} candidates in output")
            if "expected_salary_min" in df.columns:
                violators = df[df["expected_salary_min"] > 18]
                print(f"   Candidates with salary_min > 18 LPA in output: {len(violators)} (should be 0)")

        # GitHub signal check for non-technical JDs
        if key in ["hr", "accountant"]:
            print(f"\nGitHub signal leak check (should NOT influence non-tech JD):")
            intent_path = Path(csv_path).parent / "jd_intent.json"
            if intent_path.exists():
                intent = json.loads(intent_path.read_text())
                github_flag = intent.get("requires_technical_github_signals", "NOT FOUND")
                print(f"   requires_technical_github_signals = {github_flag}")
                if github_flag == True:
                    print(f"   WARNING — GitHub signals are being applied to a non-technical JD!")
                else:
                    print(f"   PASS — GitHub signals correctly excluded")

        summary_rows.append({
            "JD": label,
            "Fit@20": f"{len(fit_in_top20)}/{len(fit_anchors)}" if fit_anchors else "N/A",
            "Contam@100": len(nonfit_in_top100) if nonfit_anchors else "N/A",
            "NDCG@20": f"{ndcg:.3f}" if fit_anchors else "N/A",
            "Sem Range": f"{df.head(20)['semantic_score'].max()-df.head(20)['semantic_score'].min():.3f}" if "semantic_score" in df.columns else "N/A",
        })

    # Summary table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE (for methodology presentation)")
    print("="*80)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n\nAll evaluation results above. Copy into methodology PDF.")

if __name__ == "__main__":
    evaluate_all()