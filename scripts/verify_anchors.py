#!/usr/bin/env python3
"""
Verifies that all anchor candidates exist in the dataset and output files.
Produces scripts/anchors_verified.json with confirmed anchors and their ranks.
Usage: python scripts/verify_anchors.py
"""
import json
import pandas as pd
from pathlib import Path

OUTPUTS = {
    "de":         ("data/outputs/de/ranked_output.csv",         "Senior Data Engineer"),
    "hr":         ("data/outputs/hr/ranked_output.csv",         "HR Manager"),
    "analyst":    ("data/outputs/analyst/ranked_output.csv",    "Mid Data Analyst"),
    "pm":         ("data/outputs/pm/ranked_output.csv",         "Project Manager (Implicit)"),
    "accountant": ("data/outputs/accountant/ranked_output.csv", "Senior Accountant"),
}

def verify():
    anchors_path = Path("scripts/anchors.json")
    if not anchors_path.exists():
        print("ERROR: scripts/anchors.json not found")
        return

    anchors = json.loads(anchors_path.read_text())
    verified = {}

    print("=" * 70)
    print("ANCHOR VERIFICATION")
    print("=" * 70)

    for key, (csv_path, label) in OUTPUTS.items():
        print(f"\n--- {label} ({key}) ---")
        df = pd.read_csv(csv_path) if Path(csv_path).exists() else None

        anchor_data = anchors.get(key, {})
        fit = anchor_data.get("fit_anchors", [])
        nonfit = anchor_data.get("nonfit_anchors", [])
        verified_anchor_data = {"label": label, "fit_anchors": [], "nonfit_anchors": []}

        if df is not None:
            all_ids = df["candidate_id"].tolist()
        else:
            all_ids = []
            print(f"  [SKIP] No output file")

        for a in fit:
            rank = all_ids.index(a) + 1 if a in all_ids else "NOT FOUND"
            print(f"  Fit anchor {a}: rank {rank}")
            verified_anchor_data["fit_anchors"].append({"id": a, "rank": rank})

        for a in nonfit:
            rank = all_ids.index(a) + 1 if a in all_ids else "NOT IN TOP-500"
            print(f"  Non-fit anchor {a}: rank {rank}")
            verified_anchor_data["nonfit_anchors"].append({"id": a, "rank": rank})

        verified[key] = verified_anchor_data

    out_path = Path("scripts/anchors_verified.json")
    out_path.write_text(json.dumps(verified, indent=2))
    print(f"\nWritten to {out_path}")

if __name__ == "__main__":
    verify()
