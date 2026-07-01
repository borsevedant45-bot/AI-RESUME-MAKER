#!/usr/bin/env python3
"""
Explainability quality check — validates grounding and prints structured report.
Usage: python scripts/check_explanations.py --output data/outputs/de/
"""
import argparse
import json
import pandas as pd
from pathlib import Path

def check_explanations(output_dir):
    csv_path = Path(output_dir) / "ranked_output.csv"
    json_path = Path(output_dir) / "ranked_output.json"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run the query pipeline first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n{'='*70}")
    print(f"EXPLAINABILITY CHECK — {output_dir}")
    print(f"{'='*70}")
    print(f"Total candidates: {len(df)}")

    if "grounding_validated" in df.columns:
        passed = df["grounding_validated"].sum()
        print(f"Grounding passed: {passed}/{len(df)}")
        if passed < len(df):
            failed = df[df["grounding_validated"] == False]
            print(f"FAILED grounding: {failed['candidate_id'].tolist()}")

    explanation_cols = ["match_summary", "skill_alignment", "seniority_assessment",
                       "trajectory_signal", "platform_summary", "flags"]

    print(f"\nExplanation completeness:")
    for col in explanation_cols:
        if col in df.columns:
            empty = df[col].isna().sum() + (df[col] == "").sum()
            print(f"  {col:25s}: {len(df)-empty}/{len(df)} populated")

    print(f"\n{'-'*70}")
    print("TOP 3 EXPLANATIONS (read these manually for accuracy):")
    print(f"{'-'*70}")
    for _, row in df.head(3).iterrows():
        print(f"\nRank {row.get('rank', '?')} | ID: {row.get('candidate_id', '?')} | Composite: {row.get('composite_score', '?'):.3f}")
        for col in explanation_cols:
            if col in df.columns and pd.notna(row[col]) and row[col]:
                print(f"\n[{col.upper().replace('_',' ')}]")
                print(f"  {str(row[col])[:400]}")

    generic_phrases = [
        "relevant experience", "good fit", "strong candidate",
        "meets the requirements", "suitable for", "applicable skills"
    ]
    print(f"\n{'-'*70}")
    print("SPECIFICITY CHECK — scanning for generic phrases:")
    found_any = False
    for col in explanation_cols:
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            text = str(row.get(col, "")).lower()
            for phrase in generic_phrases:
                if phrase in text:
                    print(f"  WARNING: '{phrase}' found in {col} for rank {row.get('rank', '?')}")
                    found_any = True
    if not found_any:
        print("  (No warnings above = good specificity)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/outputs/de/",
                       help="Output directory to check")
    args = parser.parse_args()
    check_explanations(args.output)

if __name__ == "__main__":
    main()
