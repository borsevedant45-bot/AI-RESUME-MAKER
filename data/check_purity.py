import pandas as pd
import json
from pathlib import Path

feat = pd.read_parquet("data/processed/candidate_features.parquet")
outputs = {
    "de": "data/outputs/de/ranked_output.csv",
    "hr": "data/outputs/hr/ranked_output.csv",
    "analyst": "data/outputs/analyst/ranked_output.csv",
    "pm": "data/outputs/pm/ranked_output.csv",
    "accountant": "data/outputs/accountant/ranked_output.csv",
}

jd_names = {
    "de": "Senior Data Engineer",
    "hr": "HR Manager",
    "analyst": "Mid Data Analyst",
    "pm": "Project Manager",
    "accountant": "Senior Accountant",
}

for key, path in outputs.items():
    df = pd.read_csv(path)
    nulls = df.isnull().sum().sum()
    grounded = df["grounding_validated"].sum()
    title_map = feat.set_index("candidate_id")["current_title"].to_dict()
    top5_ids = [df.iloc[i]["candidate_id"] for i in range(min(5, len(df)))]
    top5_titles = [title_map.get(cid, "UNKNOWN") for cid in top5_ids]
    top5_scores = [df.iloc[i]["composite_score"] for i in range(min(5, len(df)))]
    print(f"=== {jd_names[key]} ({key}) ===")
    print(f"  Nulls: {nulls} | Grounded: {grounded}/20")
    print(f"  Top-5 titles:")
    for i, (cid, title, score) in enumerate(zip(top5_ids, top5_titles, top5_scores)):
        print(f"    #{i+1}: {cid} ({score:.3f}) — {title}")
    print()
