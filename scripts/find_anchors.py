#!/usr/bin/env python3
"""
Anchor identification helper — finds candidate IDs for manual anchor selection.
Run BEFORE the pipeline to avoid unconscious grading bias.
Usage: python scripts/find_anchors.py --jd [de|hr|analyst|pm|accountant]
"""
import argparse
import json
import pandas as pd
from pathlib import Path

def load_parquet(path="data/processed/candidate_features.parquet"):
    df = pd.read_parquet(path)
    return df

def load_raw_profiles(path="data/raw/candidates.jsonl", limit=None):
    profiles = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            p = json.loads(line)
            profiles[p["candidate_id"]] = p
    return profiles

def print_profile_summary(cid, profiles, df):
    p = profiles.get(cid)
    if not p:
        print(f"  [Profile not found for {cid}]")
        return
    row = df[df["candidate_id"] == cid]
    print(f"\n{'='*60}")
    print(f"Candidate ID : {cid}")
    print(f"Title        : {p.get('profile', {}).get('current_title', 'N/A')}")
    print(f"Experience   : {p.get('profile', {}).get('experience_years', 'N/A')} years")
    print(f"Industry     : {p.get('profile', {}).get('industry', 'N/A')}")
    print(f"Open to Work : {p.get('redrob_signals', {}).get('open_to_work', 'N/A')}")
    print(f"Notice (days): {p.get('redrob_signals', {}).get('notice_period_days', 'N/A')}")
    print(f"Salary Min   : {p.get('redrob_signals', {}).get('expected_salary_min', 'N/A')} LPA")
    print(f"Salary Max   : {p.get('redrob_signals', {}).get('expected_salary_max', 'N/A')} LPA")
    if not row.empty:
        print(f"Latest Sen.  : {row['latest_seniority'].values[0]}")
        print(f"Promo Rate   : {row['promotion_rate'].values[0]:.2f}")
        print(f"Avg Tenure   : {row['avg_tenure_months'].values[0]:.1f} mo")
        print(f"Hopping Flag : {row['job_hopping_flag'].values[0]}")
    print(f"\nSkills (top 6):")
    for s in p.get("skills", [])[:6]:
        print(f"  - {s['name']} ({s['proficiency']}, {s.get('duration_years', '?')}yr, {s.get('endorsements', 0)} endorse)")
    print(f"\nCareer History:")
    for r in p.get("career_history", [])[:3]:
        print(f"  [{r.get('duration_months', '?')} mo] {r.get('title', '?')} @ {r.get('company', '?')}")
        desc = r.get("description", "")[:120]
        if desc:
            print(f"    {desc}...")
    certs = p.get("certifications", [])
    if certs:
        print(f"\nCertifications: {', '.join(c['name'] for c in certs)}")

def find_fit_candidates(profiles, df, skill_keywords, title_keywords, n=10):
    results = []
    for cid, p in profiles.items():
        skill_names = [s["name"].lower() for s in p.get("skills", [])]
        title = p.get("profile", {}).get("current_title", "").lower()
        career_titles = [r.get("title", "").lower() for r in p.get("career_history", [])]

        skill_match = sum(1 for kw in skill_keywords if any(kw.lower() in s for s in skill_names))
        title_match = any(kw.lower() in title or any(kw.lower() in ct for ct in career_titles)
                         for kw in title_keywords)

        if skill_match >= 2 and title_match:
            results.append((cid, skill_match))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:n]]

def find_nonfit_candidates(profiles, avoid_titles, avoid_skills, n=5):
    results = []
    for cid, p in profiles.items():
        title = p.get("profile", {}).get("current_title", "").lower()
        skill_names = [s["name"].lower() for s in p.get("skills", [])]

        title_mismatch = any(t.lower() in title for t in avoid_titles)
        skill_mismatch = not any(s.lower() in skill_names for s in avoid_skills)

        if title_mismatch and skill_mismatch:
            results.append(cid)
        if len(results) >= n:
            break
    return results

ANCHOR_CONFIGS = {
    "de": {
        "label": "Senior Data Engineer",
        "fit_skills": ["kafka", "bigquery", "airflow", "python", "sql", "gcp", "spark"],
        "fit_titles": ["data engineer", "analytics engineer", "data platform"],
        "nonfit_titles": ["civil engineer", "content writer", "hr manager"],
        "nonfit_skills": ["kafka", "bigquery", "spark", "airflow"],
    },
    "hr": {
        "label": "HR Manager",
        "fit_skills": ["recruitment", "performance management", "employee relations", "hris", "hr"],
        "fit_titles": ["hr manager", "human resources", "people operations", "talent acquisition"],
        "nonfit_titles": ["data engineer", "civil engineer", "mechanical engineer"],
        "nonfit_skills": ["recruitment", "hris", "employee relations"],
    },
    "analyst": {
        "label": "Mid Data Analyst (Stretch)",
        "fit_skills": ["sql", "tableau", "power bi", "python", "analytics", "excel"],
        "fit_titles": ["data analyst", "business analyst", "bi analyst"],
        "nonfit_titles": ["civil engineer", "sales executive", "content writer"],
        "nonfit_skills": ["sql", "tableau", "analytics"],
    },
    "pm": {
        "label": "Program Lead (Implicit JD)",
        "fit_skills": ["agile", "project management", "scrum", "stakeholder management", "jira"],
        "fit_titles": ["project manager", "program manager", "delivery manager", "operations manager"],
        "nonfit_titles": ["civil engineer", "data engineer", "accountant"],
        "nonfit_skills": ["project management", "agile", "scrum"],
    },
    "accountant": {
        "label": "Senior Accountant (Salary+Location Constraint)",
        "fit_skills": ["accounting", "financial reporting", "gst", "tds", "tally", "sap", "excel"],
        "fit_titles": ["accountant", "finance", "chartered accountant"],
        "nonfit_titles": ["data engineer", "civil engineer", "hr manager"],
        "nonfit_skills": ["accounting", "financial reporting", "gst"],
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True, choices=list(ANCHOR_CONFIGS.keys()),
                        help="Which JD to find anchors for")
    parser.add_argument("--show", type=int, default=5, help="How many fit candidates to show")
    args = parser.parse_args()

    cfg = ANCHOR_CONFIGS[args.jd]
    print(f"\n=== Anchor Finder for: {cfg['label']} ===")
    print("Loading data (this takes ~30s for 100K profiles)...")

    df = load_parquet()
    profiles = load_raw_profiles()

    print(f"\n--- OBVIOUS FIT candidates (skill+title match) ---")
    fit_ids = find_fit_candidates(profiles, df, cfg["fit_skills"], cfg["fit_titles"], n=args.show)
    print(f"Found {len(fit_ids)} candidates. Showing top {args.show}:")
    for cid in fit_ids:
        print_profile_summary(cid, profiles, df)

    print(f"\n\n--- OBVIOUS NON-FIT candidates ---")
    nonfit_ids = find_nonfit_candidates(profiles, cfg["nonfit_titles"], cfg["nonfit_skills"], n=3)
    print(f"Found {len(nonfit_ids)} candidates:")
    for cid in nonfit_ids:
        print_profile_summary(cid, profiles, df)

    print(f"\n\n=== RECORD THESE ANCHOR IDs ===")
    print(f"Fit anchors   : {fit_ids[:3]}")
    print(f"Non-fit anchors: {nonfit_ids}")
    print("\nCopy these into scripts/anchors.json for use in evaluation.")

if __name__ == "__main__":
    main()
