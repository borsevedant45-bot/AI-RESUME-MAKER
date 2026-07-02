import pandas as pd

jds = [
    ("DE (Senior Data Engineer)", "data/outputs/de/"),
    ("HR (HR Manager)", "data/outputs/hr/"),
    ("Analyst (Mid Data Analyst)", "data/outputs/analyst/"),
    ("PM (Project Manager)", "data/outputs/pm/"),
    ("Accountant (Senior Accountant)", "data/outputs/accountant/"),
]

for label, path in jds:
    df = pd.read_csv(path + "ranked_output.csv")
    g = df["grounding_validated"].value_counts().to_dict()
    nulls = df.isnull().sum().sum()
    labels = df["match_summary"].apply(
        lambda x: "strong"
        if "strong match" in str(x)
        else ("moderate" if "moderate" in str(x) else ("cautious" if "cautious" in str(x) else "fallback"))
    )
    quality = labels.value_counts().to_dict()
    sr = df["semantic_score"]
    cr = df["composite_score"]
    print(f"{label}:")
    print(f"  Grounding: {g} | Nulls: {nulls}")
    print(f"  Match quality: {quality}")
    print(f"  Semantic: [{sr.min():.3f}, {sr.max():.3f}] spread={sr.max()-sr.min():.3f}")
    print(f"  Composite: [{cr.min():.3f}, {cr.max():.3f}] spread={cr.max()-cr.min():.3f}")
    r1 = df.iloc[0]
    print(f"  Rank 1: {r1['candidate_id']} comp={r1['composite_score']:.3f} | {str(r1['match_summary'])[:100]}")
    print()
