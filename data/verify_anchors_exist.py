import json
from pathlib import Path
import pandas as pd

feat = pd.read_parquet("data/processed/candidate_features.parquet")
all_ids = set(feat["candidate_id"].tolist())

anchors = json.loads(Path("scripts/anchors.json").read_text())
for key, data in anchors.items():
    for role_type in ["fit_anchors", "nonfit_anchors"]:
        for cid in data[role_type]:
            exists = cid in all_ids
            print(f"{key}/{role_type}: {cid} exists={exists}")
