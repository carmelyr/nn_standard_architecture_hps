import os
import json
from pathlib import Path

RESULTS_DIR = "results"
OUT_DIR = "dataset_meta"
os.makedirs(OUT_DIR, exist_ok=True)

dataset_seen = {}

for root, _, files in os.walk(RESULTS_DIR):
    for f in files:
        if f.endswith(".json"):
            path = Path(root) / f
            with open(path, "r") as fp:
                data = json.load(fp)
            if "dataset_stats" in data:
                ds = data["dataset_stats"]
                name = ds["name"]
                if name not in dataset_seen:
                    meta = {
                        "dataset_name": name,
                        "input_length": ds["input_shape"][0],
                        "n_channels": ds["input_shape"][1],
                        "n_classes": ds["num_classes"]
                    }
                    out_path = Path(OUT_DIR) / f"{name}_meta.json"
                    with open(out_path, "w") as out:
                        json.dump(meta, out, indent=2)
                    dataset_seen[name] = True
                    print(f"[OK] Wrote {out_path}")
