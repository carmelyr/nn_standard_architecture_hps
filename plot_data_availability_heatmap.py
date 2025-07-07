import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
models = sorted([d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))])
datasets = sorted({ds for m in models for ds in os.listdir(os.path.join(RESULTS_DIR, m)) if ds != ".DS_Store"})

# initializes DataFrames for availability and annotations
availability = pd.DataFrame(0.0, index=datasets, columns=models)
annotations = pd.DataFrame("", index=datasets, columns=models)
seed_pattern = re.compile(r"_seed(\d+)\.json$")

for model in models:
    for dataset in datasets:
        dataset_path = os.path.join(RESULTS_DIR, model, dataset)
        if not os.path.isdir(dataset_path):
            continue

        config_1_path = os.path.join(dataset_path, "config_1")          # directory for configuration 1
        config_100_path = os.path.join(dataset_path, "config_100")      # directory for configuration 100

        count = 0
        if os.path.isdir(config_100_path):
            seeds_found = set()
            # collects all seeds from config_100
            for fname in os.listdir(config_100_path):
                if fname.endswith(".json") and dataset in fname:
                    match = seed_pattern.search(fname)
                    if match:
                        seeds_found.add(int(match.group(1)))
            count = min(len(seeds_found), 5)
            availability.loc[dataset, model] = count

            if os.path.isdir(config_1_path):
                seeds_in_1 = set()
                for fname in os.listdir(config_1_path):
                    if fname.endswith(".json") and dataset in fname:
                        match = seed_pattern.search(fname)
                        if match:
                            seeds_in_1.add(int(match.group(1)))

                if len(seeds_in_1) > len(seeds_found):
                    annotations.loc[dataset, model] = f"{count} (i)"
                else:
                    annotations.loc[dataset, model] = str(count)
            else:
                annotations.loc[dataset, model] = str(count)

        elif os.path.isdir(config_1_path):
            availability.loc[dataset, model] = 0
            annotations.loc[dataset, model] = "0 (i)"

plt.figure(figsize=(12, len(datasets) * 0.25 + 1.5))
sns.heatmap(availability, cmap="YlOrRd", annot=annotations, fmt="s", linewidths=0.5,
            cbar_kws={"label": "Completed Seeds"})
plt.title("Seed Completion Status per Dataset and Model\n(i) = In Progress (config_1 exists, config_100 not yet or incomplete)")
plt.xlabel("Model")
plt.ylabel("Dataset")
plt.tight_layout()
plt.show()
