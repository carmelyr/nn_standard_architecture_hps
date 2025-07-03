import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"

# initializes the results directory
models = sorted([d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))])
datasets = set()

for model in models:
    model_dir = os.path.join(RESULTS_DIR, model)
    for dataset in os.listdir(model_dir):
        datasets.add(dataset)

datasets = sorted(d for d in datasets if d != ".DS_Store")

# creates a DataFrame to hold the availability of seeds
# rows are datasets, columns are models
availability = pd.DataFrame(0, index=datasets, columns=models)

seed_pattern = re.compile(r"_seed(\d+)\.json$")

# iterates through each model and dataset to check for completed seeds
for model in models:
    for dataset in datasets:
        dataset_path = os.path.join(RESULTS_DIR, model, dataset)
        if not os.path.isdir(dataset_path):
            continue

        seeds_found = set()
        config_path = os.path.join(dataset_path, "config_100")
        if os.path.isdir(config_path):
            for fname in os.listdir(config_path):
                if fname.endswith(".json") and dataset in fname:
                    match = seed_pattern.search(fname)
                    if match:
                        seeds_found.add(int(match.group(1)))


        availability.loc[dataset, model] = min(len(seeds_found), 5)

# heatmap plot
plt.figure(figsize=(12, len(datasets) * 0.25 + 1.5))
sns.heatmap(availability, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5, cbar_kws={"label": "Number of Completed Seeds (Max = 5)"})
plt.title("Heatmap of Seed Completion per Dataset and Model")
plt.xlabel("Model")
plt.ylabel("Dataset")
plt.tight_layout()
plt.show()
