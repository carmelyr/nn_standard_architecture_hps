import os
import re
import csv
from collections import defaultdict

RESULTS_DIR = "results"
CSV_OUTPUT = "missing_config_seed_report.csv"
EXPECTED_CONFIGS = range(1, 101)
EXPECTED_SEEDS = range(1, 6)  # seed1 to seed5

def check_missing():
    missing = defaultdict(lambda: defaultdict(lambda: {"missing_configs": [], "missing_seeds": {}}))

    for model in os.listdir(RESULTS_DIR):
        model_path = os.path.join(RESULTS_DIR, model)
        if not os.path.isdir(model_path):
            continue

        for dataset in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            # Get all valid config directories
            available_configs = {
                int(match.group(1)): d for d in os.listdir(dataset_path)
                if (match := re.match(r"config_(\d+)", d))
            }

            for config_num in EXPECTED_CONFIGS:
                if config_num not in available_configs:
                    missing[model][dataset]["missing_configs"].append(config_num)
                else:
                    config_path = os.path.join(dataset_path, available_configs[config_num])
                    for seed in EXPECTED_SEEDS:
                        expected_file = f"{dataset}_config_{config_num}_seed{seed}.json"
                        full_path = os.path.join(config_path, expected_file)
                        if not os.path.isfile(full_path):
                            if config_num not in missing[model][dataset]["missing_seeds"]:
                                missing[model][dataset]["missing_seeds"][config_num] = []
                            missing[model][dataset]["missing_seeds"][config_num].append(seed)

    return missing

def save_missing_to_csv(missing, filename):
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['model', 'dataset', 'config_number', 'missing_type', 'missing_seeds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model, datasets in missing.items():
            for dataset, data in datasets.items():
                for config_num in data["missing_configs"]:
                    writer.writerow({
                        'model': model,
                        'dataset': dataset,
                        'config_number': config_num,
                        'missing_type': 'missing_config',
                        'missing_seeds': ''
                    })
                for config_num, seeds in data["missing_seeds"].items():
                    writer.writerow({
                        'model': model,
                        'dataset': dataset,
                        'config_number': config_num,
                        'missing_type': 'missing_seed',
                        'missing_seeds': ','.join(str(s) for s in sorted(seeds))
                    })

    print(f"\n✅ Missing configuration and seed data saved to: {filename}")

def filter_empty_models(missing):
    """Remove models with no missing configs or seeds"""
    filtered = defaultdict(dict)
    for model, datasets in missing.items():
        for dataset, data in datasets.items():
            if data["missing_configs"] or data["missing_seeds"]:
                filtered[model][dataset] = data
    return filtered

if __name__ == "__main__":
    raw_missing = check_missing()
    filtered_missing = filter_empty_models(raw_missing)

    if filtered_missing:
        save_missing_to_csv(filtered_missing, CSV_OUTPUT)
    else:
        print("✅ All models have complete config_1–100 and seed1–5 results.")
