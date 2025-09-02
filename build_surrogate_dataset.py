# This module builds a surrogate dataset for hyperparameter optimization.

import os
import json
import re
import pandas as pd
import hashlib
from collections import defaultdict

MODEL_KEYS = {
    "FCNN": {
        "hidden_units", "num_layers", "dropout_rate", 
        "learning_rate", "activation", "weight_decay"
    },
    "CNN": {
        "num_filters", "num_layers", "kernel_size", "pooling",
        "pooling_size", "dropout_rate", "learning_rate", "activation"
    },
    "LSTM": {
        "hidden_units", "num_layers", "dropout_rate",
        "learning_rate", "output_activation", "bidirectional"
    },
    "GRU": {
        "hidden_units", "num_layers", "dropout_rate",
        "learning_rate", "output_activation", "bidirectional"
    },
    "Transformer": {
        "num_heads", "hidden_units", "ff_dim", "num_layers",
        "pooling", "dropout_rate", "learning_rate", "activation"
    }
}

# this method generates a hash from the configuration dictionary
def config_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

# this method extracts the model configuration and validation accuracies from a JSON file
# returns the dataset name, config dictionary, and a list of accuracies
def extract_model_config(path, model, max_epoch=20):
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading JSON file: {path}")
            return None

    config = data.get("hyperparameters", {}).copy()
    dataset = data.get("dataset_stats", {}).get("name", "unknown")

    # checks if fold_logs exist
    fold_logs = data.get("fold_logs", [])
    if not fold_logs:
        print(f"Missing fold_logs: {path}")
        return None

    fold_acc = []
    for fold in fold_logs:
        if "val_accuracy" not in fold:
            print(f"Missing val_accuracy in fold: {path}")
            continue
            
        acc_list = fold["val_accuracy"]
        
        if len(acc_list) == 0:
            print(f"Empty val_accuracy list in {path}")
            continue

        normalized_acc = []
        for i, acc_val in enumerate(acc_list):
            # converts to float to handle any string values
            try:
                acc_val = float(acc_val)
            except (ValueError, TypeError):
                print(f"Invalid accuracy value in {path}: {acc_val}")
                continue
                
            # checks if values are in percentage format and normalizes
            if acc_val > 1.0:
                acc_val = acc_val / 100.0
                if i == 0:
                    print(f"Converting percentage to fraction in {path}")

            # clamps values to [0, 1] range
            acc_val = max(0.0, min(1.0, acc_val))
            normalized_acc.append(acc_val)
        
        if not normalized_acc:
            print(f"No valid accuracy values in fold for {path}")
            continue

        # pads or truncates to max_epoch
        if len(normalized_acc) < max_epoch:
            # pads with the last available value
            last_val = normalized_acc[-1]
            padded_acc = normalized_acc + [last_val] * (max_epoch - len(normalized_acc))
            fold_acc.append(padded_acc)
        else:
            fold_acc.append(normalized_acc[:max_epoch])

    if not fold_acc:
        print(f"No valid folds found in {path}")
        return None

    # averages across available folds
    acc = pd.DataFrame(fold_acc).mean(axis=0).tolist()
    
    # final validation: ensures all accuracies are in [0, 1] range
    acc = [max(0.0, min(1.0, val)) for val in acc]
    
    config_id = config_hash(config)

    return dataset, config_id, config, acc

# this method collects all JSON files from a directory and its subdirectories
def collect_json_files(root_dir):
    jsons = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".json"):
                jsons.append(os.path.join(dirpath, f))
    return jsons

# this method processes the results of a specific model
def process_model_results(model, result_root, output_root, max_epoch=20):
    if not os.path.exists(result_root):
        print(f"No results found for {model} at {result_root}")
        return

    # groups by dataset and config_id to deduplicate
    grouped = defaultdict(lambda: {"config": None, "accuracies": []})
    json_files = collect_json_files(result_root)
    
    if not json_files:
        print(f"No JSON files found for {model} in {result_root}")
        return

    processed_files = 0
    skipped_files = 0
    
    for path in json_files:
        result = extract_model_config(path, model=model, max_epoch=max_epoch)
        if result is None:
            skipped_files += 1
            continue
            
        dataset, config_id, config, acc = result
        grouped[(dataset, config_id)]["config"] = config
        grouped[(dataset, config_id)]["accuracies"].append(acc)
        processed_files += 1
        
        if processed_files % 100 == 0:
            print(f"Processed {processed_files}/{len(json_files)} files")

    if not grouped:
        print(f"No valid configurations found for {model}")
        return

    # groups by dataset and counts unique configurations
    dataset_configs = defaultdict(list)
    for (dataset, config_id), entry in grouped.items():
        if entry["accuracies"]:                     # only includes configs with valid accuracies
            dataset_configs[dataset].append((config_id, entry))

    os.makedirs(output_root, exist_ok=True)

    for dataset, configs in dataset_configs.items():
        available_configs = len(configs)
        print(f"  Dataset {dataset}: {available_configs} unique configurations")
        
        # all available configs
        selected_configs = configs

        rows = []
        for j, (config_id, entry) in enumerate(selected_configs):
            # creates a new DataFrame each time to avoid any reference issues
            acc_data = []
            for acc_list in entry["accuracies"]:
                acc_data.append(acc_list.copy())
            
            acc_matrix = pd.DataFrame(acc_data)
            acc_avg = acc_matrix.mean(axis=0)
            acc_std = acc_matrix.std(axis=0)

            row = {k: v for k, v in entry["config"].items() if k in MODEL_KEYS[model]}
            row["dataset"] = dataset
            row["group_id"] = config_id
            row["repeat_idx"] = j

            for i in range(max_epoch):
                avg_val = float(acc_avg.iloc[i])
                std_val = float(acc_std.iloc[i]) if pd.notna(acc_std.iloc[i]) else 0.0
                
                if avg_val > 1.0:
                    print(f"WARNING: Found value > 1.0 in {dataset} config {config_id}: {avg_val}")
                    avg_val = min(1.0, avg_val)
                
                row[f"val_acc_{i+1}"] = avg_val
                row[f"val_acc_std_{i+1}"] = std_val

            rows.append(row)

        if rows:
            out_dir = os.path.join(output_root, dataset)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{model}_epochs_1-{max_epoch}.csv")
            
            df_all = pd.DataFrame(rows)
            df_all.to_csv(out_path, index=False)
            
            unique_groups = df_all["group_id"].nunique()
            print(f"    Saved {len(rows)} samples ({unique_groups} unique groups) to {out_path}")

# this function builds surrogate datasets for all specified models
def build_all_surrogate_datasets(max_epoch=20):
    models = ["FCNN", "CNN", "LSTM", "GRU", "Transformer"]

    for model in models:
        result_root = os.path.join("results", model)
        output_root = os.path.join("surrogate_datasets", model)
            
        process_model_results(model=model, result_root=result_root, output_root=output_root, max_epoch=max_epoch)

if __name__ == "__main__":
    build_all_surrogate_datasets(max_epoch=20)
    