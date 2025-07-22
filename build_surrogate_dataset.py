import os
import json
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
# groups results from repeated runs of the same hyperparameter config
def config_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

# this method extracts the model configuration and validation accuracies from a JSON file
# returns the dataset name, config ID, config dictionary, and a list of accuracies
def extract_model_config(path, model, max_epoch=20):
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading JSON file: {path}")
            return None

    config = data.get("hyperparameters", {}).copy()
    dataset = data.get("dataset_stats", {}).get("name", "unknown")

    # skip if hyperparameter config is incomplete for the given model
    if not all(key in config for key in MODEL_KEYS[model]):
        return None

    # --- FIXED BLOCK: Get val_accuracy from fold_logs ---
    fold_logs = data.get("fold_logs", [])
    if not fold_logs or any("val_accuracy" not in f for f in fold_logs):
        print(f"Missing val_accuracy in fold_logs: {path}")
        return None

    fold_acc = []
    for fold in fold_logs:
        acc_list = fold["val_accuracy"]
        if len(acc_list) < max_epoch:
            print(f"Skipping {path} — fold has only {len(acc_list)} epochs")
            return None
        fold_acc.append(acc_list[:max_epoch])

    # average across folds
    acc = pd.DataFrame(fold_acc).mean(axis=0).tolist()
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

    grouped = defaultdict(lambda: {"config": None, "accuracies": []})
    json_files = collect_json_files(result_root)
    
    if not json_files:
        print(f"No JSON files found for {model} in {result_root}")
        return

    print(f"\nProcessing {model}: Found {len(json_files)} JSON files")

    processed_files = 0
    for path in json_files:
        result = extract_model_config(path, model=model, max_epoch=max_epoch)
        if result is None:
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

    # output directory
    os.makedirs(output_root, exist_ok=True)

    print(f"CSV files for {model}")
    for (dataset, _), entry in grouped.items():
        if not entry["accuracies"]:
            continue

        # calculates average accuracy across seeds
        acc_matrix = pd.DataFrame(entry["accuracies"])
        acc_avg = acc_matrix.mean(axis=0)

        # creates row with only the relevant hyperparameters for the model
        row = {k: v for k, v in entry["config"].items() if k in MODEL_KEYS[model]}
        row["dataset"] = dataset
        for i in range(max_epoch):
            row[f"val_acc_{i+1}"] = acc_avg[i]


        out_dir = os.path.join(output_root, dataset)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{model}_epochs_1-{max_epoch}.csv")
        
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path)
            df_new = pd.DataFrame([row])
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = pd.DataFrame([row])

        df_all.to_csv(out_path, index=False)
    
    print(f"Completed processing {model}")

def build_all_surrogate_datasets(max_epoch=20):
    #models = ["FCNN", "CNN", "LSTM", "GRU", "Transformer"]
    models = ["CNN"]

    for model in models:
        result_root = os.path.join("results", model)
        output_root = os.path.join("surrogate_datasets", model)
            
        process_model_results(model=model, result_root=result_root, output_root=output_root, max_epoch=max_epoch)

if __name__ == "__main__":
    build_all_surrogate_datasets(max_epoch=20)