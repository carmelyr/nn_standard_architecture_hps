import os
import glob
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

pink_palette = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585", "#DB7093", "#FFC0CB"]
sns.set_palette(pink_palette)


class SuccessiveHalvingSimulator:

    # initializes the simulator with curve data and computes final scores
    # curve_data: a 4D numpy array with shape (configs, seeds, folds, epochs)
    # final_epoch: the last epoch to consider for final scores
    # final_scores: a 1D numpy array with the final scores for each config
    # configs: number of configurations
    # seeds: number of seeds
    # folds: number of folds
    # epochs: number of epochs in the curve data
    # final_scores: average final scores across seeds and folds for each config
    def __init__(self, curve_data: np.ndarray, final_epoch: int = 1000):
        self.curves = curve_data
        self.configs, self.seeds, self.folds, self.epochs = curve_data.shape
        self.final_scores = np.nanmean(self.curves[:, :, :, -1], axis=(1, 2))   # true final accuracy (average over all seeds and folds at the last epoch)

    # this function simulates the successive halving process
    def simulate(self, budget_epochs: List[int], top_k: int = 1):
        top_k_hits = []         # stores the hit probabilities for each budget epoch
        regrets = []            # stores the regrets for each budget epoch
        
        for budget in budget_epochs:
            epoch_idx = min(budget - 1, self.epochs - 1)
            
            scores = np.nanmean(self.curves[:, :, :, epoch_idx], axis=2)  # average over folds, shape becomes [configs, seeds]
            
            dataset_hits = []       # stores hit probabilities for each dataset
            dataset_regrets = []    # stores regrets for each dataset
        
            for seed_idx in range(self.seeds):
                seed_scores = scores[:, seed_idx]       # scores for the current seed across all configs
                valid_mask = ~np.isnan(seed_scores)     # filter out NaN scores
                
                if not np.any(valid_mask):
                    continue
                    
                valid_scores = seed_scores[valid_mask]  
                valid_configs = np.where(valid_mask)[0]
                
                # selects top-k configs at budget epoch
                top_k_indices_budget = np.argsort(seed_scores[valid_mask])[-top_k:]
                selected_configs = valid_configs[top_k_indices_budget]

                # gets final scores of those selected configs
                selected_final_scores = self.final_scores[selected_configs]

                # computes regret
                best_final_score = np.nanmax(self.final_scores[valid_mask])
                regret = best_final_score - np.nanmax(selected_final_scores)

                # computes hit
                # whether any selected config is in final top-k
                top_k_indices_final = np.argsort(self.final_scores[valid_mask])[-top_k:]
                hit = int(any(c in valid_configs[top_k_indices_final] for c in selected_configs))

                dataset_hits.append(hit)
                dataset_regrets.append(regret)
            
            # stores average over seeds for this dataset
            if dataset_hits:
                top_k_hits.append(np.mean(dataset_hits))
                regrets.append(np.mean(dataset_regrets))
            else:
                top_k_hits.append(np.nan)
                regrets.append(np.nan)
        
        return top_k_hits, regrets

# this function loads curve data from JSON files, extracts the relevant information, and returns it as a numpy array
def load_curve_data_from_json_from_files(file_list: List[str], max_epochs: int = 1000) -> np.ndarray:
    configs = {}

    for path in file_list:
        fname = os.path.basename(path)
        import re

        match = re.search(r"_config_(\d+)_seed(\d+)", fname)        # creates a regex pattern to match the config and seed IDs
        if not match:
            continue

        config_id = int(match.group(1))                # extracts the config ID from the filename
        seed_id = int(match.group(2))                  # extracts the seed ID from the filename     


        with open(path) as f:
            data = json.load(f)

        # checks if the data has the required structure
        if config_id not in configs:
            configs[config_id] = {}
        configs[config_id][seed_id] = data

    config_ids = sorted(configs.keys())
    num_configs = len(config_ids)

    if num_configs == 0:
        raise ValueError("No valid config files found.")

    # determines the number of seeds and folds based on the first config
    # num_seeds: maximum number of seeds across all configs
    # first_seed: the first seed data to determine the number of folds
    # num_folds: number of folds in the first seed data
    # curves: a 4D numpy array to store the accuracy values for each config, seed, fold, and epoch
    num_seeds = max(len(configs[c]) for c in config_ids)
    first_seed = list(configs[config_ids[0]].values())[0]
    num_folds = len(first_seed["fold_logs"])
    curves = np.full((num_configs, num_seeds, num_folds, max_epochs), fill_value=np.nan)    # stores the curves in a 4D numpy array

    # fills the curves array with accuracy values from the JSON files
    # i: config index, seed_idx: seed index, fold_idx: fold index,
    # length: number of epochs to consider
    # acc: accuracy values for the current fold
    # curves: 4D numpy array with shape (num_configs, num_seeds, num_folds, max_epochs)
    for i, config_id in enumerate(config_ids):
        for seed_id, seed_data in configs[config_id].items():
            seed_idx = seed_id - 1
            for fold_idx, fold in enumerate(seed_data["fold_logs"]):
                acc = fold["val_accuracy"]
                length = min(len(acc), max_epochs)
                curves[i, seed_idx, fold_idx, :length] = acc[:length]           # creates an empty matrix and fills it with the accuracy values from the JSON files

    return curves


# this function plots boxplots for the hit probabilities and regrets for each top-k value
# all_hits_by_k: a dictionary with top-k values as keys and lists of hit probabilities
# all_regrets_by_k: a dictionary with top-k values as keys and lists of regrets
# budget_epochs: a list of budget epochs to consider
# output_dir: directory to save the plots
# model: the name of the model being processed
def plot_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs: List[int], output_dir: str, model: str):
    os.makedirs(output_dir, exist_ok=True)

    for k in all_hits_by_k:
        # converts to numpy arrays and ensure correct shape
        hits_array = np.array(all_hits_by_k[k])[:, :len(budget_epochs)]
        regrets_array = np.array(all_regrets_by_k[k])[:, :len(budget_epochs)]
        
        # cleans data by replacing any remaining NaN with 0
        hits_array = np.nan_to_num(hits_array, nan=0)
        regrets_array = np.nan_to_num(regrets_array, nan=0)

        print(f"Plotting k={k} | Budget epochs: {len(budget_epochs)} | "f"Hits shape: {hits_array.shape} | Regrets shape: {regrets_array.shape}")

        plt.figure(figsize=(10, 4), constrained_layout=True)
        
        # Top-k hit plot
        ax1 = plt.subplot(1, 2, 1)
        sns.boxplot(data=hits_array, ax=ax1, palette=pink_palette[:len(budget_epochs)])
        ax1.set_xticks(range(len(budget_epochs)))
        ax1.set_xticklabels(budget_epochs)
        ax1.set_title(f"Top-{k} Hit Probability ({model})")
        ax1.set_xlabel("Epoch Budget")
        ax1.set_ylabel("Hit Probability")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True)

        # Regret plot
        ax2 = plt.subplot(1, 2, 2)
        sns.boxplot(data=regrets_array, ax=ax2, palette=pink_palette[:len(budget_epochs)])
        ax2.set_xticks(range(len(budget_epochs)))
        ax2.set_xticklabels(budget_epochs)
        ax2.set_title(f"Regret ({model})")
        ax2.set_xlabel("Epoch Budget")
        ax2.set_ylabel("Regret")
        ax2.grid(True)

        plt.suptitle(f"Successive Halving Results for {model} (Top-{k})")
        plt.savefig(os.path.join(output_dir, f"{model}_top{k}_results.png"), dpi=300)
        plt.close()



def plot_k_axis_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs: List[int], output_dir: str, model: str):
    os.makedirs(output_dir, exist_ok=True)
    num_budgets = len(budget_epochs)

    for budget_idx, budget in enumerate(budget_epochs):
        hits_per_k = []
        regrets_per_k = []

        for k in sorted(all_hits_by_k.keys()):
            hits_array = np.array(all_hits_by_k[k])
            regrets_array = np.array(all_regrets_by_k[k])

            # ensures the arrays are sliced to match the budget epochs
            hits_array = hits_array[:, :len(budget_epochs)]
            regrets_array = regrets_array[:, :len(budget_epochs)]

            # skips if dataset has too few budget columns
            if budget_idx >= hits_array.shape[1] or budget_idx >= regrets_array.shape[1]:
                continue

            hits_per_k.append(hits_array[:, budget_idx])
            regrets_per_k.append(regrets_array[:, budget_idx])

        # converts to array: shape (k, datasets) -> (datasets, k)
        hits_per_k = np.array(hits_per_k).T
        regrets_per_k = np.array(regrets_per_k).T
        k_values = sorted(all_hits_by_k.keys())

        print(f"Budget epochs: {len(budget_epochs)}, Hits array shape: {hits_array.shape}, Regrets array shape: {regrets_array.shape}")

        # Top-k hit
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=hits_per_k, palette=pink_palette)
        plt.xticks(ticks=range(len(k_values)), labels=[f"k={k}" for k in k_values])
        plt.title(f"Probabilities of Selecting Final k-Best (Epoch={budget}) ({model})")
        plt.xlabel("Top-k Value")
        plt.ylabel("Hit Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_k_axis_hit_epoch{budget}.png"))
        plt.close()

        # Regret
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=regrets_per_k, palette=pink_palette)
        plt.xticks(ticks=range(len(k_values)), labels=[f"k={k}" for k in k_values])
        plt.yscale("log")
        plt.title(f"Regret for Final k-Best (Epoch={budget}) ({model})")
        plt.xlabel("Top-k Value")
        plt.ylabel("Regret (log scale)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_k_axis_regret_epoch{budget}.png"))
        plt.close()


# this function saves the hit probabilities and regrets for each top-k value to CSV files
def save_csv(all_hits_by_k, all_regrets_by_k, budget_epochs: List[int], output_dir: str, model: str, dataset_names: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    for k in all_hits_by_k:
        hit_file = os.path.join(output_dir, f"{model}_top{k}_hit.csv")
        regret_file = os.path.join(output_dir, f"{model}_top{k}_regret.csv")

        with open(hit_file, "w", newline="") as f_hit, open(regret_file, "w", newline="") as f_regret:
            hit_writer = csv.writer(f_hit)
            regret_writer = csv.writer(f_regret)

            hit_writer.writerow(["Dataset"] + budget_epochs)
            regret_writer.writerow(["Dataset"] + budget_epochs)

            for name, hit_row, regret_row in zip(dataset_names, all_hits_by_k[k], all_regrets_by_k[k]):
                hit_writer.writerow([name] + hit_row)
                regret_writer.writerow([name] + regret_row)


# this function runs the successive halving simulation for all models and datasets
# it processes each model, loads the curve data, simulates the successive halving process, and generates plots and CSV files for the results
def run_all_models():
    models = ["CNN", "GRU"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    plot_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/boxplots"
    csv_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/csv"
    max_epochs = 100
    budget_epochs = [5, 10, 20, 50, 75, 100]
    top_k_values = [1, 2, 3, 4]

    for model in models:
        print(f"\n=== Processing {model} ===")
        base_dir = os.path.join(base_root, model)
        all_hits_by_k = {k: [] for k in top_k_values}
        all_regrets_by_k = {k: [] for k in top_k_values}

        dataset_names = []

        for dataset in sorted(os.listdir(base_dir)):
            dataset_path = os.path.join(base_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            if dataset == "classification_ozone":
                # hardcoded pattern for classification_ozone
                json_files = glob.glob(os.path.join(dataset_path, "config_*", "classification_ozone_config_*_seed*.json"))
            else:
                # prefix to match other datasets
                json_files = glob.glob(os.path.join(dataset_path, "config_*", f"{dataset}_config_*_seed*.json"))

            print(f"{model}/{dataset} → Found {len(json_files)} files")

            if len(json_files) == 0:
                continue

            try:
                curves = load_curve_data_from_json_from_files(json_files, max_epochs=max_epochs)
                sim = SuccessiveHalvingSimulator(curves)
                dataset_names.append(dataset)

                for k in top_k_values:
                    hits, regrets = sim.simulate(budget_epochs=budget_epochs, top_k=k)
                    all_hits_by_k[k].append(hits)
                    all_regrets_by_k[k].append(regrets)

                print(f"Processed {model}/{dataset}")
            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

        plot_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs, os.path.join(plot_root, model), model)
        save_csv(all_hits_by_k, all_regrets_by_k, budget_epochs, os.path.join(csv_root, model), model, dataset_names)
        plot_k_axis_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs, os.path.join(plot_root, model), model)


if __name__ == "__main__":
    run_all_models()
