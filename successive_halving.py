import os
import glob
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

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

        # computes last valid value per fold/seed, then average across all
        last_values = np.zeros((self.configs, self.seeds, self.folds))
        for i in range(self.configs):
            for j in range(self.seeds):
                for k in range(self.folds):
                    curve = self.curves[i, j, k]
                    valid = ~np.isnan(curve)
                    if np.any(valid):
                        last_values[i, j, k] = curve[valid][-1]
                    else:
                        last_values[i, j, k] = np.nan
        self.final_scores = np.nanmean(last_values, axis=(1, 2))


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
                regret = best_final_score - np.nanmax(selected_final_scores) # !!!!!!!!!!

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
    
    def simulate_custom_schedule(self, schedule: List[Tuple[int, int]], top_k: int = 1):
        """
        Simulates a custom successive halving schedule.
        Args:
        - schedule: List of tuples (num_configs, num_epochs) for each round.
        - top_k: Number of final top configs to consider for hit and regret.

        Returns:
        - Tuple of two lists:
            - hit probabilities for each round
            - regrets for each round
        """
        # starts with all config indices
        current_indices = list(range(self.configs))
        hits_per_round = []
        regrets_per_round = []

        for round_idx, (num_to_keep, epoch_budget) in enumerate(schedule):
            if not current_indices:
                # no more configs to evaluate
                hits_per_round.append(np.nan)
                regrets_per_round.append(np.nan)
                continue

            epoch_idx = min(epoch_budget - 1, self.epochs - 1)
            
            round_hits = []
            round_regrets = []

            for seed_idx in range(self.seeds):
                # extracts validation performance at the current epoch
                perf = np.nanmean(self.curves[current_indices, seed_idx, :, epoch_idx], axis=1)

                # filters valid configs
                valid_mask = ~np.isnan(perf)
                if not np.any(valid_mask):
                    continue

                valid_configs = np.array(current_indices)[valid_mask]
                valid_perf = perf[valid_mask]

                # gets indices of top performers to keep
                top_indices = np.argsort(valid_perf)[-num_to_keep:]
                selected = valid_configs[top_indices]

                # Hit = did we select a final top-k config
                final_scores = self.final_scores[valid_configs]
                top_k_final = valid_configs[np.argsort(final_scores)[-top_k:]]

                hit = int(any(cfg in top_k_final for cfg in selected))
                regret = np.nanmax(final_scores) - np.nanmax(self.final_scores[selected])

                round_hits.append(hit)
                round_regrets.append(regret)

            hits_per_round.append(np.mean(round_hits) if round_hits else np.nan)
            regrets_per_round.append(np.mean(round_regrets) if round_regrets else np.nan)

            # updates current_indices for next round using the average across ALL seeds (and folds) at this round's epoch
            if round_hits:  # only updates if valid results this round
                # perf_avg has shape (len(current_indices),) — mean over seeds and folds at epoch_idx
                perf_avg = np.nanmean(self.curves[current_indices, :, :, epoch_idx], axis=(1, 2))
                valid_mask_avg = ~np.isnan(perf_avg)

                if np.any(valid_mask_avg):
                    valid_configs_avg = np.array(current_indices)[valid_mask_avg]
                    valid_perf_avg = perf_avg[valid_mask_avg]

                    keep = min(num_to_keep, len(valid_configs_avg))
                    if keep > 0:
                        top_idx = np.argsort(valid_perf_avg)[-keep:][::-1]   # highest first
                        selected_avg = valid_configs_avg[top_idx]
                        current_indices = list(selected_avg)
                    else:
                        current_indices = []

                    if len(current_indices) <= 1:
                        break
                else:
                    current_indices = []
                    break

        return hits_per_round, regrets_per_round

    def plot_config_elimination(self, schedule: List[Tuple[int, int]], use_average: bool = True, output_dir: str = None, dataset_name: str = "Dataset"):
        """
        Visualizes how configurations are eliminated during successive halving rounds.
        Args:
            - schedule: List of tuples (num_configs, num_epochs) for each round.
            - use_average: If True, average across all seeds/folds (matches simulation). If False, use seed 0 only.
            - output_dir: Directory to save the plot (if None, plot is shown).
            - dataset_name: Name of the dataset for plot title.
        """
        # tracks which configs survive each round
        current_indices = list(range(self.configs))
        survival_matrix = np.zeros((self.configs, len(schedule)), dtype=bool)
        round_labels = []
        
        for round_idx, (num_to_keep, epoch_budget) in enumerate(schedule):
            if not current_indices:
                break
                
            epoch_idx = min(epoch_budget - 1, self.epochs - 1)
            
            if use_average:
                # averages across ALL seeds and folds
                perf = np.nanmean(self.curves[current_indices, :, :, epoch_idx], axis=(1, 2))
            else:
                perf = np.nanmean(self.curves[current_indices, 0, :, epoch_idx], axis=1)
            
            valid_mask = ~np.isnan(perf)
            if not np.any(valid_mask):
                break
                
            valid_configs = np.array(current_indices)[valid_mask]
            valid_perf = perf[valid_mask]

            if len(valid_configs) <= num_to_keep:
                selected = valid_configs
            else:
                top_indices = np.argsort(valid_perf)[-num_to_keep:]
                selected = valid_configs[top_indices]
            
            # marks surviving configs in the matrix
            for config_id in selected:
                survival_matrix[config_id, round_idx] = True
                
            round_labels.append(f"Round {round_idx+1}\n({num_to_keep} configs, {epoch_budget} epochs)")
            
            current_indices = list(selected)
            if len(selected) <= 1:
                break

        fig_width = min(14, 4 + len(round_labels) * 2)  
        fig_height = min(15, max(10, self.configs * 0.10))   
        
        plt.figure(figsize=(fig_width, fig_height))

        colors = ['#F0F0F0', '#FF1493']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        survival_matrix_plot = survival_matrix[:, :len(round_labels)]
        im = plt.imshow(survival_matrix_plot, 
                       cmap=cmap, aspect='auto', interpolation='nearest')
        
        plt.xlabel('Successive Halving Rounds', fontsize=11, fontweight='bold')
        plt.ylabel('Configuration ID', fontsize=11, fontweight='bold')
        
        title_parts = dataset_name.split('_')
        if len(title_parts) > 1:
            short_title = f"{title_parts[0]} - {title_parts[1]}"
        else:
            short_title = dataset_name
        plt.title(f'Config Elimination: {short_title}', fontsize=12, fontweight='bold', pad=10)
        
        short_labels = []
        for round_idx, (num_to_keep, epoch_budget) in enumerate(schedule[:len(round_labels)]):
            short_labels.append(f"Round {round_idx+1}\nKeep: {num_to_keep}\nEpochs: {epoch_budget}")
        
        x_positions = range(len(round_labels))
        plt.xticks(x_positions, short_labels, fontsize=9)
        plt.tick_params(axis='x', which='both', length=0)

        for i in range(len(round_labels)):
            plt.axvline(x=i-0.5, color='#333333', linestyle='-', alpha=0.7, linewidth=2)
        plt.axvline(x=len(round_labels)-0.5, color='#333333', linestyle='-', alpha=0.7, linewidth=2)
        
        y_ticks = list(range(self.configs))
        y_labels = [str(i + 1) for i in y_ticks]  
        plt.yticks(y_ticks, y_labels, fontsize=6) 
        
        plt.gca().set_yticks([i - 0.5 for i in range(1, self.configs)], minor=True)
        
        plt.grid(True, alpha=0.2, linewidth=0.5, which='major')
        plt.grid(True, alpha=0.1, linewidth=0.3, which='minor')

        cbar = plt.colorbar(im, shrink=0.8, pad=0.02)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Eliminated', 'Surviving'], fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        if len(current_indices) > 0:
            final_configs = current_indices
            for config_id in final_configs:
                plt.axhline(y=config_id, color='red', linewidth=1.5, alpha=0.8)
            

            plt.plot([], [], color='red', linewidth=1.5, alpha=0.8, 
                    label=f'Final: {len(final_configs)} configs')
            plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
        

        total_eliminated = self.configs - len(current_indices) if current_indices else self.configs
        elimination_rate = (total_eliminated / self.configs) * 100
        plt.text(0.02, 0.98, f'Elimination: {elimination_rate:.1f}%', 
                transform=plt.gca().transAxes, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout(pad=1.0)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if use_average:
                filename = f"{dataset_name}_config_elimination_averaged.png"
            else:
                filename = f"{dataset_name}_config_elimination_seed0.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved elimination plot: {os.path.join(output_dir, filename)}")
        else:
            plt.show()


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
                if not acc:
                    continue  # skip if empty
                length = min(len(acc), max_epochs)
                padded = np.full(max_epochs, fill_value=acc[-1])    # pad with last value
                padded[:length] = acc[:length]
                curves[i, seed_idx, fold_idx, :] = padded           # creates an empty matrix and fills it with the accuracy values from the JSON files

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
    #models = ["CNN", "GRU", "LSTM", "Transformer", "FCNN"]
    models = ["CNN"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    plot_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/boxplots"
    csv_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/csv"
    max_epochs = 100
    top_k_values = [1, 2, 3, 4]

    for model in models:
        custom_output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/custom_schedule"
        os.makedirs(custom_output_dir, exist_ok=True)

        print(f"\n=== Processing {model} ===")
        base_dir = os.path.join(base_root, model)

        # 5 halving schedules
        schedules = {
            "Standard": [(100, 1), (50, 10), (10, 30), (1, 100)],
            "Aggressive": [(100, 1), (10, 20), (1, 100)],
            "Conservative": [(100, 5), (75, 20), (50, 50), (25, 75), (1, 100)],
            "VeryAggressive": [(100, 1), (5, 50), (1, 100)],
            "ExponentialElimination": [(100, 1), (50, 5), (25, 15), (12, 30), (6, 50), (3, 70), (1, 100)]
        }

        # Initialize data structures for each schedule
        schedule_results = {}
        for schedule_name in schedules.keys():
            schedule_results[schedule_name] = {k: {'hits': [], 'regrets': []} for k in top_k_values}

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
                
                for schedule_name, schedule in schedules.items():
                    custom_hits, custom_regrets = sim.simulate_custom_schedule(schedule, top_k=1)
                    print(f"[{model}/{dataset}] {schedule_name} Schedule Hits: {custom_hits}")
                    print(f"[{model}/{dataset}] {schedule_name} Schedule Regrets: {custom_regrets}")

                    schedule_csv_path = os.path.join(custom_output_dir, f"{model}_{schedule_name.lower()}_schedule.csv")

                    if not os.path.exists(schedule_csv_path):
                        with open(schedule_csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["Dataset", "Round", "Configs", "Epochs", "Hit Probability", "Regret"])
                    
                    with open(schedule_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        for i, ((num_cfgs, budget), hit, regret) in enumerate(zip(schedule, custom_hits, custom_regrets), 1):
                            writer.writerow([dataset, i, num_cfgs, budget, round(hit, 4), round(regret, 4)])

                    elimination_output_dir = f"/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/elimination_plots/{model}/{schedule_name.lower()}"
                    sim.plot_config_elimination(
                        schedule=schedule, 
                        use_average=True, 
                        output_dir=elimination_output_dir, 
                        dataset_name=f"{model}_{dataset}_{schedule_name}"
                    )

                    for k in top_k_values:
                        final_hits, final_regrets = sim.simulate_custom_schedule(schedule, top_k=k)

                        if final_hits and final_regrets:
                            final_hit = final_hits[-1] if not np.isnan(final_hits[-1]) else 0.0
                            final_regret = final_regrets[-1] if not np.isnan(final_regrets[-1]) else 0.0
                            schedule_results[schedule_name][k]['hits'].append(final_hit)
                            schedule_results[schedule_name][k]['regrets'].append(final_regret)
                        else:
                            schedule_results[schedule_name][k]['hits'].append(0.0)
                            schedule_results[schedule_name][k]['regrets'].append(0.0)

                dataset_names.append(dataset)
                print(f"Processed {model}/{dataset}")
            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

        for schedule_name in schedules.keys():
            schedule_plot_dir = os.path.join(plot_root, model, schedule_name.lower())
            schedule_csv_dir = os.path.join(csv_root, model, schedule_name.lower())
            
            all_hits_by_k = {k: [schedule_results[schedule_name][k]['hits']] for k in top_k_values}
            all_regrets_by_k = {k: [schedule_results[schedule_name][k]['regrets']] for k in top_k_values}
            
            all_hits_by_k = {k: np.array(v).T for k, v in all_hits_by_k.items()}
            all_regrets_by_k = {k: np.array(v).T for k, v in all_regrets_by_k.items()}
            
            plot_boxplots_single_schedule(all_hits_by_k, all_regrets_by_k, schedule_plot_dir, f"{model}_{schedule_name}")
            save_csv_single_schedule(all_hits_by_k, all_regrets_by_k, schedule_csv_dir, f"{model}_{schedule_name}", dataset_names)
            plot_k_axis_boxplots_single_schedule(all_hits_by_k, all_regrets_by_k, schedule_plot_dir, f"{model}_{schedule_name}")

def plot_boxplots_single_schedule(all_hits_by_k, all_regrets_by_k, output_dir: str, model_schedule: str):
    os.makedirs(output_dir, exist_ok=True)

    for k in all_hits_by_k:
        hits_array = np.array(all_hits_by_k[k])
        regrets_array = np.array(all_regrets_by_k[k])
        
        hits_array = np.nan_to_num(hits_array, nan=0)
        regrets_array = np.nan_to_num(regrets_array, nan=0)

        print(f"Plotting {model_schedule} k={k} | Hits shape: {hits_array.shape} | Regrets shape: {regrets_array.shape}")

        plt.figure(figsize=(10, 4), constrained_layout=True)

        ax1 = plt.subplot(1, 2, 1)
        hits_data = hits_array.flatten() if hits_array.ndim > 1 else hits_array
        ax1.hist(hits_data, bins=20, alpha=0.7, color='pink', edgecolor='black')
        ax1.set_title(f"Top-{k} Hit Probability Distribution ({model_schedule})")
        ax1.set_xlabel("Hit Probability")
        ax1.set_ylabel("Frequency")
        ax1.set_xlim(-0.05, 1.05)
        ax1.grid(True)

        ax2 = plt.subplot(1, 2, 2)
        regrets_data = regrets_array.flatten() if regrets_array.ndim > 1 else regrets_array
        ax2.hist(regrets_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title(f"Regret Distribution ({model_schedule})")
        ax2.set_xlabel("Regret")
        ax2.set_ylabel("Frequency")
        ax2.grid(True)

        plt.suptitle(f"Final Results for {model_schedule} (Top-{k})")
        plt.savefig(os.path.join(output_dir, f"{model_schedule}_top{k}_results.png"), dpi=300)
        plt.close()


def save_csv_single_schedule(all_hits_by_k, all_regrets_by_k, output_dir: str, model_schedule: str, dataset_names: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    for k in all_hits_by_k:
        hit_file = os.path.join(output_dir, f"{model_schedule}_top{k}_hit.csv")
        regret_file = os.path.join(output_dir, f"{model_schedule}_top{k}_regret.csv")

        with open(hit_file, "w", newline="") as f_hit, open(regret_file, "w", newline="") as f_regret:
            hit_writer = csv.writer(f_hit)
            regret_writer = csv.writer(f_regret)

            hit_writer.writerow(["Dataset", "Final_Hit_Probability"])
            regret_writer.writerow(["Dataset", "Final_Regret"])

            hits_data = all_hits_by_k[k].flatten() if all_hits_by_k[k].ndim > 1 else all_hits_by_k[k]
            regrets_data = all_regrets_by_k[k].flatten() if all_regrets_by_k[k].ndim > 1 else all_regrets_by_k[k]

            for name, hit_val, regret_val in zip(dataset_names, hits_data, regrets_data):
                hit_writer.writerow([name, hit_val])
                regret_writer.writerow([name, regret_val])


def plot_k_axis_boxplots_single_schedule(all_hits_by_k, all_regrets_by_k, output_dir: str, model_schedule: str):
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = sorted(all_hits_by_k.keys())
    hits_per_k = []
    regrets_per_k = []

    for k in k_values:
        hits_data = all_hits_by_k[k].flatten() if all_hits_by_k[k].ndim > 1 else all_hits_by_k[k]
        regrets_data = all_regrets_by_k[k].flatten() if all_regrets_by_k[k].ndim > 1 else all_regrets_by_k[k]
        hits_per_k.append(hits_data)
        regrets_per_k.append(regrets_data)

    # Top-k hit
    plt.figure(figsize=(6, 4))
    plt.boxplot(hits_per_k, labels=[f"k={k}" for k in k_values])
    plt.title(f"Hit Probability by k-value ({model_schedule})")
    plt.xlabel("Top-k Value")
    plt.ylabel("Hit Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_schedule}_k_axis_hit.png"))
    plt.close()

    # Regret
    plt.figure(figsize=(6, 4))
    plt.boxplot(regrets_per_k, labels=[f"k={k}" for k in k_values])
    plt.yscale("log")
    plt.title(f"Regret by k-value ({model_schedule})")
    plt.xlabel("Top-k Value")
    plt.ylabel("Regret (log scale)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_schedule}_k_axis_regret.png"))
    plt.close()


if __name__ == "__main__":
    run_all_models()
