# This script serves as the main entry point for the successive halving analysis.
# It directs all analysis components: schedules, multi-objective, and resource-constrained.

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sh_core import (SuccessiveHalvingSimulator, load_curve_data_from_json_from_files)
from sh_resource_constraints import create_resource_constraint_analysis

pink_palette = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585", "#DB7093", "#FFC0CB"]
sns.set_palette(pink_palette)

# this function runs the entire analysis process
# it processes each model and dataset, loads curve data, simulates schedules, and generates plots and CSV files
def run_all_models():
    models = ["CNN", "GRU", "LSTM", "Transformer", "FCNN"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    plot_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/boxplots"
    csv_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/csv"
    max_epochs = 100
    top_k_values = [1, 2, 3, 4]

    budget_schedules = {
        "Moderate": [1, 10, 30, 100],
        "Aggressive": [1, 20, 100],
        "Conservative": [5, 20, 50, 75, 100],
        "VeryAggressive": [1, 50, 100],
        "Standard": [1, 5, 15, 30, 50, 70, 100]
    }
    
    dropout_schedules = {
        "Linear": [100, 80, 60, 40, 20, 1],
        "Standard": [100, 50, 25, 12, 6, 3, 1],
        "Aggressive": [100, 20, 5, 1],
        "Gradual": [100, 90, 80, 70, 60, 50, 40, 1],
        "TwoStage": [100, 50, 25, 1]
    }

    hybrid_schedules = create_hybrid_schedules(budget_schedules, dropout_schedules)
    
    all_budget_epochs = set()
    for schedule in hybrid_schedules.values():
        for _, epochs in schedule:
            all_budget_epochs.add(epochs)
    
    budget_epochs = sorted(list(all_budget_epochs))

    # First pass: collect global maximum regret across all models
    print("Collecting global regret maximum across all models...")
    global_regret_max = float('-inf')

    for model in models:
        base_dir = os.path.join(base_root, model)
        for dataset in sorted(os.listdir(base_dir)):
            dataset_path = os.path.join(base_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue
            
            if dataset == "classification_ozone":
                json_files = glob.glob(os.path.join(dataset_path, "config_*", "classification_ozone_config_*_seed*.json"))
            else:
                json_files = glob.glob(os.path.join(dataset_path, "config_*", f"{dataset}_config_*_seed*.json"))
            
            if len(json_files) == 0:
                continue
            
            try:
                curves = load_curve_data_from_json_from_files(json_files, max_epochs=max_epochs)
                sim = SuccessiveHalvingSimulator(curves)
                
                for k in top_k_values:
                    hits, regrets = sim.simulate(budget_epochs=budget_epochs, top_k=k)
                    regrets_array = np.array(regrets)
                    global_regret_max = max(global_regret_max, np.max(regrets_array))
            except Exception as e:
                continue

    # Add padding to global regret limit
    regrets_padding = global_regret_max * 0.05
    global_regret_ylim = (0, global_regret_max + regrets_padding)
    
    print(f"Global regret y-limit: {global_regret_ylim}")

    # Second pass: process models with consistent regret y-limits
    for model in models:
        hybrid_output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/hybrid_schedules"
        os.makedirs(hybrid_output_dir, exist_ok=True)
        
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
                json_files = glob.glob(os.path.join(dataset_path, "config_*", "classification_ozone_config_*_seed*.json"))
            else:
                json_files = glob.glob(os.path.join(dataset_path, "config_*", f"{dataset}_config_*_seed*.json"))

            print(f"{model}/{dataset} → Found {len(json_files)} files")

            if len(json_files) == 0:
                continue

            try:
                curves = load_curve_data_from_json_from_files(json_files, max_epochs=max_epochs)
                sim = SuccessiveHalvingSimulator(curves)
                
                hybrid_results = {}
                for schedule_name, schedule in hybrid_schedules.items():
                    custom_hits, custom_regrets, _ = sim.simulate_custom_schedule(schedule, top_k=1)
                    hybrid_results[schedule_name] = {'hits': custom_hits, 'regrets': custom_regrets, 'schedule': schedule}
                    
                    print(f"[{model}/{dataset}] {schedule_name} Hybrid Schedule - Final Hit: {custom_hits[-1]:.3f}")

                save_hybrid_results_csv(hybrid_results, model, dataset, hybrid_output_dir)

                dataset_names.append(dataset)

                for k in top_k_values:
                    hits, regrets = sim.simulate(budget_epochs=budget_epochs, top_k=k)
                    all_hits_by_k[k].append(hits)
                    all_regrets_by_k[k].append(regrets)

            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

        plot_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs, os.path.join(plot_root, model), model, global_regret_ylim)
        save_csv(all_hits_by_k, all_regrets_by_k, budget_epochs, os.path.join(csv_root, model), model, dataset_names)
        plot_k_axis_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs, os.path.join(plot_root, model), model, global_regret_ylim)

# this function creates hybrid schedules by combining budget and dropout schedules
# it takes the budget and dropout schedules as input and produces a set of hybrid schedules
# output is a dictionary mapping schedule names to their (num_configs, epochs) tuples
def create_hybrid_schedules(budget_schedules, dropout_schedules):
    hybrid_schedules = {}
    
    for budget_name, budget_epochs in budget_schedules.items():
        for dropout_name, dropout_configs in dropout_schedules.items():
            # creates hybrid name
            hybrid_name = f"{budget_name}_{dropout_name}"
            
            # combines budget and dropout schedules
            # takes minimum length to avoid index errors
            min_length = min(len(budget_epochs), len(dropout_configs))
            
            # creates schedule as list of (num_configs, epochs) tuples
            hybrid_schedule = []
            for i in range(min_length):
                hybrid_schedule.append((dropout_configs[i], budget_epochs[i]))
            
            # ensures that it ends with 1 configuration at max epochs
            if hybrid_schedule[-1][0] != 1:
                hybrid_schedule.append((1, budget_epochs[-1] if budget_epochs else 100))
            
            hybrid_schedules[hybrid_name] = hybrid_schedule
    
    return hybrid_schedules

# this function saves hybrid results to CSV files
def save_hybrid_results_csv(hybrid_results, model, dataset, output_dir):
    model_dir = os.path.join(output_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    
    csv_path = os.path.join(model_dir, f"{model}_{dataset}_hybrid_results.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hybrid_Schedule", "Round", "Configs", "Epochs", "Hit_Probability", "Regret"])
        
        for schedule_name, results in hybrid_results.items():
            schedule = results['schedule']
            hits = results['hits']
            regrets = results['regrets']
            
            for i, ((num_cfgs, budget), hit, regret) in enumerate(zip(schedule, hits, regrets), 1):
                writer.writerow([schedule_name, i, num_cfgs, budget, round(hit, 4), round(regret, 6)])
    
    summary_csv_path = os.path.join(model_dir, f"{model}_{dataset}_hybrid_summary.csv")
    
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hybrid_Schedule", "Final_Hit_Probability", "Final_Regret", "Total_Rounds"])
        
        for schedule_name, results in hybrid_results.items():
            final_hit = results['hits'][-1] if results['hits'] else 0
            final_regret = results['regrets'][-1] if results['regrets'] else float('inf')
            total_rounds = len(results['schedule'])
            
            writer.writerow([schedule_name, round(final_hit, 4), round(final_regret, 6), total_rounds])

# this function plots boxplots for hit probabilities and regrets
def plot_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs: List[int], output_dir: str, model: str, global_regret_ylim=None):
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
        ax1.set_xticklabels(budget_epochs, fontsize=15)
        ax1.set_title(f"Top-{k} Hit Probability ({model})", fontsize=18, fontweight='bold')
        ax1.set_xlabel("Epoch Budget", fontsize=16)
        ax1.set_ylabel("Hit Probability", fontsize=16)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=16)

        # Regret plot
        ax2 = plt.subplot(1, 2, 2)
        sns.boxplot(data=regrets_array, ax=ax2, palette=pink_palette[:len(budget_epochs)])
        ax2.set_xticks(range(len(budget_epochs)))
        ax2.set_xticklabels(budget_epochs, fontsize=15)
        ax2.set_title(f"Regret ({model})", fontsize=18, fontweight='bold')
        ax2.set_xlabel("Epoch Budget", fontsize=16)
        ax2.set_ylabel("Regret", fontsize=16)
        
        # Apply global regret y-limit if provided
        if global_regret_ylim:
            ax2.set_ylim(global_regret_ylim)
        
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=16)

        plt.suptitle(f"Successive Halving Results for {model} (Top-{k})", fontsize=18, fontweight='bold')
        plt.savefig(os.path.join(output_dir, f"{model}_top{k}_results.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

# this function plots boxplots for hit probabilities and regrets
def plot_k_axis_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs: List[int], output_dir: str, model: str, global_regret_ylim=None):
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
        plt.xticks(ticks=range(len(k_values)), labels=[f"k={k}" for k in k_values], fontsize=12)
        plt.title(f"Probabilities of Selecting Final k-Best (Epoch={budget}) ({model})", fontsize=14, fontweight='bold')
        plt.xlabel("Top-k Value", fontsize=12)
        plt.ylabel("Hit Probability", fontsize=12)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_k_axis_hit_epoch{budget}.pdf"), 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        # Regret
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=regrets_per_k, palette=pink_palette)
        plt.xticks(ticks=range(len(k_values)), labels=[f"k={k}" for k in k_values], fontsize=12)
        plt.title(f"Regret for Final k-Best (Epoch={budget}) ({model})", fontsize=14, fontweight='bold')
        plt.xlabel("Top-k Value", fontsize=12)
        plt.ylabel("Regret", fontsize=12)
        
        # Apply global regret y-limit if provided (remove log scale when using consistent limits)
        if global_regret_ylim:
            plt.ylim(global_regret_ylim)
        else:
            plt.yscale("log")
            plt.ylabel("Regret (log scale)", fontsize=12)
        
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_k_axis_regret_epoch{budget}.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

# this function saves hybrid results to CSV files
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

# runs the main analysis
def main():
    # runs main analysis with hybrid schedules
    print("\nRunning main model analysis with 25 hybrid schedules...")
    run_all_models()
    
    # runs resource constraint analysis
    print("\nCreating resource-constrained analysis...")
    #create_resource_constraint_analysis()

if __name__ == "__main__":
    main()
