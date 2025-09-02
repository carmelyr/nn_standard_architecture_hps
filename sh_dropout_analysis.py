# This module provides utilities for analyzing dropout schedules in neural network training.

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sh_core import (SuccessiveHalvingSimulator, load_curve_data_from_json_from_files)
from sh_visualization import create_dropout_schedule_analysis

pink_palette = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585", "#DB7093", "#FFC0CB"]
sns.set_palette(pink_palette)

# this function creates a comparative analysis of dropout schedules across all models and datasets
def create_comparative_dropout_analysis():
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/comparative_dropout_analysis"
    max_epochs = 100
    
    schedules = {
        "Moderate": [(100, 1), (50, 10), (10, 30), (1, 100)],
        "Aggressive": [(100, 1), (10, 20), (1, 100)],
        "Conservative": [(100, 5), (75, 20), (50, 50), (25, 75), (1, 100)],
        "VeryAggressive": [(100, 1), (5, 50), (1, 100)],
        "Standard": [(100, 1), (50, 5), (25, 15), (12, 30), (6, 50), (3, 70), (1, 100)]
    }
    
    os.makedirs(output_dir, exist_ok=True)

    all_dropout_data = {}           # stores dropout data for all models and datasets

    for model in models:
        print(f"\n=== Analyzing Dropout Schedules for {model} ===")
        base_dir = os.path.join(base_root, model)
        all_dropout_data[model] = {}
        
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
                
                dataset_dropout_data = {}
                for schedule_name, schedule in schedules.items():
                    _, _, dropout_counts = sim.simulate_custom_schedule(schedule, top_k=1)
                    dataset_dropout_data[schedule_name] = dropout_counts
                
                all_dropout_data[model][dataset] = dataset_dropout_data
                print(f"Processed {model}/{dataset}")
                
            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

    create_dropout_summary_plots(all_dropout_data, schedules, output_dir)       # creates dropout summary plots
    save_dropout_summary_csv(all_dropout_data, schedules, output_dir)           # saves dropout summary results to CSV

# this function creates summary plots showing average dropout patterns by model and schedule
def create_dropout_summary_plots(all_dropout_data, schedules, output_dir):
    models = list(all_dropout_data.keys())
    schedule_names = list(schedules.keys())
    
    avg_dropout_data = {}
    
    for model in models:
        avg_dropout_data[model] = {}
        for schedule_name in schedule_names:
            dropout_lists = []
            for dataset_data in all_dropout_data[model].values():
                if schedule_name in dataset_data:
                    dropout_lists.append(dataset_data[schedule_name])
            
            if dropout_lists:
                max_len = max(len(lst) for lst in dropout_lists)
                padded_lists = []
                for lst in dropout_lists:
                    padded = lst + [0] * (max_len - len(lst))
                    padded_lists.append(padded)
                avg_dropout_data[model][schedule_name] = np.mean(padded_lists, axis=0)
            else:
                avg_dropout_data[model][schedule_name] = []
    
    plt.figure(figsize=(12, 8))
    schedule_colors = plt.cm.Set1(np.linspace(0, 1, len(schedule_names)))   # color map for schedules
    schedule_labels_added = set()   # keeps track of added schedule labels
    
    for model in models:
        for j, schedule_name in enumerate(schedule_names):
            if schedule_name in avg_dropout_data[model] and len(avg_dropout_data[model][schedule_name]) > 0:
                rounds = range(1, len(avg_dropout_data[model][schedule_name]) + 1)

                label = None
                if schedule_name not in schedule_labels_added:
                    label = schedule_name
                    schedule_labels_added.add(schedule_name)
                
                plt.plot(rounds, avg_dropout_data[model][schedule_name], 'o-', label=label, color=schedule_colors[j], linewidth=2, markersize=6, alpha=0.8)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xlabel('Round')
    plt.ylabel('Average Configurations Eliminated')
    plt.title('Average Dropout per Round - All Models and Schedules')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_dropout_comparison.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Saved average dropout comparison plot")

# this function saves detailed dropout analysis to CSV files
def save_dropout_summary_csv(all_dropout_data, schedules, output_dir):
    csv_path = os.path.join(output_dir, "comprehensive_dropout_analysis.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Dataset', 'Schedule', 'Round', 'Configs_Eliminated', 'Total_Configs', 'Elimination_Rate_%'])
        
        for model, model_data in all_dropout_data.items():
            for dataset, dataset_data in model_data.items():
                for schedule_name, dropout_counts in dataset_data.items():
                    total_configs = 100
                    remaining = total_configs
                    
                    for round_idx, eliminated in enumerate(dropout_counts):
                        elimination_rate = (eliminated / remaining * 100) if remaining > 0 else 0   # elimination rate calculation
                        writer.writerow([model, dataset, schedule_name, round_idx + 1, eliminated, remaining, f"{elimination_rate:.2f}"])
                        remaining = max(0, remaining - eliminated)
    
    print(f"Saved comprehensive dropout analysis: {csv_path}")

if __name__ == "__main__":
    create_comparative_dropout_analysis()
