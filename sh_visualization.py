# This module contains visualization and plotting utilities for successive halving analysis.

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sh_core import SuccessiveHalvingSimulator

pink_palette = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585", "#DB7093", "#FFC0CB"]
sns.set_palette(pink_palette)

# this function visualizes configuration elimination
def plot_config_elimination(sim: SuccessiveHalvingSimulator, schedule: List[tuple], use_average: bool = True, output_dir: str = None, dataset_name: str = "Dataset"):
    # tracks, which configs survive each round
    current_indices = list(range(sim.configs))
    survival_matrix = np.zeros((sim.configs, len(schedule)), dtype=bool)
    round_labels = []
    
    for round_idx, (num_to_keep, epoch_budget) in enumerate(schedule):
        if not current_indices:
            break
            
        epoch_idx = min(epoch_budget - 1, sim.epochs - 1)
        
        if use_average:
            # averages across all seeds and folds for each configuration
            perf = np.nanmean(sim.curves[current_indices, :, :, epoch_idx], axis=(1, 2))
        else:
            # uses first seed only, but averages across all folds
            perf = np.nanmean(sim.curves[current_indices, 0, :, epoch_idx], axis=1)

        valid_mask = ~np.isnan(perf)        # masks NaN values

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
    fig_height = min(15, max(10, sim.configs * 0.10))   
    
    plt.figure(figsize=(fig_width, fig_height))

    colors = ['#F0F0F0', '#FF1493']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    survival_matrix_plot = survival_matrix[:, :len(round_labels)]
    im = plt.imshow(survival_matrix_plot, cmap=cmap, aspect='auto', interpolation='nearest')
    
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
    
    y_ticks = list(range(sim.configs))
    y_labels = [str(i + 1) for i in y_ticks]  
    plt.yticks(y_ticks, y_labels, fontsize=6) 
    
    plt.gca().set_yticks([i - 0.5 for i in range(1, sim.configs)], minor=True)
    
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
        
        plt.plot([], [], color='red', linewidth=1.5, alpha=0.8, label=f'Final: {len(final_configs)} configs')
        plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    total_eliminated = sim.configs - len(current_indices) if current_indices else sim.configs
    elimination_rate = (total_eliminated / sim.configs) * 100
    plt.text(0.02, 0.98, f'Elimination: {elimination_rate:.1f}%', transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), verticalalignment='top')
    plt.tight_layout(pad=1.0)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if use_average:
            filename = f"{dataset_name}_config_elimination_averaged.pdf"
        else:
            filename = f"{dataset_name}_config_elimination_seed0.pdf"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved elimination plot: {os.path.join(output_dir, filename)}")
    else:
        plt.show()

# this function creates dropout schedule analysis plots
def create_dropout_schedule_analysis(sim: SuccessiveHalvingSimulator, schedules: dict, output_dir: str = None, dataset_name: str = "Dataset"):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # calculates dropout schedules for each strategy
    dropout_analysis = {}
    
    for schedule_name, schedule in schedules.items():
        _, _, dropout_schedule = sim.simulate_custom_schedule(schedule, top_k=1)
        
        # calculates additional metrics
        total_eliminated = sum(dropout_schedule)
        elimination_rates = []
        remaining_configs = sim.configs
        
        for eliminated in dropout_schedule:
            if remaining_configs > 0:
                rate = (eliminated / remaining_configs) * 100
                elimination_rates.append(rate)
                remaining_configs -= eliminated
            else:
                elimination_rates.append(0)
        
        dropout_analysis[schedule_name] = {
            'schedule': schedule,
            'dropout_counts': dropout_schedule,
            'elimination_rates': elimination_rates,
            'total_eliminated': total_eliminated,
            'final_remaining': max(0, sim.configs - total_eliminated)
        }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    colors = plt.cm.Set3(np.linspace(0, 1, len(schedules)))
    
    # Plot 1: Absolute dropout counts per round
    for i, (name, data) in enumerate(dropout_analysis.items()):
        rounds = range(1, len(data['dropout_counts']) + 1)
        ax1.bar([r + i*0.15 for r in rounds], data['dropout_counts'], width=0.15, label=name, color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Configurations Eliminated')
    ax1.set_title(f'Configurations Eliminated per Round\n{dataset_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Elimination rates (percentage of remaining configs eliminated)
    for i, (name, data) in enumerate(dropout_analysis.items()):
        rounds = range(1, len(data['elimination_rates']) + 1)
        ax2.plot(rounds, data['elimination_rates'], 'o-', label=name, color=colors[i], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Elimination Rate (%)')
    ax2.set_title(f'Elimination Rate per Round\n{dataset_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Plot 3: Remaining configurations over rounds
    for i, (name, data) in enumerate(dropout_analysis.items()):
        remaining = [sim.configs]
        for eliminated in data['dropout_counts']:
            remaining.append(remaining[-1] - eliminated)
        rounds = range(len(remaining))
        ax3.plot(rounds, remaining, 'o-', label=name, color=colors[i], linewidth=2, markersize=6)
    
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Remaining Configurations')
    ax3.set_title(f'Remaining Configurations Over Time\n{dataset_name}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Summary statistics
    schedule_names = list(dropout_analysis.keys())
    total_eliminated = [data['total_eliminated'] for data in dropout_analysis.values()]
    final_remaining = [data['final_remaining'] for data in dropout_analysis.values()]
    
    x_pos = np.arange(len(schedule_names))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, total_eliminated, width, label='Total Eliminated', color='lightcoral', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, final_remaining, width, label='Final Remaining', color='lightblue', alpha=0.7)
    
    ax4.set_xlabel('Schedule Strategy')
    ax4.set_ylabel('Number of Configurations')
    ax4.set_title(f'Total Elimination Summary\n{dataset_name}')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(schedule_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    
    if output_dir:
        filename = f"{dataset_name}_dropout_schedule_analysis.pdf"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved dropout schedule analysis: {os.path.join(output_dir, filename)}")
        
        csv_filename = f"{dataset_name}_dropout_schedule_report.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Schedule', 'Round', 'Configs_to_Keep', 'Epochs', 'Configs_Eliminated', 'Elimination_Rate_%', 'Remaining_Configs'])
            
            for schedule_name, data in dropout_analysis.items():
                remaining = sim.configs
                for round_idx, (eliminated, rate) in enumerate(zip(data['dropout_counts'], data['elimination_rates'])):
                    configs_to_keep, epochs = data['schedule'][round_idx] if round_idx < len(data['schedule']) else (0, 0)
                    writer.writerow([schedule_name, round_idx + 1, configs_to_keep, epochs, eliminated, f"{rate:.2f}", remaining - eliminated])
                    remaining -= eliminated
        
        print(f"Saved dropout schedule report: {csv_path}")
    else:
        plt.show()
    
    return dropout_analysis

# this function plots boxplots for hit probabilities and regrets
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
        plt.savefig(os.path.join(output_dir, f"{model}_top{k}_results.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

# this function plots boxplots with k-values on x-axis for each budget epoch.
def plot_k_axis_boxplots(all_hits_by_k, all_regrets_by_k, budget_epochs: List[int], output_dir: str, model: str):
    os.makedirs(output_dir, exist_ok=True)

    for budget_idx, budget in enumerate(budget_epochs):
        hits_per_k = []
        regrets_per_k = []

        for k in sorted(all_hits_by_k.keys()):
            hits_array = np.array(all_hits_by_k[k])
            regrets_array = np.array(all_regrets_by_k[k])

            # only slices if necessary, checks dimensions first
            if hits_array.shape[1] > len(budget_epochs):
                hits_array = hits_array[:, :len(budget_epochs)]
            if regrets_array.shape[1] > len(budget_epochs):
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

        print(f"Budget epochs: {len(budget_epochs)}, Hits array shape: {hits_array.shape}, "f"Regrets array shape: {regrets_array.shape}")

        # Top-k hit
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=hits_per_k, palette=pink_palette)
        plt.xticks(ticks=range(len(k_values)), labels=[f"k={k}" for k in k_values])
        plt.title(f"Probabilities of Selecting Final k-Best (Epoch={budget}) ({model})")
        plt.xlabel("Top-k Value")
        plt.ylabel("Hit Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_k_axis_hit_epoch{budget}.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
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
        plt.savefig(os.path.join(output_dir, f"{model}_k_axis_regret_epoch{budget}.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

# this function plots boxplots for a single schedule.
def plot_boxplots_single_schedule(all_hits_by_k, all_regrets_by_k, output_dir: str, model_schedule: str):
    os.makedirs(output_dir, exist_ok=True)

    for k in all_hits_by_k:
        hits_array = np.array(all_hits_by_k[k])
        regrets_array = np.array(all_regrets_by_k[k])
        
        hits_array = np.nan_to_num(hits_array, nan=0)
        regrets_array = np.nan_to_num(regrets_array, nan=0)

        print(f"Plotting {model_schedule} k={k} | Hits shape: {hits_array.shape} | "f"Regrets shape: {regrets_array.shape}")

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
        plt.savefig(os.path.join(output_dir, f"{model_schedule}_top{k}_results.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

# this function plots boxplots for a single schedule.
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
    plt.savefig(os.path.join(output_dir, f"{model_schedule}_k_axis_hit.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
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
    plt.savefig(os.path.join(output_dir, f"{model_schedule}_k_axis_regret.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# this function saves the results to CSV files.
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

# this function saves the results to CSV files for a single schedule.
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
