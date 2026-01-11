# This script provides resource-constrained analysis for successive halving.
# Simulates performance under fixed timeout or epoch budget constraints.

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple
from sh_core import (SuccessiveHalvingSimulator, load_curve_data_from_json_from_files, compute_epoch_times_from_json_files, generate_synthetic_epoch_times)

pink_palette = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585", "#DB7093", "#FFC0CB"]
sns.set_palette(pink_palette)

# this function creates hybrid schedules by combining budget and dropout schedules
# It takes two dictionaries as input and returns a new dictionary with the combined schedules.
# The combined schedules will have the format: {hybrid_name: [(num_configs, epochs), ...]}
# budget_schedules: Dict of budget allocation strategies
# dropout_schedules: Dict of dropout strategies
def create_hybrid_schedules(budget_schedules, dropout_schedules):
    hybrid_schedules = {}
    
    for budget_name, budget_epochs in budget_schedules.items():
        for dropout_name, dropout_configs in dropout_schedules.items():
            hybrid_name = f"{budget_name}_{dropout_name}"
            min_length = min(len(budget_epochs), len(dropout_configs))
            hybrid_schedule = []
            for i in range(min_length):
                hybrid_schedule.append((dropout_configs[i], budget_epochs[i]))
            hybrid_schedules[hybrid_name] = hybrid_schedule
    
    return hybrid_schedules

# this class simulates resource-constrained scenarios for successive halving
class ResourceConstrainedSimulator(SuccessiveHalvingSimulator):

    # this function simulates successive halving under different resource constraints.
    # It takes a schedule and a set of resource constraints as input.
    # schedule: List of tuples (num_configs, num_epochs) for each round
    # resource_constraints: Dict with resource limits (max_epochs, max_time_hours)
    # top_k: Number of final top configs to consider
    # returns: Dict containing performance metrics under different constraints
    def simulate_with_resource_constraints(self, schedule: List[Tuple[int, int]], resource_constraints: dict, top_k: int = 1):
        results = {}
        
        # simulates under each constraint
        for constraint_name, constraint_value in resource_constraints.items():
            if constraint_name == 'max_epochs':
                perf_at_timeout, perf_at_convergence, runtime_info = self._simulate_epoch_constraint(schedule, constraint_value, top_k)
            elif constraint_name == 'max_time_hours':
                perf_at_timeout, perf_at_convergence, runtime_info = self._simulate_time_constraint(schedule, constraint_value, top_k)
            else:
                continue
                
            results[constraint_name] = {
                'constraint_value': constraint_value,
                'performance_at_timeout': perf_at_timeout,
                'performance_at_convergence': perf_at_convergence,
                'runtime_info': runtime_info
            }
        
        return results

    # this function simulates performance under epoch budget constraint
    def _simulate_epoch_constraint(self, schedule: List[Tuple[int, int]], max_epochs: int, top_k: int):
        # calculates total epochs needed for the full schedule
        total_epochs_needed = 0
        configs_remaining = self.configs
        
        for num_to_keep, epochs_per_config in schedule:
            total_epochs_needed += configs_remaining * epochs_per_config
            configs_remaining = min(num_to_keep, configs_remaining)

        if total_epochs_needed <= max_epochs:
            hits, regrets, dropout_counts = self.simulate_custom_schedule(schedule, top_k)
            perf_at_timeout = hits[-1] if hits else 0.0
            perf_at_convergence = hits[-1] if hits else 0.0
            
            runtime_info = {
                'total_epochs_used': total_epochs_needed,
                'budget_utilization': total_epochs_needed / max_epochs,
                'completed_rounds': len(schedule),
                'early_stopping': False
            }
        else:
            # truncate schedule to fit budget
            truncated_schedule = self._create_truncated_schedule(schedule, max_epochs)
            hits, regrets, dropout_counts = self.simulate_custom_schedule(truncated_schedule, top_k)
            perf_at_timeout = hits[-1] if hits else 0.0
            
            # estimates convergence performance (what we would get with full schedule)
            full_hits, _, _ = self.simulate_custom_schedule(schedule, top_k)
            perf_at_convergence = full_hits[-1] if full_hits else 0.0
            
            runtime_info = {
                'total_epochs_used': max_epochs,
                'budget_utilization': 1.0,
                'completed_rounds': len(truncated_schedule),
                'early_stopping': True
            }
        
        return perf_at_timeout, perf_at_convergence, runtime_info

    # this function simulates performance under time budget constraint
    def _simulate_time_constraint(self, schedule: List[Tuple[int, int]], max_time_hours: float, top_k: int):
        max_time_seconds = max_time_hours * 3600    # converts hours to seconds
        total_time_needed = 0
        configs_remaining = self.configs
        
        for num_to_keep, epochs_per_config in schedule:
            avg_epoch_time = np.nanmean(self.epoch_times[:configs_remaining, :])    # calculates average epoch time
            round_time = configs_remaining * epochs_per_config * avg_epoch_time     # calculates total time for this round
            total_time_needed += round_time                                         # updates total time needed
            configs_remaining = min(num_to_keep, configs_remaining)
        
        # if enough time budget, run full schedule
        if total_time_needed <= max_time_seconds:
            hits, regrets, dropout_counts = self.simulate_custom_schedule(schedule, top_k)
            perf_at_timeout = hits[-1] if hits else 0.0
            perf_at_convergence = hits[-1] if hits else 0.0
            
            runtime_info = {
                'total_time_used_hours': total_time_needed / 3600,
                'budget_utilization': total_time_needed / max_time_seconds,
                'completed_rounds': len(schedule),
                'early_stopping': False
            }
        else:
            # truncate schedule to fit time budget
            truncated_schedule = self._create_time_truncated_schedule(schedule, max_time_seconds)
            hits, regrets, dropout_counts = self.simulate_custom_schedule(truncated_schedule, top_k)
            perf_at_timeout = hits[-1] if hits else 0.0
            
            # estimates convergence performance
            full_hits, _, _ = self.simulate_custom_schedule(schedule, top_k)
            perf_at_convergence = full_hits[-1] if full_hits else 0.0
            
            runtime_info = {
                'total_time_used_hours': max_time_hours,
                'budget_utilization': 1.0,
                'completed_rounds': len(truncated_schedule),
                'early_stopping': True
            }
        
        return perf_at_timeout, perf_at_convergence, runtime_info

    # this function creates a truncated schedule that fits within the epoch budget
    def _create_truncated_schedule(self, schedule: List[Tuple[int, int]], max_epochs: int):
        truncated_schedule = []
        epochs_used = 0
        configs_remaining = self.configs
        
        for num_to_keep, epochs_per_config in schedule:
            epochs_needed = configs_remaining * epochs_per_config
            
            if epochs_used + epochs_needed <= max_epochs:
                # can fit this round completely
                truncated_schedule.append((num_to_keep, epochs_per_config))
                epochs_used += epochs_needed
                configs_remaining = min(num_to_keep, configs_remaining)
            else:
                # partial round - reduce epochs per config
                remaining_budget = max_epochs - epochs_used
                if remaining_budget > 0 and configs_remaining > 0:
                    reduced_epochs = remaining_budget // configs_remaining
                    if reduced_epochs > 0:
                        truncated_schedule.append((num_to_keep, reduced_epochs))
                break
        
        return truncated_schedule

    # this function creates a truncated schedule that fits within the time budget
    def _create_time_truncated_schedule(self, schedule: List[Tuple[int, int]], max_time_seconds: float):
        truncated_schedule = []
        time_used = 0
        configs_remaining = self.configs
        
        for num_to_keep, epochs_per_config in schedule:
            avg_epoch_time = np.nanmean(self.epoch_times[:configs_remaining, :])
            time_needed = configs_remaining * epochs_per_config * avg_epoch_time
            
            if time_used + time_needed <= max_time_seconds:
                # can fit this round completely
                truncated_schedule.append((num_to_keep, epochs_per_config))
                time_used += time_needed
                configs_remaining = min(num_to_keep, configs_remaining)
            else:
                # partial round - reduce epochs per config
                remaining_time = max_time_seconds - time_used
                if remaining_time > 0 and configs_remaining > 0:
                    reduced_epochs = int(remaining_time / (configs_remaining * avg_epoch_time))
                    if reduced_epochs > 0:
                        truncated_schedule.append((num_to_keep, reduced_epochs))
                break
        
        return truncated_schedule

# this function creates a resource constraint analysis
# it uses real epoch times if available; otherwise, falls back to synthetic.
def create_resource_constraint_analysis():
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/resource_constraint_analysis"
    max_epochs = 100

    # budget schedules
    schedules = {
        "Moderate": [(100, 1), (50, 10), (10, 30), (1, 100)],  
        "Aggressive": [(100, 1), (10, 20), (1, 100)],
        "Conservative": [(100, 5), (75, 20), (50, 50), (25, 75), (1, 100)],
        "VeryAggressive": [(100, 1), (5, 50), (1, 100)],
        "Standard": [(100, 1), (50, 5), (25, 15), (12, 30), (6, 50), (3, 70), (1, 100)]
    }

    # resource constraints
    resource_constraints = {
        'max_epochs_1000': 1000,
        'max_epochs_2000': 2000,
        'max_epochs_5000': 5000,
        'max_time_hours_10': 10,
        'max_time_hours_24': 24,
        'max_time_hours_72': 72
    }

    os.makedirs(output_dir, exist_ok=True)
    all_constraint_results = {}

    for model in models:
        print(f"\n=== Analyzing Resource Constraints for {model} ===")
        base_dir = os.path.join(base_root, model)
        all_constraint_results[model] = {}

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
                curves = load_curve_data_from_json_from_files(json_files, max_epochs=max_epochs)    # loads curve data
                epoch_times = compute_epoch_times_from_json_files(json_files)                       # computes epoch times

                # fallback to synthetic if epoch_times is None
                if epoch_times is None:
                    epoch_times = generate_synthetic_epoch_times(curves.shape[0], curves.shape[1], model)

                sim = ResourceConstrainedSimulator(curves, epoch_times=epoch_times)     # initializes the simulator

                dataset_results = {}

                for schedule_name, schedule in schedules.items():
                    print(f"Processing {model}/{dataset} - {schedule_name}")
                    runtime_stats = sim.compute_schedule_runtime(schedule)

                    constraint_results = {}     # stores results for each constraint

                    # epoch budgets
                    for key, value in resource_constraints.items():
                        if key.startswith('max_epochs_'):
                            res = sim.simulate_with_resource_constraints(schedule, {'max_epochs': value}, top_k=1)
                            constraint_results[f'max_epochs_{value}'] = res['max_epochs']

                    # time budgets
                    for key, value in resource_constraints.items():
                        if key.startswith('max_time_hours_'):
                            res = sim.simulate_with_resource_constraints(schedule, {'max_time_hours': value}, top_k=1)
                            constraint_results[f'max_time_hours_{value}'] = res['max_time_hours']

                    dataset_results[schedule_name] = {
                        'runtime_stats': runtime_stats,
                        'constraint_results': constraint_results
                    }

                all_constraint_results[model][dataset] = dataset_results
                print(f"Completed {model}/{dataset}")

            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

    # visualizations and a analysis csv report
    create_runtime_comparison_plots(all_constraint_results, schedules, output_dir)
    create_constraint_performance_plots(all_constraint_results, schedules, resource_constraints, output_dir)
    save_constraint_analysis_csv(all_constraint_results, schedules, output_dir)
    create_resource_constrained_plot_622(all_constraint_results, schedules, output_dir)

    return all_constraint_results

# this function creates resource-constrained plots for the analysis
def create_resource_constrained_plot_622(all_results, schedules, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    models = list(all_results.keys())
    schedule_names = list(schedules.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(schedule_names)))

    constraint_scenarios = [
        ('max_epochs', 1000, '1000 Epochs'),
        ('max_epochs', 2000, '2000 Epochs'),
        ('max_epochs', 5000, '5000 Epochs'),
        ('max_time_hours', 10, '10 Hours'),
        ('max_time_hours', 24, '24 Hours'),
        ('max_time_hours', 72, '72 Hours')
    ]

    for model_idx, model in enumerate(models):
        if model_idx >= len(axes.flatten()):
            break

        ax = axes.flatten()[model_idx]

        for schedule_idx, schedule_name in enumerate(schedule_names):
            timeout_performances = []
            convergence_performances = []
            scenario_labels = []

            for constraint_type, constraint_value, label in constraint_scenarios:
                key = f"{constraint_type}_{constraint_value}"
                scenario_timeout = []
                scenario_convergence = []

                for dataset_data in all_results[model].values():
                    if schedule_name not in dataset_data:
                        continue
                    cres = dataset_data[schedule_name]['constraint_results']
                    if key in cres:
                        result = cres[key]
                        scenario_timeout.append(result['performance_at_timeout'])
                        scenario_convergence.append(result['performance_at_convergence'])

                if scenario_timeout and scenario_convergence:
                    timeout_performances.append(float(np.mean(scenario_timeout)))
                    convergence_performances.append(float(np.mean(scenario_convergence)))
                    scenario_labels.append(label)

            if timeout_performances and convergence_performances:
                ax.scatter(convergence_performances, timeout_performances,
                           c=[colors[schedule_idx]], label=schedule_name,
                           alpha=0.7, s=80, marker='o')
                max_val = max(max(convergence_performances), max(timeout_performances))
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Performance at Convergence')
        ax.set_ylabel('Performance at Timeout')
        ax.set_title(f'{model}\nResource-Constrained Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    # hides unused subplots
    if len(models) < len(axes.flatten()):
        axes.flatten()[-1].set_visible(False)

    plt.suptitle('Resource-Constrained Analysis\nTimeout vs Convergence Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_constrained_analysis.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # creates summary table
    create_constraint_summary_table(all_results, schedules, output_dir)
    print("Saved resource-constrained analysis plot")

# this function creates a summary table for the resource-constrained analysis
def create_constraint_summary_table(all_results, schedules, output_dir):
    summary_data = []
    
    for model, model_data in all_results.items():
        for dataset, dataset_data in model_data.items():
            for schedule_name, schedule_data in dataset_data.items():
                constraint_results = schedule_data['constraint_results']
                
                for constraint_type, constraint_data in constraint_results.items():
                    summary_data.append({
                        'Model': model,
                        'Dataset': dataset,
                        'Schedule': schedule_name,
                        'Constraint_Type': constraint_type,
                        'Constraint_Value': constraint_data['constraint_value'],
                        'Performance_Timeout': constraint_data['performance_at_timeout'],
                        'Performance_Convergence': constraint_data['performance_at_convergence'],
                        'Performance_Gap': constraint_data['performance_at_convergence'] - constraint_data['performance_at_timeout'],
                        'Budget_Utilization': constraint_data['runtime_info']['budget_utilization'],
                        'Early_Stopping': constraint_data['runtime_info']['early_stopping']
                    })
    
    if summary_data:
        df = pd.DataFrame(summary_data)

        # creates aggregated summary by model and constraint
        pivot_summary = df.groupby(['Model', 'Constraint_Type', 'Schedule']).agg({'Performance_Gap': ['mean', 'std'], 'Budget_Utilization': 'mean', 'Early_Stopping': 'mean'}).round(4)
        
        # csv
        summary_csv = os.path.join(output_dir, "constraint_summary_table.csv")
        pivot_summary.to_csv(summary_csv)
        print(f"Saved constraint summary table: {summary_csv}")
        
        # creates visualization of the summary
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # this function replaces near-zero values with exact zero to avoid -0.000 display
        def clean_zero_values(data):
            cleaned = data.copy()
            cleaned[abs(cleaned) < 1e-10] = 0.0
            return cleaned
        
        # performance gap by model and constraint
        gap_data = df.groupby(['Model', 'Constraint_Type'])['Performance_Gap'].mean().unstack()     # average performance gap
        gap_data_cleaned = clean_zero_values(gap_data)                                              # replaces near-zero values with exact zero
        sns.heatmap(gap_data_cleaned, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0])           # creates heatmap
        axes[0].set_title('Average Performance Gap by Model and Constraint')
        axes[0].set_ylabel('Model')

        # budget utilization by model and constraint
        util_data = df.groupby(['Model', 'Constraint_Type'])['Budget_Utilization'].mean().unstack() # average budget utilization
        util_data_cleaned = clean_zero_values(util_data)                                            # replaces near-zero values with exact zero
        sns.heatmap(util_data_cleaned, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1])            # creates heatmap
        axes[1].set_title('Average Budget Utilization by Model and Constraint')
        axes[1].set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "constraint_summary_heatmaps.pdf"), 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print("Saved constraint summary heatmaps")

# this function creates runtime comparison plots
def create_runtime_comparison_plots(all_results, schedules, output_dir):
    runtime_dir = os.path.join(output_dir, "runtime_comparisons")
    os.makedirs(runtime_dir, exist_ok=True)
    models = list(all_results.keys())
    schedule_names = list(schedules.keys())
    runtime_data = {}
    
    for model in models:
        runtime_data[model] = {}
        for schedule_name in schedule_names:
            total_times = []
            total_epochs = []
            computational_savings = []
            
            for dataset_data in all_results[model].values():
                if schedule_name in dataset_data:
                    stats = dataset_data[schedule_name]['runtime_stats']['summary']
                    total_times.append(stats['total_time_hours'])
                    total_epochs.append(stats['total_epochs'])
                    computational_savings.append(stats['computational_savings'])
            
            runtime_data[model][schedule_name] = {
                'total_times': total_times,                     # list of total times for each dataset
                'total_epochs': total_epochs,                   # list of total epochs for each dataset
                'computational_savings': computational_savings  # list of computational savings for each dataset
            }
    
    # Plot 1: total runtime by model and schedule
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # runtime comparison (top left plot)
    ax1 = axes[0, 0]
    x_pos = np.arange(len(models))
    width = 0.15
    colors = plt.cm.Set3(np.linspace(0, 1, len(schedule_names)))

    # creates a grouped bar chart
    for i, schedule_name in enumerate(schedule_names):
        # calculates average times per model
        avg_times = [np.mean(runtime_data[model][schedule_name]['total_times']) if runtime_data[model][schedule_name]['total_times'] else 0 for model in models]
        ax1.bar(x_pos + i*width, avg_times, width, label=schedule_name, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Runtime (hours)')
    ax1.set_title('Average Runtime by Model and Schedule')
    ax1.set_xticks(x_pos + width * 2)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # computational savings (top right plot)
    ax2 = axes[0, 1]
    for i, schedule_name in enumerate(schedule_names):
        avg_savings = [np.mean(runtime_data[model][schedule_name]['computational_savings']) if runtime_data[model][schedule_name]['computational_savings'] else 0 for model in models]
        ax2.bar(x_pos + i*width, [s*100 for s in avg_savings], width, label=schedule_name, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Computational Savings (%)')
    ax2.set_title('Computational Savings by Model and Schedule')
    ax2.set_xticks(x_pos + width * 2)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # runtime distribution (bottom left plot)
    ax3 = axes[1, 0]
    all_runtimes_by_schedule = []
    labels = []
    for schedule_name in schedule_names:
        schedule_runtimes = []
        for model in models:
            schedule_runtimes.extend(runtime_data[model][schedule_name]['total_times'])
        all_runtimes_by_schedule.append(schedule_runtimes)
        labels.append(schedule_name)
    
    ax3.boxplot(all_runtimes_by_schedule, labels=labels)
    ax3.set_ylabel('Runtime (hours)')
    ax3.set_title('Runtime Distribution by Schedule')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # epochs vs time scatter (bottom right plot)
    ax4 = axes[1, 1]
    for i, schedule_name in enumerate(schedule_names):
        all_epochs = []
        all_times = []
        for model in models:
            all_epochs.extend(runtime_data[model][schedule_name]['total_epochs'])
            all_times.extend(runtime_data[model][schedule_name]['total_times'])
        
        if all_epochs and all_times:
            ax4.scatter(all_epochs, all_times, label=schedule_name, color=colors[i], alpha=0.6, s=30)
    
    ax4.set_xlabel('Total Epochs')
    ax4.set_ylabel('Total Time (hours)')
    ax4.set_title('Epochs vs Runtime Relationship')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(runtime_dir, "runtime_comparison_analysis.pdf"), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Saved runtime comparison analysis")

# this function creates plots showing performance under different resource constraints
def create_constraint_performance_plots(all_results, schedules, constraints, output_dir):
    constraint_dir = os.path.join(output_dir, "constraint_performance")
    os.makedirs(constraint_dir, exist_ok=True)

    models = list(all_results.keys())
    schedule_names = list(schedules.keys())

    for constraint_type in ['max_epochs', 'max_time_hours']:
        # extract numeric values from constraints dict in ascending order
        constraint_values = sorted(int(v) for k, v in constraints.items() if k.startswith(constraint_type))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for model_idx, model in enumerate(models):
            if model_idx >= len(axes):
                break
            ax = axes[model_idx]

            for schedule_name in schedule_names:
                avg_timeout_by_value = []
                avg_convergence_by_value = []

                for val in constraint_values:
                    key = f"{constraint_type}_{val}"
                    timeout_perfs = []
                    convergence_perfs = []
                    for dataset_data in all_results[model].values():
                        if schedule_name not in dataset_data:
                            continue
                        cres = dataset_data[schedule_name]['constraint_results']
                        if key in cres:
                            timeout_perfs.append(cres[key]['performance_at_timeout'])
                            convergence_perfs.append(cres[key]['performance_at_convergence'])
                    avg_timeout_by_value.append(np.mean(timeout_perfs) if timeout_perfs else np.nan)
                    avg_convergence_by_value.append(np.mean(convergence_perfs) if convergence_perfs else np.nan)

                if np.isfinite(avg_timeout_by_value).any():
                    ax.plot(constraint_values, avg_timeout_by_value, 'o-', label=f'{schedule_name} (Timeout)', alpha=0.7)
                if np.isfinite(avg_convergence_by_value).any():
                    ax.plot(constraint_values, avg_convergence_by_value, 's--', label=f'{schedule_name} (Convergence)', alpha=0.7)

            ax.set_xlabel(f'{constraint_type.replace("_", " ").title()}')
            ax.set_ylabel('Performance (Hit Probability)')
            ax.set_title(f'{model} - Performance vs {constraint_type.replace("_", " ").title()}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(constraint_dir, f"performance_vs_{constraint_type}.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved constraint performance plot for {constraint_type}")

# this function saves constraint analysis results to CSV files
def save_constraint_analysis_csv(all_results, schedules, output_dir):
    csv_dir = os.path.join(output_dir, "csv_reports")
    os.makedirs(csv_dir, exist_ok=True)

    # runtime analysis CSV
    runtime_csv = os.path.join(csv_dir, "runtime_analysis.csv")
    with open(runtime_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Schedule', 'Total_Epochs', 'Total_Time_Hours', 'Total_Time_Days', 'Computational_Savings_%', 'Rounds_Completed'])

        for model, model_data in all_results.items():
            for dataset, dataset_data in model_data.items():
                for schedule_name, schedule_data in dataset_data.items():
                    stats = schedule_data['runtime_stats']['summary']
                    writer.writerow([model, dataset, schedule_name, stats['total_epochs'], f"{stats['total_time_hours']:.2f}", f"{stats['total_time_days']:.2f}", f"{stats['computational_savings']*100:.2f}", stats['rounds_completed']])

    # constraint performance CSV
    constraint_csv = os.path.join(csv_dir, "constraint_performance.csv")
    with open(constraint_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Schedule', 'Constraint_Key', 'Constraint_Value', 'Performance_At_Timeout', 'Performance_At_Convergence', 'Budget_Utilization', 'Early_Stopping'])

        for model, model_data in all_results.items():
            for dataset, dataset_data in model_data.items():
                for schedule_name, schedule_data in dataset_data.items():
                    for key, constraint_data in schedule_data['constraint_results'].items():
                        writer.writerow([model, dataset, schedule_name, key, constraint_data['constraint_value'], f"{constraint_data['performance_at_timeout']:.4f}", f"{constraint_data['performance_at_convergence']:.4f}", f"{constraint_data['runtime_info']['budget_utilization']:.4f}", constraint_data['runtime_info']['early_stopping']])

    print(f"Saved runtime analysis CSV: {runtime_csv}")
    print(f"Saved constraint performance CSV: {constraint_csv}")

# this function creates the hybrid resource constraint analysis
# uses all hybrid schedules (25) under resource constraints
def create_hybrid_resource_constraint_analysis():
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/hybrid_constraint_analysis"
    max_epochs = 100

    # creates all 25 hybrid schedules from budget and dropout schedules (each combination)
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
    print(f"Testing ALL {len(hybrid_schedules)} hybrid schedules under constraints")

    resource_constraints = {'max_epochs': [1000, 2000, 5000], 'max_time_hours': [10, 24, 72]}

    os.makedirs(output_dir, exist_ok=True)
    all_hybrid_results = {}

    for model in models:
        print(f"\n=== Analyzing {model} with ALL Hybrid Schedules ===")
        base_dir = os.path.join(base_root, model)
        all_hybrid_results[model] = {}

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
                curves = load_curve_data_from_json_from_files(json_files, max_epochs=max_epochs)            # loads training curves
                epoch_times = compute_epoch_times_from_json_files(json_files)                               # computes epoch times

                if epoch_times is None:
                    epoch_times = generate_synthetic_epoch_times(curves.shape[0], curves.shape[1], model)   # generates synthetic epoch times if not found

                sim = ResourceConstrainedSimulator(curves, epoch_times=epoch_times)     # initializes the simulator
                dataset_results = {}

                # tests all 25 hybrid schedules
                for hybrid_name, hybrid_schedule in hybrid_schedules.items():
                    print(f"  Processing {hybrid_name} on {dataset}")
                    
                    hybrid_constraint_results = {}
                    
                    # tests under all resource constraints
                    for constraint_type, constraint_values in resource_constraints.items():
                        for constraint_value in constraint_values:
                            try:
                                res = sim.simulate_with_resource_constraints(hybrid_schedule, {constraint_type: constraint_value}, top_k=1)
                                key = f"{constraint_type}_{constraint_value}"
                                hybrid_constraint_results[key] = res[constraint_type]
                            except Exception as e:
                                print(f"Error with {constraint_type}={constraint_value}: {e}")

                    # stores results for this hybrid
                    dataset_results[hybrid_name] = {'constraint_results': hybrid_constraint_results, 'schedule': hybrid_schedule}

                all_hybrid_results[model][dataset] = dataset_results
                print(f"  Completed {dataset} with {len(hybrid_schedules)} hybrids")

            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

    analyze_hybrid_performance_patterns(all_hybrid_results, output_dir)     # analyzes hybrid performance patterns
    find_best_hybrids_per_constraint(all_hybrid_results, output_dir)        # finds best hybrids per constraint
    create_hybrid_robustness_ranking(all_hybrid_results, output_dir)        # creates hybrid robustness ranking

    comprehensive_df = save_all_hybrids_per_constraint_csv(all_hybrid_results, output_dir)  # saves comprehensive results
    
    if comprehensive_df is not None:
        create_hybrid_pivot_tables(comprehensive_df, output_dir)       # creates hybrid pivot tables

    return all_hybrid_results

# this function analyzes hybrid performance patterns
def analyze_hybrid_performance_patterns(all_hybrid_results, output_dir):
    
    # separate budget and dropout effects
    budget_performance = {}  
    dropout_performance = {}
    
    for model, model_data in all_hybrid_results.items():
        for dataset, dataset_data in model_data.items():
            for hybrid_name, hybrid_data in dataset_data.items():
                budget_type, dropout_type = hybrid_name.split('_', 1)
                
                # gets average performance under tight constraints
                tight_constraint_perf = []
                for key, result in hybrid_data['constraint_results'].items():
                    if 'max_epochs_1000' in key or 'max_time_hours_10' in key:
                        tight_constraint_perf.append(result['performance_at_timeout'])

                # gets average performance under loose constraints
                loose_constraint_perf = []
                for key, result in hybrid_data['constraint_results'].items():
                    if 'max_epochs_1000' not in key and 'max_time_hours_10' not in key:
                        loose_constraint_perf.append(result['performance_at_timeout'])

                if tight_constraint_perf:
                    avg_perf = np.mean(tight_constraint_perf)
                    
                    if budget_type not in budget_performance:
                        budget_performance[budget_type] = []
                    budget_performance[budget_type].append(avg_perf)
                
                    if dropout_type not in dropout_performance:
                        dropout_performance[dropout_type] = []
                    dropout_performance[dropout_type].append(avg_perf)
    
    # Calculate means
    budget_means = {k: np.mean(v) for k, v in budget_performance.items()}
    dropout_means = {k: np.mean(v) for k, v in dropout_performance.items()}
    
    # plots budget strategy effectiveness
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(budget_means.keys(), budget_means.values(), color=pink_palette)
    ax1.set_title('Budget Strategy Performance Under Tight Constraints', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average Performance', fontsize=14)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim(0.0, 1.0)  # Fixed y-axis limits from 0 to 1
    ax1.grid(True, alpha=0.3)  # Add grid for better readability
    
    ax2.bar(dropout_means.keys(), dropout_means.values(), color=pink_palette)
    ax2.set_title('Dropout Strategy Performance Under Tight Constraints', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Average Performance', fontsize=14)
    ax2.tick_params(axis='x', rotation=45, labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylim(0.0, 1.0)  # Fixed y-axis limits from 0 to 1
    ax2.grid(True, alpha=0.3)  # Add grid for better readability
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_strategy_effectiveness.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("Best budget strategies under constraints:")
    for strategy, perf in sorted(budget_means.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {perf:.3f}")
        
    print("\nBest dropout strategies under constraints:")
    for strategy, perf in sorted(dropout_means.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {perf:.3f}")

# this function finds the best hybrid for each constraint
def find_best_hybrids_per_constraint(all_hybrid_results, output_dir):
    
    constraint_winners = {}  # constraint -> (hybrid_name, avg_performance)
    
    # groups by constraint type and value
    for model, model_data in all_hybrid_results.items():
        for dataset, dataset_data in model_data.items():
            for hybrid_name, hybrid_data in dataset_data.items():
                for constraint_key, result in hybrid_data['constraint_results'].items():
                    perf = result['performance_at_timeout']
                    
                    if constraint_key not in constraint_winners:
                        constraint_winners[constraint_key] = {}
                    
                    if hybrid_name not in constraint_winners[constraint_key]:
                        constraint_winners[constraint_key][hybrid_name] = []
                    
                    constraint_winners[constraint_key][hybrid_name].append(perf)

    # finds the best hybrid for each constraint
    final_winners = {}
    for constraint_key, hybrid_perfs in constraint_winners.items():
        hybrid_averages = {hybrid: np.mean(perfs) for hybrid, perfs in hybrid_perfs.items()}
        best_hybrid = max(hybrid_averages, key=hybrid_averages.get)
        final_winners[constraint_key] = (best_hybrid, hybrid_averages[best_hybrid])
    
    with open(os.path.join(output_dir, "best_hybrids_per_constraint.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Constraint', 'Best_Hybrid', 'Average_Performance', 'Budget_Strategy', 'Dropout_Strategy'])
        
        for constraint, (hybrid, perf) in final_winners.items():
            budget, dropout = hybrid.split('_', 1)
            writer.writerow([constraint, hybrid, f"{perf:.4f}", budget, dropout])
    
    print("Best hybrid per constraint:")
    for constraint, (hybrid, perf) in sorted(final_winners.items()):
        print(f"  {constraint}: {hybrid} ({perf:.3f})")

# this function creates a ranking of hybrids based on their robustness
def create_hybrid_robustness_ranking(all_hybrid_results, output_dir):
    
    hybrid_robustness = {}  # hybrid_name -> robustness_score
    
    for model, model_data in all_hybrid_results.items():
        for dataset, dataset_data in model_data.items():
            for hybrid_name, hybrid_data in dataset_data.items():
                performance_gaps = []
                
                for constraint_key, result in hybrid_data['constraint_results'].items():
                    perf_timeout = result['performance_at_timeout']
                    perf_convergence = result['performance_at_convergence']
                    
                    # ensures convergence performance is at least as good as timeout
                    perf_convergence = max(perf_timeout, perf_convergence)
                    
                    # calculates relative performance loss (0 = no loss, 1 = total loss)
                    if perf_convergence > 0:
                        relative_gap = (perf_convergence - perf_timeout) / perf_convergence
                    else:
                        relative_gap = 0.0
                    
                    # clamps to [0, 1] range
                    relative_gap = max(0.0, min(1.0, relative_gap))
                    performance_gaps.append(relative_gap)
                
                if performance_gaps:
                    # Robustness = 1 - average_relative_gap
                    # in [0, 1] range
                    robustness_score = 1.0 - np.mean(performance_gaps)
                    
                    if hybrid_name not in hybrid_robustness:
                        hybrid_robustness[hybrid_name] = []
                    hybrid_robustness[hybrid_name].append(robustness_score)
    
    # average robustness across all models/datasets
    final_robustness = {hybrid: np.mean(scores) for hybrid, scores in hybrid_robustness.items()}
    
    # creates robustness ranking
    robustness_ranking = sorted(final_robustness.items(), key=lambda x: x[1], reverse=True)
    
    # saves ranking with validation
    with open(os.path.join(output_dir, "hybrid_robustness_ranking.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Hybrid', 'Robustness_Score', 'Budget_Strategy', 'Dropout_Strategy', 'Score_Valid'])
        
        for rank, (hybrid, score) in enumerate(robustness_ranking, 1):
            budget, dropout = hybrid.split('_', 1)
            score_valid = 0.0 <= score <= 1.0
            writer.writerow([rank, hybrid, f"{score:.4f}", budget, dropout, score_valid])
    
    print("Top 10 most robust hybrids:")
    for rank, (hybrid, score) in enumerate(robustness_ranking[:10], 1):
        print(f"  {rank}. {hybrid}: {score:.3f}")
    
    # checks for invalid scores
    invalid_scores = [(hybrid, score) for hybrid, score in final_robustness.items() 
                     if not (0.0 <= score <= 1.0)]
    if invalid_scores:
        print(f"\nWarning: {len(invalid_scores)} hybrids have invalid robustness scores:")
        for hybrid, score in invalid_scores[:5]:
            print(f"  {hybrid}: {score:.3f}")
        
    # plots robustness distribution
    plt.figure(figsize=(12, 8))
    hybrids, scores = zip(*robustness_ranking)
    
    # color bars by validity
    colors = [pink_palette[0] if 0.0 <= score <= 1.0 else 'red' for score in scores]
    
    plt.bar(range(len(hybrids)), scores, color=colors)
    plt.xlabel('Hybrid Rank')
    plt.ylabel('Robustness Score')
    plt.title('Hybrid Schedule Robustness Ranking\n(Red bars indicate invalid scores)')
    plt.xticks(range(0, len(hybrids), 5), range(1, len(hybrids)+1, 5))
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Max Valid Score')
    plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, label='Min Valid Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_robustness_distribution.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# this function saves all hybrids' performance under all constraints to a CSV file
def save_all_hybrids_per_constraint_csv(all_hybrid_results, output_dir):
    print("\n=== Saving All Hybrids Per Constraint CSV ===")
    
    all_data = []
    
    # collects all hybrid performances under all constraints
    for model, model_data in all_hybrid_results.items():
        for dataset, dataset_data in model_data.items():
            for hybrid_name, hybrid_data in dataset_data.items():
                budget_strategy, dropout_strategy = hybrid_name.split('_', 1)
                
                for constraint_key, result in hybrid_data['constraint_results'].items():
                    # parses constraint type and value
                    if constraint_key.startswith('max_epochs_'):
                        constraint_type = 'max_epochs'
                        constraint_value = int(constraint_key.split('_')[-1])
                    elif constraint_key.startswith('max_time_hours_'):
                        constraint_type = 'max_time_hours'
                        constraint_value = int(constraint_key.split('_')[-1])
                    else:
                        continue
                    
                    # calculates performance metrics
                    perf_timeout = result['performance_at_timeout']
                    perf_convergence = result['performance_at_convergence']
                    
                    # ensures convergence >= timeout
                    perf_convergence = max(perf_timeout, perf_convergence)

                    # calculates performance gap and retention
                    performance_gap = perf_convergence - perf_timeout
                    if perf_convergence > 0:
                        performance_retention = perf_timeout / perf_convergence
                        relative_gap = performance_gap / perf_convergence
                    else:
                        performance_retention = 1.0
                        relative_gap = 0.0

                    runtime_info = result['runtime_info']
                    
                    all_data.append({
                        'Model': model,
                        'Dataset': dataset,
                        'Hybrid_Name': hybrid_name,
                        'Budget_Strategy': budget_strategy,
                        'Dropout_Strategy': dropout_strategy,
                        'Constraint_Type': constraint_type,
                        'Constraint_Value': constraint_value,
                        'Performance_At_Timeout': perf_timeout,
                        'Performance_At_Convergence': perf_convergence,
                        'Performance_Gap': performance_gap,
                        'Performance_Retention': performance_retention,
                        'Relative_Performance_Gap': relative_gap,
                        'Budget_Utilization': runtime_info.get('budget_utilization', 0),
                        'Early_Stopping': runtime_info.get('early_stopping', False),
                        'Completed_Rounds': runtime_info.get('completed_rounds', 0)
                    })
    
    if all_data:
        # creates DataFrame for easier manipulation
        df = pd.DataFrame(all_data)
        
        # saves comprehensive CSV
        comprehensive_csv = os.path.join(output_dir, "all_hybrids_per_constraint_comprehensive.csv")
        df.to_csv(comprehensive_csv, index=False, float_format='%.6f')
        print(f"Saved comprehensive hybrid analysis: {comprehensive_csv}")
        
        # creates summary statistics
        create_hybrid_constraint_summary_tables(df, output_dir)
        
        return df
    else:
        print("No data to save!")
        return None

# this function creates summary tables from the comprehensive hybrid data
def create_hybrid_constraint_summary_tables(df, output_dir):
    # 1. Average performance by hybrid across all constraints
    hybrid_avg = df.groupby(['Hybrid_Name', 'Budget_Strategy', 'Dropout_Strategy']).agg({
        'Performance_At_Timeout': ['mean', 'std', 'min', 'max'],
        'Performance_Retention': ['mean', 'std'],
        'Budget_Utilization': 'mean',
        'Early_Stopping': 'mean'
    }).round(4)

    hybrid_avg.columns = ['_'.join(col).strip() for col in hybrid_avg.columns]
    hybrid_avg = hybrid_avg.reset_index()
    hybrid_avg.to_csv(os.path.join(output_dir, "hybrid_performance_summary.csv"), index=False)
    
    # 2. Performance by constraint type and value
    constraint_summary = df.groupby(['Constraint_Type', 'Constraint_Value', 'Hybrid_Name']).agg({
        'Performance_At_Timeout': 'mean',
        'Performance_Retention': 'mean',
        'Budget_Utilization': 'mean'
    }).round(4).reset_index()
    
    constraint_summary.to_csv(os.path.join(output_dir, "performance_by_constraint.csv"), index=False)
    
    # 3. Best and worst hybrids per constraint
    best_worst_by_constraint = []
    
    for (constraint_type, constraint_value), group in df.groupby(['Constraint_Type', 'Constraint_Value']):
        avg_perf = group.groupby('Hybrid_Name')['Performance_At_Timeout'].mean()
        
        best_hybrid = avg_perf.idxmax()
        best_performance = avg_perf.max()
        worst_hybrid = avg_perf.idxmin()
        worst_performance = avg_perf.min()
        
        best_worst_by_constraint.append({
            'Constraint_Type': constraint_type,
            'Constraint_Value': constraint_value,
            'Best_Hybrid': best_hybrid,
            'Best_Performance': best_performance,
            'Worst_Hybrid': worst_hybrid,
            'Worst_Performance': worst_performance,
            'Performance_Range': best_performance - worst_performance
        })
    
    best_worst_df = pd.DataFrame(best_worst_by_constraint)
    best_worst_df.to_csv(os.path.join(output_dir, "best_worst_hybrids_per_constraint.csv"), index=False, float_format='%.4f')
    
    # 4. Budget vs Dropout strategy comparison
    budget_comparison = df.groupby(['Budget_Strategy', 'Constraint_Type', 'Constraint_Value']).agg({
        'Performance_At_Timeout': 'mean',
        'Performance_Retention': 'mean',
        'Budget_Utilization': 'mean'
    }).round(4).reset_index()
    
    budget_comparison.to_csv(os.path.join(output_dir, "budget_strategy_comparison.csv"), index=False)
    
    dropout_comparison = df.groupby(['Dropout_Strategy', 'Constraint_Type', 'Constraint_Value']).agg({
        'Performance_At_Timeout': 'mean',
        'Performance_Retention': 'mean',
        'Budget_Utilization': 'mean'
    }).round(4).reset_index()
    
    dropout_comparison.to_csv(os.path.join(output_dir, "dropout_strategy_comparison.csv"), index=False)
    
    # 5. Model-specific hybrid rankings
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        model_ranking = model_df.groupby('Hybrid_Name').agg({
            'Performance_At_Timeout': 'mean',
            'Performance_Retention': 'mean',
            'Budget_Utilization': 'mean'
        }).round(4).sort_values('Performance_At_Timeout', ascending=False).reset_index()
        
        model_ranking['Rank'] = range(1, len(model_ranking) + 1)
        model_ranking.to_csv(os.path.join(output_dir, f"hybrid_ranking_{model}.csv"), index=False)

# this function creates pivot tables for easy analysis of hybrid performance
def create_hybrid_pivot_tables(df, output_dir):    
    # Pivot 1: Hybrids vs Constraints (Performance at Timeout)
    pivot1 = df.pivot_table(
        values='Performance_At_Timeout',
        index='Hybrid_Name',
        columns=['Constraint_Type', 'Constraint_Value'],
        aggfunc='mean'
    ).round(4)
    
    pivot1.to_csv(os.path.join(output_dir, "pivot_hybrids_vs_constraints_timeout.csv"))
    
    # Pivot 2: Hybrids vs Constraints (Performance Retention)
    pivot2 = df.pivot_table(
        values='Performance_Retention',
        index='Hybrid_Name', 
        columns=['Constraint_Type', 'Constraint_Value'],
        aggfunc='mean'
    ).round(4)
    
    pivot2.to_csv(os.path.join(output_dir, "pivot_hybrids_vs_constraints_retention.csv"))
    
    # Pivot 3: Budget Strategy vs Constraints
    pivot3 = df.pivot_table(
        values='Performance_At_Timeout',
        index='Budget_Strategy',
        columns=['Constraint_Type', 'Constraint_Value'],
        aggfunc='mean'
    ).round(4)
    
    pivot3.to_csv(os.path.join(output_dir, "pivot_budget_strategy_vs_constraints.csv"))
    
    # Pivot 4: Dropout Strategy vs Constraints  
    pivot4 = df.pivot_table(
        values='Performance_At_Timeout',
        index='Dropout_Strategy',
        columns=['Constraint_Type', 'Constraint_Value'],
        aggfunc='mean'
    ).round(4)
    
    pivot4.to_csv(os.path.join(output_dir, "pivot_dropout_strategy_vs_constraints.csv"))

# this function creates the main analysis using all hybrid schedules under resource constraints
def create_hybrid_resource_constraint_analysis():
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/hybrid_constraint_analysis"
    max_epochs = 100

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
    print(f"Testing ALL {len(hybrid_schedules)} hybrid schedules under constraints")

    resource_constraints = {'max_epochs': [1000, 2000, 5000], 'max_time_hours': [10, 24, 72]}

    os.makedirs(output_dir, exist_ok=True)
    all_hybrid_results = {}

    for model in models:
        print(f"\n=== Analyzing {model} with ALL Hybrid Schedules ===")
        base_dir = os.path.join(base_root, model)
        all_hybrid_results[model] = {}

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
                epoch_times = compute_epoch_times_from_json_files(json_files)
                if epoch_times is None:
                    epoch_times = generate_synthetic_epoch_times(curves.shape[0], curves.shape[1], model)

                sim = ResourceConstrainedSimulator(curves, epoch_times=epoch_times)
                dataset_results = {}

                # tests all 25 hybrid schedules
                for hybrid_name, hybrid_schedule in hybrid_schedules.items():
                    print(f"  Processing {hybrid_name} on {dataset}")
                    
                    hybrid_constraint_results = {}
                    
                    # tests under all resource constraints
                    for constraint_type, constraint_values in resource_constraints.items():
                        for constraint_value in constraint_values:
                            try:
                                res = sim.simulate_with_resource_constraints(hybrid_schedule, {constraint_type: constraint_value}, top_k=1)
                                key = f"{constraint_type}_{constraint_value}"
                                hybrid_constraint_results[key] = res[constraint_type]
                            except Exception as e:
                                print(f"Error with {constraint_type}={constraint_value}: {e}")

                    # stores results for this hybrid
                    dataset_results[hybrid_name] = {
                        'constraint_results': hybrid_constraint_results,
                        'schedule': hybrid_schedule
                    }

                all_hybrid_results[model][dataset] = dataset_results
                print(f"  Completed {dataset} with {len(hybrid_schedules)} hybrids")

            except Exception as e:
                print(f"Skipped {model}/{dataset} due to error: {e}")

    analyze_hybrid_performance_patterns(all_hybrid_results, output_dir)
    find_best_hybrids_per_constraint(all_hybrid_results, output_dir)
    create_hybrid_robustness_ranking(all_hybrid_results, output_dir)
    
    comprehensive_df = save_all_hybrids_per_constraint_csv(all_hybrid_results, output_dir)
    if comprehensive_df is not None:
        create_hybrid_pivot_tables(comprehensive_df, output_dir)
    
    return all_hybrid_results

if __name__ == "__main__":
    
    try:
        all_hybrid_results = create_hybrid_resource_constraint_analysis()
        print("\nHybrid resource constraint analysis completed successfully.")
    except Exception as e:
        print(f"\nError running hybrid analysis: {e}")
        
        try:
            all_results = create_resource_constraint_analysis()
            print("\nOriginal resource constraint analysis completed successfully.")
        except Exception as e2:
            print(f"\nError running original analysis: {e2}")
