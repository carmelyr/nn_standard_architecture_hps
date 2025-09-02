# This module performs multi-objective analysis.

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sh_core import (load_curve_data_from_json_from_files)
import re

pink_palette = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585", "#DB7093", "#FFC0CB"]
sns.set_palette(pink_palette)

# this function calculates the model size
# config: model configuration dictionary
# model_type: type of the model (e.g., CNN, RNN)
# input_shape: shape of the input data
# num_classes: number of output classes
# returns: total number of trainable parameters
def calculate_model_size(config, model_type, input_shape, num_classes):
    if model_type == "FCNN":
        return _calculate_fcnn_size(config, input_shape, num_classes)
    elif model_type == "CNN":
        return _calculate_cnn_size(config, input_shape, num_classes)
    elif model_type == "LSTM":
        return _calculate_lstm_size(config, input_shape, num_classes)
    elif model_type == "GRU":
        return _calculate_gru_size(config, input_shape, num_classes)
    elif model_type == "Transformer":
        return _calculate_transformer_size(config, input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# calculates FCNN size (parameter count)
def _calculate_fcnn_size(config, input_shape, num_classes):
    hidden_units = config["hidden_units"]
    num_layers = config["num_layers"]
    
    # flattens input size
    input_size = int(np.prod(input_shape))
    
    total_params = 0
    in_features = input_size
    
    # hidden layers
    for _ in range(num_layers):
        # Linear layer: weight + bias
        total_params += in_features * hidden_units + hidden_units
        # BatchNorm1d: weight + bias
        total_params += hidden_units + hidden_units
        in_features = hidden_units
    
    # output layer
    total_params += hidden_units * num_classes + num_classes
    
    return total_params

# calculates CNN size (parameter count)
def _calculate_cnn_size(config, input_shape, num_classes):
    num_filters = config["num_filters"]
    kernel_size = config["kernel_size"]
    num_layers = config.get("num_layers", 1)
    
    seq_len, num_features = input_shape[0], input_shape[-1]
    
    total_params = 0
    in_channels = num_features
    current_seq_len = seq_len
    
    # Convolutional layers
    for _ in range(num_layers):
        # Conv1d: (in_channels * kernel_size * out_channels) + bias
        total_params += in_channels * kernel_size * num_filters + num_filters
        # BatchNorm1d: weight + bias
        total_params += num_filters + num_filters
        
        # updates for next layer
        in_channels = num_filters

        # accounts for pooling reduction
        if current_seq_len > 4:
            pooling_size = config.get("pooling_size", 2)
            current_seq_len = current_seq_len // pooling_size
    
    # final classifier
    flatten_size = current_seq_len * num_filters
    total_params += flatten_size * num_classes + num_classes
    
    return total_params

# calculates LSTM size (parameter count)
def _calculate_lstm_size(config, input_shape, num_classes):
    hidden_units = config["hidden_units"]
    num_layers = config["num_layers"]
    bidirectional = config.get("bidirectional", False)
    
    seq_len, num_features = input_shape
    directions = 2 if bidirectional else 1
    
    total_params = 0
    
    # LSTM parameters: 4 gates * (input_size + hidden_size + 1) * hidden_size
    for layer in range(num_layers):
        input_size = num_features if layer == 0 else hidden_units * directions

        # input to hidden weights (4 gates)
        total_params += 4 * input_size * hidden_units
        # hidden to hidden weights (4 gates)
        total_params += 4 * hidden_units * hidden_units
        # biases (4 gates)
        total_params += 4 * hidden_units

        # if bidirectional, doubles the parameters for backward direction
        if bidirectional:
            total_params += 4 * input_size * hidden_units
            total_params += 4 * hidden_units * hidden_units
            total_params += 4 * hidden_units

    # classifier layers
    classifier_input = hidden_units * directions
    # first linear layer
    total_params += classifier_input * (hidden_units * 2) + (hidden_units * 2)
    # second linear layer
    total_params += (hidden_units * 2) * num_classes + num_classes
    
    return total_params

# calculates GRU size (parameter count)
def _calculate_gru_size(config, input_shape, num_classes):
    hidden_units = config["hidden_units"]
    num_layers = config["num_layers"]
    bidirectional = config.get("bidirectional", False)
    
    seq_len, num_features = input_shape if len(input_shape) == 2 else (input_shape[0], 1)
    directions = 2 if bidirectional else 1
    
    total_params = 0
    
    # GRU parameters: 3 gates * (input_size + hidden_size + 1) * hidden_size
    for layer in range(num_layers):
        input_size = num_features if layer == 0 else hidden_units * directions

        # input to hidden weights (3 gates)
        total_params += 3 * input_size * hidden_units
        # hidden to hidden weights (3 gates)
        total_params += 3 * hidden_units * hidden_units
        # biases (3 gates)
        total_params += 3 * hidden_units
        
        # if bidirectional, doubles the parameters
        if bidirectional:
            total_params += 3 * input_size * hidden_units
            total_params += 3 * hidden_units * hidden_units
            total_params += 3 * hidden_units

    # classifier layers
    classifier_input = hidden_units * directions
    # LayerNorm
    total_params += classifier_input + classifier_input  # weight + bias
    # first linear layer
    total_params += classifier_input * hidden_units + hidden_units
    # second linear layer
    total_params += hidden_units * num_classes + num_classes
    
    return total_params

# calculates Transformer size (parameter count)
def _calculate_transformer_size(config, input_shape, num_classes):
    num_heads = config["num_heads"]
    hidden_units = config["hidden_units"]
    ff_dim = config["ff_dim"]
    num_layers = config["num_layers"]
    
    seq_len, num_features = input_shape
    
    # adjusts hidden_units to be divisible by num_heads
    hidden_units = ((hidden_units // num_heads) + 1) * num_heads
    
    total_params = 0

    # embedding layer
    total_params += num_features * hidden_units + hidden_units      # Linear
    total_params += hidden_units + hidden_units                     # LayerNorm
    # Positional encoding is not trainable
    
    # Transformer encoder layers
    for _ in range(num_layers):
        # Multi-head attention
        # Q, K, V projections
        total_params += 3 * (hidden_units * hidden_units + hidden_units)
        # output projection
        total_params += hidden_units * hidden_units + hidden_units

        # feed-forward network
        total_params += hidden_units * ff_dim + ff_dim          # first linear
        total_params += ff_dim * hidden_units + hidden_units    # second linear

        # layer norms (2 per layer)
        total_params += 2 * (hidden_units + hidden_units)

    # classifier head
    total_params += hidden_units * hidden_units + hidden_units  # first linear
    total_params += hidden_units + hidden_units                 # layer norm
    total_params += hidden_units * num_classes + num_classes    # final linear

    return total_params

# this function extracts the Pareto frontier from the given performance and model size data
# identifies the points that are not dominated by any other points
# performance: array of performance values (higher is better)
# model_size: array of model sizes (lower is better)
def extract_pareto_frontier(performance, model_size):
    points = np.column_stack((model_size, performance))
    n_points = len(points)
    
    if n_points == 0:
        return np.array([])
    
    # sorts by model size (ascending)
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    pareto_indices = []
    max_performance_so_far = -np.inf
    
    for i, (size, perf) in enumerate(sorted_points):
        # if this point has better performance than all previous points with smaller/equal size
        if perf > max_performance_so_far:
            pareto_indices.append(sorted_indices[i])
            max_performance_so_far = perf
    
    return np.array(pareto_indices)

# this function extracts dataset information from JSON files
def get_dataset_info_from_files(json_files):
    if not json_files:
        return None, None
        
    for file_path in json_files:
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            dataset_stats = data.get('dataset_stats', {})
            input_shape = dataset_stats.get('input_shape')
            num_classes = dataset_stats.get('num_classes')
            
            if input_shape is not None and num_classes is not None:
                return tuple(input_shape), num_classes
        except Exception:
            continue
    
    return None, None

# this function creates the Pareto analysis combining performance and model size
# analyzes all configurations and extracts Pareto frontiers for each model
def create_pareto_analysis():
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    base_root = "/Users/carmely/GitHub/nn_standard_architecture_hps/results"
    output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/pareto_analysis"
    max_epochs = 100

    os.makedirs(output_dir, exist_ok=True)
    all_model_data = {}

    for model in models:
        print(f"\n=== Creating Pareto Analysis for {model} ===")
        base_dir = os.path.join(base_root, model)
        
        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            continue

        model_configurations = []
        total_datasets = 0
        total_json_files = 0
        total_valid_configs = 0

        for dataset in sorted(os.listdir(base_dir)):
            dataset_path = os.path.join(base_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            total_datasets += 1
            print(f"Processing {model}/{dataset}")

            if dataset == "classification_ozone":
                json_files = glob.glob(os.path.join(dataset_path, "config_*", "classification_ozone_config_*_seed*.json"))
            else:
                json_files = glob.glob(os.path.join(dataset_path, "config_*", f"{dataset}_config_*_seed*.json"))

            total_json_files += len(json_files)
            
            if len(json_files) == 0:
                print(f"  No JSON files found for {model}/{dataset}")
                continue

            input_shape, num_classes = get_dataset_info_from_files(json_files)
            if input_shape is None or num_classes is None:
                print(f"  Could not extract dataset info for {model}/{dataset}")
                continue

            try:
                curves = load_curve_data_from_json_from_files(json_files, max_epochs=max_epochs)
                if curves.shape[0] == 0:
                    print(f"  No valid curves for {model}/{dataset}")
                    continue

                # processes each configuration (groups by config ID and averages across seeds)
                config_data = {}
                dataset_valid_configs = 0
                
                for json_file in json_files:
                    try:
                        import json
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        config = data.get('hyperparameters', {})
                        if not config:
                            continue
          
                        match = re.search(r'config_(\d+)', json_file)
                        if not match:
                            continue
                        config_id = int(match.group(1))

                        # calculates model size (same for all seeds of this config)
                        model_size = calculate_model_size(config, model, input_shape, num_classes)

                        # gets performance (final validation accuracy)
                        performance = data.get('avg_val_accuracy', 0.0)
                        if performance == 0.0:
                            val_accuracies = data.get('val_accuracy', [])
                            if val_accuracies:
                                performance = np.mean(val_accuracies)

                        # groups by config ID
                        if config_id not in config_data:
                            config_data[config_id] = {'model_size': model_size, 'performances': [], 'config': config, 'dataset': dataset}

                        config_data[config_id]['performances'].append(performance)
                        
                    except Exception as e:
                        print(f"    Error processing {json_file}: {e}")
                        continue

                # averages across seeds and adds to model data
                for config_id, data in config_data.items():
                    if len(data['performances']) > 0:
                        # averages performance across all seeds for this configuration
                        avg_performance = np.mean(data['performances'])
                        
                        model_configurations.append({
                            'model': model,
                            'dataset': dataset,
                            'config_id': config_id,
                            'model_size': data['model_size'],
                            'performance': avg_performance,
                            'config': data['config'],
                            'num_seeds': len(data['performances'])  # tracks how many seeds were averaged
                        })
                        dataset_valid_configs += 1

                print(f"{dataset}: {len(json_files)} JSON files -> {dataset_valid_configs} unique configs (averaged across seeds)")
                
                total_valid_configs += dataset_valid_configs
                print(f"{dataset}: {len(json_files)} JSON files -> {dataset_valid_configs} valid configs")

            except Exception as e:
                print(f"Error processing {model}/{dataset}: {e}")
                continue

        print(f"\n{model} SUMMARY:")
        print(f"Total datasets processed: {total_datasets}")
        print(f"Total JSON files found: {total_json_files}")
        print(f"Total valid configurations: {total_valid_configs}")
        print(f"Configurations plotted: {len(model_configurations)}")
        print(f"Expected if 100 configs/dataset: {total_datasets * 100}")

        all_model_data[model] = model_configurations

    print(f"\n{'='*50}")
    print("OVERALL SUMMARY:")
    total_points = 0
    for model, configs in all_model_data.items():
        print(f"{model}: {len(configs)} points")
        total_points += len(configs)
    print(f"TOTAL POINTS PLOTTED: {total_points}")
    print(f"{'='*50}")

    # Pareto analysis plots
    create_pareto_plots_with_frontiers(all_model_data, output_dir)
    save_pareto_csv_with_frontiers(all_model_data, output_dir)
    
    return all_model_data

# this function creates Pareto plots with frontiers
def create_pareto_plots_with_frontiers(all_model_data, output_dir):
    if not all_model_data:
        print("No data available for Pareto analysis")
        return

    # Plot 1: All models on one plot with Pareto frontiers
    plt.figure(figsize=(14, 10))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (model, configurations) in enumerate(all_model_data.items()):
        if not configurations:
            continue
            
        model_sizes = np.array([config['model_size'] for config in configurations])
        performances = np.array([config['performance'] for config in configurations])
        
        # plots all points
        plt.scatter(model_sizes, performances, c=colors[i], marker=markers[i], label=f'{model} (all configs)', alpha=0.6, s=50)

        # extracts and plots Pareto frontier
        pareto_indices = extract_pareto_frontier(performances, model_sizes)
        if len(pareto_indices) > 0:
            pareto_sizes = model_sizes[pareto_indices]
            pareto_perfs = performances[pareto_indices]
            
            # sorts for plotting line
            sort_idx = np.argsort(pareto_sizes)
            pareto_sizes_sorted = pareto_sizes[sort_idx]
            pareto_perfs_sorted = pareto_perfs[sort_idx]
            
            plt.plot(pareto_sizes_sorted, pareto_perfs_sorted, color=colors[i], linewidth=2, linestyle='-', label=f'{model} Pareto frontier')
            
            # highlights Pareto points
            plt.scatter(pareto_sizes, pareto_perfs, c=colors[i], marker=markers[i], s=120, edgecolors='black', linewidth=2)

    plt.xlabel('Model Size', fontsize=14, fontweight='bold')
    plt.ylabel('Performance (Validation Accuracy)', fontsize=14, fontweight='bold')
    plt.title('Pareto Analysis: Performance vs Model Size\nAll Models with Pareto Frontiers', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, 1.05)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_performance_vs_model_size_all.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # Plot 2: Individual plots for each model
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (model, configurations) in enumerate(all_model_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if not configurations:
            ax.set_title(f'{model}\nNo Data Available', fontsize=14, fontweight='bold')
            ax.set_visible(False)
            continue

        # groups configurations by dataset
        datasets = {}
        for config in configurations:
            dataset = config['dataset']
            if dataset not in datasets:
                datasets[dataset] = {'sizes': [], 'perfs': []}
            datasets[dataset]['sizes'].append(config['model_size'])
            datasets[dataset]['perfs'].append(config['performance'])
        
        # plots each dataset
        dataset_colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        for j, (dataset, data) in enumerate(datasets.items()):
            ax.scatter(data['sizes'], data['perfs'], 
                      c=[dataset_colors[j]], alpha=0.6, s=30,
                      label=dataset if len(datasets) <= 5 else None)

        # extracts and plots overall Pareto frontier
        all_sizes = np.array([config['model_size'] for config in configurations])
        all_perfs = np.array([config['performance'] for config in configurations])
        
        pareto_indices = extract_pareto_frontier(all_perfs, all_sizes)
        if len(pareto_indices) > 0:
            pareto_sizes = all_sizes[pareto_indices]
            pareto_perfs = all_perfs[pareto_indices]
            
            # sorts for plotting
            sort_idx = np.argsort(pareto_sizes)
            pareto_sizes_sorted = pareto_sizes[sort_idx]
            pareto_perfs_sorted = pareto_perfs[sort_idx]
            
            ax.plot(pareto_sizes_sorted, pareto_perfs_sorted, 'r-', linewidth=2, label='Pareto Frontier')
            ax.scatter(pareto_sizes, pareto_perfs, c='red', s=80, edgecolors='black', linewidth=1)

        ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance (Val Accuracy)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model}\n{len(configurations)} Configurations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        if len(datasets) <= 5:
            ax.legend(fontsize=10, loc='lower left')

    # hides unused subplots
    for i in range(len(all_model_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_individual_models.pdf"), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# this function creates frontier curves with points for all models
def create_frontier_curves_with_points_all_models(all_model_data, output_dir):
    if not all_model_data:
        print("No data available for Pareto frontier-only plot")
        return

    plt.figure(figsize=(12, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (model, configurations) in enumerate(all_model_data.items()):
        if not configurations:
            continue

        # extracts arrays
        model_sizes = np.array([c['model_size'] for c in configurations], dtype=float)
        performances = np.array([c['performance'] for c in configurations], dtype=float)

        # computes frontier
        pareto_indices = extract_pareto_frontier(performances, model_sizes)
        if len(pareto_indices) == 0:
            continue

        frontier_sizes = model_sizes[pareto_indices]
        frontier_perfs = performances[pareto_indices]

        # sorts for smooth curve
        sort_idx = np.argsort(frontier_sizes)
        frontier_sizes = frontier_sizes[sort_idx]
        frontier_perfs = frontier_perfs[sort_idx]

        # plots frontier curve with same color
        plt.plot(
            frontier_sizes,
            frontier_perfs,
            linewidth=2.5,
            color=colors[i],
            marker=markers[i],
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=1,
            label=f"{model} Pareto frontier",
        )

    plt.xlabel('Model Size', fontsize=14, fontweight='bold')
    plt.ylabel('Performance (Validation Accuracy)', fontsize=14, fontweight='bold')
    plt.title('Pareto Frontiers: Performance vs Model Size (Frontier Points Only)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left', fontsize=12)

    plt.tick_params(axis='both', which='major', labelsize=12)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()

    out_path = os.path.join(output_dir, "pareto_frontiers_points_only_all_models.pdf")
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# this function saves Pareto analysis results to CSV files
def save_pareto_csv_with_frontiers(all_model_data, output_dir):
    csv_path = os.path.join(output_dir, "pareto_all_configurations.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Config_ID', 'Model_Size_Parameters', 'Performance_Val_Accuracy_Avg', 'Num_Seeds_Averaged', 'Is_Pareto_Optimal'])
        
        for model, configurations in all_model_data.items():
            if not configurations:
                continue

            # determines Pareto optimal points
            model_sizes = np.array([config['model_size'] for config in configurations])
            performances = np.array([config['performance'] for config in configurations])
            pareto_indices = extract_pareto_frontier(performances, model_sizes)
            pareto_set = set(pareto_indices)
            
            for i, config in enumerate(configurations):
                is_pareto = 'Yes' if i in pareto_set else 'No'
                num_seeds = config.get('num_seeds', 1)
                writer.writerow([config['model'], config['dataset'], config['config_id'], config['model_size'], f"{config['performance']:.4f}", num_seeds, is_pareto])

    # saves Pareto frontiers only
    frontier_csv_path = os.path.join(output_dir, "pareto_frontiers_only.csv")
    with open(frontier_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Config_ID', 'Model_Size_Parameters', 'Performance_Val_Accuracy_Avg', 'Num_Seeds_Averaged'])
        
        for model, configurations in all_model_data.items():
            if not configurations:
                continue
                
            model_sizes = np.array([config['model_size'] for config in configurations])
            performances = np.array([config['performance'] for config in configurations])
            pareto_indices = extract_pareto_frontier(performances, model_sizes)
            
            for idx in pareto_indices:
                config = configurations[idx]
                num_seeds = config.get('num_seeds', 1)
                writer.writerow([
                    config['model'], config['dataset'], config['config_id'],
                    config['model_size'], f"{config['performance']:.4f}", num_seeds
                ])

if __name__ == "__main__":
    all_model_data = create_pareto_analysis()
    output_dir = "/Users/carmely/GitHub/nn_standard_architecture_hps/successive_halving_plots/pareto_analysis"
    create_frontier_curves_with_points_all_models(all_model_data, output_dir)
