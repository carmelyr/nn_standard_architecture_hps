# this module contains the core functionality for the successive halving simulator

import os
import json
import numpy as np
from typing import List, Tuple
import re

# this class is the core simulator for SH with resource tracking and constraint analysis
class SuccessiveHalvingSimulator:
    # this function initializes the simulator with curve data and computes final scores
    # curve_data: 4D numpy array with shape (configs, seeds, folds, epochs)
    # final_epoch: The last epoch to consider for final scores
    # epoch_times: (optional) 2D numpy array with shape (configs, seeds) representing average time per epoch
    def __init__(self, curve_data: np.ndarray, final_epoch: int = 1000, epoch_times: np.ndarray = None):
        self.curves = curve_data
        self.configs, self.seeds, self.folds, self.epochs = curve_data.shape
        
        # stores epoch timing information (if provided)
        self.epoch_times = epoch_times if epoch_times is not None else np.ones((self.configs, self.seeds))

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

    # this function simulates the basic successive halving process
    def simulate(self, budget_epochs: List[int], top_k: int = 1):
        top_k_hits = []
        regrets = []
        
        for budget in budget_epochs:
            epoch_idx = min(budget - 1, self.epochs - 1)
            
            scores = np.nanmean(self.curves[:, :, :, epoch_idx], axis=2)
            
            dataset_hits = []
            dataset_regrets = []
        
            for seed_idx in range(self.seeds):
                seed_scores = scores[:, seed_idx]
                valid_mask = ~np.isnan(seed_scores)
                
                if not np.any(valid_mask):
                    continue
                    
                valid_scores = seed_scores[valid_mask]
                valid_configs = np.where(valid_mask)[0]
                
                # selects top-k configs at budget epoch
                top_k_indices_budget = np.argsort(seed_scores[valid_mask])[-top_k:]
                selected_configs = valid_configs[top_k_indices_budget]

                # gets final scores of selected configs
                selected_final_scores = self.final_scores[selected_configs]

                # computes regret
                best_final_score = np.nanmax(self.final_scores[valid_mask])
                regret = best_final_score - np.nanmax(selected_final_scores)

                # computes hit
                top_k_indices_final = np.argsort(self.final_scores[valid_mask])[-top_k:]
                hit = int(any(c in valid_configs[top_k_indices_final] for c in selected_configs))

                dataset_hits.append(hit)
                dataset_regrets.append(regret)
            
            if dataset_hits:
                top_k_hits.append(np.mean(dataset_hits))
                regrets.append(np.mean(dataset_regrets))
            else:
                top_k_hits.append(np.nan)
                regrets.append(np.nan)
        
        return top_k_hits, regrets

    # this function retrieves the global best score across all configurations
    def get_global_best(self):
        # accuracy case: higher is better
        return float(np.max(self.final_scores))

    # this function retrieves the final top-k scores at the end of a schedule
    def get_final_topk_scores(self, schedule, k):
        # simulates the schedule to get final survivors
        current_indices = list(range(self.configs))
        
        for round_idx, (num_to_keep, epoch_budget) in enumerate(schedule):
            if not current_indices or len(current_indices) <= 1:
                break
            
            epoch_idx = min(epoch_budget - 1, self.epochs - 1)
            
            # average performance across all seeds and folds
            perf = np.nanmean(self.curves[current_indices, :, :, epoch_idx], axis=(1, 2))
            valid_mask = ~np.isnan(perf)
            
            if not np.any(valid_mask):
                break
            
            valid_configs = np.array(current_indices)[valid_mask]
            valid_perf = perf[valid_mask]
            
            # keeps top performers
            if len(valid_configs) <= num_to_keep:
                current_indices = list(valid_configs)
            else:
                top_indices = np.argsort(valid_perf)[-num_to_keep:]
                current_indices = list(valid_configs[top_indices])
    
        # gets final scores of survivors
        if current_indices:
            survivor_scores = self.final_scores[current_indices]
            # sorts in descending order and takes top-k
            sorted_scores = np.sort(survivor_scores)[::-1]
            return sorted_scores[:min(k, len(sorted_scores))]
        else:
            return np.array([])

    # this function simulates a custom successive halving schedule
    # schedule: List of tuples (num_configs, num_epochs) for each round
    # top_k: Number of final top configs to consider for hit and regret
    # returns: Tuple of three lists: hit probabilities, regrets, dropout schedule
    def simulate_custom_schedule(self, schedule: List[Tuple[int, int]], top_k: int = 1):
        current_indices = list(range(self.configs))
        hits_per_round = []
        regrets_per_round = []
        dropout_schedule = []

        for round_idx, (num_to_keep, epoch_budget) in enumerate(schedule):
            if not current_indices:
                hits_per_round.append(np.nan)
                regrets_per_round.append(np.nan)
                dropout_schedule.append(0)
                continue

            initial_count = len(current_indices)
            epoch_idx = min(epoch_budget - 1, self.epochs - 1)
            
            round_hits = []
            round_regrets = []

            for seed_idx in range(self.seeds):
                if not current_indices:
                    continue
                    
                # extract validation performance at the current epoch
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

            # updates current_indices for next round
            if round_hits:
                perf_avg = np.nanmean(self.curves[current_indices, :, :, epoch_idx], axis=(1, 2))
                valid_mask_avg = ~np.isnan(perf_avg)

                if np.any(valid_mask_avg):
                    valid_configs_avg = np.array(current_indices)[valid_mask_avg]
                    valid_perf_avg = perf_avg[valid_mask_avg]

                    keep = min(num_to_keep, len(valid_configs_avg))
                    if keep > 0:
                        top_idx = np.argsort(valid_perf_avg)[-keep:][::-1]
                        selected_avg = valid_configs_avg[top_idx]
                        final_count = len(selected_avg)
                        dropout_schedule.append(initial_count - final_count)
                        current_indices = list(selected_avg)
                    else:
                        dropout_schedule.append(initial_count)
                        current_indices = []

                    if len(current_indices) <= 1:
                        break
                else:
                    dropout_schedule.append(initial_count)
                    current_indices = []
                    break
            else:
                dropout_schedule.append(initial_count)
                current_indices = []

        return hits_per_round, regrets_per_round, dropout_schedule

    # this function simulates a dual schedule
    def simulate_dual_schedule(self, budget_epochs: List[int], dropout_keeps: List[int], top_k: int = 1):
        rounds = min(len(budget_epochs), len(dropout_keeps))
        paired = [(dropout_keeps[r], budget_epochs[r]) for r in range(rounds)]
        return self.simulate_custom_schedule(paired, top_k=top_k)

    # this function computes the runtime statistics for a given schedule
    # returns: Dict with runtime breakdown by round and overall statistics
    def compute_schedule_runtime(self, schedule: List[Tuple[int, int]]):
        runtime_breakdown = []
        total_epochs = 0
        total_time_seconds = 0
        configs_remaining = self.configs
        
        for round_idx, (num_to_keep, epochs_per_config) in enumerate(schedule):
            if configs_remaining == 0:
                break
                
            # calculates for this round
            round_epochs = configs_remaining * epochs_per_config
            
            # average time per epoch for remaining configs
            avg_epoch_time = np.nanmean(self.epoch_times[:configs_remaining, :])
            round_time_seconds = round_epochs * avg_epoch_time
            
            round_info = {
                'round': round_idx + 1,
                'configs_evaluated': configs_remaining,
                'epochs_per_config': epochs_per_config,
                'total_epochs_round': round_epochs,
                'avg_epoch_time_seconds': avg_epoch_time,
                'total_time_round_seconds': round_time_seconds,
                'total_time_round_hours': round_time_seconds / 3600,
                'configs_surviving': min(num_to_keep, configs_remaining)
            }
            
            runtime_breakdown.append(round_info)
            
            total_epochs += round_epochs
            total_time_seconds += round_time_seconds
            configs_remaining = min(num_to_keep, configs_remaining)
        
        summary = {
            'total_epochs': total_epochs,
            'total_time_seconds': total_time_seconds,
            'total_time_hours': total_time_seconds / 3600,
            'total_time_days': total_time_seconds / (3600 * 24),
            'rounds_completed': len(runtime_breakdown),
            'final_configs': configs_remaining,
            'computational_savings': 1 - (total_epochs / (self.configs * self.epochs)) if self.epochs > 0 else 0
        }
        
        return {
            'breakdown': runtime_breakdown,
            'summary': summary
        }

# this function loads curve data from JSON files
# returns: 4D numpy array with shape (num_configs, num_seeds, num_folds, max_epochs)
def load_curve_data_from_json_from_files(file_list: List[str], max_epochs: int = 1000) -> np.ndarray:
    configs = {}

    for path in file_list:
        fname = os.path.basename(path)
        match = re.search(r"_config_(\d+)_seed(\d+)", fname)
        if not match:
            continue

        config_id = int(match.group(1))
        seed_id = int(match.group(2))

        with open(path) as f:
            data = json.load(f)

        if config_id not in configs:
            configs[config_id] = {}
        configs[config_id][seed_id] = data

    config_ids = sorted(configs.keys())
    num_configs = len(config_ids)

    if num_configs == 0:
        raise ValueError("No valid config files found.")

    # determines the number of seeds and folds
    num_seeds = max(len(configs[c]) for c in config_ids)
    first_seed = list(configs[config_ids[0]].values())[0]
    num_folds = len(first_seed["fold_logs"])
    curves = np.full((num_configs, num_seeds, num_folds, max_epochs), fill_value=np.nan)

    # fills the curves array with accuracy values
    for i, config_id in enumerate(config_ids):
        for seed_id, seed_data in configs[config_id].items():
            seed_idx = seed_id - 1
            for fold_idx, fold in enumerate(seed_data["fold_logs"]):
                acc = fold["val_accuracy"]
                if not acc:
                    continue
                length = min(len(acc), max_epochs)
                padded = np.full(max_epochs, fill_value=acc[-1])
                padded[:length] = acc[:length]
                curves[i, seed_idx, fold_idx, :] = padded

    return curves

# this function computes epoch times from JSON files
# returns: 2D numpy array with shape (num_configs, num_seeds) or None if not found
def compute_epoch_times_from_json_files(file_list: List[str]):
    times = {}

    for path in file_list:
        fname = os.path.basename(path)
        m = re.search(r"_config_(\d+)_seed(\d+)", fname)
        if not m:
            continue
        cfg = int(m.group(1))
        seed = int(m.group(2))

        with open(path) as f:
            data = json.load(f)

        cand = []
        # seed-level timing
        if isinstance(data, dict) and 'epoch_times' in data and isinstance(data['epoch_times'], list) and data['epoch_times']:
            cand.append(np.nanmean(data['epoch_times']))

        # fold-level timing
        if 'fold_logs' in data and isinstance(data['fold_logs'], list):
            for fold in data['fold_logs']:
                if not isinstance(fold, dict):
                    continue
                for key in ('epoch_times', 'epoch_durations'):
                    if key in fold and isinstance(fold[key], list) and fold[key]:
                        cand.append(np.nanmean(fold[key]))

        if cand:
            times.setdefault(cfg, {})[seed] = float(np.nanmean(cand))

    if not times:
        return None

    config_ids = sorted(times.keys())
    num_configs = len(config_ids)
    max_seed = 1
    for cfg in config_ids:
        if times[cfg]:
            max_seed = max(max_seed, max(times[cfg].keys()))

    arr = np.full((num_configs, max_seed), np.nan, dtype=float)
    for i, cfg in enumerate(config_ids):
        for seed_id, v in times[cfg].items():
            arr[i, seed_id - 1] = v

    if np.isnan(arr).all():
        return None

    global_mean = np.nanmean(arr)
    arr = np.where(np.isnan(arr), global_mean, arr)
    return arr

# this function generates synthetic epoch times
# returns: 2D numpy array with shape (num_configs, num_seeds)
def generate_synthetic_epoch_times(num_configs, num_seeds, model_type):
    # synthetic base times in seconds per epoch
    base_times = {
        'CNN': 15.0,
        'FCNN': 5.0, 
        'GRU': 25.0,
        'LSTM': 30.0,
        'Transformer': 45.0
    }
    
    base_time = base_times.get(model_type, 20.0)
    
    # adds variance
    config_multipliers = np.random.lognormal(0, 0.3, num_configs)   # config-level variance of 30%
    seed_multipliers = np.random.normal(1.0, 0.1, num_seeds)        # seed-level variance of 10%

    epoch_times = np.outer(config_multipliers, seed_multipliers) * base_time
    return np.abs(epoch_times)
