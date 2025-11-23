#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM-TD3 Experiment Visualization Script
=======================================
Generates academic-quality plots from the results of run_comprehensive_comparison.py.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

COLORS = {
    'CAM_TD3': '#d62728',      # Red
    'CAM-TD3': '#d62728',
    'TD3': '#1f77b4',          # Blue
    'DDPG': '#ff7f0e',         # Orange
    'SAC': '#2ca02c',          # Green
    'Greedy': '#9467bd',       # Purple
    'Local-Only': '#8c564b',   # Brown
    'Remote-Only': '#e377c2',  # Pink
    'Random': '#7f7f7f',       # Gray
    'No_Migration': '#bcbd22', # Olive
    'No_Cache': '#17becf'      # Cyan
}

def load_result(filepath: str) -> Dict[str, Any]:
    """Load a single result JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️ Failed to load {filepath}: {e}")
        return {}

def get_final_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract final converged metrics from result data."""
    if not data or 'final_performance' not in data:
        return {}
    
    perf = data['final_performance']
    # Fallback to episode metrics if final_performance is missing/empty
    if not perf and 'episode_metrics' in data:
        metrics = data['episode_metrics']
        return {
            'avg_delay': np.mean(metrics['avg_delay'][-20:]),
            'avg_energy': np.mean(metrics['total_energy'][-20:]),
            'completion_rate': np.mean(metrics['task_completion_rate'][-20:]),
            'cache_hit_rate': np.mean(metrics.get('cache_hit_rate', [0])[-20:])
        }
        
    return {
        'avg_delay': perf.get('avg_delay', 0),
        'avg_energy': perf.get('avg_energy', 0),
        'completion_rate': perf.get('avg_completion', 0), # Note: key might be avg_completion or completion_rate
        'cache_hit_rate': perf.get('avg_cache_hit', 0) # Check key
    }

def plot_bar_comparison(results: Dict[str, Dict[str, float]], metric: str, title: str, ylabel: str, filename: str):
    """Generate a bar chart comparing algorithms."""
    algorithms = list(results.keys())
    values = [results[algo].get(metric, 0) for algo in algorithms]
    
    # Sort by value (optional, maybe better to keep fixed order)
    # fixed_order = ['CAM_TD3', 'TD3', 'DDPG', 'SAC', 'Greedy', 'Local-Only', 'Remote-Only', 'Random']
    # algorithms = [algo for algo in fixed_order if algo in results]
    # values = [results[algo].get(metric, 0) for algo in algorithms]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(algorithms, values, color=[COLORS.get(algo, '#333333') for algo in algorithms], alpha=0.8)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}' if height < 1 else f'{height:.1f}',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"📊 Saved {filename}")

def plot_line_sensitivity(x_values: List[float], results: Dict[str, List[float]], 
                         xlabel: str, ylabel: str, title: str, filename: str):
    """Generate a line chart for sensitivity analysis."""
    plt.figure(figsize=(10, 6))
    
    for algo, y_values in results.items():
        plt.plot(x_values, y_values, marker='o', label=algo, color=COLORS.get(algo, '#333333'))
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"📊 Saved {filename}")

def process_exp1_core():
    """Process Experiment 1: Core Comparison"""
    folder = "results/comparison/exp1_core"
    if not os.path.exists(folder):
        print(f"⚠️ Folder {folder} not found.")
        return

    metrics_data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            algo_name = filename.replace(".json", "").replace("cam_td3", "CAM_TD3").replace("local-only", "Local-Only").replace("remote-only", "Remote-Only")
            # Capitalize properly if needed, but filename is usually lowercase from run script
            # Map lowercase filename to display name
            name_map = {
                "cam_td3": "CAM_TD3", "td3": "TD3", "ddpg": "DDPG", "sac": "SAC",
                "greedy": "Greedy", "local-only": "Local-Only", "remote-only": "Remote-Only", "random": "Random"
            }
            display_name = name_map.get(filename.replace(".json", ""), filename.replace(".json", "").upper())
            
            data = load_result(os.path.join(folder, filename))
            metrics = get_final_metrics(data)
            if metrics:
                metrics_data[display_name] = metrics

    if not metrics_data:
        print("⚠️ No data found for Experiment 1")
        return

    # Plot Delay
    plot_bar_comparison(metrics_data, 'avg_delay', 'Average Task Delay Comparison', 'Delay (s)', 
                       f"{folder}/comparison_delay.png")
    # Plot Energy
    plot_bar_comparison(metrics_data, 'avg_energy', 'Average System Energy Comparison', 'Energy (J)', 
                       f"{folder}/comparison_energy.png")
    # Plot Completion Rate
    plot_bar_comparison(metrics_data, 'completion_rate', 'Task Completion Rate Comparison', 'Completion Rate', 
                       f"{folder}/comparison_completion.png")

def process_exp2_load():
    """Process Experiment 2: Load Analysis"""
    folder = "results/comparison/exp2_load"
    if not os.path.exists(folder):
        return

    rates = [1.5, 2.5, 3.5]
    algos = ["CAM_TD3", "TD3", "Greedy"]
    
    delay_data = {algo: [] for algo in algos}
    energy_data = {algo: [] for algo in algos}
    
    for rate in rates:
        for algo in algos:
            filename = f"{algo.lower()}_rate_{rate}.json"
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                data = load_result(filepath)
                metrics = get_final_metrics(data)
                delay_data[algo].append(metrics.get('avg_delay', 0))
                energy_data[algo].append(metrics.get('avg_energy', 0))
            else:
                delay_data[algo].append(0)
                energy_data[algo].append(0)

    plot_line_sensitivity(rates, delay_data, 'Task Arrival Rate (tasks/s)', 'Average Delay (s)',
                         'Impact of Task Load on Delay', f"{folder}/sensitivity_load_delay.png")
    plot_line_sensitivity(rates, energy_data, 'Task Arrival Rate (tasks/s)', 'Average Energy (J)',
                         'Impact of Task Load on Energy', f"{folder}/sensitivity_load_energy.png")

def process_exp3_scale():
    """Process Experiment 3: Scalability"""
    folder = "results/comparison/exp3_scale"
    if not os.path.exists(folder):
        return

    counts = [10, 20, 30]
    algos = ["CAM_TD3", "TD3", "Greedy"]
    
    delay_data = {algo: [] for algo in algos}
    
    for count in counts:
        for algo in algos:
            filename = f"{algo.lower()}_veh_{count}.json"
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                data = load_result(filepath)
                metrics = get_final_metrics(data)
                delay_data[algo].append(metrics.get('avg_delay', 0))
            else:
                delay_data[algo].append(0)

    plot_line_sensitivity(counts, delay_data, 'Number of Vehicles', 'Average Delay (s)',
                         'Scalability Analysis (Delay)', f"{folder}/sensitivity_scale_delay.png")

def process_exp4_ablation():
    """Process Experiment 4: Ablation"""
    folder = "results/comparison/exp4_ablation"
    if not os.path.exists(folder):
        return

    metrics_data = {}
    variants = ["cam_td3_full", "no_migration", "no_cache"]
    display_names = {"cam_td3_full": "CAM_TD3 (Full)", "no_migration": "No Migration", "no_cache": "No Cache"}
    
    for var in variants:
        filename = f"{var}.json"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            data = load_result(filepath)
            metrics = get_final_metrics(data)
            if metrics:
                metrics_data[display_names[var]] = metrics

    if metrics_data:
        plot_bar_comparison(metrics_data, 'avg_delay', 'Ablation Study: Delay', 'Delay (s)',
                           f"{folder}/ablation_delay.png")
        plot_bar_comparison(metrics_data, 'avg_energy', 'Ablation Study: Energy', 'Energy (J)',
                           f"{folder}/ablation_energy.png")

def main():
    print("🎨 Generating Experiment Plots...")
    process_exp1_core()
    process_exp2_load()
    process_exp3_scale()
    process_exp4_ablation()
    print("✅ Plotting complete.")

if __name__ == "__main__":
    main()
