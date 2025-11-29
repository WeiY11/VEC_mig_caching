#!/usr/bin/env python3
"""
Benchmarks Visualization Script

Reads the JSON output from `run_benchmarks_vs_optimized_td3.py` and generates:
1. Reward & Cost Training Curves (Line charts with std dev shading)
2. Final Metrics Comparison (Bar charts)
3. Parameter Sensitivity Analysis (Line charts over sweep parameters)
4. HTML Summary Report

Usage:
    python Benchmarks/visualize_benchmarks.py --input results/benchmarks_sweeps/sweep_latest.json --output results/benchmarks_sweeps/report_latest
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure plot style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("deep")
ALGO_COLORS = {
    "td3": "#1f77b4",       # Blue
    "optimized_td3_ref": "#d62728", # Red
    "sac": "#2ca02c",       # Green
    "ddpg": "#ff7f0e",      # Orange
    "local": "#7f7f7f",     # Gray
    "heuristic": "#9467bd", # Purple
    "sa": "#8c564b",        # Brown
}

def load_data(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def smooth_curve(data: np.ndarray, window: int = 10) -> np.ndarray:
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values

def get_algo_color(algo_name: str) -> str:
    return ALGO_COLORS.get(algo_name.lower(), "#333333")

# -----------------------------------------------------------------------------
# 1. Training Curves (Reward & Cost Lines)
# -----------------------------------------------------------------------------

def plot_training_curves(experiment_data: Dict[str, Any], output_dir: str, exp_name: str):
    """
    Generates:
    - Reward Curve
    - Avg Delay Curve
    - Total Energy Curve
    """
    runs_data = experiment_data.get("runs", {})
    
    # Metrics configuration
    # Key in JSON -> (Y-Label, Title, Filename)
    metrics_config = [
        ("episode_rewards", "Reward", "Training Reward Convergence", "reward_curve"),
        ("avg_task_delay", "Avg Delay (s)", "Training Delay Convergence", "delay_curve"),
        ("total_energy_consumption", "Energy (J)", "Training Energy Convergence", "energy_curve"),
        ("task_completion_rate", "Completion Rate", "Task Completion Rate", "completion_curve"),
    ]
    
    for metric_key, ylabel, title, fname in metrics_config:
        plt.figure(figsize=(10, 6))
        has_data = False
        
        for algo, runs in runs_data.items():
            all_series = []
            for run in runs:
                # Check if metric is in run directly (rewards) or in episode_metrics
                series = None
                if metric_key in run:
                    series = run[metric_key]
                elif "episode_metrics" in run and metric_key in run["episode_metrics"]:
                    series = run["episode_metrics"][metric_key]
                
                if series:
                    # Handle flat lists (single value repeated) or full curves
                    if len(series) == 1 and "episodes" in run:
                         # Replicate for visualization if it's a single final value (e.g. SA)
                         series = series * run["episodes"]
                    
                    all_series.append(smooth_curve(np.array(series)))
            
            if not all_series:
                continue
                
            has_data = True
            # Pad sequences
            max_len = max(len(s) for s in all_series)
            padded_series = []
            for s in all_series:
                padded = np.pad(s, (0, max_len - len(s)), mode='edge')
                padded_series.append(padded)
            
            data_matrix = np.vstack(padded_series)
            mean_curve = np.mean(data_matrix, axis=0)
            std_curve = np.std(data_matrix, axis=0)
            x = range(1, len(mean_curve) + 1)
            
            color = get_algo_color(algo)
            plt.plot(x, mean_curve, label=algo, color=color, linewidth=2)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)
            
        if has_data:
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.title(f"{title} ({exp_name})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{fname}_{exp_name}.png"), dpi=150)
            plt.close()

# -----------------------------------------------------------------------------
# 2. Final Metrics Comparison (Bar Charts)
# -----------------------------------------------------------------------------

def plot_metrics_bars(experiment_data: Dict[str, Any], output_dir: str, exp_name: str):
    """
    Generates bar charts for final converged metrics.
    """
    runs_data = experiment_data.get("runs", {})
    
    metrics_to_plot = [
        ("avg_task_delay", "Avg Delay (s)"),
        ("total_energy_consumption", "Total Energy (J)"),
        ("task_completion_rate", "Completion Rate"),
        ("cache_hit_rate", "Cache Hit Rate"),
        ("dropped_tasks", "Dropped Tasks")
    ]
    
    # Prepare data for DataFrame
    data_records = []
    
    for algo, runs in runs_data.items():
        for run in runs:
            # Extract final metric (tail average)
            for key, label in metrics_to_plot:
                val = None
                if "episode_metrics" in run and key in run["episode_metrics"]:
                    series = run["episode_metrics"][key]
                    if series:
                        # Take average of last 10% or last 10 episodes
                        tail_len = max(1, int(len(series) * 0.1))
                        val = np.mean(series[-tail_len:])
                
                if val is not None:
                    data_records.append({
                        "Algorithm": algo,
                        "Metric": label,
                        "Value": val
                    })
    
    if not data_records:
        return

    df = pd.DataFrame(data_records)
    
    # Plot grouped bar chart for each metric (or subplots)
    # Let's do separate plots for clarity
    for key, label in metrics_to_plot:
        subset = df[df["Metric"] == label]
        if subset.empty:
            continue
            
        plt.figure(figsize=(8, 6))
        sns.barplot(data=subset, x="Algorithm", y="Value", palette=ALGO_COLORS, capsize=.1, errorbar="sd")
        plt.title(f"Final {label} Comparison ({exp_name})")
        plt.ylabel(label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"bar_{key}_{exp_name}.png"), dpi=150)
        plt.close()

# -----------------------------------------------------------------------------
# 3. Parameter Sensitivity (Line Charts)
# -----------------------------------------------------------------------------

def plot_sensitivity(all_results: Dict[str, Any], output_dir: str):
    """
    Detects sweeps and plots metrics vs parameter.
    """
    experiments = all_results.get("experiments", {})
    param_groups = {} 
    
    for exp_key, exp_data in experiments.items():
        if "=" in exp_key:
            param, val = exp_key.split("=")
            if param not in param_groups:
                param_groups[param] = {}
            try:
                val = float(val)
            except:
                pass
            param_groups[param][val] = exp_data
            
    metrics_to_plot = [
        ("episode_rewards", "Reward"),
        ("avg_task_delay", "Avg Delay (s)"),
        ("total_energy_consumption", "Energy (J)")
    ]

    for param, val_map in param_groups.items():
        sorted_vals = sorted(val_map.keys())
        
        for metric_key, ylabel in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            has_data = False
            
            # Collect data: {algo: [ (val, mean, std), ... ]}
            algo_data = {}
            
            for val in sorted_vals:
                exp_data = val_map[val]
                runs_data = exp_data.get("runs", {})
                
                for algo, runs in runs_data.items():
                    vals = []
                    for run in runs:
                        series = None
                        if metric_key == "episode_rewards" and "episode_rewards" in run:
                            series = run["episode_rewards"]
                        elif "episode_metrics" in run and metric_key in run["episode_metrics"]:
                            series = run["episode_metrics"][metric_key]
                        
                        if series:
                            tail_len = max(1, int(len(series) * 0.1))
                            vals.append(np.mean(series[-tail_len:]))
                    
                    if vals:
                        if algo not in algo_data:
                            algo_data[algo] = []
                        algo_data[algo].append((val, np.mean(vals), np.std(vals)))
            
            # Plot
            for algo, points in algo_data.items():
                if not points: continue
                has_data = True
                points.sort(key=lambda x: x[0])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                yerrs = [p[2] for p in points]
                
                color = get_algo_color(algo)
                plt.errorbar(xs, ys, yerr=yerrs, label=algo, marker='o', capsize=5, color=color, linewidth=2)
            
            if has_data:
                plt.xlabel(param.capitalize())
                plt.ylabel(ylabel)
                plt.title(f"Sensitivity: {ylabel} vs {param}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sensitivity_{param}_{metric_key}.png"), dpi=150)
                plt.close()

# -----------------------------------------------------------------------------
# HTML Report Generation
# -----------------------------------------------------------------------------

def generate_html_report(output_dir: str, json_path: str):
    images = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    
    # Group images
    sensitivity_imgs = [img for img in images if "sensitivity" in img]
    curve_imgs = [img for img in images if "curve" in img]
    bar_imgs = [img for img in images if "bar" in img]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmarks Visualization Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: #f5f5f5; color: #333; }}
            .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.05); }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 20px; color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }}
            .meta {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #eee; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; margin-top: 20px; }}
            .chart-card {{ background: white; border: 1px solid #eee; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: transform 0.2s; }}
            .chart-card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            img {{ max_width: 100%; height: auto; border-radius: 4px; }}
            .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Benchmarks Comparison Report</h1>
            <div class="meta">
                <p><strong>Source Data:</strong> {os.path.basename(json_path)}</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now()}</p>
                <p><strong>Path:</strong> {os.path.abspath(json_path)}</p>
            </div>
            
            <h2>1. Final Metrics Comparison</h2>
            <p>Comparison of converged performance metrics across algorithms.</p>
            <div class="chart-grid">
                {''.join([f'<div class="chart-card"><img src="{img}" loading="lazy"></div>' for img in bar_imgs])}
            </div>
            
            <h2>2. Training Convergence Curves</h2>
            <p>Training progress over episodes. Shaded areas represent standard deviation across seeds.</p>
            <div class="chart-grid">
                {''.join([f'<div class="chart-card"><img src="{img}" loading="lazy"></div>' for img in curve_imgs])}
            </div>
            
            <h2>3. Parameter Sensitivity Analysis</h2>
            <p>Performance variation under different environmental parameters (e.g., bandwidth, vehicle count).</p>
            <div class="chart-grid">
                {''.join([f'<div class="chart-card"><img src="{img}" loading="lazy"></div>' for img in sensitivity_imgs])}
            </div>
            
            <div class="footer">
                Generated by Benchmarks Visualization Script
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to sweep results JSON")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)
        
    data = load_data(args.input)
    ensure_dir(args.output)
    
    print(f"Loading data from {args.input}...")
    
    # 1. Plot Training Curves
    experiments = data.get("experiments", {})
    for exp_name, exp_data in experiments.items():
        # Only plot curves for 'base' to avoid clutter, or all if few
        if exp_name == "base" or len(experiments) < 5:
            print(f"Generating training curves for {exp_name}...")
            plot_training_curves(exp_data, args.output, exp_name)
            print(f"Generating bar charts for {exp_name}...")
            plot_metrics_bars(exp_data, args.output, exp_name)
        
    # 2. Plot Sensitivity
    print("Generating sensitivity analysis...")
    plot_sensitivity(data, args.output)
    
    # 3. Generate HTML
    print("Generating HTML report...")
    generate_html_report(args.output, args.input)
    
    print(f"Visualization complete. Report saved to {os.path.join(args.output, 'report.html')}")

if __name__ == "__main__":
    main()
