"""Recompute TD3 and baseline metrics with unified normalization and refined plotting."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

BASELINE_FILE = Path("results/offloading_comparison/vehicle_sweep_20251014_011307.json")
TD3_FILES: Dict[int, Path] = {
    8: Path("results/single_agent/td3/8/training_results_20251012_234250.json"),
    12: Path("results/single_agent/td3/12/training_results_20251012_122337.json"),
    16: Path("results/single_agent/td3/16/training_results_20251010_182446.json"),
    20: Path("results/single_agent/td3/20/training_results_20251011_194418.json"),
    24: Path("results/single_agent/td3/24/training_results_20251011_205701.json"),
}
OUTPUT_DIR = Path("academic_figures/td3_real_metrics_unified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAIL_EPISODES = 200  # use last 200 episodes for TD3 statistics

plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'LocalOnly': '#e74c3c',
    'RSUOnly': '#3498db',
    'UAVOnly': '#2ecc71',
    'Random': '#9b59b6',
    'RoundRobin': '#f39c12',
    'NearestNode': '#1abc9c',
    'LoadBalance': '#34495e',
    'MinDelay': '#e67e22',
    'TD3': '#c0392b',
}

MARKERS = {
    'LocalOnly': 'o',
    'RSUOnly': 's',
    'UAVOnly': '^',
    'Random': 'D',
    'RoundRobin': 'v',
    'NearestNode': 'h',
    'LoadBalance': '*',
    'MinDelay': 'p',
    'TD3': 'X',
}

LINESTYLES = {
    'LocalOnly': '--',
    'RSUOnly': '--',
    'UAVOnly': '--',
    'Random': ':',
    'RoundRobin': ':',
    'NearestNode': '-.',
    'LoadBalance': '-.',
    'MinDelay': '-.',
    'TD3': '-',
}


def tail_mean(values: List[float], tail: int) -> float:
    if not values:
        return float('nan')
    return float(np.mean(values[-tail:]))


def load_td3_metrics() -> Dict[int, Dict[str, float]]:
    metrics = {}
    for vehicles, path in TD3_FILES.items():
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        epi_metrics = data.get('episode_metrics', {})
        avg_delay = tail_mean(epi_metrics.get('avg_delay', []), TAIL_EPISODES)
        avg_energy_j = tail_mean(epi_metrics.get('total_energy', []), TAIL_EPISODES)
        avg_completion = tail_mean(epi_metrics.get('task_completion_rate', []), TAIL_EPISODES)
        cost = 2.0 * avg_delay + 1.2 * (avg_energy_j / 1000.0)
        metrics[vehicles] = {
            'strategy': 'TD3',
            'vehicles': vehicles,
            'avg_delay': avg_delay,
            'avg_energy_j': avg_energy_j,
            'avg_energy_kj': avg_energy_j / 1000.0,
            'avg_completion': avg_completion,
            'avg_weighted_cost': cost,
        }
    return metrics


def load_baseline_metrics() -> List[Dict[str, float]]:
    with BASELINE_FILE.open('r', encoding='utf-8') as f:
        data = json.load(f)
    values: List[int] = data['values']
    rows: List[Dict[str, float]] = []
    for strategy, records in data['results'].items():
        for vehicles, record in zip(values, records):
            if vehicles not in TD3_FILES:
                continue
            avg_delay = record['avg_delay']
            avg_energy_j = record['avg_energy']
            avg_completion = record['avg_completion_rate']
            cost = 2.0 * avg_delay + 1.2 * (avg_energy_j / 1000.0)
            rows.append({
                'strategy': strategy,
                'vehicles': vehicles,
                'avg_delay': avg_delay,
                'avg_energy_j': avg_energy_j,
                'avg_energy_kj': avg_energy_j / 1000.0,
                'avg_completion': avg_completion,
                'avg_weighted_cost': cost,
            })
    return rows


def build_dataframe() -> pd.DataFrame:
    rows = load_baseline_metrics()
    rows.extend(load_td3_metrics().values())
    df = pd.DataFrame(rows)
    df.sort_values(['vehicles', 'strategy'], inplace=True)
    return df


def plot_metrics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = sorted(df['vehicles'].unique())
    metrics = {
        '(a) Average Delay (s)': ('avg_delay', False),
        '(b) Energy Consumption (kJ)': ('avg_energy_kj', False),
        '(c) Task Completion Rate (%)': ('avg_completion', True),
        '(d) Weighted Cost (↓)': ('avg_weighted_cost', False),
    }
    for (title, (metric, is_percentage)), ax in zip(metrics.items(), axes.flatten()):
        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy].set_index('vehicles').reindex(x)
            y = subset[metric].values
            if is_percentage:
                y = y * 100.0
            ax.plot(
                x,
                y,
                label=strategy,
                color=COLORS.get(strategy, '#7f8c8d'),
                marker=MARKERS.get(strategy, 'o'),
                linestyle=LINESTYLES.get(strategy, '-'),
                linewidth=3 if strategy == 'TD3' else 1.5,
                alpha=0.9 if strategy == 'TD3' else 0.7,
            )
        ax.set_xlabel('Number of Vehicles')
        ylabel = title.split('(')[0].strip()
        ax.set_ylabel(ylabel if not ylabel.endswith('↓') else 'Weighted Cost')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xticks(x)
    plt.tight_layout()
    path = OUTPUT_DIR / 'td3_vs_baseline_unified'
    plt.savefig(f"{path}.pdf")
    plt.savefig(f"{path}.png")
    plt.close()
    print(f"[OK] 保存图表: {path}.(pdf/png)")


def plot_weighted_cost(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = sorted(df['vehicles'].unique())
    strategies = ['LocalOnly', 'RSUOnly', 'UAVOnly', 'Random', 'TD3']
    for strategy in strategies:
        subset = df[df['strategy'] == strategy].set_index('vehicles').reindex(x)
        y = subset['avg_weighted_cost'].values
        ax.plot(
            x,
            y,
            label=strategy,
            color=COLORS.get(strategy, '#7f8c8d'),
            marker=MARKERS.get(strategy, 'o'),
            linestyle=LINESTYLES.get(strategy, '-'),
            linewidth=3 if strategy == 'TD3' else 1.5,
            markersize=10 if strategy == 'TD3' else 6,
        )
        if strategy == 'TD3':
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords='offset points', xytext=(0, -12), ha='center', color=COLORS['TD3'])
    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel('Weighted Cost (unified)')
    ax.set_title('Unified Cost Comparison (TD3 vs. Baselines)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(x)
    plt.tight_layout()
    path = OUTPUT_DIR / 'td3_weighted_cost_unified'
    plt.savefig(f"{path}.pdf")
    plt.savefig(f"{path}.png")
    plt.close()
    print(f"[OK] 保存图表: {path}.(pdf/png)")


def save_report(df: pd.DataFrame) -> None:
    report = []
    report.append("# TD3与Baseline统一归一指标对比\n\n")
    report.append(f"生成时间: {np.datetime64('now')}\n\n")
    report.append("## 1. 统一成本公式\n\n")
    report.append("`cost = 2.0 * delay + 1.2 * (energy / 1000)`\n\n")
    report.append("## 2. 指标表 (delay: s, energy: kJ)\n\n")
    pivot = df.pivot(index='vehicles', columns='strategy', values='avg_weighted_cost')
    report.append(pivot.round(3).to_markdown())
    report.append("\n\n")
    report.append("### 能耗 (kJ) 对照表\n\n")
    energy_table = df.pivot(index='vehicles', columns='strategy', values='avg_energy_kj')
    report.append(energy_table.round(3).to_markdown())
    report.append("\n\n")
    report.append("## 3. 观察\n\n")
    td3 = df[df['strategy'] == 'TD3'].set_index('vehicles')
    for strategy in ['LocalOnly', 'RSUOnly', 'UAVOnly', 'Random']:
        base = df[df['strategy'] == strategy].set_index('vehicles')
        if base.empty:
            continue
        report.append(f"### {strategy}\n")
        for vehicles in td3.index:
            delta = td3.loc[vehicles, 'avg_weighted_cost'] - base.loc[vehicles, 'avg_weighted_cost']
            report.append(f"- {vehicles}辆车: TD3 - {strategy} = {delta:+.3f}\n")
        report.append("\n")
    report_path = OUTPUT_DIR / 'td3_vs_baseline_unified_report.md'
    report_path.write_text(''.join(report), encoding='utf-8')
    print(f"[OK] 保存报告: {report_path}")


def main() -> None:
    df = build_dataframe()
    csv_path = OUTPUT_DIR / 'td3_vs_baseline_unified_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"[OK] 保存数据表: {csv_path}")
    plot_metrics(df)
    plot_weighted_cost(df)
    save_report(df)
    print("[COMPLETE] 统一归一结果生成完毕")


if __name__ == '__main__':
    main()
