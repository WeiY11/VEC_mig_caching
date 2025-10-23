#!/usr/bin/env python3
"""
Generate publication-ready comparison plots across algorithms and vehicle counts.

该脚本自动扫描训练/评估结果 JSON 文件，提取任务完成率、平均时延、能耗效率、数据丢失率等指标，
并输出符合论文标准的折线图与汇总数据表。支持 TD3/CAM-TD3 等深度强化学习算法，以及
RoundRobin、SimulatedAnnealing 等启发式或元启发式方法。

用法示例：
    python visualization/generate_paper_comparison.py \\
        --results-root results \\
        --algorithms TD3 CAM-TD3 TD3_Xuance RoundRobin Random LocalOnly RSUOnly SimulatedAnnealing \\
        --vehicle-counts 8 12 16 20 \\
        --output-dir academic_figures/paper_comparison
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 样式配置：符合 IEEE/Springer 要求 ---
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

ALGORITHM_LABELS = {
    "td3": "TD3",
    "cam-td3": "CAM-TD3",
    "td3_xuance": "TD3-xuance",
    "td3-xuance": "TD3-xuance",
    "random": "Random",
    "roundrobin": "RoundRobin",
    "localonly": "LocalOnly",
    "rsuonly": "RSUOnly",
    "simulatedannealing": "SimulatedAnnealing",
    "greedyenergy": "GreedyEnergy",
}

ALGORITHM_COLORS = {
    "TD3": "#c0392b",
    "CAM-TD3": "#8e44ad",
    "TD3-xuance": "#2980b9",
    "Random": "#9b59b6",
    "RoundRobin": "#f39c12",
    "LocalOnly": "#e74c3c",
    "RSUOnly": "#3498db",
    "SimulatedAnnealing": "#2ecc71",
    "GreedyEnergy": "#16a085",
}

ALGORITHM_MARKERS = {
    "TD3": "X",
    "CAM-TD3": "D",
    "TD3-xuance": "s",
    "Random": "o",
    "RoundRobin": "v",
    "LocalOnly": "^",
    "RSUOnly": "P",
    "SimulatedAnnealing": "h",
    "GreedyEnergy": "*",
}


@dataclass
class RunMetrics:
    algorithm: str
    label: str
    category: str
    vehicles: int
    seed: Optional[int]
    episodes: Optional[int]
    avg_latency: Optional[float]
    avg_latency_std: Optional[float]
    completion_rate: Optional[float]
    completion_rate_std: Optional[float]
    avg_energy: Optional[float]
    avg_energy_std: Optional[float]
    energy_efficiency: Optional[float]
    data_loss_rate: Optional[float]
    data_loss_rate_std: Optional[float]
    training_time_hours: Optional[float]
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate algorithm metrics and generate publication-quality plots.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing training_results_*.json or *_seed_*.json files.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list({label for label in ALGORITHM_LABELS.values()}),
        help="Target algorithm names to include (case-insensitive).",
    )
    parser.add_argument(
        "--vehicle-counts",
        nargs="+",
        type=int,
        default=[8, 12, 16, 20],
        help="Vehicle counts to consider when aggregating results.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.2,
        help="Fraction of the latest episodes used for DRL tail statistics (0-1).",
    )
    parser.add_argument(
        "--tail-min",
        type=int,
        default=50,
        help="Minimum number of episodes to include in DRL tail statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("academic_figures/paper_comparison"),
        help="Directory to store generated plots and aggregated CSV.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only compute and export aggregated metrics without plotting.",
    )
    return parser.parse_args()


def normalise_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "-")


def find_snapshot(path: Path, cache: Dict[Path, Dict]) -> Optional[Dict]:
    """Search upwards for comparison_config_snapshot.json to recover scenario info."""
    for parent in [path.parent, path.parent.parent, path.parent.parent.parent]:
        if parent is None:
            continue
        snapshot_path = parent / "comparison_config_snapshot.json"
        if snapshot_path.exists():
            if snapshot_path not in cache:
                with snapshot_path.open("r", encoding="utf-8") as fh:
                    cache[snapshot_path] = json.load(fh)
            return cache[snapshot_path]
    return None


def compute_tail_stats(values: Iterable[float], tail_fraction: float, tail_min: int) -> Tuple[Optional[float], Optional[float]]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None, None
    total = len(filtered)
    tail_len = max(1, min(total, max(tail_min, int(math.ceil(total * tail_fraction)))))
    tail = np.array(filtered[-tail_len:], dtype=float)
    mean = float(np.mean(tail))
    std = float(np.std(tail, ddof=1)) if tail_len > 1 else 0.0
    return mean, std


def parse_drl_training(path: Path, data: Dict, tail_fraction: float, tail_min: int) -> RunMetrics:
    algorithm = data.get("algorithm", path.parent.name)
    algorithm_key = normalise_name(algorithm)
    label = ALGORITHM_LABELS.get(algorithm_key, algorithm)
    category = data.get("agent_type", "drl")
    scenario = data.get("override_scenario") or {}
    system_cfg = data.get("system_config") or {}
    vehicles = scenario.get("num_vehicles") or system_cfg.get("num_vehicles")
    if vehicles is None:
        raise ValueError(f"Unable to infer vehicle count for {path}")

    seed = system_cfg.get("random_seed")
    training_cfg = data.get("training_config") or {}
    episodes = training_cfg.get("num_episodes")
    training_time = training_cfg.get("training_time_hours")

    episode_metrics = data.get("episode_metrics") or {}
    avg_delay, avg_delay_std = compute_tail_stats(
        episode_metrics.get("avg_delay", []), tail_fraction, tail_min
    )
    completion_rate, completion_std = compute_tail_stats(
        episode_metrics.get("task_completion_rate", []), tail_fraction, tail_min
    )
    energy, energy_std = compute_tail_stats(
        episode_metrics.get("total_energy", []), tail_fraction, tail_min
    )
    data_loss, data_loss_std = compute_tail_stats(
        episode_metrics.get("data_loss_ratio_bytes", []), tail_fraction, tail_min
    )

    if completion_rate is None and data.get("final_performance"):
        completion_rate = data["final_performance"].get("avg_completion")
        completion_std = 0.0
    if avg_delay is None and data.get("final_performance"):
        avg_delay = data["final_performance"].get("avg_delay")
        avg_delay_std = 0.0

    energy_eff = None
    if energy and completion_rate is not None:
        # 两个值均为 episode 均值；能耗单位：焦耳
        energy_eff = completion_rate / energy if energy != 0 else None

    return RunMetrics(
        algorithm=algorithm,
        label=label,
        category=category,
        vehicles=int(vehicles),
        seed=seed,
        episodes=episodes,
        avg_latency=avg_delay,
        avg_latency_std=avg_delay_std,
        completion_rate=completion_rate,
        completion_rate_std=completion_std,
        avg_energy=energy,
        avg_energy_std=energy_std,
        energy_efficiency=energy_eff,
        data_loss_rate=data_loss,
        data_loss_rate_std=data_loss_std,
        training_time_hours=training_time,
        source_file=str(path),
    )


def parse_summary_result(
    path: Path,
    data: Dict,
    snapshot_cache: Dict[Path, Dict],
) -> Optional[RunMetrics]:
    algorithm = data.get("algorithm")
    if not algorithm:
        return None
    algorithm_key = normalise_name(algorithm)
    label = ALGORITHM_LABELS.get(algorithm_key, algorithm)
    category = data.get("category", "heuristic")

    snapshot = find_snapshot(path, snapshot_cache)
    scenario = (snapshot or {}).get("scenario") or {}
    vehicles = scenario.get("num_vehicles")
    if vehicles is None:
        # 部分 sweep 的 scenario 写在 per_seed_runs.json 中，可尝试再解析
        per_seed_runs = path.parent.parent / "per_seed_runs.json"
        if per_seed_runs.exists():
            with per_seed_runs.open("r", encoding="utf-8") as fh:
                runs = json.load(fh)
            first_run = runs[0] if runs else {}
            vehicles = first_run.get("scenario", {}).get("num_vehicles")
    if vehicles is None:
        # 若仍无法识别，则回退为当前目录名中的数值（如 vehicle_scaling_8）
        for token in path.parts[::-1]:
            if token.isdigit():
                vehicles = int(token)
                break
    if vehicles is None:
        return None

    summary = data.get("summary") or {}
    avg_delay = summary.get("avg_delay")
    completion_rate = summary.get("avg_completion_rate")
    avg_energy = summary.get("avg_energy")
    data_loss = summary.get("data_loss_ratio")
    data_loss_alt = summary.get("data_loss_ratio_bytes")
    if data_loss is None and data_loss_alt is not None:
        data_loss = data_loss_alt

    energy_eff = None
    if avg_energy and completion_rate is not None:
        energy_eff = completion_rate / avg_energy if avg_energy != 0 else None

    return RunMetrics(
        algorithm=algorithm,
        label=label,
        category=category,
        vehicles=int(vehicles),
        seed=data.get("seed"),
        episodes=data.get("episodes"),
        avg_latency=avg_delay,
        avg_latency_std=0.0 if avg_delay is not None else None,
        completion_rate=completion_rate,
        completion_rate_std=0.0 if completion_rate is not None else None,
        avg_energy=avg_energy,
        avg_energy_std=0.0 if avg_energy is not None else None,
        energy_efficiency=energy_eff,
        data_loss_rate=data_loss,
        data_loss_rate_std=0.0 if data_loss is not None else None,
        training_time_hours=summary.get("training_time_hours"),
        source_file=str(path),
    )


def collect_metrics(
    results_root: Path,
    algorithms: List[str],
    tail_fraction: float,
    tail_min: int,
) -> List[RunMetrics]:
    target_keys = {normalise_name(name) for name in algorithms}
    metrics: List[RunMetrics] = []
    snapshot_cache: Dict[Path, Dict] = {}

    patterns = ("training_results_*.json", "*_seed_*.json", "*_aggregated.json")
    for pattern in patterns:
        for path in results_root.rglob(pattern):
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            if "final_performance" in data or "episode_metrics" in data:
                algorithm = data.get("algorithm", "")
                if normalise_name(algorithm) not in target_keys:
                    continue
                run = parse_drl_training(path, data, tail_fraction, tail_min)
                metrics.append(run)
            elif data.get("summary") and data.get("algorithm"):
                if normalise_name(data.get("algorithm", "")) not in target_keys:
                    continue
                run = parse_summary_result(path, data, snapshot_cache)
                if run:
                    metrics.append(run)

    return metrics


def build_dataframe(metrics: List[RunMetrics]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        rows.append(
            {
                "algorithm": m.algorithm,
                "label": m.label,
                "category": m.category,
                "vehicles": m.vehicles,
                "seed": m.seed,
                "episodes": m.episodes,
                "avg_latency": m.avg_latency,
                "avg_latency_std": m.avg_latency_std,
                "completion_rate": m.completion_rate,
                "completion_rate_std": m.completion_rate_std,
                "avg_energy": m.avg_energy,
                "avg_energy_std": m.avg_energy_std,
                "energy_efficiency": m.energy_efficiency,
                "data_loss_rate": m.data_loss_rate,
                "data_loss_rate_std": m.data_loss_rate_std,
                "training_time_hours": m.training_time_hours,
                "source_file": m.source_file,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.sort_values(["label", "vehicles", "seed"], inplace=True)
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds for each algorithm/vehicle combination."""
    metrics = ["avg_latency", "completion_rate", "avg_energy", "energy_efficiency", "data_loss_rate"]
    grouped = df.groupby(["label", "vehicles"], dropna=False)
    summary_rows = []
    for (label, vehicles), group in grouped:
        entry = {
            "label": label,
            "vehicles": vehicles,
            "num_runs": int(group.shape[0]),
        }
        for metric in metrics:
            series = group[metric].dropna()
            entry[f"{metric}_mean"] = float(series.mean()) if not series.empty else math.nan
            entry[f"{metric}_std"] = float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0
            entry[f"{metric}_ci95"] = (
                1.96 * entry[f"{metric}_std"] / math.sqrt(series.shape[0])
                if series.shape[0] > 1
                else 0.0
            )
        summary_rows.append(entry)
    summary = pd.DataFrame(summary_rows)
    summary.sort_values(["vehicles", "label"], inplace=True)
    return summary


def plot_metric(ax, summary: pd.DataFrame, metric: str, ylabel: str, percentage: bool = False, invert: bool = False):
    labels = summary["label"].unique()
    vehicle_values = sorted(summary["vehicles"].unique())
    for label in labels:
        subset = summary[summary["label"] == label].set_index("vehicles").reindex(vehicle_values)
        if subset.empty:
            continue
        y = subset[f"{metric}_mean"].values
        if percentage:
            y = y * 100.0
        ax.plot(
            vehicle_values,
            y,
            label=label,
            color=ALGORITHM_COLORS.get(label, "#7f8c8d"),
            marker=ALGORITHM_MARKERS.get(label, "o"),
            linewidth=2.4 if label in ("TD3", "CAM-TD3") else 1.8,
            alpha=0.95 if label in ("TD3", "CAM-TD3") else 0.8,
        )
        std = subset[f"{metric}_ci95"].values
        if percentage:
            std = std * 100.0
        if np.any(np.isfinite(std)) and subset["num_runs"].fillna(0).astype(int).gt(1).any():
            ax.fill_between(
                vehicle_values,
                y - std,
                y + std,
                color=ALGORITHM_COLORS.get(label, "#95a5a6"),
                alpha=0.15,
                linewidth=0,
            )
    ax.set_xlabel("Number of Vehicles")
    ax.set_ylabel(ylabel)
    if invert:
        ax.invert_yaxis()
    ax.grid(True)
    ax.set_xticks(vehicle_values)


def generate_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_metric(axes[0, 0], summary, "completion_rate", "(a) Task Completion Rate (%)", percentage=True)
    plot_metric(axes[0, 1], summary, "avg_latency", "(b) Average Latency (s)")
    # 能耗效率单位为 completed tasks per Joule，可乘 1e3 便于阅读
    efficiency_summary = summary.copy()
    efficiency_summary["energy_efficiency_mean"] = efficiency_summary["energy_efficiency_mean"] * 1e3
    efficiency_summary["energy_efficiency_ci95"] = efficiency_summary["energy_efficiency_ci95"] * 1e3
    plot_metric(axes[1, 0], efficiency_summary, "energy_efficiency", "(c) Energy Efficiency (1e-3 tasks/J)")
    plot_metric(axes[1, 1], summary, "data_loss_rate", "(d) Data Loss Ratio (%)", percentage=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    figure_path = output_dir / "algorithm_comparison_metrics"
    plt.savefig(f"{figure_path}.pdf")
    plt.savefig(f"{figure_path}.png")
    plt.close(fig)
    print(f"[OK] 图表已生成: {figure_path}.pdf / .png")


def main() -> None:
    args = parse_args()
    metrics = collect_metrics(args.results_root, args.algorithms, args.tail_fraction, args.tail_min)
    if not metrics:
        raise SystemExit("未在指定目录中找到匹配的结果文件，请先运行训练/比较实验。")

    df = build_dataframe(metrics)
    summary = aggregate_metrics(df[df["vehicles"].isin(args.vehicle_counts)])
    if summary.empty:
        raise SystemExit("未能聚合到指定车辆规模的数据，请检查 --vehicle-counts 或结果目录。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / "aggregated_metrics.csv"
    summary.to_csv(metrics_csv, index=False)
    print(f"[OK] 指标汇总表已保存: {metrics_csv}")

    runs_csv = args.output_dir / "per_run_metrics.csv"
    df.to_csv(runs_csv, index=False)
    print(f"[OK] 单次运行指标已保存: {runs_csv}")

    if not args.skip_plots:
        generate_plots(summary, args.output_dir)


if __name__ == "__main__":
    main()

