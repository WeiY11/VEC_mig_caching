#!/usr/bin/env python3
"""
Helper plots for the unified algorithm comparison workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.setdefault("axes.prop_cycle", plt.cycler(color=plt.cm.tab10.colors))


def _sanitise(values: List[float]) -> List[float]:
    return [float(v) if v is not None else 0.0 for v in values]


def generate_metric_overview_chart(
    aggregated_results: Dict[str, Dict],
    metrics: List[str],
    output_dir: Path,
) -> Optional[Path]:
    """
    Build a grid of bar charts (mean ± std) for the selected metrics across algorithms.

    Args:
        aggregated_results: Output block returned by AlgorithmComparisonRunner.run_all.
        metrics: Ordered list of metric keys to plot.
        output_dir: Destination directory for the generated figure.

    Returns:
        Path to the saved figure, or None when no metric can be rendered.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = []
    for metric in metrics:
        if any(
            data.get("summary", {}).get(metric, {}).get("mean") is not None
            for data in aggregated_results.values()
        ):
            metrics_to_plot.append(metric)

    if not metrics_to_plot:
        return None

    num_metrics = len(metrics_to_plot)
    cols = min(3, num_metrics)
    rows = math.ceil(num_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]  # type: ignore
    elif cols == 1:
        axes = [[ax] for ax in axes]  # type: ignore

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        labels = []
        means = []
        stds = []
        for key, data in aggregated_results.items():
            metric_block = data.get("summary", {}).get(metric)
            if not metric_block or metric_block.get("mean") is None:
                continue
            labels.append(data.get("label", key))
            means.append(metric_block.get("mean"))
            stds.append(metric_block.get("std") or 0.0)

        if not labels:
            ax.axis("off")
            continue

        ax.bar(labels, _sanitise(means), yerr=_sanitise(stds), capsize=4, color="#2563eb", alpha=0.85)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Hide any remaining empty subplots
    total_slots = rows * cols
    for blank_idx in range(len(metrics_to_plot), total_slots):
        row = blank_idx // cols
        col = blank_idx % cols
        axes[row][col].axis("off")

    fig.tight_layout()
    fig_path = output_dir / "metric_overview.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _prepare_axis(values: List) -> Tuple[List[float], List[str], bool]:
    numeric_values: List[float] = []
    labels = [str(v) for v in values]
    is_numeric = True
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            is_numeric = False
            break
    if not is_numeric:
        numeric_values = list(range(len(values)))
    return numeric_values, labels, is_numeric


def generate_sweep_line_plots(
    sweep_data: Dict[str, Dict],
    output_dir: Path,
) -> Dict[str, str]:
    """
    Render line charts for scenario sweep results.

    Returns:
        Mapping of metric name to saved figure path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = sweep_data.get("values", [])
    metrics = sweep_data.get("metrics", [])
    algorithms = sweep_data.get("algorithms", {})

    x_positions, x_labels, is_numeric = _prepare_axis(values)
    figure_paths: Dict[str, str] = {}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False

        for alg_id, alg_data in algorithms.items():
            entries = alg_data.get("metrics", {}).get(metric, [])
            means = [entry.get("mean") for entry in entries]
            stds = [entry.get("std") for entry in entries]

            if not entries or all(m is None for m in means):
                continue

            y_vals = np.array([m if m is not None else np.nan for m in means], dtype=float)
            std_vals = np.array([s if s is not None else 0.0 for s in stds], dtype=float)

            ax.plot(x_positions, y_vals, marker="o", label=alg_data.get("label", alg_id))
            ax.fill_between(
                x_positions,
                y_vals - std_vals,
                y_vals + std_vals,
                alpha=0.15,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        if is_numeric:
            ax.set_xlabel(sweep_data.get("label", sweep_data.get("parameter", "")))
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel(sweep_data.get("label", sweep_data.get("parameter", "")))

        if sweep_data.get("unit"):
            ax.set_xlabel(f"{ax.get_xlabel()} ({sweep_data['unit']})")

        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig_path = output_dir / f"{sweep_data.get('name', 'sweep')}_{metric}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_paths[metric] = str(fig_path)

    return figure_paths


__all__ = ["generate_metric_overview_chart", "generate_sweep_line_plots"]
