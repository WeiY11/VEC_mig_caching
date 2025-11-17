#!/usr/bin/env python3
"""
TD3 任务数据大小对比实验
==========================

核心目的
--------
- 评估不同任务数据大小区间对系统性能的影响
- 对比六种策略（或指定子集）在数据密集度变化下的表现
- 为论文绘图与参数敏感性分析提供一站式脚本
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.td3_strategy_suite.strategy_runner import (
    enrich_with_normalized_costs,
    run_strategy_suite,
    strategy_label,
)
from experiments.td3_strategy_suite.visualization_utils import (
    add_line_charts,
    print_chart_summary,
)
from experiments.td3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
    get_default_scenario_overrides,
,
    validate_td3_episodes,
)
from experiments.td3_strategy_suite.parameter_presets import default_data_size_configs

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

# 🎯 基于当前任务数据大小配置动态生成（默认划分三档）
DEFAULT_DATA_SIZE_CONFIGS: List[Tuple[int, int, str]] = default_data_size_configs()


def parse_data_sizes(value: str) -> List[Tuple[int, int, str]]:
    if not value or value.strip().lower() == "default":
        return [tuple(cfg) for cfg in DEFAULT_DATA_SIZE_CONFIGS]

    configs: List[Tuple[int, int, str]] = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid data size format: {item}. Expected 'min,max'")
        min_kb, max_kb = int(parts[0]), int(parts[1])
        label = f"{min_kb}-{max_kb}KB"
        configs.append((min_kb, max_kb, label))
    return configs


def run_single_config(
    min_kb: int,
    max_kb: int,
    label: str,
    episodes: int,
    seed: int,
    silent: bool,
    suite_dir: Path,
    strategy_keys: List[str],
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Running data size configuration: {label}")
    print(f"{'=' * 60}")

    override_scenario = get_default_scenario_overrides(
        task_data_size_min_kb=min_kb,
        task_data_size_max_kb=max_kb,
    )

    config_dir = suite_dir / f"{min_kb}_{max_kb}"
    config_dir.mkdir(parents=True, exist_ok=True)

    raw = run_strategy_suite(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategy_keys,
    )
    enriched = enrich_with_normalized_costs(raw)

    for strat_key in strategy_keys:
        metrics = enriched[strat_key]
        detail_path = config_dir / f"{strat_key}.json"
        detail_payload = {
            "strategy": strat_key,
            "strategy_label": strategy_label(strat_key),
            "min_kb": min_kb,
            "max_kb": max_kb,
            **metrics,
        }
        detail_path.write_text(json.dumps(detail_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"  - {strategy_label(strat_key)}: "
            f"Cost={metrics['raw_cost']:.4f} Delay={metrics['avg_delay']:.4f}s "
            f"Energy={metrics['avg_energy']:.2f}J"
        )

    return {
        "label": label,
        "min_kb": min_kb,
        "max_kb": max_kb,
        "avg_kb": (min_kb + max_kb) / 2.0,
        "strategies": enriched,
        "episodes": episodes,
        "seed": seed,
    }


def plot_results(results: List[Dict[str, Any]], suite_dir: Path, strategy_keys: List[str]) -> None:
    avg_sizes = [record["avg_kb"] for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(
                avg_sizes,
                values,
                marker="o",
                linewidth=2,
                label=strategy_label(strat_key),
            )
        plt.xlabel("Average Task Data Size (KB)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Data Size on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "data_size_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "data_size_vs_delay.png")
    make_chart("avg_energy", "Average Energy (J)", "data_size_vs_energy.png")
    make_chart("normalized_cost", "Normalized Cost", "data_size_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "data_size_vs_cost.png",
        "data_size_vs_delay.png",
        "data_size_vs_energy.png",
        "data_size_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strategy performance across different task data sizes."
    )
    parser.add_argument(
        "--data-sizes",
        type=str,
        default="default",
        help="Ranges in 'min,max;...' format or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="data_size",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="data_size",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    data_size_configs = parse_data_sizes(args.data_sizes)

    # 🎯 验证TD3训练轮次
    validate_td3_episodes(common.episodes, strategy_keys)
    
    suite_dir = build_suite_path(common)
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for min_kb, max_kb, label in data_size_configs:
        results.append(
            run_single_config(
                min_kb=min_kb,
                max_kb=max_kb,
                label=label,
                episodes=common.episodes,
                seed=common.seed,
                silent=common.silent,
                suite_dir=suite_dir,
                strategy_keys=strategy_keys,
            )
        )

    summary = {
        "experiment_type": "data_size_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    print("\nData Size Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Data Size':<18}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (18 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['label']:<18}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
