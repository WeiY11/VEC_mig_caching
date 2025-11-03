#!/usr/bin/env python3
"""
CAMTD3 本地计算资源成本敏感性实验
==============================

关注点
------
- 调整车辆本地 CPU 频率，观察总成本、时延成本、能耗成本的变化
- 对比六种策略（或指定子集）在本地能力提升情况下的收益
- 输出论文可用的多维指标图表
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
)
from experiments.camtd3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
)
from utils.unified_reward_calculator import UnifiedRewardCalculator

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_CPU_FREQS = [1.2, 2.0, 2.8]  # 优化: 5配置→3配置 (低/中/高)

_reward_calculator: UnifiedRewardCalculator | None = None


def _get_reward_calculator() -> UnifiedRewardCalculator:
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator(algorithm="general")
    return _reward_calculator


def parse_cpu_frequencies(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CPU_FREQS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def cost_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    from config import config as global_config  # 延迟引入避免循环依赖

    weight_delay = float(global_config.rl.reward_weight_delay)
    weight_energy = float(global_config.rl.reward_weight_energy)

    calc = _get_reward_calculator()
    delay_norm = max(calc.delay_normalizer, 1e-6)
    energy_norm = max(calc.energy_normalizer, 1e-6)

    metrics["delay_cost"] = weight_delay * (metrics["avg_delay"] / delay_norm)
    metrics["energy_cost"] = weight_energy * (metrics["avg_energy"] / energy_norm)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    cpu_freqs = [float(record["cpu_freq_ghz"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(cpu_freqs, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Local CPU Frequency (GHz)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Local CPU Frequency on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "local_cpu_vs_total_cost.png")
    make_chart("delay_cost", "Delay Cost Component", "local_cpu_vs_delay_cost.png")
    make_chart("energy_cost", "Energy Cost Component", "local_cpu_vs_energy_cost.png")
    make_chart("normalized_cost", "Normalized Cost", "local_cpu_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "local_cpu_vs_total_cost.png",
        "local_cpu_vs_delay_cost.png",
        "local_cpu_vs_energy_cost.png",
        "local_cpu_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cost impact of local computing resources across strategies."
    )
    parser.add_argument(
        "--cpu-frequencies",
        type=str,
        default="default",
        help="Comma-separated CPU frequencies (GHz) or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="local_resource_cost",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="local_resource_cost",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    cpu_freqs = parse_cpu_frequencies(args.cpu_frequencies)
    configs: List[Dict[str, object]] = []
    for freq in cpu_freqs:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "vehicle_cpu_freq": float(freq) * 1e9,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{freq:.1f}ghz",
                "label": f"{freq:.1f} GHz",
                "overrides": overrides,
                "cpu_freq_ghz": freq,
            }
        )

    suite_dir = build_suite_path(common)
    results = evaluate_configs(
        configs=configs,
        episodes=common.episodes,
        seed=common.seed,
        silent=common.silent,
        suite_path=suite_dir,
        strategies=strategy_keys,
        per_strategy_hook=cost_hook,
    )

    summary = {
        "experiment_type": "local_resource_cost_sensitivity",
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

    print("\nLocal Resource Cost Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'CPU (GHz)':<12}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['cpu_freq_ghz']:<12.1f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
