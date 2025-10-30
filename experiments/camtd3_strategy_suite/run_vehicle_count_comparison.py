#!/usr/bin/env python3
"""
CAMTD3 车辆数量对比实验（六策略版本）
==========================================

【功能】
评估不同车辆数量对系统性能的影响，对比六种策略的可扩展性。
通过扫描不同的车辆数量配置，分析：
- 系统规模如何影响决策性能
- 各策略在不同规模下的适应能力
- 系统可扩展性和容量规划

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 系统可扩展性评估（Scalability）
- 验证CAMTD3在不同网络规模下的性能

【实验设计】
扫描参数: num_vehicles (车辆数量)
- 小规模: 6 辆（基础场景）
- 中小规模: 9 辆
- 标准规模: 12 辆（默认配置）
- 中大规模: 15 辆
- 大规模: 18 辆（高密度场景）

固定参数:
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本（时延+能耗）
- 平均时延（车辆越多竞争越激烈）
- 平均能耗（受负载影响）
- 归一化成本

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_vehicle_count_comparison.py \\
    --episodes 100 --suite-id vehicle_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_vehicle_count_comparison.py \\
    --episodes 500 --seed 42 --suite-id vehicle_paper

# 自定义车辆数量配置
python experiments/camtd3_strategy_suite/run_vehicle_count_comparison.py \\
    --vehicle-counts "6,12,18,24" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_vehicle_count_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 5配置 × 6策略）：约1.5-2.5小时
- 完整实验（500轮 × 5配置 × 6策略）：约6-9小时

【输出图表】
- vehicle_count_vs_cost.png: 车辆数 vs 平均成本
- vehicle_count_vs_delay.png: 车辆数 vs 平均时延
- vehicle_count_vs_energy.png: 车辆数 vs 平均能耗
- vehicle_count_vs_normalized_cost.png: 车辆数 vs 归一化成本

【论文贡献】
展示CAMTD3在不同网络规模下的优势，证明其良好的可扩展性
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import (
    STRATEGY_KEYS,
    evaluate_configs,
    strategy_label,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_VEHICLE_COUNTS = [6, 9, 12, 15, 18]


def parse_vehicle_counts(value: str) -> List[int]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_VEHICLE_COUNTS)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def plot_results(results: List[Dict[str, object]], suite_path: Path) -> None:
    vehicle_counts = [int(r["num_vehicles"]) for r in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in STRATEGY_KEYS:
            values = [r["strategies"][strat_key][metric] for r in results]
            plt.plot(vehicle_counts, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Number of Vehicles")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Vehicle Count on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_path / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "vehicle_count_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "vehicle_count_vs_delay.png")
    make_chart("avg_energy", "Average Energy (J)", "vehicle_count_vs_energy.png")
    make_chart("normalized_cost", "Normalized Cost", "vehicle_count_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "vehicle_count_vs_cost.png",
        "vehicle_count_vs_delay.png",
        "vehicle_count_vs_energy.png",
        "vehicle_count_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_path / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategy performance across different vehicle counts.")
    parser.add_argument("--vehicle-counts", type=str, default="default", help="Comma-separated vehicle counts.")
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"vehicle_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier.",
    )
    parser.add_argument("--output-root", type=str, default="results/parameter_sensitivity", help="Output root directory.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    vehicle_counts = parse_vehicle_counts(args.vehicle_counts)
    episodes = args.episodes or DEFAULT_EPISODES
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    configs: List[Dict[str, object]] = []
    for count in vehicle_counts:
        overrides = {
            "num_vehicles": count,
            "num_rsus": 4,
            "num_uavs": 2,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{count}veh",
                "label": f"{count} Vehicles",
                "overrides": overrides,
                "num_vehicles": count,
            }
        )

    suite_path = Path(args.output_root) / args.suite_id
    results = evaluate_configs(
        configs=configs,
        episodes=episodes,
        seed=seed,
        silent=args.silent,
        suite_path=suite_path,
    )

    summary = {
        "experiment_type": "vehicle_count_sensitivity",
        "suite_id": args.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": episodes,
        "seed": seed,
        "results": results,
    }
    summary_path = suite_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_path)

    print("\nVehicle Count Sensitivity Analysis Completed")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Vehicles':<12}", end="")
    for strat_key in STRATEGY_KEYS:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(STRATEGY_KEYS)))
    for record in results:
        print(f"{record['num_vehicles']:<12}", end="")
        for strat_key in STRATEGY_KEYS:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
