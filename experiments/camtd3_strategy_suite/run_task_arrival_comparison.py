#!/usr/bin/env python3
"""
CAMTD3 任务到达率对比实验（六策略版本）
==========================================

【功能】
评估不同任务到达率对系统性能的影响，对比六种策略在不同负载强度下的表现。
通过扫描不同的任务到达率配置，分析：
- 系统负载如何影响总成本和时延
- 各策略在高负载场景下的鲁棒性
- 系统容量的上限和瓶颈

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 系统可扩展性评估
- 高负载场景下的性能对比
Experiment design:
Sweep parameter: task_arrival_rate (tasks/s)
- Light load: 0.8 tasks/s
- Balanced load: 1.0 tasks/s
- Standard load: 1.2 tasks/s (default)
- High load: 1.4 tasks/s
- Stress load: 1.6 tasks/s



固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本（时延+能耗）
- 平均时延（负载越高越大）
- 任务完成率（高负载下的关键指标）
- 归一化成本

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --episodes 100 --suite-id arrival_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --episodes 500 --seed 42 --suite-id arrival_paper

# ????????????tasks/s?

python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --arrival-rates "0.5,1.0,1.5,2.0" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 6配置 × 6策略）：约1.5-3小时
- 完整实验（500轮 × 6配置 × 6策略）：约6-10小时

【输出图表】
- arrival_rate_vs_cost.png: 到达率 vs 平均成本
- arrival_rate_vs_delay.png: 到达率 vs 平均时延
- arrival_rate_vs_completion.png: 到达率 vs 任务完成率
- arrival_rate_vs_normalized_cost.png: 到达率 vs 归一化成本
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
DEFAULT_ARRIVAL_RATES = [0.8, 1.0, 1.2, 1.4, 1.6]


def parse_arrival_rates(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_ARRIVAL_RATES)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def plot_results(results: List[Dict[str, object]], suite_path: Path) -> None:
    arrival_rates = [float(r["arrival_rate"]) for r in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in STRATEGY_KEYS:
            values = [r["strategies"][strat_key][metric] for r in results]
            plt.plot(arrival_rates, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Task Arrival Rate (tasks/s)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Arrival Rate on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_path / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "arrival_rate_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "arrival_rate_vs_delay.png")
    make_chart("completion_rate", "Completion Rate", "arrival_rate_vs_completion.png")
    make_chart("normalized_cost", "Normalized Cost", "arrival_rate_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "arrival_rate_vs_cost.png",
        "arrival_rate_vs_delay.png",
        "arrival_rate_vs_completion.png",
        "arrival_rate_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_path / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategy performance across task arrival rates.")
    parser.add_argument("--arrival-rates", type=str, default="default", help="Comma-separated arrival rates.")
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"arrival_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier.",
    )
    parser.add_argument("--output-root", type=str, default="results/parameter_sensitivity", help="Output root directory.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    arrival_rates = parse_arrival_rates(args.arrival_rates)
    episodes = args.episodes or DEFAULT_EPISODES
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    configs: List[Dict[str, object]] = []
    for rate in arrival_rates:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "task_arrival_rate": float(rate),
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{rate:.2f}",
                "label": f"{rate:.2f} tasks/s",
                "overrides": overrides,
                "arrival_rate": rate,
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
        "experiment_type": "task_arrival_sensitivity",
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

    print("\nArrival Rate Sensitivity Analysis Completed")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Arrival Rate':<18}", end="")
    for strat_key in STRATEGY_KEYS:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (18 + 18 * len(STRATEGY_KEYS)))
    for record in results:
        print(f"{record['arrival_rate']:<18.2f}", end="")
        for strat_key in STRATEGY_KEYS:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
