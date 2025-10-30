#!/usr/bin/env python3
"""
CAMTD3 任务数据大小对比实验（六策略版本）
==========================================

【功能】
评估不同任务数据大小对系统性能的影响，对比六种策略在不同数据负载下的表现。
通过扫描不同的任务数据大小范围，分析：
- 数据传输开销如何影响系统成本
- 各策略对数据大小变化的适应能力
- 大数据任务下的决策优化

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 数据密集型场景下的性能评估
- 验证CAMTD3在不同任务负载下的鲁棒性

【实验设计】
扫描参数: task_data_size (任务数据大小 KB)
- 小任务: 50-150 KB（轻量级任务）
- 中小任务: 100-300 KB（常规任务）
- 中等任务: 200-500 KB（标准配置）
- 中大任务: 300-800 KB（数据密集型）
- 大任务: 500-1000 KB（重负载场景）

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本（时延+能耗）
- 平均时延（数据大小影响传输时间）
- 平均能耗（数据大小影响传输能耗）
- 归一化成本（便于对比）

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --episodes 100 --suite-id datasize_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --episodes 500 --seed 42 --suite-id datasize_paper

# 自定义数据大小配置（格式：min,max; ...）
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --data-sizes "100,200;200,400;400,800" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 5配置 × 6策略）：约1.5-2.5小时
- 完整实验（500轮 × 5配置 × 6策略）：约6-9小时

【输出图表】
- data_size_vs_cost.png: 数据大小 vs 平均成本
- data_size_vs_delay.png: 数据大小 vs 平均时延
- data_size_vs_energy.png: 数据大小 vs 平均能耗
- data_size_vs_normalized_cost.png: 数据大小 vs 归一化成本
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import (
    STRATEGY_KEYS,
    enrich_with_normalized_costs,
    run_strategy_suite,
    strategy_label,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

DEFAULT_DATA_SIZE_CONFIGS: List[Tuple[int, int, str]] = [
    (50, 150, "Small (50-150KB)"),
    (100, 300, "Medium-Small (100-300KB)"),
    (200, 500, "Medium (200-500KB)"),
    (300, 800, "Medium-Large (300-800KB)"),
    (500, 1000, "Large (500-1000KB)"),
]


def parse_data_sizes(value: str) -> List[Tuple[int, int, str]]:
    """
    解析数据大小配置字符串
    
    【功能】
    将命令行输入的数据大小字符串解析为(min, max, label)元组列表
    
    【参数】
    value: str - 数据大小字符串，格式为 "100,200;200,400" 或 "default"
    
    【返回值】
    List[Tuple[int, int, str]] - (最小值, 最大值, 标签)元组列表
    
    【示例】
    "100,300;500,800" -> [(100, 300, "100-300KB"), (500, 800, "500-800KB")]
    """
    if not value or value.strip().lower() == "default":
        return [tuple(cfg) for cfg in DEFAULT_DATA_SIZE_CONFIGS]
    
    configs: List[Tuple[int, int, str]] = []
    for item in value.split(";"):
        parts = item.strip().split(",")
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
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个数据大小配置的实验
    
    【功能】
    对指定的数据大小范围，训练并评估六种策略的性能
    
    【参数】
    min_kb: int - 最小任务数据大小（KB）
    max_kb: int - 最大任务数据大小（KB）
    label: str - 配置标签（用于展示）
    episodes: int - 训练轮数
    seed: int - 随机种子
    silent: bool - 是否静默模式
    suite_path: Path - 输出目录路径
    
    【返回值】
    Dict[str, Any] - 包含所有策略性能指标的字典
    
    【实验流程】
    1. 配置任务数据大小参数
    2. 固定网络拓扑（12车辆+4RSU+2UAV）
    3. 训练六种策略
    4. 保存每个策略的详细结果
    5. 返回汇总指标
    """
    print(f"\n{'='*60}")
    print(f"Running data size configuration: {label}")
    print(f"{'='*60}")

    override_scenario = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "task_data_size_min_kb": min_kb,
        "task_data_size_max_kb": max_kb,
        "override_topology": True,
    }

    config_dir = suite_path / f"{min_kb}_{max_kb}"
    config_dir.mkdir(parents=True, exist_ok=True)

    strategies_raw = run_strategy_suite(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
    )
    strategies = enrich_with_normalized_costs(strategies_raw)

    for strat_key, metrics in strategies.items():
        detail_path = config_dir / f"{strat_key}.json"
        detail_path.write_text(
            json.dumps(
                {
                    "strategy": strat_key,
                    "strategy_label": strategy_label(strat_key),
                    "min_kb": min_kb,
                    "max_kb": max_kb,
                    **metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    for strat_key in STRATEGY_KEYS:
        metrics = strategies[strat_key]
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
        "strategies": strategies,
        "episodes": episodes,
        "seed": seed,
    }


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    avg_sizes = [r["avg_kb"] for r in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in STRATEGY_KEYS:
            values = [r["strategies"][strat_key][metric] for r in results]
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
        plt.savefig(suite_path / filename, dpi=300, bbox_inches="tight")
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
        print(f"  - {suite_path / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategy performance across different task data sizes.")
    parser.add_argument("--data-sizes", type=str, default="default", help="Ranges in 'min,max;...' format or 'default'.")
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"data_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier.",
    )
    parser.add_argument("--output-root", type=str, default="results/parameter_sensitivity", help="Output root directory.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    data_size_configs = parse_data_sizes(args.data_sizes)
    episodes = args.episodes or DEFAULT_EPISODES
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for min_kb, max_kb, label in data_size_configs:
        entry = run_single_config(
            min_kb=min_kb,
            max_kb=max_kb,
            label=label,
            episodes=episodes,
            seed=seed,
            silent=args.silent,
            suite_path=suite_path,
        )
        results.append(entry)

    summary = {
        "experiment_type": "data_size_sensitivity",
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

    print(f"\nData Size Sensitivity Analysis Completed")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Data Size':<18}", end="")
    for strat_key in STRATEGY_KEYS:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (18 + 18 * len(STRATEGY_KEYS)))
    for record in results:
        print(f"{record['label']:<18}", end="")
        for strat_key in STRATEGY_KEYS:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
