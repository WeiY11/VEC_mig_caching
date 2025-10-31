#!/usr/bin/env python3
"""
CAMTD3 边缘节点配置对比实验（六策略版本）
==========================================

【功能】
评估不同边缘节点配置（RSU+UAV）对系统性能的影响，对比六种策略的适应能力。
通过扫描不同的RSU和UAV数量组合，分析：
- 边缘计算资源如何影响系统成本
- 各策略在不同基础设施配置下的性能
- 资源投入与性能提升的关系（成本效益分析）

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 基础设施配置优化
- 验证CAMTD3对边缘节点配置的适应性

【实验设计】
Sweep parameter: (num_rsus, num_uavs)
- Minimal: (2, 0) - RSU only
- Balanced: (3, 1) - 3 RSUs + 1 UAV
- Standard: (4, 2) - default mix
- Mid-High: (5, 2) - extra RSU coverage
- High-end: (6, 3) - added aerial support

固定参数:
- 车辆数: 12
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本
- 平均时延（受边缘节点覆盖影响）
- 单节点成本（cost_per_node）：评估资源利用效率
- 归一化成本

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --episodes 100 --suite-id edge_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --episodes 500 --seed 42 --suite-id edge_paper

# 自定义配置（格式：rsu,uav,label; ...）
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --configurations "2,0,MinConfig;4,2,Standard;8,4,MaxConfig" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 6配置 × 6策略）：约2-3小时
- 完整实验（500轮 × 6配置 × 6策略）：约7-10小时

【输出图表】
- edge_config_vs_cost.png: 节点配置 vs 平均成本
- edge_config_vs_delay.png: 节点配置 vs 平均时延
- edge_config_vs_cost_per_node.png: 节点配置 vs 单节点成本
- edge_config_vs_normalized_cost.png: 节点配置 vs 归一化成本

【论文贡献】
为VEC系统基础设施规划提供指导，展示不同配置下的性能权衡
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
DEFAULT_CONFIGS: List[Tuple[int, int, str]] = [
    (2, 0, "2 RSU, 0 UAV"),
    (3, 1, "3 RSU, 1 UAV"),
    (4, 2, "4 RSU, 2 UAV"),
    (5, 2, "5 RSU, 2 UAV"),
    (6, 3, "6 RSU, 3 UAV"),
]


def parse_configurations(value: str) -> List[Tuple[int, int, str]]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CONFIGS)
    configs: List[Tuple[int, int, str]] = []
    for item in value.split(";"):
        parts = item.strip().split(",")
        if len(parts) < 2:
            raise ValueError(f"Invalid edge node specification: {item}")
        num_rsus = int(parts[0])
        num_uavs = int(parts[1])
        label = parts[2].strip() if len(parts) > 2 else f"{num_rsus} RSU, {num_uavs} UAV"
        configs.append((num_rsus, num_uavs, label))
    return configs


def edge_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    total_nodes = int(config["num_rsus"]) + int(config["num_uavs"])
    metrics["total_nodes"] = total_nodes
    metrics["cost_per_node"] = metrics["raw_cost"] / max(total_nodes, 1)


def plot_results(results: List[Dict[str, object]], suite_path: Path) -> None:
    labels = [r["label"] for r in results]
    x_positions = range(len(results))

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in STRATEGY_KEYS:
            values = [r["strategies"][strat_key][metric] for r in results]
            plt.plot(x_positions, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xticks(x_positions, labels, rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Edge Node Configuration on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_path / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "edge_config_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "edge_config_vs_delay.png")
    make_chart("cost_per_node", "Cost per Node", "edge_config_vs_cost_per_node.png")
    make_chart("normalized_cost", "Normalized Cost", "edge_config_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "edge_config_vs_cost.png",
        "edge_config_vs_delay.png",
        "edge_config_vs_cost_per_node.png",
        "edge_config_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_path / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategy performance across edge node configurations.")
    parser.add_argument(
        "--configurations",
        type=str,
        default="default",
        help="Semicolon-separated list like '4,2,Label;6,2,Label'.",
    )
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"edge_node_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier.",
    )
    parser.add_argument("--output-root", type=str, default="results/parameter_sensitivity", help="Output root directory.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    node_configs = parse_configurations(args.configurations)
    episodes = args.episodes or DEFAULT_EPISODES
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    configs: List[Dict[str, object]] = []
    for num_rsus, num_uavs, label in node_configs:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": int(num_rsus),
            "num_uavs": int(num_uavs),
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{num_rsus}rsu_{num_uavs}uav",
                "label": label,
                "overrides": overrides,
                "num_rsus": int(num_rsus),
                "num_uavs": int(num_uavs),
            }
        )

    suite_path = Path(args.output_root) / args.suite_id
    results = evaluate_configs(
        configs=configs,
        episodes=episodes,
        seed=seed,
        silent=args.silent,
        suite_path=suite_path,
        per_strategy_hook=edge_hook,
    )

    summary = {
        "experiment_type": "edge_node_configuration",
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

    print("\nEdge Node Configuration Analysis Completed")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'RSU/UAV':<12}", end="")
    for strat_key in STRATEGY_KEYS:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(STRATEGY_KEYS)))
    for record in results:
        label = f"{record['num_rsus']} / {record['num_uavs']}"
        print(f"{label:<12}", end="")
        for strat_key in STRATEGY_KEYS:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
