#!/usr/bin/env python3
"""
TD3 混合负载场景对比实验
============================

研究目标
--------
- 评估不同任务类型组合对系统性能的影响
- 模拟真实场景：时延敏感、计算密集、数据密集型任务的混合
- 分析各策略在异构任务负载下的适应能力
- 验证算法对不同QoS需求的支持能力

场景设计
--------
1. 计算密集型 (Compute-Intensive): 高CPU密度，小数据量
2. 数据密集型 (Data-Intensive): 大数据量，中等CPU需求
3. 时延敏感型 (Latency-Critical): 低时延要求，中等资源需求
4. 均衡混合型 (Balanced): 各类型任务均衡分布
5. 极端混合型 (Extreme): 所有类型同时出现，压力最大

学术价值
--------
- 支撑论文中关于"异构任务QoS保证"的讨论
- 验证算法在真实复杂场景下的泛化能力
- 为多样化应用场景提供性能基准
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ========== 添加项目根目录到 Python 路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.td3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
    tail_mean,
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
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

# 混合负载配置：(complexity, data_size_kb, arrival_rate, label)
WORKLOAD_CONFIGS = [
    {
        "key": "compute_intensive",
        "label": "Compute-Intensive",
        "description": "High CPU density with lightweight payloads",
        "task_complexity_multiplier": 1.8,
        "task_compute_density": 650.0,
        "task_data_size_kb": 180.0,
        "task_arrival_rate": 1.2,
    },
    {
        "key": "data_intensive",
        "label": "Data-Intensive",
        "description": "Large data payloads and moderate compute demand",
        "task_complexity_multiplier": 1.3,
        "task_compute_density": 420.0,
        "task_data_size_kb": 580.0,
        "task_arrival_rate": 1.0,
    },
    {
        "key": "latency_critical",
        "label": "Latency-Critical",
        "description": "Burst arrival patterns requiring tight latency",
        "task_complexity_multiplier": 1.1,
        "task_compute_density": 360.0,
        "task_data_size_kb": 240.0,
        "task_arrival_rate": 1.8,
    },
    {
        "key": "balanced",
        "label": "Balanced Mix",
        "description": "Even blend of latency-, compute-, and data-oriented tasks",
        "task_complexity_multiplier": 1.4,
        "task_compute_density": 470.0,
        "task_data_size_kb": 360.0,
        "task_arrival_rate": 1.4,
    },
    {
        "key": "extreme",
        "label": "Extreme Blend",
        "description": "Simultaneous heavy compute, data, and arrival bursts",
        "task_complexity_multiplier": 2.1,
        "task_compute_density": 720.0,
        "task_data_size_kb": 620.0,
        "task_arrival_rate": 2.0,
    },
]


def workload_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """计算负载相关指标"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    
    # 计算QoS指标
    delay_list = episode_metrics.get("avg_delay", [])
    if delay_list:
        metrics["delay_std"] = float(np.std(delay_list))  # 时延稳定性
        metrics["delay_95percentile"] = float(np.percentile(delay_list, 95))  # 尾延迟
    else:
        metrics["delay_std"] = 0.0
        metrics["delay_95percentile"] = 0.0
    
    # 系统吞吐量（任务数/时间）
    metrics["throughput"] = metrics["completion_rate"] * config.get("task_arrival_rate", 0.3)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """生成多维度对比图"""
    
    workload_labels = [record["workload_label"] for record in results]
    x_pos = range(len(workload_labels))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========== 子图1: 平均成本 ==========
    ax = axes[0, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax.plot(x_pos, values, marker="o", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workload_labels, rotation=20, ha="right")
    ax.set_ylabel("Average Cost")
    ax.set_title("Cost Across Workload Types")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图2: 任务完成率 ==========
    ax = axes[0, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        ax.plot(x_pos, values, marker="s", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workload_labels, rotation=20, ha="right")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Task Completion Across Workloads")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图3: 平均时延 ==========
    ax = axes[0, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["avg_delay"] for record in results]
        ax.plot(x_pos, values, marker="^", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workload_labels, rotation=20, ha="right")
    ax.set_ylabel("Average Delay (s)")
    ax.set_title("Delay Performance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图4: 时延稳定性 ==========
    ax = axes[1, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["delay_std"] for record in results]
        ax.plot(x_pos, values, marker="D", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workload_labels, rotation=20, ha="right")
    ax.set_ylabel("Delay Std Dev (s)")
    ax.set_title("Delay Stability")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图5: 尾延迟(95th) ==========
    ax = axes[1, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["delay_95percentile"] for record in results]
        ax.plot(x_pos, values, marker="v", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workload_labels, rotation=20, ha="right")
    ax.set_ylabel("95th Percentile Delay (s)")
    ax.set_title("Tail Latency (QoS)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图6: 系统吞吐量 ==========
    ax = axes[1, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["throughput"] for record in results]
        ax.plot(x_pos, values, marker="*", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workload_labels, rotation=20, ha="right")
    ax.set_ylabel("Throughput (tasks/s)")
    ax.set_title("System Throughput")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    output_path = suite_dir / "mixed_workload_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nChart saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TD3 strategies under mixed workload scenarios."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="mixed_workload",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="mixed_workload",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for workload_config in WORKLOAD_CONFIGS:
        overrides = {
            "task_complexity_multiplier": workload_config["task_complexity_multiplier"],
            "task_compute_density": workload_config["task_compute_density"],
            "task_data_size_kb": workload_config["task_data_size_kb"],
            "task_arrival_rate": workload_config["task_arrival_rate"],
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": workload_config["key"],
                "label": workload_config["label"],
                "overrides": overrides,
                "workload_label": workload_config["label"],
                "workload_description": workload_config["description"],
                "task_arrival_rate": workload_config["task_arrival_rate"],
                "task_data_size_kb": workload_config["task_data_size_kb"],
                "task_compute_density": workload_config["task_compute_density"],
                "task_complexity_multiplier": workload_config["task_complexity_multiplier"],
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
        per_strategy_hook=workload_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "mixed_workload_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "workload_configs": WORKLOAD_CONFIGS,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nMixed Workload Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Workload Type':<20}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (20 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['workload_label']:<20}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

