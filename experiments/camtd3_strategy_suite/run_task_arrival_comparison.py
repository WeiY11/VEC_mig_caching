#!/usr/bin/env python3
"""
CAMTD3 任务到达率对比实验
=========================

【功能】
评估不同任务到达率（负载强度）对系统性能的影响。
通过扫描不同的任务生成速率，分析：
- 系统在不同负载压力下的性能表现
- 系统的最大吞吐能力
- 负载增加时的性能退化规律

【论文对应】
- 负载可扩展性分析（Load Scalability Analysis）
- 评估系统承载能力和瓶颈
- 验证CAMTD3在高负载下的鲁棒性

【实验设计】
扫描参数: task_arrival_rate (每车每秒任务数)
- 轻载: 0.3 tasks/s/vehicle
- 中轻载: 0.5 tasks/s/vehicle
- 标准: 1.0 tasks/s/vehicle  (默认)
- 中重载: 1.5 tasks/s/vehicle
- 重载: 2.0 tasks/s/vehicle
- 超重载: 3.0 tasks/s/vehicle

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本
- 任务完成率（重要！）
- 任务丢弃率
- 系统吞吐量

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --episodes 100 --suite-id arrival_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --episodes 500 --seed 42 --suite-id arrival_paper

# 自定义到达率（单位：tasks/s/vehicle）
python experiments/camtd3_strategy_suite/run_task_arrival_comparison.py \\
    --arrival-rates "0.5,1.0,1.5,2.0" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 6配置）：约1.5-3小时
- 完整实验（500轮 × 6配置）：约6-9小时

【输出图表】
- arrival_rate_vs_cost.png: 到达率 vs 平均成本
- arrival_rate_vs_completion.png: 到达率 vs 任务完成率
- arrival_rate_vs_throughput.png: 到达率 vs 系统吞吐量
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from train_single_agent import _apply_global_seed_from_env, train_single_algorithm

# ========== 默认实验参数 ==========
DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

# ========== 任务到达率配置 (tasks/s/vehicle) ==========
DEFAULT_ARRIVAL_RATES = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]


def parse_arrival_rates(value: str) -> List[float]:
    """解析任务到达率配置字符串"""
    if not value or value.strip().lower() == "default":
        return DEFAULT_ARRIVAL_RATES
    
    rates = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not rates:
        raise ValueError("Arrival rates list cannot be empty")
    
    return rates


def run_single_config(
    arrival_rate: float,
    args: argparse.Namespace,
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个任务到达率配置的训练
    
    【功能】
    使用指定的任务到达率训练CAMTD3，并收集性能指标。
    
    【参数】
    arrival_rate: float - 任务到达率（tasks/s/vehicle）
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含性能指标的字典
    """
    print(f"\n{'='*60}")
    print(f"Running: Task Arrival Rate = {arrival_rate} tasks/s/vehicle")
    print(f"{'='*60}")
    
    # ========== 步骤1: 设置随机种子 ==========
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()
    
    # ========== 步骤2: 构建场景覆盖配置 ==========
    override_scenario = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "task_arrival_rate": arrival_rate,  # tasks/s/vehicle
        "override_topology": True,
    }
    
    # ========== 步骤3: 执行训练 ==========
    episodes = args.episodes or DEFAULT_EPISODES
    results = train_single_algorithm(
        "CAMTD3",
        num_episodes=episodes,
        silent_mode=args.silent,
        override_scenario=override_scenario,
        use_enhanced_cache=True,
        disable_migration=False,
        enforce_offload_mode=None,
    )
    
    # ========== 步骤4: 提取性能指标 ==========
    episode_metrics = results.get("episode_metrics", {})
    
    def tail_mean(values):
        if not values:
            return 0.0
        seq = list(map(float, values))
        subset = seq[len(seq) // 2:] if len(seq) >= 100 else seq
        return float(np.mean(subset))
    
    avg_delay = tail_mean(episode_metrics.get("avg_delay", []))
    avg_energy = tail_mean(episode_metrics.get("total_energy", []))
    completion_rate = tail_mean(episode_metrics.get("task_completion_rate", []))
    
    # 计算统一代价
    weight_delay = float(config.rl.reward_weight_delay)
    weight_energy = float(config.rl.reward_weight_energy)
    avg_cost = weight_delay * avg_delay + weight_energy * (avg_energy / 1000.0)
    
    # ========== 步骤5: 计算吞吐量指标 ==========
    # 系统总到达率（tasks/s）
    total_arrival_rate = arrival_rate * 12  # 12辆车
    
    # 有效吞吐量（tasks/s）= 到达率 × 完成率
    effective_throughput = total_arrival_rate * completion_rate
    
    # 任务丢弃率
    drop_rate = 1.0 - completion_rate
    
    # ========== 步骤6: 构建结果字典 ==========
    result_dict = {
        "arrival_rate": arrival_rate,
        "total_arrival_rate": total_arrival_rate,
        "avg_cost": avg_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "completion_rate": completion_rate,
        "drop_rate": drop_rate,
        "effective_throughput": effective_throughput,
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤7: 保存结果到文件 ==========
    result_path = suite_path / f"arrival_{arrival_rate:.1f}.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Total Arrival : {total_arrival_rate:.1f} tasks/s")
    print(f"  Avg Cost      : {avg_cost:.4f}")
    print(f"  Avg Delay     : {avg_delay:.4f} s")
    print(f"  Avg Energy    : {avg_energy:.2f} J")
    print(f"  Completion    : {completion_rate:.3f}")
    print(f"  Drop Rate     : {drop_rate:.3f}")
    print(f"  Throughput    : {effective_throughput:.2f} tasks/s")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制任务到达率对性能的影响：
    1. 到达率 vs 平均成本
    2. 到达率 vs 任务完成率
    3. 到达率 vs 系统吞吐量
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    arrival_rates = [r["arrival_rate"] for r in results]
    costs = [r["avg_cost"] for r in results]
    completion_rates = [r["completion_rate"] for r in results]
    throughputs = [r["effective_throughput"] for r in results]
    drop_rates = [r["drop_rate"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: 到达率 vs 平均成本 ==========
    fig, ax = plt.subplots()
    ax.plot(arrival_rates, costs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Task Arrival Rate (tasks/s/vehicle)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Task Arrival Rate on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(arrival_rates, costs):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "arrival_rate_vs_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: 到达率 vs 任务完成率和丢弃率（双Y轴） ==========
    fig, ax1 = plt.subplots()
    
    color_comp = '#A23B72'
    ax1.set_xlabel('Task Arrival Rate (tasks/s/vehicle)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Completion Rate', color=color_comp, fontsize=13, fontweight='bold')
    line1 = ax1.plot(arrival_rates, completion_rates, 'o-', linewidth=2, markersize=8, 
                     color=color_comp, label='Completion Rate')
    ax1.tick_params(axis='y', labelcolor=color_comp)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_drop = '#F18F01'
    ax2.set_ylabel('Drop Rate', color=color_drop, fontsize=13, fontweight='bold')
    line2 = ax2.plot(arrival_rates, drop_rates, 's-', linewidth=2, markersize=8, 
                     color=color_drop, label='Drop Rate')
    ax2.tick_params(axis='y', labelcolor=color_drop)
    ax2.set_ylim(0, 1.05)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Impact of Task Arrival Rate on Task Completion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(suite_path / "arrival_rate_vs_completion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: 到达率 vs 系统吞吐量 ==========
    fig, ax = plt.subplots()
    
    # 绘制理论最大吞吐量（虚线）
    max_throughput = [r * 12 for r in arrival_rates]
    ax.plot(arrival_rates, max_throughput, '--', linewidth=1.5, color='gray', 
            label='Theoretical Max', alpha=0.6)
    
    # 绘制实际吞吐量
    ax.plot(arrival_rates, throughputs, 'o-', linewidth=2, markersize=8, 
            color='#00A896', label='Actual Throughput')
    
    ax.set_xlabel('Task Arrival Rate (tasks/s/vehicle)', fontsize=13, fontweight='bold')
    ax.set_ylabel('System Throughput (tasks/s)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Task Arrival Rate on System Throughput', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(arrival_rates, throughputs):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "arrival_rate_vs_throughput.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'arrival_rate_vs_cost.png'}")
    print(f"  - {suite_path / 'arrival_rate_vs_completion.png'}")
    print(f"  - {suite_path / 'arrival_rate_vs_throughput.png'}")
    print(f"{'='*60}")


def main() -> None:
    """脚本主入口函数"""
    parser = argparse.ArgumentParser(
        description="Evaluate CAMTD3 performance across different task arrival rates."
    )
    parser.add_argument(
        "--arrival-rates",
        type=str,
        default="default",
        help="Comma-separated arrival rates (tasks/s/vehicle), e.g., '0.5,1.0,1.5,2.0'. Use 'default' for presets.",
    )
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default: 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default: 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"task_arrival_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier for result grouping.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/parameter_sensitivity",
        help="Root directory for outputs.",
    )
    parser.add_argument("--silent", action="store_true", help="Run training in silent mode.")
    
    args = parser.parse_args()
    
    # 解析配置
    arrival_rates = parse_arrival_rates(args.arrival_rates)
    
    # 准备输出目录
    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)
    
    # 循环运行各配置
    results = []
    for rate in arrival_rates:
        result = run_single_config(rate, args, suite_path)
        results.append(result)
    
    # 保存汇总结果
    summary = {
        "experiment_type": "task_arrival_sensitivity",
        "suite_id": args.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": args.episodes or DEFAULT_EPISODES,
        "seed": args.seed or DEFAULT_SEED,
        "results": results,
    }
    
    summary_path = suite_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # 生成对比图表
    plot_results(results, suite_path)
    
    # 打印最终摘要
    print(f"\n{'='*60}")
    print("Task Arrival Rate Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'Rate':<10} {'Cost':<10} {'Completion':<12} {'Throughput':<12}")
    print("-" * 44)
    for r in results:
        print(f"{r['arrival_rate']:<10.1f} {r['avg_cost']:<10.4f} {r['completion_rate']:<12.3f} {r['effective_throughput']:<12.2f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

