#!/usr/bin/env python3
"""
CAMTD3 车辆移动速度对比实验
===========================

【功能】
评估不同车辆移动速度对系统性能的影响。
通过扫描不同的车辆速度配置，分析：
- 移动性如何影响通信质量和系统性能
- 高速场景下的性能退化
- 切换（Handover）对系统的影响

【论文对应】
- 移动性影响分析（Mobility Impact Analysis）
- 评估VEC系统在不同移动场景下的鲁棒性
- 验证CAMTD3对移动性的适应能力

【实验设计】
扫描参数: vehicle_speed (车辆移动速度 m/s)
- 静止/低速: 5 m/s  (~18 km/h, 城市慢速)
- 城市: 10 m/s       (~36 km/h, 城市正常)
- 标准: 15 m/s       (~54 km/h, 默认配置)
- 快速: 20 m/s       (~72 km/h, 城市快速)
- 高速: 25 m/s       (~90 km/h, 高速公路)
- 超高速: 30 m/s     (~108 km/h, 高速公路快速)

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本
- 通信成功率
- 平均时延（受移动性影响）
- 切换次数

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_mobility_speed_comparison.py \\
    --episodes 100 --suite-id mobility_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_mobility_speed_comparison.py \\
    --episodes 500 --seed 42 --suite-id mobility_paper

# 自定义速度配置（单位：m/s）
python experiments/camtd3_strategy_suite/run_mobility_speed_comparison.py \\
    --speeds "10,15,20,25" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 6配置）：约1.5-3小时
- 完整实验（500轮 × 6配置）：约6-9小时

【输出图表】
- mobility_speed_vs_cost.png: 移动速度 vs 平均成本
- mobility_speed_vs_delay.png: 移动速度 vs 平均时延
- mobility_speed_vs_completion.png: 移动速度 vs 任务完成率
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

# ========== 车辆速度配置 (m/s) ==========
DEFAULT_SPEEDS = [5, 10, 15, 20, 25, 30]


def parse_speeds(value: str) -> List[float]:
    """解析车辆速度配置字符串"""
    if not value or value.strip().lower() == "default":
        return DEFAULT_SPEEDS
    
    speeds = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not speeds:
        raise ValueError("Speeds list cannot be empty")
    
    return speeds


def run_single_config(
    speed_ms: float,
    args: argparse.Namespace,
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个移动速度配置的训练
    
    【功能】
    使用指定的车辆速度训练CAMTD3，并收集性能指标。
    
    【参数】
    speed_ms: float - 车辆速度（m/s）
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含性能指标的字典
    """
    # 计算对应的km/h
    speed_kmh = speed_ms * 3.6
    
    print(f"\n{'='*60}")
    print(f"Running: Vehicle Speed = {speed_ms} m/s ({speed_kmh:.1f} km/h)")
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
        "vehicle_speed_ms": speed_ms,  # m/s
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
    
    # ========== 步骤5: 构建结果字典 ==========
    result_dict = {
        "speed_ms": speed_ms,
        "speed_kmh": speed_kmh,
        "avg_cost": avg_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "completion_rate": completion_rate,
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤6: 保存结果到文件 ==========
    result_path = suite_path / f"speed_{int(speed_ms)}ms.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Avg Cost      : {avg_cost:.4f}")
    print(f"  Avg Delay     : {avg_delay:.4f} s")
    print(f"  Avg Energy    : {avg_energy:.2f} J")
    print(f"  Completion    : {completion_rate:.3f}")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制移动速度对性能的影响：
    1. 移动速度 vs 平均成本
    2. 移动速度 vs 平均时延
    3. 移动速度 vs 任务完成率
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    speeds_kmh = [r["speed_kmh"] for r in results]
    costs = [r["avg_cost"] for r in results]
    delays = [r["avg_delay"] for r in results]
    completion_rates = [r["completion_rate"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: 移动速度 vs 平均成本 ==========
    fig, ax = plt.subplots()
    ax.plot(speeds_kmh, costs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Vehicle Speed (km/h)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Vehicle Mobility on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数据点
    for x, y in zip(speeds_kmh, costs):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 添加速度场景标签
    ax.axvline(x=36, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(36, ax.get_ylim()[1]*0.95, 'Urban', ha='center', fontsize=9, color='gray')
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(90, ax.get_ylim()[1]*0.95, 'Highway', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(suite_path / "mobility_speed_vs_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: 移动速度 vs 平均时延 ==========
    fig, ax = plt.subplots()
    ax.plot(speeds_kmh, delays, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Vehicle Speed (km/h)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Delay (s)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Vehicle Mobility on Task Delay', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(speeds_kmh, delays):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 添加速度场景标签
    ax.axvline(x=36, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(suite_path / "mobility_speed_vs_delay.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: 移动速度 vs 任务完成率 ==========
    fig, ax = plt.subplots()
    ax.plot(speeds_kmh, completion_rates, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Vehicle Speed (km/h)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Task Completion Rate', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Vehicle Mobility on Task Completion', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(speeds_kmh, completion_rates):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 添加速度场景标签
    ax.axvline(x=36, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(36, 0.05, 'Urban', ha='center', fontsize=9, color='gray')
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(90, 0.05, 'Highway', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(suite_path / "mobility_speed_vs_completion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'mobility_speed_vs_cost.png'}")
    print(f"  - {suite_path / 'mobility_speed_vs_delay.png'}")
    print(f"  - {suite_path / 'mobility_speed_vs_completion.png'}")
    print(f"{'='*60}")


def main() -> None:
    """脚本主入口函数"""
    parser = argparse.ArgumentParser(
        description="Evaluate CAMTD3 performance across different vehicle mobility speeds."
    )
    parser.add_argument(
        "--speeds",
        type=str,
        default="default",
        help="Comma-separated speeds in m/s (e.g., '10,15,20,25'). Use 'default' for presets.",
    )
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default: 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default: 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"mobility_speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    speeds = parse_speeds(args.speeds)
    
    # 准备输出目录
    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)
    
    # 循环运行各配置
    results = []
    for speed in speeds:
        result = run_single_config(speed, args, suite_path)
        results.append(result)
    
    # 保存汇总结果
    summary = {
        "experiment_type": "mobility_speed_sensitivity",
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
    print("Vehicle Mobility Speed Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'Speed(m/s)':<12} {'Speed(km/h)':<12} {'Cost':<10} {'Delay':<10} {'Completion':<12}")
    print("-" * 56)
    for r in results:
        print(f"{r['speed_ms']:<12.1f} {r['speed_kmh']:<12.1f} {r['avg_cost']:<10.4f} {r['avg_delay']:<10.4f} {r['completion_rate']:<12.3f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

