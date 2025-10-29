#!/usr/bin/env python3
"""
CAMTD3 带宽资源对成本影响实验
=============================

【功能】
评估不同网络带宽对系统性能的影响。
通过扫描不同的信道带宽配置，分析：
- 网络带宽如何影响卸载时延和能耗
- 带宽瓶颈对系统性能的限制
- 带宽与计算资源的协同优化

【论文对应】
- 网络资源优化（Network Resource Optimization）
- 评估通信瓶颈对边缘计算的影响
- 验证CAMTD3在不同网络条件下的鲁棒性

【实验设计】
扫描参数: channel_bandwidth (信道带宽)
- 窄带: 5 MHz   (受限网络)
- 中窄带: 10 MHz
- 标准窄带: 15 MHz
- 标准: 20 MHz  (默认配置，符合5G NR)
- 宽带: 40 MHz
- 超宽带: 80 MHz
- 极宽带: 100 MHz

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
主要关注: 平均总成本 (Average Total Cost)
- 计算方法: ω_T·时延 + ω_E·能耗
- 预期: 带宽增加会降低时延，但存在收益递减

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_bandwidth_cost_comparison.py \\
    --episodes 100 --suite-id bandwidth_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_bandwidth_cost_comparison.py \\
    --episodes 500 --seed 42 --suite-id bandwidth_paper

# 自定义带宽配置（单位MHz）
python experiments/camtd3_strategy_suite/run_bandwidth_cost_comparison.py \\
    --bandwidths "5,10,20,40,80" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 7配置）：约1.5-3小时
- 完整实验（500轮 × 7配置）：约7-10小时

【输出图表】
- bandwidth_vs_total_cost.png: 带宽 vs 总成本
- bandwidth_vs_delay.png: 带宽 vs 平均时延
- bandwidth_vs_throughput.png: 带宽 vs 系统吞吐量
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
# 确保可以导入项目模块（脚本在experiments/camtd3_strategy_suite/，需要回到根目录）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from train_single_agent import _apply_global_seed_from_env, train_single_algorithm

# ========== 默认实验参数 ==========
DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

# ========== 带宽配置 (MHz) ==========
DEFAULT_BANDWIDTHS = [5, 10, 15, 20, 40, 80, 100]


def parse_bandwidths(value: str) -> List[int]:
    """
    解析带宽配置字符串
    
    【功能】
    将用户输入的带宽字符串解析为整数列表。
    
    【参数】
    value: str - 格式: "bw1,bw2,bw3,..." 或 "default"
        例: "5,10,20,40,80" (单位MHz)
    
    【返回值】
    List[int] - 带宽列表（MHz）
    
    【示例】
    parse_bandwidths("5,10,20")
    # -> [5, 10, 20]
    """
    if not value or value.strip().lower() == "default":
        return DEFAULT_BANDWIDTHS
    
    bandwidths = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not bandwidths:
        raise ValueError("Bandwidths list cannot be empty")
    
    return bandwidths


def run_single_config(
    bandwidth_mhz: int,
    args: argparse.Namespace,
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个带宽配置的训练
    
    【功能】
    使用指定的信道带宽训练CAMTD3，并收集性能指标。
    
    【参数】
    bandwidth_mhz: int - 信道带宽（MHz）
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含性能指标的字典
        {
          "bandwidth_mhz": 20,
          "avg_cost": 12.34,
          "avg_delay": 0.15,
          "avg_energy": 450.0,
          "completion_rate": 0.98,
          "avg_throughput_mbps": 15.6
        }
    """
    print(f"\n{'='*60}")
    print(f"Running: Channel Bandwidth = {bandwidth_mhz} MHz")
    print(f"{'='*60}")
    
    # ========== 步骤1: 设置随机种子 ==========
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()
    
    # ========== 步骤2: 构建场景覆盖配置 ==========
    # 修改信道带宽，其他参数保持默认
    override_scenario = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "channel_bandwidth_mhz": bandwidth_mhz,
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
        disable_migration=False,  # 使用完整CAMTD3
        enforce_offload_mode=None,
    )
    
    # ========== 步骤4: 提取性能指标 ==========
    episode_metrics = results.get("episode_metrics", {})
    
    # 计算后50%轮次的稳定均值
    def tail_mean(values):
        if not values:
            return 0.0
        seq = list(map(float, values))
        subset = seq[len(seq) // 2:] if len(seq) >= 100 else seq
        return float(np.mean(subset))
    
    avg_delay = tail_mean(episode_metrics.get("avg_delay", []))
    avg_energy = tail_mean(episode_metrics.get("total_energy", []))
    completion_rate = tail_mean(episode_metrics.get("task_completion_rate", []))
    
    # ========== 步骤5: 计算成本和吞吐量 ==========
    weight_delay = float(config.rl.reward_weight_delay)      # ω_T = 2.0
    weight_energy = float(config.rl.reward_weight_energy)    # ω_E = 1.2
    
    # 总成本
    avg_cost = weight_delay * avg_delay + weight_energy * (avg_energy / 1000.0)
    
    # 估算系统吞吐量（Mbps）
    # 简化计算：假设平均任务大小350KB，每步12个任务，按时延计算吞吐
    avg_task_size_mb = 0.35  # MB
    num_tasks_per_step = 12
    if avg_delay > 0:
        avg_throughput_mbps = (avg_task_size_mb * num_tasks_per_step) / avg_delay
    else:
        avg_throughput_mbps = 0.0
    
    # ========== 步骤6: 构建结果字典 ==========
    result_dict = {
        "bandwidth_mhz": bandwidth_mhz,
        "avg_cost": avg_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "completion_rate": completion_rate,
        "avg_throughput_mbps": avg_throughput_mbps,
        "weight_delay": weight_delay,
        "weight_energy": weight_energy,
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤7: 保存结果到文件 ==========
    result_path = suite_path / f"bandwidth_{bandwidth_mhz}mhz.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Avg Cost     : {avg_cost:.4f}")
    print(f"  Avg Delay    : {avg_delay:.4f} s")
    print(f"  Avg Energy   : {avg_energy:.2f} J")
    print(f"  Throughput   : {avg_throughput_mbps:.2f} Mbps")
    print(f"  Completion   : {completion_rate:.3f}")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制带宽对性能的影响曲线：
    1. 带宽 vs 总成本
    2. 带宽 vs 平均时延
    3. 带宽 vs 系统吞吐量
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    bandwidths = [r["bandwidth_mhz"] for r in results]
    costs = [r["avg_cost"] for r in results]
    delays = [r["avg_delay"] for r in results]
    throughputs = [r["avg_throughput_mbps"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: 带宽 vs 总成本 ==========
    fig, ax = plt.subplots()
    ax.plot(bandwidths, costs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Channel Bandwidth (MHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Total Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Bandwidth on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数据点
    for x, y in zip(bandwidths, costs):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 标注最优点
    min_idx = np.argmin(costs)
    ax.plot(bandwidths[min_idx], costs[min_idx], 'r*', markersize=15, 
            label=f'Optimal: {bandwidths[min_idx]} MHz')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(suite_path / "bandwidth_vs_total_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: 带宽 vs 平均时延 ==========
    fig, ax = plt.subplots()
    ax.plot(bandwidths, delays, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Channel Bandwidth (MHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Delay (s)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Bandwidth on Task Delay', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(bandwidths, delays):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "bandwidth_vs_delay.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: 带宽 vs 系统吞吐量 ==========
    fig, ax = plt.subplots()
    ax.plot(bandwidths, throughputs, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Channel Bandwidth (MHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('System Throughput (Mbps)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Bandwidth on System Throughput', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(bandwidths, throughputs):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "bandwidth_vs_throughput.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'bandwidth_vs_total_cost.png'}")
    print(f"  - {suite_path / 'bandwidth_vs_delay.png'}")
    print(f"  - {suite_path / 'bandwidth_vs_throughput.png'}")
    print(f"{'='*60}")


def main() -> None:
    """
    脚本主入口函数
    
    【执行流程】
    1. 解析命令行参数
    2. 准备输出目录
    3. 循环运行各带宽配置
    4. 汇总结果到summary.json
    5. 生成对比图表
    6. 打印最终摘要（含最优配置）
    """
    # ========== 步骤1: 构建参数解析器 ==========
    parser = argparse.ArgumentParser(
        description="Evaluate impact of network bandwidth on CAMTD3 performance."
    )
    parser.add_argument(
        "--bandwidths",
        type=str,
        default="default",
        help="Comma-separated bandwidths in MHz (e.g., '5,10,20,40,80'). Use 'default' for preset configs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Training episodes per configuration (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"bandwidth_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier for result grouping.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/parameter_sensitivity",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Run training in silent mode.",
    )
    
    args = parser.parse_args()
    
    # ========== 步骤2: 解析带宽配置 ==========
    bandwidths = parse_bandwidths(args.bandwidths)
    
    # ========== 步骤3: 准备输出目录 ==========
    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 步骤4: 循环运行各配置 ==========
    results = []
    for bandwidth in bandwidths:
        result = run_single_config(bandwidth, args, suite_path)
        results.append(result)
    
    # ========== 步骤5: 保存汇总结果 ==========
    summary = {
        "experiment_type": "bandwidth_cost_sensitivity",
        "suite_id": args.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": args.episodes or DEFAULT_EPISODES,
        "seed": args.seed or DEFAULT_SEED,
        "results": results,
    }
    
    summary_path = suite_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # ========== 步骤6: 生成对比图表 ==========
    plot_results(results, suite_path)
    
    # ========== 步骤7: 找出最优配置 ==========
    min_cost_idx = np.argmin([r["avg_cost"] for r in results])
    optimal_result = results[min_cost_idx]
    
    # ========== 步骤8: 打印最终摘要 ==========
    print(f"\n{'='*60}")
    print("Bandwidth Cost Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'BW (MHz)':<10} {'Cost':<10} {'Delay (s)':<12} {'Throughput (Mbps)':<18}")
    print("-" * 50)
    for r in results:
        marker = " *" if r == optimal_result else ""
        print(f"{r['bandwidth_mhz']:<10} {r['avg_cost']:<10.4f} {r['avg_delay']:<12.4f} {r['avg_throughput_mbps']:<18.2f}{marker}")
    
    print(f"\n{'='*60}")
    print("OPTIMAL CONFIGURATION:")
    print(f"  Bandwidth   : {optimal_result['bandwidth_mhz']} MHz")
    print(f"  Total Cost  : {optimal_result['avg_cost']:.4f}")
    print(f"  Delay       : {optimal_result['avg_delay']:.4f} s")
    print(f"  Throughput  : {optimal_result['avg_throughput_mbps']:.2f} Mbps")
    print(f"{'='*60}")
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

