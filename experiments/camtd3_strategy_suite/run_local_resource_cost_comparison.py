#!/usr/bin/env python3
"""
CAMTD3 本地计算资源对成本影响实验
=================================

【功能】
评估不同本地计算资源对系统成本的影响。
通过扫描不同的本地CPU频率配置，分析：
- 本地计算能力如何影响系统总成本
- 本地资源增强是否能降低整体成本
- 本地资源与边缘资源的协同优化

【论文对应】
- 资源配置优化（Resource Allocation Optimization）
- 评估本地计算资源的投资回报率
- 验证CAMTD3在不同资源配置下的适应性

【实验设计】
扫描参数: vehicle_cpu_frequency (车辆本地CPU频率)
- 极弱本地: 0.8 GHz
- 弱本地: 1.2 GHz
- 中等本地: 1.6 GHz
- 标准: 2.0 GHz    (默认配置)
- 中强本地: 2.4 GHz
- 强本地: 2.8 GHz
- 极强本地: 3.2 GHz

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
主要关注: 平均总成本 (Average Total Cost)
- 计算方法: ω_T·时延 + ω_E·能耗
- 预期: 存在最优的本地资源配置点

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --episodes 100 --suite-id local_cost_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --episodes 500 --seed 42 --suite-id local_cost_paper

# 自定义CPU频率配置（单位GHz）
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --cpu-frequencies "0.8,1.2,1.6,2.0,2.4,2.8,3.2" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 7配置）：约1.5-3小时
- 完整实验（500轮 × 7配置）：约7-10小时

【输出图表】
- local_cpu_vs_total_cost.png: 本地CPU频率 vs 总成本
- local_cpu_vs_delay_energy.png: 本地CPU频率 vs 时延和能耗（双Y轴）
- local_cpu_vs_breakdown.png: 各成本组成的堆叠柱状图
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

# ========== 本地CPU频率配置 (GHz) ==========
DEFAULT_CPU_FREQUENCIES = [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]


def parse_cpu_frequencies(value: str) -> List[float]:
    """
    解析CPU频率配置字符串
    
    【功能】
    将用户输入的CPU频率字符串解析为浮点数列表。
    
    【参数】
    value: str - 格式: "freq1,freq2,freq3,..." 或 "default"
        例: "0.8,1.2,1.6,2.0,2.4,2.8,3.2" (单位GHz)
    
    【返回值】
    List[float] - CPU频率列表（GHz）
    
    【示例】
    parse_cpu_frequencies("1.0,2.0,3.0")
    # -> [1.0, 2.0, 3.0]
    """
    if not value or value.strip().lower() == "default":
        return DEFAULT_CPU_FREQUENCIES
    
    freqs = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not freqs:
        raise ValueError("CPU frequencies list cannot be empty")
    
    return freqs


def run_single_config(
    cpu_freq_ghz: float,
    args: argparse.Namespace,
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个本地CPU频率配置的训练
    
    【功能】
    使用指定的本地CPU频率训练CAMTD3，并收集成本相关指标。
    
    【参数】
    cpu_freq_ghz: float - 本地CPU频率（GHz）
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含成本组成和性能指标的字典
        {
          "cpu_freq_ghz": 2.0,
          "avg_total_cost": 12.34,
          "avg_delay": 0.15,
          "avg_energy": 450.0,
          "delay_cost": 0.30,
          "energy_cost": 0.54,
          ...
        }
    """
    print(f"\n{'='*60}")
    print(f"Running: Local CPU Frequency = {cpu_freq_ghz} GHz")
    print(f"{'='*60}")
    
    # ========== 步骤1: 设置随机种子 ==========
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()
    
    # ========== 步骤2: 构建场景覆盖配置 ==========
    # 修改本地CPU频率，其他参数保持默认
    override_scenario = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "vehicle_cpu_frequency_ghz": cpu_freq_ghz,
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
    
    # ========== 步骤5: 计算成本组成 ==========
    weight_delay = float(config.rl.reward_weight_delay)      # ω_T = 2.0
    weight_energy = float(config.rl.reward_weight_energy)    # ω_E = 1.2
    
    # 时延成本组成
    delay_cost = weight_delay * avg_delay
    
    # 能耗成本组成（归一化到与时延相当的量级）
    energy_cost = weight_energy * (avg_energy / 1000.0)
    
    # 总成本
    avg_total_cost = delay_cost + energy_cost
    
    # ========== 步骤6: 构建结果字典 ==========
    result_dict = {
        "cpu_freq_ghz": cpu_freq_ghz,
        "avg_total_cost": avg_total_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "delay_cost": delay_cost,
        "energy_cost": energy_cost,
        "completion_rate": completion_rate,
        "weight_delay": weight_delay,
        "weight_energy": weight_energy,
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤7: 保存结果到文件 ==========
    result_path = suite_path / f"cpu_{cpu_freq_ghz}ghz.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Total Cost   : {avg_total_cost:.4f}")
    print(f"    - Delay Cost : {delay_cost:.4f} (weight={weight_delay})")
    print(f"    - Energy Cost: {energy_cost:.4f} (weight={weight_energy})")
    print(f"  Avg Delay    : {avg_delay:.4f} s")
    print(f"  Avg Energy   : {avg_energy:.2f} J")
    print(f"  Completion   : {completion_rate:.3f}")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制本地CPU频率对成本的影响曲线：
    1. CPU频率 vs 总成本
    2. CPU频率 vs 时延和能耗（双Y轴）
    3. 成本组成堆叠柱状图
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    cpu_freqs = [r["cpu_freq_ghz"] for r in results]
    total_costs = [r["avg_total_cost"] for r in results]
    delays = [r["avg_delay"] for r in results]
    energies = [r["avg_energy"] for r in results]
    delay_costs = [r["delay_cost"] for r in results]
    energy_costs = [r["energy_cost"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: CPU频率 vs 总成本 ==========
    fig, ax = plt.subplots()
    ax.plot(cpu_freqs, total_costs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Local CPU Frequency (GHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Total Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Local Computing Resource on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数据点
    for x, y in zip(cpu_freqs, total_costs):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 标注最优点
    min_idx = np.argmin(total_costs)
    ax.plot(cpu_freqs[min_idx], total_costs[min_idx], 'r*', markersize=15, 
            label=f'Optimal: {cpu_freqs[min_idx]:.1f} GHz')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(suite_path / "local_cpu_vs_total_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: CPU频率 vs 时延和能耗（双Y轴）==========
    fig, ax1 = plt.subplots()
    
    color_delay = '#A23B72'
    ax1.set_xlabel('Local CPU Frequency (GHz)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Delay (s)', color=color_delay, fontsize=13, fontweight='bold')
    line1 = ax1.plot(cpu_freqs, delays, 'o-', linewidth=2, markersize=8, 
                     color=color_delay, label='Delay')
    ax1.tick_params(axis='y', labelcolor=color_delay)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_energy = '#F18F01'
    ax2.set_ylabel('Average Energy (J)', color=color_energy, fontsize=13, fontweight='bold')
    line2 = ax2.plot(cpu_freqs, energies, 's-', linewidth=2, markersize=8, 
                     color=color_energy, label='Energy')
    ax2.tick_params(axis='y', labelcolor=color_energy)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Impact of Local CPU on Delay and Energy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(suite_path / "local_cpu_vs_delay_energy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: 成本组成堆叠柱状图 ==========
    fig, ax = plt.subplots()
    
    x_pos = np.arange(len(cpu_freqs))
    width = 0.6
    
    p1 = ax.bar(x_pos, delay_costs, width, label='Delay Cost', color='#A23B72')
    p2 = ax.bar(x_pos, energy_costs, width, bottom=delay_costs, 
                label='Energy Cost', color='#F18F01')
    
    ax.set_xlabel('Local CPU Frequency (GHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost', fontsize=13, fontweight='bold')
    ax.set_title('Cost Breakdown by Local Computing Resource', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{f:.1f}' for f in cpu_freqs])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(suite_path / "local_cpu_vs_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'local_cpu_vs_total_cost.png'}")
    print(f"  - {suite_path / 'local_cpu_vs_delay_energy.png'}")
    print(f"  - {suite_path / 'local_cpu_vs_breakdown.png'}")
    print(f"{'='*60}")


def main() -> None:
    """
    脚本主入口函数
    
    【执行流程】
    1. 解析命令行参数
    2. 准备输出目录
    3. 循环运行各CPU频率配置
    4. 汇总结果到summary.json
    5. 生成对比图表
    6. 打印最终摘要（含最优配置）
    """
    # ========== 步骤1: 构建参数解析器 ==========
    parser = argparse.ArgumentParser(
        description="Evaluate impact of local computing resources on system cost."
    )
    parser.add_argument(
        "--cpu-frequencies",
        type=str,
        default="default",
        help="Comma-separated CPU frequencies in GHz (e.g., '0.8,1.2,1.6,2.0,2.4,2.8,3.2'). Use 'default' for preset configs.",
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
        default=f"local_resource_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    
    # ========== 步骤2: 解析CPU频率配置 ==========
    cpu_frequencies = parse_cpu_frequencies(args.cpu_frequencies)
    
    # ========== 步骤3: 准备输出目录 ==========
    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 步骤4: 循环运行各配置 ==========
    results = []
    for cpu_freq in cpu_frequencies:
        result = run_single_config(cpu_freq, args, suite_path)
        results.append(result)
    
    # ========== 步骤5: 保存汇总结果 ==========
    summary = {
        "experiment_type": "local_resource_cost_sensitivity",
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
    min_cost_idx = np.argmin([r["avg_total_cost"] for r in results])
    optimal_result = results[min_cost_idx]
    
    # ========== 步骤8: 打印最终摘要 ==========
    print(f"\n{'='*60}")
    print("Local Resource Cost Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'CPU (GHz)':<12} {'Total Cost':<12} {'Delay':<10} {'Energy':<10}")
    print("-" * 44)
    for r in results:
        marker = " *" if r == optimal_result else ""
        print(f"{r['cpu_freq_ghz']:<12.1f} {r['avg_total_cost']:<12.4f} {r['avg_delay']:<10.4f} {r['avg_energy']:<10.2f}{marker}")
    
    print(f"\n{'='*60}")
    print("OPTIMAL CONFIGURATION:")
    print(f"  CPU Frequency: {optimal_result['cpu_freq_ghz']:.1f} GHz")
    print(f"  Total Cost   : {optimal_result['avg_total_cost']:.4f}")
    print(f"  Delay        : {optimal_result['avg_delay']:.4f} s")
    print(f"  Energy       : {optimal_result['avg_energy']:.2f} J")
    print(f"{'='*60}")
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

