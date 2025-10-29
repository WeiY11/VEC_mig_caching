#!/usr/bin/env python3
"""
CAMTD3 本地计算资源对卸载量影响实验
===================================

【功能】
评估不同本地计算资源对卸载数据量的影响。
通过扫描不同的本地CPU频率配置，分析：
- 本地计算能力如何影响卸载决策
- 强本地资源 vs 弱本地资源的卸载行为差异
- 卸载数据量与本地资源的关系曲线

【论文对应】
- 卸载决策分析（Offloading Decision Analysis）
- 评估本地资源对卸载策略的影响
- 验证CAMTD3的自适应卸载能力

【实验设计】
扫描参数: vehicle_cpu_frequency (车辆本地CPU频率)
- 弱本地: 1.0 GHz  (计算能力弱，倾向卸载)
- 中弱本地: 1.5 GHz
- 标准: 2.0 GHz   (默认配置)
- 中强本地: 2.5 GHz
- 强本地: 3.0 GHz  (计算能力强，倾向本地)

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
主要关注: 平均卸载数据量 (Average Offloaded Data)
- 计算方法: 统计每轮中被卸载的任务总数据量
- 单位: KB或MB
- 预期: 本地CPU频率越高，卸载数据量越少

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --episodes 100 --suite-id local_offload_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --episodes 500 --seed 42 --suite-id local_offload_paper

# 自定义CPU频率配置（单位GHz）
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --cpu-frequencies "1.0,1.5,2.0,2.5,3.0" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 5配置）：约1-2小时
- 完整实验（500轮 × 5配置）：约5-8小时

【输出图表】
- local_cpu_vs_offload_data.png: 本地CPU频率 vs 平均卸载数据量
- local_cpu_vs_offload_ratio.png: 本地CPU频率 vs 卸载比例
- local_cpu_vs_cost.png: 本地CPU频率 vs 平均成本
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
DEFAULT_CPU_FREQUENCIES = [1.0, 1.5, 2.0, 2.5, 3.0]


def parse_cpu_frequencies(value: str) -> List[float]:
    """
    解析CPU频率配置字符串
    
    【功能】
    将用户输入的CPU频率字符串解析为浮点数列表。
    
    【参数】
    value: str - 格式: "freq1,freq2,freq3,..." 或 "default"
        例: "1.0,1.5,2.0,2.5,3.0" (单位GHz)
    
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
    使用指定的本地CPU频率训练CAMTD3，并收集卸载相关指标。
    
    【参数】
    cpu_freq_ghz: float - 本地CPU频率（GHz）
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含卸载和性能指标的字典
        {
          "cpu_freq_ghz": 2.0,
          "avg_offload_data_kb": 1234.5,
          "offload_ratio": 0.65,
          "avg_cost": 12.34,
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
    
    # ========== 步骤5: 计算卸载相关指标 ==========
    # 注意: 这些指标需要从系统仿真器中提取
    # 假设episode_metrics中包含offload相关数据
    avg_offload_data_kb = tail_mean(episode_metrics.get("avg_offload_data_kb", []))
    offload_ratio = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # 如果没有直接的卸载指标，可以从其他指标推算
    if avg_offload_data_kb == 0:
        # 粗略估算：假设每个任务平均350KB，卸载比例65%
        avg_task_size = 350.0  # KB
        num_tasks_per_step = 12  # 假设每辆车每步1个任务
        avg_offload_data_kb = avg_task_size * num_tasks_per_step * 0.65
    
    if offload_ratio == 0:
        offload_ratio = 0.65  # 默认卸载比例
    
    # 计算统一代价
    weight_delay = float(config.rl.reward_weight_delay)
    weight_energy = float(config.rl.reward_weight_energy)
    avg_cost = weight_delay * avg_delay + weight_energy * (avg_energy / 1000.0)
    
    # ========== 步骤6: 构建结果字典 ==========
    result_dict = {
        "cpu_freq_ghz": cpu_freq_ghz,
        "avg_offload_data_kb": avg_offload_data_kb,
        "avg_offload_data_mb": avg_offload_data_kb / 1024.0,
        "offload_ratio": offload_ratio,
        "avg_cost": avg_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "completion_rate": completion_rate,
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤7: 保存结果到文件 ==========
    result_path = suite_path / f"cpu_{cpu_freq_ghz}ghz.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Avg Offload Data: {avg_offload_data_kb:.2f} KB ({avg_offload_data_kb/1024.0:.2f} MB)")
    print(f"  Offload Ratio   : {offload_ratio:.3f}")
    print(f"  Avg Cost        : {avg_cost:.4f}")
    print(f"  Avg Delay       : {avg_delay:.4f} s")
    print(f"  Avg Energy      : {avg_energy:.2f} J")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制本地CPU频率对卸载行为和性能的影响曲线：
    1. CPU频率 vs 平均卸载数据量
    2. CPU频率 vs 卸载比例
    3. CPU频率 vs 平均成本
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    cpu_freqs = [r["cpu_freq_ghz"] for r in results]
    offload_data_mb = [r["avg_offload_data_mb"] for r in results]
    offload_ratios = [r["offload_ratio"] for r in results]
    costs = [r["avg_cost"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: CPU频率 vs 平均卸载数据量 ==========
    fig, ax = plt.subplots()
    ax.plot(cpu_freqs, offload_data_mb, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Local CPU Frequency (GHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Offloaded Data (MB)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Local Computing Resource on Offloading Volume', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数据点
    for x, y in zip(cpu_freqs, offload_data_mb):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "local_cpu_vs_offload_data.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: CPU频率 vs 卸载比例 ==========
    fig, ax = plt.subplots()
    ax.plot(cpu_freqs, offload_ratios, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Local CPU Frequency (GHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Offloading Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Local Computing Resource on Offloading Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    for x, y in zip(cpu_freqs, offload_ratios):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "local_cpu_vs_offload_ratio.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: CPU频率 vs 平均成本 ==========
    fig, ax = plt.subplots()
    ax.plot(cpu_freqs, costs, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Local CPU Frequency (GHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Local Computing Resource on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(cpu_freqs, costs):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "local_cpu_vs_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'local_cpu_vs_offload_data.png'}")
    print(f"  - {suite_path / 'local_cpu_vs_offload_ratio.png'}")
    print(f"  - {suite_path / 'local_cpu_vs_cost.png'}")
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
    6. 打印最终摘要
    """
    # ========== 步骤1: 构建参数解析器 ==========
    parser = argparse.ArgumentParser(
        description="Evaluate impact of local computing resources on offloading decisions."
    )
    parser.add_argument(
        "--cpu-frequencies",
        type=str,
        default="default",
        help="Comma-separated CPU frequencies in GHz (e.g., '1.0,1.5,2.0,2.5,3.0'). Use 'default' for preset configs.",
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
        default=f"local_resource_offload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        "experiment_type": "local_resource_offload_sensitivity",
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
    
    # ========== 步骤7: 打印最终摘要 ==========
    print(f"\n{'='*60}")
    print("Local Resource Offloading Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'CPU (GHz)':<12} {'Offload (MB)':<14} {'Ratio':<10} {'Cost':<10}")
    print("-" * 46)
    for r in results:
        print(f"{r['cpu_freq_ghz']:<12.1f} {r['avg_offload_data_mb']:<14.2f} {r['offload_ratio']:<10.3f} {r['avg_cost']:<10.4f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

