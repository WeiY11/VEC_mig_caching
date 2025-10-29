#!/usr/bin/env python3
"""
CAMTD3 数据大小对比实验
======================

【功能】
评估不同任务数据大小对系统性能的影响。
通过扫描不同的数据量配置（KB级到MB级），分析：
- 数据大小如何影响时延和能耗
- 大数据任务是否更倾向于卸载
- 最优数据大小的选择范围

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 评估数据大小对卸载决策的影响
- 验证系统在不同数据规模下的鲁棒性

【实验设计】
扫描参数: task_data_size_range (任务数据大小范围)
- 小数据: [50KB, 150KB]
- 中小数据: [100KB, 300KB]
- 中等数据: [200KB, 500KB]  (默认配置)
- 中大数据: [300KB, 800KB]
- 大数据: [500KB, 1000KB]

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --episodes 100 --suite-id data_size_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --episodes 500 --seed 42 --suite-id data_size_paper

# 自定义数据大小范围（单位KB）
python experiments/camtd3_strategy_suite/run_data_size_comparison.py \\
    --data-sizes "50,150;100,300;200,500" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 5配置）：约1-2小时
- 完整实验（500轮 × 5配置）：约5-8小时

【输出图表】
- data_size_vs_cost.png: 数据大小 vs 平均成本
- data_size_vs_delay.png: 数据大小 vs 平均时延
- data_size_vs_energy.png: 数据大小 vs 平均能耗
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

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
DEFAULT_EPISODES = 500  # 默认训练轮数（参数扫描建议用500）
DEFAULT_SEED = 42       # 默认随机种子

# ========== 数据大小配置 (KB) ==========
# 格式: (最小值, 最大值, 描述标签)
DEFAULT_DATA_SIZE_CONFIGS = [
    (50, 150, "Small (50-150KB)"),
    (100, 300, "Medium-Small (100-300KB)"),
    (200, 500, "Medium (200-500KB)"),  # 默认配置
    (300, 800, "Medium-Large (300-800KB)"),
    (500, 1000, "Large (500-1000KB)"),
]


def parse_data_sizes(value: str) -> List[Tuple[int, int, str]]:
    """
    解析数据大小配置字符串
    
    【功能】
    将用户输入的数据大小范围字符串解析为配置列表。
    
    【参数】
    value: str - 格式: "min1,max1;min2,max2;..." (单位KB)
        例: "50,150;100,300;200,500"
    
    【返回值】
    List[Tuple[int, int, str]] - [(min_kb, max_kb, label), ...]
    
    【示例】
    parse_data_sizes("50,150;100,300")
    # -> [(50, 150, "50-150KB"), (100, 300, "100-300KB")]
    """
    if not value or value.strip().lower() == "default":
        return DEFAULT_DATA_SIZE_CONFIGS
    
    configs = []
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
    args: argparse.Namespace,
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个数据大小配置的训练
    
    【功能】
    使用指定的数据大小范围训练CAMTD3，并收集性能指标。
    
    【参数】
    min_kb: int - 最小数据大小（KB）
    max_kb: int - 最大数据大小（KB）
    label: str - 配置标签
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含性能指标的字典
        {
          "label": "200-500KB",
          "min_kb": 200,
          "max_kb": 500,
          "avg_cost": 12.34,
          "avg_delay": 0.15,
          "avg_energy": 450.0,
          "completion_rate": 0.98
        }
    """
    print(f"\n{'='*60}")
    print(f"Running: Data Size = {label}")
    print(f"{'='*60}")
    
    # ========== 步骤1: 设置随机种子 ==========
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()
    
    # ========== 步骤2: 构建场景覆盖配置 ==========
    # 修改任务数据大小范围，其他参数保持默认
    override_scenario = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "task_data_size_min_kb": min_kb,
        "task_data_size_max_kb": max_kb,
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
    
    # 计算统一代价
    weight_delay = float(config.rl.reward_weight_delay)
    weight_energy = float(config.rl.reward_weight_energy)
    avg_cost = weight_delay * avg_delay + weight_energy * (avg_energy / 1000.0)
    
    # ========== 步骤5: 构建结果字典 ==========
    result_dict = {
        "label": label,
        "min_kb": min_kb,
        "max_kb": max_kb,
        "avg_kb": (min_kb + max_kb) / 2.0,
        "avg_cost": avg_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "completion_rate": completion_rate,
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤6: 保存结果到文件 ==========
    result_path = suite_path / f"data_{min_kb}_{max_kb}kb.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Avg Cost   : {avg_cost:.4f}")
    print(f"  Avg Delay  : {avg_delay:.4f} s")
    print(f"  Avg Energy : {avg_energy:.2f} J")
    print(f"  Completion : {completion_rate:.3f}")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制数据大小对性能的影响曲线：
    1. 数据大小 vs 平均成本
    2. 数据大小 vs 平均时延
    3. 数据大小 vs 平均能耗
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    avg_sizes = [r["avg_kb"] for r in results]
    labels = [r["label"] for r in results]
    costs = [r["avg_cost"] for r in results]
    delays = [r["avg_delay"] for r in results]
    energies = [r["avg_energy"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: 数据大小 vs 平均成本 ==========
    fig, ax = plt.subplots()
    ax.plot(avg_sizes, costs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Average Task Data Size (KB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Data Size on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数据点
    for x, y, label in zip(avg_sizes, costs, labels):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "data_size_vs_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: 数据大小 vs 平均时延 ==========
    fig, ax = plt.subplots()
    ax.plot(avg_sizes, delays, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Average Task Data Size (KB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Delay (s)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Data Size on Task Delay', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(avg_sizes, delays):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "data_size_vs_delay.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: 数据大小 vs 平均能耗 ==========
    fig, ax = plt.subplots()
    ax.plot(avg_sizes, energies, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Average Task Data Size (KB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Energy (J)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Data Size on Energy Consumption', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(avg_sizes, energies):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(suite_path / "data_size_vs_energy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'data_size_vs_cost.png'}")
    print(f"  - {suite_path / 'data_size_vs_delay.png'}")
    print(f"  - {suite_path / 'data_size_vs_energy.png'}")
    print(f"{'='*60}")


def main() -> None:
    """
    脚本主入口函数
    
    【执行流程】
    1. 解析命令行参数
    2. 准备输出目录
    3. 循环运行各数据大小配置
    4. 汇总结果到summary.json
    5. 生成对比图表
    6. 打印最终摘要
    """
    # ========== 步骤1: 构建参数解析器 ==========
    parser = argparse.ArgumentParser(
        description="Evaluate CAMTD3 performance across different task data sizes."
    )
    parser.add_argument(
        "--data-sizes",
        type=str,
        default="default",
        help="Data size ranges in format 'min1,max1;min2,max2;...' (KB). Use 'default' for preset configs.",
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
        default=f"data_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    
    # ========== 步骤2: 解析数据大小配置 ==========
    data_size_configs = parse_data_sizes(args.data_sizes)
    
    # ========== 步骤3: 准备输出目录 ==========
    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 步骤4: 循环运行各配置 ==========
    results = []
    for min_kb, max_kb, label in data_size_configs:
        result = run_single_config(min_kb, max_kb, label, args, suite_path)
        results.append(result)
    
    # ========== 步骤5: 保存汇总结果 ==========
    summary = {
        "experiment_type": "data_size_sensitivity",
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
    print("Data Size Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'Data Size':<25} {'Avg Cost':<12} {'Avg Delay':<12} {'Avg Energy':<12}")
    print("-" * 61)
    for r in results:
        print(f"{r['label']:<25} {r['avg_cost']:<12.4f} {r['avg_delay']:<12.4f} {r['avg_energy']:<12.2f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

