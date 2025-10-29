#!/usr/bin/env python3
"""
CAMTD3 边缘节点配置对比实验
============================

【功能】
评估不同边缘节点配置对系统性能的影响。
通过扫描不同的RSU和UAV数量组合，分析：
- 边缘节点数量如何影响系统性能
- RSU vs UAV的性能差异
- 最优的基础设施部署方案

【论文对应】
- 基础设施配置优化（Infrastructure Deployment Optimization）
- 评估边缘节点投资回报率（ROI）
- 验证CAMTD3在不同规模部署下的性能

【实验设计】
扫描参数: (num_rsus, num_uavs) 组合
- 最小配置: (2 RSU, 0 UAV)
- 仅RSU: (4 RSU, 0 UAV)
- 标准配置: (4 RSU, 2 UAV) - 默认
- RSU+UAV: (6 RSU, 2 UAV)
- 密集配置: (6 RSU, 4 UAV)
- 超密集: (8 RSU, 4 UAV)

固定参数:
- 车辆数: 12
- 数据大小: [200KB, 500KB]
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本（时延+能耗）
- 任务完成率
- 节点利用率

【使用示例】
```bash
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --episodes 100 --suite-id edge_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --episodes 500 --seed 42 --suite-id edge_paper

# 自定义配置（格式: rsu1,uav1;rsu2,uav2;...）
python experiments/camtd3_strategy_suite/run_edge_node_comparison.py \\
    --node-configs "2,0;4,2;6,4" --episodes 300
```

【预计运行时间】
- 快速测试（100轮 × 6配置）：约1.5-3小时
- 完整实验（500轮 × 6配置）：约6-9小时

【输出图表】
- edge_nodes_vs_cost.png: 边缘节点数 vs 平均成本
- edge_nodes_vs_completion.png: 边缘节点数 vs 任务完成率
- rsu_uav_heatmap.png: RSU-UAV配置热力图
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# ========== 边缘节点配置 (RSU数, UAV数) ==========
DEFAULT_NODE_CONFIGS = [
    (2, 0, "Minimal (2R+0U)"),
    (4, 0, "RSU-only (4R+0U)"),
    (4, 2, "Standard (4R+2U)"),      # 默认配置
    (6, 2, "Enhanced (6R+2U)"),
    (6, 4, "Dense (6R+4U)"),
    (8, 4, "Ultra-Dense (8R+4U)"),
]


def parse_node_configs(value: str) -> List[Tuple[int, int, str]]:
    """
    解析边缘节点配置字符串
    
    【功能】
    将用户输入的节点配置字符串解析为配置列表。
    
    【参数】
    value: str - 格式: "rsu1,uav1;rsu2,uav2;..." 或 "default"
        例: "2,0;4,2;6,4"
    
    【返回值】
    List[Tuple[int, int, str]] - [(num_rsus, num_uavs, label), ...]
    
    【示例】
    parse_node_configs("2,0;4,2;6,4")
    # -> [(2, 0, "2R+0U"), (4, 2, "4R+2U"), (6, 4, "6R+4U")]
    """
    if not value or value.strip().lower() == "default":
        return DEFAULT_NODE_CONFIGS
    
    configs = []
    for item in value.split(";"):
        parts = item.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid config format: {item}. Expected 'rsu,uav'")
        num_rsus, num_uavs = int(parts[0]), int(parts[1])
        label = f"{num_rsus}R+{num_uavs}U"
        configs.append((num_rsus, num_uavs, label))
    
    return configs


def run_single_config(
    num_rsus: int,
    num_uavs: int,
    label: str,
    args: argparse.Namespace,
    suite_path: Path,
) -> Dict[str, Any]:
    """
    运行单个边缘节点配置的训练
    
    【功能】
    使用指定的RSU和UAV数量训练CAMTD3，并收集性能指标。
    
    【参数】
    num_rsus: int - RSU数量
    num_uavs: int - UAV数量
    label: str - 配置标签
    args: argparse.Namespace - 命令行参数
    suite_path: Path - Suite输出目录
    
    【返回值】
    Dict[str, Any] - 包含性能指标的字典
    """
    print(f"\n{'='*60}")
    print(f"Running: Edge Nodes = {label}")
    print(f"{'='*60}")
    
    # ========== 步骤1: 设置随机种子 ==========
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()
    
    # ========== 步骤2: 构建场景覆盖配置 ==========
    override_scenario = {
        "num_vehicles": 12,
        "num_rsus": num_rsus,
        "num_uavs": num_uavs,
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
    
    # 计算总节点数
    total_nodes = num_rsus + num_uavs
    
    # ========== 步骤5: 构建结果字典 ==========
    result_dict = {
        "label": label,
        "num_rsus": num_rsus,
        "num_uavs": num_uavs,
        "total_nodes": total_nodes,
        "avg_cost": avg_cost,
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "completion_rate": completion_rate,
        "cost_per_node": avg_cost / max(total_nodes, 1),  # 每节点成本
        "episodes": episodes,
        "seed": seed,
    }
    
    # ========== 步骤6: 保存结果到文件 ==========
    result_path = suite_path / f"nodes_{num_rsus}rsu_{num_uavs}uav.json"
    result_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"  Total Nodes    : {total_nodes} ({num_rsus}R + {num_uavs}U)")
    print(f"  Avg Cost       : {avg_cost:.4f}")
    print(f"  Cost/Node      : {result_dict['cost_per_node']:.4f}")
    print(f"  Avg Delay      : {avg_delay:.4f} s")
    print(f"  Avg Energy     : {avg_energy:.2f} J")
    print(f"  Completion Rate: {completion_rate:.3f}")
    
    return result_dict


def plot_results(results: List[Dict[str, Any]], suite_path: Path) -> None:
    """
    生成对比图表
    
    【功能】
    绘制边缘节点配置对性能的影响：
    1. 总节点数 vs 平均成本
    2. 总节点数 vs 任务完成率
    3. RSU-UAV配置热力图
    
    【参数】
    results: List[Dict] - 所有配置的结果列表
    suite_path: Path - 输出目录
    """
    # ========== 提取数据 ==========
    labels = [r["label"] for r in results]
    total_nodes = [r["total_nodes"] for r in results]
    costs = [r["avg_cost"] for r in results]
    completion_rates = [r["completion_rate"] for r in results]
    rsus = [r["num_rsus"] for r in results]
    uavs = [r["num_uavs"] for r in results]
    
    # ========== 设置绘图样式 ==========
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # ========== 图1: 总节点数 vs 平均成本 ==========
    fig, ax = plt.subplots()
    ax.plot(total_nodes, costs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Total Number of Edge Nodes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Cost', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Edge Node Count on System Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数据点
    for x, y, label in zip(total_nodes, costs, labels):
        ax.annotate(f'{y:.2f}\n{label}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(suite_path / "edge_nodes_vs_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图2: 总节点数 vs 任务完成率 ==========
    fig, ax = plt.subplots()
    ax.plot(total_nodes, completion_rates, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Total Number of Edge Nodes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Task Completion Rate', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Edge Node Count on Task Completion', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    for x, y, label in zip(total_nodes, completion_rates, labels):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(suite_path / "edge_nodes_vs_completion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图3: RSU-UAV配置热力图 ==========
    # 创建网格数据
    unique_rsus = sorted(set(rsus))
    unique_uavs = sorted(set(uavs))
    
    # 创建成本矩阵
    cost_matrix = np.full((len(unique_uavs), len(unique_rsus)), np.nan)
    for r in results:
        rsu_idx = unique_rsus.index(r["num_rsus"])
        uav_idx = unique_uavs.index(r["num_uavs"])
        cost_matrix[uav_idx, rsu_idx] = r["avg_cost"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(unique_rsus)))
    ax.set_yticks(np.arange(len(unique_uavs)))
    ax.set_xticklabels(unique_rsus)
    ax.set_yticklabels(unique_uavs)
    ax.set_xlabel('Number of RSUs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of UAVs', fontsize=13, fontweight='bold')
    ax.set_title('Cost Heatmap: RSU-UAV Configuration', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(unique_uavs)):
        for j in range(len(unique_rsus)):
            if not np.isnan(cost_matrix[i, j]):
                text = ax.text(j, i, f'{cost_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Cost', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(suite_path / "rsu_uav_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Charts saved:")
    print(f"  - {suite_path / 'edge_nodes_vs_cost.png'}")
    print(f"  - {suite_path / 'edge_nodes_vs_completion.png'}")
    print(f"  - {suite_path / 'rsu_uav_heatmap.png'}")
    print(f"{'='*60}")


def main() -> None:
    """脚本主入口函数"""
    parser = argparse.ArgumentParser(
        description="Evaluate CAMTD3 performance across different edge node configurations."
    )
    parser.add_argument(
        "--node-configs",
        type=str,
        default="default",
        help="Edge node configs in format 'rsu1,uav1;rsu2,uav2;...' or 'default'.",
    )
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default: 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default: 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"edge_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    node_configs = parse_node_configs(args.node_configs)
    
    # 准备输出目录
    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)
    
    # 循环运行各配置
    results = []
    for num_rsus, num_uavs, label in node_configs:
        result = run_single_config(num_rsus, num_uavs, label, args, suite_path)
        results.append(result)
    
    # 保存汇总结果
    summary = {
        "experiment_type": "edge_node_sensitivity",
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
    print("Edge Node Configuration Sensitivity Analysis Completed")
    print(f"{'='*60}")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"\nResults Summary:")
    print(f"{'Config':<20} {'Nodes':<8} {'Cost':<10} {'Completion':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['label']:<20} {r['total_nodes']:<8} {r['avg_cost']:<10.4f} {r['completion_rate']:<12.3f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

