#!/usr/bin/env python3
"""
CAMTD3 Bandwidth Sensitivity Experiment (six strategies)
========================================================

Purpose:
- Compare six CAMTD3 strategies under varied channel bandwidth.
- Produce average cost, delay, throughput, and normalized plots.

Default sweep (MHz): 10, 20, 30, 40, 50

Default setup: 12 vehicles, 4 RSUs, 2 UAVs, 500 episodes per configuration.

Usage:
python experiments/camtd3_strategy_suite/run_bandwidth_cost_comparison.py \
    --episodes 100 --suite-id bandwidth_quick
python experiments/camtd3_strategy_suite/run_bandwidth_cost_comparison.py \
    --bandwidths "10,20,30,40,50" --episodes 300

Estimated runtime (six strategies):
- 100 episodes x 5 configs: ~1.5-2 hours
- 500 episodes x 5 configs: ~6-8 hours
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
DEFAULT_BANDWIDTHS = [10, 20, 30, 40, 50]


def parse_bandwidths(value: str) -> List[int]:
    """
    解析带宽配置字符串
    
    【功能】
    将命令行输入的带宽字符串解析为整数列表
    
    【参数】
    value: str - 带宽字符串，格式为 "5,10,20,40" 或 "default"
    
    【返回值】
    List[int] - 带宽值列表（单位：MHz）
    """
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_BANDWIDTHS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def bandwidth_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """
    带宽实验的后处理钩子函数
    
    【功能】
    计算平均吞吐量指标，用于评估通信性能。
    优先从训练记录中提取吞吐量数据，如果没有则根据任务大小和时延估算。
    
    【参数】
    strategy_key: str - 策略标识符
    metrics: Dict[str, float] - 性能指标字典（会被修改）
    config: Dict[str, object] - 配置信息
    episode_metrics: Dict[str, List[float]] - 训练过程中的指标序列
    
    【返回值】
    None - 直接修改metrics字典，添加 'avg_throughput_mbps' 字段
    
    【算法说明】
    1. 优先使用训练记录中的吞吐量数据（取后半段均值）
    2. 如果无记录，使用估算公式：吞吐量 = (任务大小 x 任务数) / 平均时延
    3. 确保吞吐量为非负值
    """
    # ========== 步骤1：从训练记录提取吞吐量 ==========
    avg_throughput = 0.0
    throughput_series = episode_metrics.get("throughput_mbps") or episode_metrics.get("avg_throughput_mbps")
    if throughput_series:
        values = list(map(float, throughput_series))
        if values:
            # 取后半段数据（收敛后的稳定值）
            half = values[len(values) // 2 :] if len(values) >= 100 else values
            avg_throughput = float(sum(half) / max(len(half), 1))
    
    # ========== 步骤2：如果无记录则估算吞吐量 ==========
    if avg_throughput <= 0:
        avg_task_size_mb = 0.35  # 假设平均任务大小350KB
        num_tasks_per_step = config.get("assumed_tasks_per_step", 12)
        if metrics["avg_delay"] > 0:
            avg_throughput = (avg_task_size_mb * num_tasks_per_step) / metrics["avg_delay"]
    
    # ========== 步骤3：保存吞吐量指标 ==========
    metrics["avg_throughput_mbps"] = max(avg_throughput, 0.0)


def plot_results(results: List[Dict[str, object]], suite_path: Path) -> None:
    """
    生成带宽敏感性分析图表
    
    【功能】
    为带宽实验生成四张对比图表：
    1. 带宽 vs 平均成本
    2. 带宽 vs 平均时延
    3. 带宽 vs 归一化成本
    4. 带宽 vs 吞吐量
    
    【参数】
    results: List[Dict] - 所有配置的实验结果
    suite_path: Path - 输出目录路径
    
    【返回值】
    None - 图表保存到suite_path目录
    
    【论文用途】
    这些图表用于论文的参数敏感性分析部分，展示：
    - 带宽资源对系统性能的影响
    - 各策略在不同带宽下的相对优势
    - 通信瓶颈的临界点
    """
    # ========== 提取带宽配置列表 ==========
    bandwidths = [int(r["bandwidth_mhz"]) for r in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        """
        生成单个对比图表
        
        【参数】
        metric: str - 指标名称（如 'raw_cost', 'avg_delay'）
        ylabel: str - Y轴标签
        filename: str - 输出文件名
        """
        # 绘制所有策略的曲线
        plt.figure(figsize=(10, 6))
        for strat_key in STRATEGY_KEYS:
            values = [r["strategies"][strat_key][metric] for r in results]
            plt.plot(bandwidths, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        
        # 设置图表样式
        plt.xlabel("Channel Bandwidth (MHz)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"Impact of Bandwidth on {ylabel}", fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 保存高分辨率图表（论文级别）
        plt.savefig(suite_path / filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ========== 生成所有图表 ==========
    make_chart("raw_cost", "Average Cost", "bandwidth_vs_total_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "bandwidth_vs_delay.png")
    make_chart("normalized_cost", "Normalized Cost", "bandwidth_vs_normalized_cost.png")
    make_chart("avg_throughput_mbps", "Average Throughput (Mbps)", "bandwidth_vs_throughput.png")

    print("\nCharts saved:")
    for name in [
        "bandwidth_vs_total_cost.png",
        "bandwidth_vs_delay.png",
        "bandwidth_vs_normalized_cost.png",
        "bandwidth_vs_throughput.png",
    ]:
        print(f"  - {suite_path / name}")


def main() -> None:
    """
    主函数：运行带宽敏感性分析实验
    
    【功能】
    1. 解析命令行参数
    2. 配置不同带宽场景
    3. 对每个场景评估六种策略
    4. 生成对比图表和汇总报告
    
    【实验流程】
    对于每个带宽配置：
      - 固定拓扑（12车辆+4RSU+2UAV）
      - 训练六种策略（各500轮）
      - 记录性能指标
    汇总所有结果并生成图表
    """
    parser = argparse.ArgumentParser(
        description="带宽敏感性分析：评估六种策略在不同信道带宽下的性能表现"
    )
    parser.add_argument("--bandwidths", type=str, default="default", help="Comma-separated bandwidth list (MHz).")
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"bandwidth_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier.",
    )
    parser.add_argument("--output-root", type=str, default="results/parameter_sensitivity", help="Output root directory.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    bandwidths = parse_bandwidths(args.bandwidths)
    episodes = args.episodes or DEFAULT_EPISODES
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    configs = []
    for bw in bandwidths:
        overrides = {
            "bandwidth": float(bw),
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{bw}mhz",
                "label": f"{bw} MHz",
                "overrides": overrides,
                "bandwidth_mhz": bw,
                "assumed_tasks_per_step": 12,
            }
        )

    suite_path = Path(args.output_root) / args.suite_id
    results = evaluate_configs(
        configs=configs,
        episodes=episodes,
        seed=seed,
        silent=args.silent,
        suite_path=suite_path,
        per_strategy_hook=bandwidth_hook,
    )

    summary = {
        "experiment_type": "bandwidth_cost_sensitivity",
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

    print("\nBandwidth Sensitivity Analysis Completed")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Bandwidth':<12}", end="")
    for strat_key in STRATEGY_KEYS:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(STRATEGY_KEYS)))
    for record in results:
        print(f"{record['bandwidth_mhz']:<12}", end="")
        for strat_key in STRATEGY_KEYS:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
