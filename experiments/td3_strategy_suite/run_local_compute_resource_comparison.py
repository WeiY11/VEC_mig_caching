#!/usr/bin/env python3
"""
TD3 本地计算资源综合对比实验
==============================

【合并说明】
本实验合并了两个原实验：
1. run_local_resource_cost_comparison.py - 本地资源成本分析
2. run_local_resource_offload_comparison.py - 本地资源对卸载影响

【研究目标】
- 评估本地CPU频率对系统性能的综合影响
- 分析本地能力提升的成本收益
- 研究本地-边缘协同的平衡点
- 观察卸载决策随本地资源变化的规律

【核心指标】
- 总成本、时延成本、能耗成本
- 卸载数据量、卸载比例
- 本地执行比例
- 完成率

【论文对应】
- 资源配置敏感性分析
- 本地-边缘协同优化
- 卸载决策影响因素

【使用示例】
```bash
# 快速测试（10轮）
python experiments/td3_strategy_suite/run_local_compute_resource_comparison.py \\
    --episodes 10 --suite-id local_quick

# 完整实验（500轮）
python experiments/td3_strategy_suite/run_local_compute_resource_comparison.py \\
    --episodes 500 --seed 42

# 自定义CPU频率
python experiments/td3_strategy_suite/run_local_compute_resource_comparison.py \\
    --cpu-frequencies "1.0,2.0,3.0" --episodes 100
```
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# ========== 添加项目根目录到Python路径 ==========
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
from utils.unified_reward_calculator import UnifiedRewardCalculator

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
# 🎯 基于当前资源池配置：总本地计算资源（12车辆共享）
# 当前配置：6 GHz总本地计算
# 对比：受限(4GHz) vs 当前(6GHz) vs 充裕(10GHz, 15GHz)
DEFAULT_CPU_FREQS = [4.0, 6.0, 8.0, 10.0, 12.0]  # 总本地计算资源 (GHz)

_reward_calculator: UnifiedRewardCalculator | None = None


def _get_reward_calculator() -> UnifiedRewardCalculator:
    """获取全局奖励计算器实例"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator(algorithm="general")
    return _reward_calculator


def parse_cpu_frequencies(value: str) -> List[float]:
    """解析CPU频率配置字符串"""
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CPU_FREQS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def comprehensive_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """
    综合指标钩子：计算成本分量和卸载指标
    
    【功能】
    1. 计算时延成本和能耗成本分量
    2. 提取卸载数据量和卸载比例
    3. 计算本地执行比例
    """
    from config import config as global_config
    
    # ========== 成本分量计算 ==========
    weight_delay = float(global_config.rl.reward_weight_delay)
    weight_energy = float(global_config.rl.reward_weight_energy)
    
    # ✅ 修复：使用与训练时完全一致的归一化因子
    calc = _get_reward_calculator()
    delay_norm = max(calc.latency_target, 1e-6)  # 0.4（与训练一致）
    energy_norm = max(calc.energy_target, 1e-6)  # 1200.0（与训练一致）
    
    metrics["delay_cost"] = weight_delay * (metrics["avg_delay"] / delay_norm)
    metrics["energy_cost"] = weight_energy * (metrics["avg_energy"] / energy_norm)
    
    # ========== 卸载指标提取 ==========
    avg_offload_data_kb = tail_mean(episode_metrics.get("avg_offload_data_kb", []))
    offload_ratio = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # 备用估算（如果指标缺失）
    if avg_offload_data_kb <= 0:
        avg_task_size_kb = float(config.get("fallback_task_size_kb", 350.0))
        tasks_per_step = int(config.get("assumed_tasks_per_step", 12))
        avg_offload_data_kb = avg_task_size_kb * tasks_per_step * 0.6
    if offload_ratio <= 0:
        offload_ratio = 0.6
    
    metrics["avg_offload_data_kb"] = avg_offload_data_kb
    metrics["avg_offload_data_mb"] = avg_offload_data_kb / 1024.0
    metrics["offload_ratio"] = offload_ratio
    metrics["local_execution_ratio"] = 1.0 - offload_ratio


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """
    生成综合对比图表
    
    【图表清单】
    1. 总成本 vs CPU频率
    2. 成本分量 (时延+能耗) vs CPU频率
    3. 卸载数据量 vs CPU频率
    4. 卸载比例 vs CPU频率
    5. 归一化成本 vs CPU频率
    6. 完成率 vs CPU频率
    """
    cpu_freqs = [float(record["cpu_freq_ghz"]) for record in results]
    
    def make_chart(metric: str, ylabel: str, filename: str, title_suffix: str = None) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(cpu_freqs, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Local CPU Frequency (GHz)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        title = f"Impact of Local CPU Frequency on {title_suffix or ylabel}"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    
    # ========== 主要图表 ==========
    make_chart("raw_cost", "Average Cost", "local_compute_vs_total_cost.png", "Total Cost")
    make_chart("avg_delay", "Average Delay (s)", "local_compute_vs_delay.png", "Delay")
    make_chart("avg_energy", "Average Energy (J)", "local_compute_vs_energy.png", "Energy")
    make_chart("completion_rate", "Task Completion Rate", "local_compute_vs_completion.png", "Completion Rate")
    make_chart("normalized_cost", "Normalized Cost", "local_compute_vs_normalized_cost.png", "Normalized Cost")
    
    # ========== 成本分量图表 ==========
    make_chart("delay_cost", "Delay Cost Component", "local_compute_vs_delay_cost.png", "Delay Cost")
    make_chart("energy_cost", "Energy Cost Component", "local_compute_vs_energy_cost.png", "Energy Cost")
    
    # ========== 卸载行为图表 ==========
    make_chart("avg_offload_data_mb", "Offloaded Data (MB)", "local_compute_vs_offload_data.png", "Offloaded Data")
    make_chart("offload_ratio", "Offload Ratio", "local_compute_vs_offload_ratio.png", "Offload Ratio")
    make_chart("local_execution_ratio", "Local Execution Ratio", "local_compute_vs_local_ratio.png", "Local Execution")
    
    # ========== 成本分量组合图（双Y轴）==========
    fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
    
    for strat_key in strategy_keys:
        delay_costs = [record["strategies"][strat_key]["delay_cost"] for record in results]
        ax1.plot(cpu_freqs, delay_costs, marker="o", linewidth=2, linestyle='-', 
                label=f"{strategy_label(strat_key)} (Delay)")
    
    ax1.set_xlabel("Local CPU Frequency (GHz)", fontsize=12)
    ax1.set_ylabel("Delay Cost Component", fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    for strat_key in strategy_keys:
        energy_costs = [record["strategies"][strat_key]["energy_cost"] for record in results]
        ax2.plot(cpu_freqs, energy_costs, marker="s", linewidth=2, linestyle='--',
                label=f"{strategy_label(strat_key)} (Energy)")
    
    ax2.set_ylabel("Energy Cost Component", fontsize=12, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    
    plt.title("Cost Components vs Local CPU Frequency", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(suite_dir / "local_compute_vs_cost_components.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*70)
    print("图表已保存:")
    print("="*70)
    chart_groups = [
        ("核心性能指标", [
            "local_compute_vs_total_cost.png",
            "local_compute_vs_delay.png",
            "local_compute_vs_energy.png",
            "local_compute_vs_completion.png",
        ]),
        ("成本分析", [
            "local_compute_vs_delay_cost.png",
            "local_compute_vs_energy_cost.png",
            "local_compute_vs_cost_components.png",
        ]),
        ("卸载行为分析", [
            "local_compute_vs_offload_data.png",
            "local_compute_vs_offload_ratio.png",
            "local_compute_vs_local_ratio.png",
        ]),
        ("归一化指标", [
            "local_compute_vs_normalized_cost.png",
        ]),
    ]
    
    for group_name, chart_files in chart_groups:
        print(f"\n{group_name}:")
        for name in chart_files:
            print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of local computing resources impact on strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 快速测试
  python %(prog)s --episodes 10 --suite-id local_quick
  
  # 完整实验
  python %(prog)s --episodes 500 --seed 42
  
  # 自定义CPU频率
  python %(prog)s --cpu-frequencies "1.0,1.5,2.0,2.5,3.0" --episodes 100
        """
    )
    parser.add_argument(
        "--cpu-frequencies",
        type=str,
        default="default",
        help="逗号分隔的CPU频率列表 (GHz) 或 'default' 使用默认配置 [1.2, 2.0, 2.8]",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="local_compute",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    
    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="local_compute",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)
    
    cpu_freqs = parse_cpu_frequencies(args.cpu_frequencies)
    
    print("="*70)
    print("TD3 本地计算资源综合对比实验")
    print("="*70)
    print(f"CPU频率配置: {cpu_freqs} GHz")
    print(f"策略数量: {len(strategy_keys)}")
    print(f"每配置训练轮数: {common.episodes}")
    print(f"随机种子: {common.seed}")
    print(f"总训练次数: {len(cpu_freqs)} × {len(strategy_keys)} = {len(cpu_freqs) * len(strategy_keys)}")
    print("="*70)
    
    # ========== 构建配置列表 ==========
    configs: List[Dict[str, object]] = []
    for freq in cpu_freqs:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "total_vehicle_compute": float(freq) * 1e9,  # 🎯 总本地计算资源(Hz)，12车辆共享
            "override_topology": True,
            "fallback_task_size_kb": 350.0,
            "assumed_tasks_per_step": 12,
        }
        configs.append(
            {
                "key": f"{freq:.1f}ghz",
                "label": f"{freq:.1f} GHz",
                "overrides": overrides,
                "cpu_freq_ghz": freq,
            }
        )
    
    # ========== 运行实验 ==========
    suite_dir = build_suite_path(common)
    results = evaluate_configs(
        configs=configs,
        episodes=common.episodes,
        seed=common.seed,
        silent=common.silent,
        suite_path=suite_dir,
        strategies=strategy_keys,
        per_strategy_hook=comprehensive_metrics_hook,
        central_resource=common.central_resource,  # 🎯 传递中央资源分配参数
    )
    
    # ========== 生成图表 ==========
    plot_results(results, suite_dir, strategy_keys)
    
    # ========== 保存详细结果 ==========
    summary_path = suite_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "local_compute_resource_comparison",
                "description": "本地计算资源综合对比（合并实验）",
                "timestamp": datetime.now().isoformat(),
                "cpu_frequencies_ghz": cpu_freqs,
                "strategies": format_strategy_list(strategy_keys),
                "episodes": common.episodes,
                "seed": common.seed,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    
    print(f"\n汇总结果: {summary_path}")
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    
    # ========== 打印关键统计 ==========
    print("\n关键发现:")
    for i, record in enumerate(results):
        freq = record["cpu_freq_ghz"]
        print(f"\nCPU频率: {freq} GHz")
        for strat_key in strategy_keys[:3]:  # 只显示前3个策略
            metrics = record["strategies"][strat_key]
            print(f"  {strategy_label(strat_key)}:")
            print(f"    - 总成本: {metrics['raw_cost']:.3f}")
            print(f"    - 卸载比例: {metrics['offload_ratio']:.2%}")
            print(f"    - 完成率: {metrics['completion_rate']:.2%}")


if __name__ == "__main__":
    main()

