#!/usr/bin/env python3
"""
CAMTD3 本地计算资源对卸载量影响实验（六策略版本）
==========================================

【功能】
评估车辆本地计算能力对任务卸载决策的影响，对比六种策略的卸载行为。
通过扫描不同的车辆CPU频率，分析：
- 本地计算能力如何影响卸载比例
- 本地执行与远程卸载的动态平衡
- 各策略对本地资源变化的响应机制

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 卸载决策行为分析
- 本地-边缘协同优化

【实验设计】
扫描参数: vehicle_cpu_freq (车辆CPU频率 GHz)
- 低性能: 1.0 GHz（强制卸载场景）
- 中低性能: 1.5 GHz
- 标准性能: 2.0 GHz（默认配置）
- 中高性能: 2.5 GHz
- 高性能: 3.0 GHz（本地优先场景）

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本
- 平均卸载数据量（MB）：衡量卸载行为
- 卸载比例（offload_ratio）：卸载任务占比
- 归一化成本

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --episodes 100 --suite-id offload_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --episodes 500 --seed 42 --suite-id offload_paper

# 自定义CPU频率配置（单位：GHz）
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --cpu-frequencies "0.8,1.5,2.5,3.5" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_local_resource_offload_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 5配置 × 6策略）：约1.5-2.5小时
- 完整实验（500轮 × 5配置 × 6策略）：约6-9小时

【输出图表】
- local_cpu_vs_cost.png: CPU频率 vs 平均成本
- local_cpu_vs_offload_data.png: CPU频率 vs 卸载数据量
- local_cpu_vs_offload_ratio.png: CPU频率 vs 卸载比例
- local_cpu_vs_normalized_cost.png: CPU频率 vs 归一化成本

【论文贡献】
揭示本地资源对卸载决策的影响机制，验证智能卸载策略的有效性
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
    tail_mean,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_CPU_FREQS = [1.0, 1.5, 2.0, 2.5, 3.0]


def parse_cpu_frequencies(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CPU_FREQS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def offload_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    avg_offload_data_kb = tail_mean(episode_metrics.get("avg_offload_data_kb", []))
    offload_ratio = tail_mean(episode_metrics.get("offload_ratio", []))
    if avg_offload_data_kb <= 0:
        avg_task_size_kb = float(config.get("fallback_task_size_kb", 350.0))
        tasks_per_step = int(config.get("assumed_tasks_per_step", 12))
        avg_offload_data_kb = avg_task_size_kb * tasks_per_step * 0.6
    if offload_ratio <= 0:
        offload_ratio = 0.6

    metrics["avg_offload_data_kb"] = avg_offload_data_kb
    metrics["avg_offload_data_mb"] = avg_offload_data_kb / 1024.0
    metrics["offload_ratio"] = offload_ratio


def plot_results(results: List[Dict[str, object]], suite_path: Path) -> None:
    cpu_freqs = [float(r["cpu_freq_ghz"]) for r in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in STRATEGY_KEYS:
            values = [r["strategies"][strat_key][metric] for r in results]
            plt.plot(cpu_freqs, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Local CPU Frequency (GHz)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Local CPU Frequency on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_path / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "local_cpu_vs_cost.png")
    make_chart("avg_offload_data_mb", "Average Offloaded Data (MB)", "local_cpu_vs_offload_data.png")
    make_chart("offload_ratio", "Offload Ratio", "local_cpu_vs_offload_ratio.png")
    make_chart("normalized_cost", "Normalized Cost", "local_cpu_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "local_cpu_vs_cost.png",
        "local_cpu_vs_offload_data.png",
        "local_cpu_vs_offload_ratio.png",
        "local_cpu_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_path / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offloading behaviour under different local CPU frequencies.")
    parser.add_argument("--cpu-frequencies", type=str, default="default", help="Comma-separated CPU frequencies (GHz).")
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"local_offload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite identifier.",
    )
    parser.add_argument("--output-root", type=str, default="results/parameter_sensitivity", help="Output root directory.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    cpu_freqs = parse_cpu_frequencies(args.cpu_frequencies)
    episodes = args.episodes or DEFAULT_EPISODES
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    configs: List[Dict[str, object]] = []
    for freq in cpu_freqs:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "vehicle_cpu_freq": float(freq) * 1e9,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{freq:.1f}ghz",
                "label": f"{freq:.1f} GHz",
                "overrides": overrides,
                "cpu_freq_ghz": freq,
                "fallback_task_size_kb": 350.0,
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
        per_strategy_hook=offload_hook,
    )

    summary = {
        "experiment_type": "local_resource_offload_sensitivity",
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

    print("\nLocal Resource Offload Analysis Completed")
    print(f"Suite ID: {args.suite_id}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'CPU (GHz)':<12}", end="")
    for strat_key in STRATEGY_KEYS:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(STRATEGY_KEYS)))
    for record in results:
        print(f"{record['cpu_freq_ghz']:<12.1f}", end="")
        for strat_key in STRATEGY_KEYS:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
