#!/usr/bin/env python3
"""
CAMTD3 本地计算资源对成本影响实验（六策略版本）
==========================================

【功能】
评估车辆本地计算能力对系统成本的影响，对比六种策略在不同本地资源下的表现。
通过扫描不同的车辆CPU频率，分析：
- 本地计算能力如何影响卸载决策
- 时延成本与能耗成本的权衡
- 本地执行与远程卸载的成本对比

【论文对应】
- 参数敏感性分析（Parameter Sensitivity Analysis）
- 本地计算vs边缘卸载权衡分析
- 验证CAMTD3对本地资源变化的适应能力

【实验设计】
扫描参数: vehicle_cpu_freq (车辆CPU频率 GHz)
- 入门性能: 1.2 GHz（轻量设备）
- 均衡性能: 1.6 GHz
- 标准性能: 2.0 GHz（默认配置）
- 强化性能: 2.4 GHz
- 高性能: 2.8 GHz（高端设备）

固定参数:
- 车辆数: 12
- RSU数: 4
- UAV数: 2
- 训练轮数: 可配置（默认500）

【核心指标】
- 平均总成本（时延+能耗）
- 时延成本分量（weight_delay × avg_delay）
- 能耗成本分量（weight_energy × avg_energy）
- 归一化成本

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --episodes 100 --suite-id local_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --episodes 500 --seed 42 --suite-id local_paper

# 自定义CPU频率配置（单位：GHz）
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --cpu-frequencies "1.0,2.0,3.0,4.0" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_local_resource_cost_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 7配置 × 6策略）：约2-3小时
- 完整实验（500轮 × 7配置 × 6策略）：约8-12小时

【输出图表】
- local_cpu_vs_total_cost.png: CPU频率 vs 总成本
- local_cpu_vs_delay_cost.png: CPU频率 vs 时延成本
- local_cpu_vs_energy_cost.png: CPU频率 vs 能耗成本
- local_cpu_vs_normalized_cost.png: CPU频率 vs 归一化成本

【论文贡献】
揭示本地计算能力对卸载决策的影响，为设备选型提供指导
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
DEFAULT_CPU_FREQS = [1.2, 1.6, 2.0, 2.4, 2.8]


def parse_cpu_frequencies(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CPU_FREQS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def cost_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    weight_delay = float(config.get("weight_delay", 0))
    weight_energy = float(config.get("weight_energy", 0))
    if not weight_delay or not weight_energy:
        from config import config as global_config  # local import to avoid circular issues

        weight_delay = float(global_config.rl.reward_weight_delay)
        weight_energy = float(global_config.rl.reward_weight_energy)

    delay_cost = weight_delay * metrics["avg_delay"]
    energy_cost = weight_energy * (metrics["avg_energy"] / 1000.0)
    metrics["delay_cost"] = delay_cost
    metrics["energy_cost"] = energy_cost


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

    make_chart("raw_cost", "Average Cost", "local_cpu_vs_total_cost.png")
    make_chart("delay_cost", "Delay Cost Component", "local_cpu_vs_delay_cost.png")
    make_chart("energy_cost", "Energy Cost Component", "local_cpu_vs_energy_cost.png")
    make_chart("normalized_cost", "Normalized Cost", "local_cpu_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "local_cpu_vs_total_cost.png",
        "local_cpu_vs_delay_cost.png",
        "local_cpu_vs_energy_cost.png",
        "local_cpu_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_path / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cost impact of local computing resources across strategies.")
    parser.add_argument("--cpu-frequencies", type=str, default="default", help="Comma-separated CPU frequencies (GHz).")
    parser.add_argument("--episodes", type=int, help="Training episodes per configuration (default 500).")
    parser.add_argument("--seed", type=int, help="Random seed (default 42).")
    parser.add_argument(
        "--suite-id",
        type=str,
        default=f"local_resource_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
            }
        )

    suite_path = Path(args.output_root) / args.suite_id
    results = evaluate_configs(
        configs=configs,
        episodes=episodes,
        seed=seed,
        silent=args.silent,
        suite_path=suite_path,
        per_strategy_hook=cost_hook,
    )

    summary = {
        "experiment_type": "local_resource_cost_sensitivity",
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

    print("\nLocal Resource Cost Analysis Completed")
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
