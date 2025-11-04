#!/usr/bin/env python3
"""
CAMTD3 通信信道质量对比实验
============================

研究目标
--------
- 评估不同信道质量条件对系统性能的影响
- 模拟优秀、良好、中等、较差、恶劣五种信道环境
- 分析通信开销在不同SNR条件下的变化
- 验证算法在弱通信环境下的鲁棒性

技术细节
--------
通过调整以下参数模拟信道质量：
1. 噪声功率 (noise_power_dbm): 影响SNR
2. 路径损耗指数 (path_loss_exponent): 影响信号衰减
3. 阴影衰落标准差 (shadow_fading_std): 影响信道波动

信道等级与3GPP标准对应：
- Excellent: LOS场景，低干扰（SNR > 20dB）
- Good: LOS场景，中等干扰（15 < SNR < 20dB）
- Medium: NLOS场景，标准干扰（10 < SNR < 15dB）
- Poor: NLOS场景，高干扰（5 < SNR < 10dB）
- Bad: 极端NLOS，严重干扰（SNR < 5dB）

学术价值
--------
- 支撑论文中关于"通信开销优化"的讨论
- 验证算法对无线环境变化的适应能力
- 为弱网络环境优化提供理论依据
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# ========== 添加项目根目录到 Python 路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
    tail_mean,
)
from experiments.camtd3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

# 信道质量配置
# 格式: (噪声功率dBm, 路径损耗指数, 阴影衰落std, 标签, 描述)
CHANNEL_CONFIGS = [{
        "key": "excellent", # 标准热噪声
        "path_loss_exponent": 3.0, }]


def channel_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """计算信道相关指标"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    
    # 通信开销相关
    metrics["comm_delay"] = tail_mean(episode_metrics.get("comm_delay", []))
    metrics["offload_ratio"] = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # 计算通信效率（完成率/通信时延）
    if metrics["comm_delay"] > 0:
        metrics["comm_efficiency"] = metrics["completion_rate"] / metrics["comm_delay"]
    else:
        metrics["comm_efficiency"] = 0.0


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """生成多维度对比图"""
    
    channel_labels = [record["channel_label"] for record in results]
    snr_values = [record["estimated_snr_db"] for record in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========== 子图1: 平均成本 vs SNR ==========
    ax = axes[0, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax.plot(snr_values, values, marker="o", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Average Cost")
    ax.set_title("Cost vs. Channel Quality (SNR)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_xaxis()  # SNR从高到低
    
    # ========== 子图2: 任务完成率 ==========
    ax = axes[0, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        ax.plot(snr_values, values, marker="s", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Task Completion vs. SNR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    
    # ========== 子图3: 通信时延 ==========
    ax = axes[0, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["comm_delay"] for record in results]
        ax.plot(snr_values, values, marker="^", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Communication Delay (s)")
    ax.set_title("Comm Delay vs. Channel Quality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    
    # ========== 子图4: 卸载率 ==========
    ax = axes[1, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["offload_ratio"] * 100 for record in results]
        ax.plot(snr_values, values, marker="D", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Offload Ratio (%)")
    ax.set_title("Offloading Decision vs. Channel Quality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    
    # ========== 子图5: 通信效率 ==========
    ax = axes[1, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["comm_efficiency"] for record in results]
        ax.plot(snr_values, values, marker="v", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Communication Efficiency")
    ax.set_title("Comm Efficiency (Completion/Delay)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    
    # ========== 子图6: 分类对比（条形图）==========
    ax = axes[1, 2]
    x_pos = range(len(channel_labels))
    width = 0.15
    for i, strat_key in enumerate(strategy_keys):
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        offset = (i - len(strategy_keys)/2) * width
        ax.bar([x + offset for x in x_pos], values, width, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channel_labels, rotation=15)
    ax.set_ylabel("Average Cost")
    ax.set_title("Cost by Channel Category")
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = suite_dir / "channel_quality_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nChart saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAMTD3 strategies under different channel quality conditions."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="channel_quality",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="channel_quality",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for channel_config in CHANNEL_CONFIGS:
        overrides = {
            "noise_power_dbm": channel_config["noise_power_dbm"],
            "path_loss_exponent": channel_config["path_loss_exponent"],
            "shadow_fading_std": channel_config["shadow_fading_std"],
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": channel_config["key"],
                "label": f"{channel_config['label']} (SNR≈{channel_config['estimated_snr_db']}dB)",
                "overrides": overrides,
                "channel_label": channel_config["label"],
                "channel_description": channel_config["description"],
                "estimated_snr_db": channel_config["estimated_snr_db"],
            }
        )

    suite_dir = build_suite_path(common)
    results = evaluate_configs(
        configs=configs,
        episodes=common.episodes,
        seed=common.seed,
        silent=common.silent,
        suite_path=suite_dir,
        strategies=strategy_keys,
        per_strategy_hook=channel_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "channel_quality_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "channel_configs": CHANNEL_CONFIGS,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nChannel Quality Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Channel Quality':<18} {'SNR(dB)':>8}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (26 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['channel_label']:<18} {record['estimated_snr_db']:>8d}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

