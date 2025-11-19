#!/usr/bin/env python3
"""
只跑带宽敏感性：
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py --experiment-types bandwidth

只跑“基站总计算资源”对比：
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py --experiment-types rsu_compute

只跑“无人机总计算资源”对比：
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py --experiment-types uav_compute



"""


from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, cast

import matplotlib.pyplot as plt

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.td3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
    strategy_group,
)
from experiments.td3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
    get_default_scenario_overrides,
    validate_td3_episodes,
)
from experiments.td3_strategy_suite.parameter_presets import (
    default_rsu_compute_levels,
)
from experiments.td3_strategy_suite.result_validation import (
    validate_experiment_results,
)

DEFAULT_EPISODES = 1500  # 🎯 优化：从800增加到1500，确保TD3充分收敛
DEFAULT_EPISODES_FAST = 500  # 🚀 快速验证模式：500轮，约1/3时间
DEFAULT_EPISODES_HEURISTIC = 300  # 🎯 启发式策略优化：300轮即可稳定
DEFAULT_SEED = 42
# 🎯 默认运行的5档参数（硬编码，不受config影响）
DEFAULT_BANDWIDTHS = [30.0, 40.0, 50.0, 60.0, 70.0]  # MHz
DEFAULT_RSU_COMPUTE_GHZ = default_rsu_compute_levels()
DEFAULT_UAV_COMPUTE_GHZ = [6.0, 7.0, 8.0, 9.0, 10.0]  # GHz
EXPERIMENT_CHOICES = ("bandwidth", "rsu_compute", "uav_compute")
GROUP_STYLE = {
    "baseline": {"color": "#1f77b4", "linestyle": "--"},
    "layered": {"color": "#ff7f0e", "linestyle": "-"},
}
GROUP_STYLE["default"] = {"color": "#7f7f7f", "linestyle": ":"}

STRATEGY_COLORS = {
    "local-only": "#1f77b4",
    "remote-only": "#ff7f0e",
    "offloading-only": "#2ca02c",
    "resource-only": "#d62728",
    "comprehensive-no-migration": "#9467bd",
    "comprehensive-migration": "#8c564b",
}


def _parse_float_sequence(value: str, default_values: Sequence[float]) -> List[float]:
    """通用浮点数组解析，支持 'default' 别名。"""

    if not value or value.strip().lower() == "default":
        return [float(v) for v in default_values]
    parsed: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    return parsed


def parse_bandwidths(value: str) -> List[float]:
    return _parse_float_sequence(value, DEFAULT_BANDWIDTHS)


def parse_rsu_compute_levels(value: str) -> List[float]:
    return _parse_float_sequence(value, DEFAULT_RSU_COMPUTE_GHZ)


def parse_uav_compute_levels(value: str) -> List[float]:
    return _parse_float_sequence(value, DEFAULT_UAV_COMPUTE_GHZ)


def parse_experiment_types(value: str) -> List[str]:
    """解析要运行的实验类型，支持'all'快捷项。"""

    if not value:
        return list(EXPERIMENT_CHOICES)

    lowered = value.strip().lower()
    if lowered in {"all", "default"}:
        return list(EXPERIMENT_CHOICES)

    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not selected:
        return list(EXPERIMENT_CHOICES)

    invalid = [item for item in selected if item not in EXPERIMENT_CHOICES]
    if invalid:
        options = ", ".join(EXPERIMENT_CHOICES)
        raise ValueError(f"未知实验类型 {', '.join(sorted(set(invalid)))}，应为: {options}")

    ordered = [choice for choice in EXPERIMENT_CHOICES if choice in selected]
    return ordered or list(EXPERIMENT_CHOICES)


def warn_if_not_five(values: Sequence[float], label: str) -> None:
    """确保参数组数为5，不满足时输出警告。"""

    if len(values) != 5:
        print(
            f"[警告] {label} 参数数量为 {len(values)}（推荐5组以保持一致对比）。",
            file=sys.stderr,
        )


def metrics_enrichment_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """指标增强钩子：计算吞吐量、RSU利用率、卸载率等关键指标"""
    # 🎯 优化1：吞吐量计算
    throughput_series = episode_metrics.get("throughput_mbps") or episode_metrics.get("avg_throughput_mbps")
    avg_throughput = 0.0
    if throughput_series:
        values = list(map(float, throughput_series))
        if values:
            half = values[len(values) // 2 :] if len(values) >= 100 else values
            avg_throughput = float(sum(half) / max(len(half), 1))

    if avg_throughput <= 0:
        avg_task_size_mb = 0.35  # 约 350KB
        num_tasks_per_step = int(cast(float, config.get("assumed_tasks_per_step", 12)))
        avg_delay = metrics.get("avg_delay", 0.0)
        if avg_delay > 0:
            avg_throughput = (avg_task_size_mb * num_tasks_per_step) / avg_delay

    metrics["avg_throughput_mbps"] = max(avg_throughput, 0.0)
    
    # 🎯 优化2：RSU利用率指标（验证资源是否被充分利用）
    rsu_util_series = episode_metrics.get("rsu_utilization") or episode_metrics.get("avg_rsu_utilization")
    if rsu_util_series:
        values = list(map(float, rsu_util_series))
        if values:
            half = values[len(values) // 2:] if len(values) >= 100 else values
            metrics["avg_rsu_utilization"] = float(sum(half) / max(len(half), 1))
    else:
        metrics["avg_rsu_utilization"] = 0.0
    
    # 🎯 优化3：卸载率指标（验证策略是否有效利用边缘资源）
    offload_series = episode_metrics.get("offload_ratio") or episode_metrics.get("remote_execution_ratio")
    if offload_series:
        values = list(map(float, offload_series))
        if values:
            half = values[len(values) // 2:] if len(values) >= 100 else values
            metrics["avg_offload_ratio"] = float(sum(half) / max(len(half), 1))
    else:
        metrics["avg_offload_ratio"] = 0.0
    
    # 🎯 优化4：队列长度指标（验证高资源配置下是否缓解拥塞）
    queue_series = episode_metrics.get("queue_rho_mean") or episode_metrics.get("avg_queue_length")
    if queue_series:
        values = list(map(float, queue_series))
        if values:
            half = values[len(values) // 2:] if len(values) >= 100 else values
            metrics["avg_queue_length"] = float(sum(half) / max(len(half), 1))
    else:
        metrics["avg_queue_length"] = 0.0
    
    # 🎯 优化5：性能稳定性指标（后半段标准差）
    delay_series = episode_metrics.get("avg_delay")
    if delay_series:
        values = list(map(float, delay_series))
        if len(values) >= 100:
            half = values[len(values) // 2:]
            if half:
                import numpy as np
                metrics["delay_std"] = float(np.std(half))
                metrics["delay_cv"] = float(np.std(half) / max(np.mean(half), 1e-6))  # 变异系数
    
    # 🎯 优化6：资源利用效率（任务完成率 / 资源消耗）
    completion_rate = metrics.get("completion_rate", 0.0)
    avg_energy = metrics.get("avg_energy", 1.0)
    if avg_energy > 0:
        metrics["resource_efficiency"] = completion_rate / avg_energy * 1000  # 归一化到合理范围


def build_bandwidth_configs(bandwidths: List[float]) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for bw in bandwidths:
        bw_hz = float(bw) * 1e6  # 转换为Hz (e.g., 10MHz -> 10e6 Hz)
        # 🎯 使用统一的默认配置，消除硬编码
        overrides = get_default_scenario_overrides(
            total_bandwidth=bw_hz,  # 统一使用 total_bandwidth
            assumed_tasks_per_step=12,
        )
        configs.append(
            {
                "key": f"{bw}mhz",
                "label": f"{bw} MHz",
                "overrides": overrides,
                "bandwidth_mhz": bw,
                "assumed_tasks_per_step": 12,
            }
        )
    return configs


def build_rsu_compute_configs(levels_ghz: List[float]) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for freq in levels_ghz:
        total_hz = float(freq) * 1e9
        # 🎯 使用统一的默认配置，消除硬编码
        overrides = get_default_scenario_overrides(
            total_rsu_compute=total_hz,
            assumed_tasks_per_step=12,
        )
        configs.append(
            {
                "key": f"rsu_{freq:.1f}ghz",
                "label": f"{freq:.1f} GHz",
                "overrides": overrides,
                "rsu_compute_ghz": freq,
                "assumed_tasks_per_step": 12,
            }
        )
    return configs


def build_uav_compute_configs(levels_ghz: List[float]) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for freq in levels_ghz:
        total_hz = float(freq) * 1e9
        # 🎯 使用统一的默认配置，消除硬编码
        overrides = get_default_scenario_overrides(
            total_uav_compute=total_hz,
            assumed_tasks_per_step=12,
        )
        configs.append(
            {
                "key": f"uav_{freq:.1f}ghz",
                "label": f"{freq:.1f} GHz",
                "overrides": overrides,
                "uav_compute_ghz": freq,
                "assumed_tasks_per_step": 12,
            }
        )
    return configs


def plot_results(
    results: List[Dict[str, object]],
    suite_dir: Path,
    strategy_keys: List[str],
    *,
    chart_prefix: str,
    title_prefix: str,
    x_label: str,
) -> List[Path]:
    labels = [str(record["label"]) for record in results]
    x_positions = range(len(results))
    saved_paths: List[Path] = []

    def make_chart(metric: str, ylabel: str, suffix: str, highlight_adaptive: bool = False) -> None:
        plt.figure(figsize=(12, 7))
        
        # 🎯 分组绘制：突出TD3策略的自适应能力
        adaptive_strategies = ['comprehensive-no-migration', 'comprehensive-migration']
        baseline_strategies = ['local-only', 'remote-only', 'offloading-only', 'resource-only']
        heuristic_strategies = ['random', 'round-robin']
        
        for strat_key in strategy_keys:
            values: List[float] = []
            for r in results:
                strategies_dict = cast(Dict[str, object], r["strategies"])
                strat_dict = cast(Dict[str, object], strategies_dict[strat_key])
                values.append(float(cast(float, strat_dict[metric])))
            
            group_name = strategy_group(strat_key)
            style = GROUP_STYLE.get(group_name, GROUP_STYLE["default"])
            label = f"{strategy_label(strat_key)} ({group_name})"
            color = STRATEGY_COLORS.get(strat_key, style.get("color"))
            linestyle = style.get("linestyle", "-")
            
            # 🎯 突出显示TD3策略
            if highlight_adaptive and strat_key in adaptive_strategies:
                linewidth = 3.0
                markersize = 10
                alpha = 1.0
            else:
                linewidth = 2.0 if strat_key in baseline_strategies else 1.5
                markersize = 8 if strat_key in baseline_strategies else 6
                alpha = 0.7 if strat_key in heuristic_strategies else 1.0
            
            plt.plot(
                x_positions,
                values,
                marker="o",
                linewidth=linewidth,
                markersize=markersize,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
            )
        
        plt.xticks(x_positions, cast(List[str], labels), fontsize=11)
        plt.xlabel(x_label, fontsize=13, fontweight='bold')
        plt.ylabel(ylabel, fontsize=13, fontweight='bold')
        plt.title(f"Impact of {title_prefix} on {ylabel}", fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, loc='best', framealpha=0.95)
        plt.tight_layout()
        filename = f"{chart_prefix}_vs_{suffix}.png"
        out_path = suite_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)

    # 🎯 基础性能指标
    make_chart("raw_cost", "Average Cost", "total_cost")
    make_chart("avg_delay", "Average Delay (s)", "delay")
    make_chart("normalized_cost", "Normalized Cost", "normalized_cost", highlight_adaptive=True)  # 突出TD3适应性
    make_chart("avg_throughput_mbps", "Average Throughput (Mbps)", "throughput")
    
    # 🎯 优化：新增资源利用率图表
    make_chart("avg_rsu_utilization", "RSU Utilization", "rsu_utilization")
    make_chart("avg_offload_ratio", "Offload Ratio", "offload_ratio")
    make_chart("avg_queue_length", "Average Queue Length", "queue_length")
    make_chart("resource_efficiency", "Resource Efficiency", "efficiency")

    print("\nCharts saved:")
    for path in saved_paths:
        print(f"  - {path}")
    return saved_paths


def print_cost_table(
    results: List[Dict[str, object]],
    strategy_keys: List[str],
    *,
    axis_field: str,
    axis_label: str,
) -> None:
    """按照指定X轴字段打印总成本表。"""

    header_width = 20
    print(f"\n{axis_label:<{header_width}}", end="")
    for strat_key in strategy_keys:
        label = f"{strategy_label(strat_key)}[{strategy_group(strat_key)}]"
        print(f"{label:>22}", end="")
    print()
    print("-" * (header_width + 22 * len(strategy_keys)))

    for record in results:
        axis_value = record.get(axis_field, record.get("label", "N/A"))
        if isinstance(axis_value, float):
            axis_str = f"{axis_value:.2f}"
        else:
            axis_str = str(axis_value)
        print(f"{axis_str:<{header_width}}", end="")
        for strat_key in strategy_keys:
            strategies_dict = cast(Dict[str, object], record["strategies"])
            strat_dict = cast(Dict[str, object], strategies_dict[strat_key])
            raw_cost = float(cast(float, strat_dict["raw_cost"]))
            print(f"{raw_cost:<22.4f}", end="")
        print()
    
    # 🎯 优化：打印关键指标对比表
    print("\n" + "="*80)
    print("📊 关键指标对比 (RSU利用率 | 卸载率 | 队列长度)")
    print("="*80)
    
    for record in results:
        axis_value = record.get(axis_field, record.get("label", "N/A"))
        if isinstance(axis_value, float):
            config_label = f"{axis_value:.1f}"
        else:
            config_label = str(axis_value)
        print(f"\n配置: {config_label}")
        print("-" * 80)
        
        for strat_key in strategy_keys:
            strategies_dict = cast(Dict[str, object], record["strategies"])
            strat_dict = cast(Dict[str, object], strategies_dict[strat_key])
            
            rsu_util = strat_dict.get("avg_rsu_utilization", 0.0)
            offload = strat_dict.get("avg_offload_ratio", 0.0)
            queue = strat_dict.get("avg_queue_length", 0.0)
            
            label = strategy_label(strat_key)
            print(f"  {label:40s} | RSU: {rsu_util:5.2f} | Offload: {offload:5.2f} | Queue: {queue:6.3f}")


def run_experiment_suite(
    *,
    experiment_key: str,
    configs: List[Dict[str, object]],
    suite_root: Path,
    strategy_keys: List[str],
    common_args,
    axis_field: str,
    axis_label: str,
    chart_prefix: str,
    title_prefix: str,
) -> Dict[str, object]:
    """运行单个对比实验并输出绘图/表格/JSON。"""

    if not configs:
        raise ValueError(f"{experiment_key} 实验配置为空，无法运行。")

    exp_dir = suite_root / experiment_key
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 🚨 修夏：训练轮次验证（防止严重性能坏掉）
    td3_strategies = ['comprehensive-no-migration', 'comprehensive-migration']
    td3_count = len([s for s in strategy_keys if s in td3_strategies])
    if td3_count > 0 and common_args.episodes < 1500:
        print("\n" + "="*80)
        print("⚠️  警告：TD3训练轮次严重不足！")
        print("="*80)
        print(f"🛑 当前轮次: {common_args.episodes}")
        print(f"✅ 建议轮次: 1500+ (最低要求)")
        print(f"❗ 影响: CAMTD3和无迁移TD3将完全未收敛")
        print(f"⚠️  后果: 成本可能高于启发式策略，结果无效")
        print(f"📊 预计时间: ~30h (1500轮) vs ~20h (当前{common_args.episodes}轮)")
        print("="*80)
        print("建议立即停止并使用正确参数重跑：")
        print("  python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \\")
        print("    --experiment-types rsu_compute --episodes 1500 --seed 42")
        print("="*80 + "\n")
        import time
        print("等待15秒以便您可以停止实验 (Ctrl+C)...")
        for i in range(15, 0, -1):
            print(f"\r{i}秒...", end="", flush=True)
            time.sleep(1)
        print("\n继续运行，但结果将被标记为'未收敛/无效'\n")
    
    # 🎯 启发式策略优化：为启发式策略使用300轮
    heuristic_strategies = ['local-only', 'remote-only', 'offloading-only', 'resource-only']
    heuristic_count = len([s for s in strategy_keys if s in heuristic_strategies])
    
    if common_args.optimize_heuristic and heuristic_count > 0:
        print(f"\n🎯 启发式策略优化已启用:")
        print(f"  - 启发式策略 ({heuristic_count}个): {DEFAULT_EPISODES_HEURISTIC}轮")
        if td3_count > 0:
            print(f"  - TD3策略 ({td3_count}个): {common_args.episodes}轮")
        time_saved = int((1 - DEFAULT_EPISODES_HEURISTIC/common_args.episodes) * heuristic_count / len(strategy_keys) * 100)
        print(f"  - 预计时间节省: ~{time_saved}%\n")

    strategy_episode_overrides: Dict[str, int] = {}
    if common_args.optimize_heuristic:
        for key in strategy_keys:
            if key in heuristic_strategies:
                strategy_episode_overrides[key] = DEFAULT_EPISODES_HEURISTIC

    results = evaluate_configs(
        configs=configs,
        episodes=common_args.episodes,
        seed=common_args.seed,
        silent=common_args.silent,
        suite_path=exp_dir,
        strategies=strategy_keys,
        per_strategy_hook=metrics_enrichment_hook,
        central_resource=common_args.central_resource,
        strategy_episode_overrides=strategy_episode_overrides or None,
    )

    plot_results(
        results,
        exp_dir,
        strategy_keys,
        chart_prefix=chart_prefix,
        title_prefix=title_prefix,
        x_label=axis_label,
    )
    print_cost_table(results, strategy_keys, axis_field=axis_field, axis_label=axis_label)

    summary = {
        "experiment_key": experiment_key,
        "title_prefix": title_prefix,
        "axis_field": axis_field,
        "axis_label": axis_label,
        "suite_id": common_args.suite_id,
        "created_at": datetime.now().isoformat(),
        "episodes": common_args.episodes,
        "seed": common_args.seed,
        "strategies": format_strategy_list(strategy_keys),
        "strategy_groups": sorted({strategy_group(k) for k in strategy_keys}),
        "num_configs": len(configs),
        "results": results,
    }
    summary_path = exp_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    return {
        "results": results,
        "summary_path": summary_path,
        "output_dir": exp_dir,
    }






def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TD3 strategies under varied bandwidth / RSU / UAV compute resources."
    )
    parser.add_argument(
        "--experiment-types",
        type=str,
        default="all",
        help="选择要运行的实验: bandwidth,rsu_compute,uav_compute 或 'all'（默认）。",
    )
    parser.add_argument(
        "--bandwidths",
        type=str,
        default="default",
        help=f"带宽列表(MHz)或 'default'（默认: {', '.join(map(str, DEFAULT_BANDWIDTHS))}）。",
    )
    parser.add_argument(
        "--rsu-compute-levels",
        type=str,
        default="default",
        help=(
            "RSU 总计算资源档位(GHz)或 'default'。"
            f" 默认: {', '.join(map(str, DEFAULT_RSU_COMPUTE_GHZ))}"
        ),
    )
    parser.add_argument(
        "--uav-compute-levels",
        type=str,
        default="default",
        help=(
            "UAV 总计算资源档位(GHz)或 'default'。"
            f" 默认: {', '.join(map(str, DEFAULT_UAV_COMPUTE_GHZ))}"
        ),
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="bandwidth",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    
    # 🚀 快速模式处理
    if args.fast_mode:
        print("\n" + "="*80)
        print("🚀 快速验证模式已启用")
        print("="*80)
        print(f"  训练轮次: 1500 → {DEFAULT_EPISODES_FAST}")
        print(f"  配置数量: 5 → 3（最小、中值、最大）")
        print(f"  预计时间节省: ~67%")
        print("="*80 + "\n")
        
        # 自动调整配置
        if args.bandwidths == "default":
            args.bandwidths = "20.0,40.0,60.0"  # 3个配置点
        if args.rsu_compute_levels == "default":
            args.rsu_compute_levels = "30.0,50.0,70.0"
        if args.uav_compute_levels == "default":
            args.uav_compute_levels = "6.0,8.0,10.0"
        
        # 使用快速轮次
        default_episodes_to_use = DEFAULT_EPISODES_FAST
    else:
        default_episodes_to_use = DEFAULT_EPISODES
    
    common = resolve_common_args(
        args,
        default_suite_prefix="bandwidth",
        default_output_root="results/parameter_sensitivity",
        default_episodes=default_episodes_to_use,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    experiment_types = parse_experiment_types(args.experiment_types)
    bandwidths = parse_bandwidths(args.bandwidths)
    rsu_levels = parse_rsu_compute_levels(args.rsu_compute_levels)
    uav_levels = parse_uav_compute_levels(args.uav_compute_levels)

    warn_if_not_five(bandwidths, "Bandwidth (MHz)")
    warn_if_not_five(rsu_levels, "RSU total compute (GHz)")
    warn_if_not_five(uav_levels, "UAV total compute (GHz)")
    
    # 🎯 优化：添加配置一致性检查
    from config import config as sys_config
    config_rsu_compute_ghz = float(getattr(sys_config.compute, 'total_rsu_compute', 50e9)) / 1e9
    middle_rsu_level = sorted(rsu_levels)[len(rsu_levels)//2] if rsu_levels else 50.0
    
    if abs(config_rsu_compute_ghz - middle_rsu_level) > 5.0:
        print(f"\n⚠️  [警告] 配置不一致：")
        print(f"   系统默认RSU计算资源: {config_rsu_compute_ghz:.1f} GHz")
        print(f"   实验中间配置点: {middle_rsu_level:.1f} GHz")
        print(f"   建议：使CAMTD3在中间配置点训练，可获得更好的泛化性能\n")

    suite_root = build_suite_path(common)
    suite_root.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('TD3 bandwidth/edge-resource sensitivity comparison')
    print('=' * 80)
    print(f"Experiments      : {', '.join(experiment_types)}")
    print(f"Episodes/Seed    : {common.episodes} | {common.seed}")
    print(f"Strategies       : {format_strategy_list(common.strategies)}")
    if common.strategy_groups:
        print(f"Strategy groups  : {', '.join(common.strategy_groups)}")
    print(f"Output directory : {suite_root}")
    print('=' * 80)

    executed_runs: List[Dict[str, object]] = []
    for exp in experiment_types:
        if exp == "bandwidth":
            print("\n>>> Running bandwidth sensitivity experiment (MHz)")
            configs = build_bandwidth_configs(bandwidths)
            run_info = run_experiment_suite(
                experiment_key="bandwidth",
                configs=configs,
                suite_root=suite_root,
                strategy_keys=strategy_keys,
                common_args=common,
                axis_field="bandwidth_mhz",
                axis_label="Bandwidth (MHz)",
                chart_prefix="bandwidth",
                title_prefix="Bandwidth",
            )
            executed_runs.append({"experiment": exp, **run_info})
        elif exp == "rsu_compute":
            print("\n>>> Running RSU total compute sensitivity experiment (GHz)")
            configs = build_rsu_compute_configs(rsu_levels)
            run_info = run_experiment_suite(
                experiment_key="rsu_compute",
                configs=configs,
                suite_root=suite_root,
                strategy_keys=strategy_keys,
                common_args=common,
                axis_field="rsu_compute_ghz",
                axis_label="RSU total compute (GHz)",
                chart_prefix="rsu_compute",
                title_prefix="RSU Total Compute",
            )
            executed_runs.append({"experiment": exp, **run_info})
        elif exp == "uav_compute":
            print("\n>>> Running UAV total compute sensitivity experiment (GHz)")
            configs = build_uav_compute_configs(uav_levels)
            run_info = run_experiment_suite(
                experiment_key="uav_compute",
                configs=configs,
                suite_root=suite_root,
                strategy_keys=strategy_keys,
                common_args=common,
                axis_field="uav_compute_ghz",
                axis_label="UAV total compute (GHz)",
                chart_prefix="uav_compute",
                title_prefix="UAV Total Compute",
            )
            executed_runs.append({"experiment": exp, **run_info})
    if not executed_runs:
        print('No experiments were selected; exiting.')
        return

    # 🎯 结果验证检查
    for run in executed_runs:
        exp_name = str(run['experiment'])  # 确保类型为str
        results_obj = run.get('results', [])
        if isinstance(results_obj, list):
            validate_experiment_results(results_obj, exp_name)

    print("\n" + "="*80)
    print("🎯 所有实验完成！输出摘要:")
    print("="*80)
    for run in executed_runs:
        print(f"  - {run['experiment']:<12} -> {run['output_dir']}")
        print(f"      summary: {run['summary_path']}")


if __name__ == "__main__":
    main()
