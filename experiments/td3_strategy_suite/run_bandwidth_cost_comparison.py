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
    get_default_scenario_overrides,  # 🎯 消除硬编码
)
from experiments.td3_strategy_suite.parameter_presets import (
    default_rsu_compute_levels,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
# 🎯 默认运行的五档参数
DEFAULT_BANDWIDTHS = [20.0, 30.0, 40.0, 50.0, 60.0]  # MHz
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


def build_bandwidth_configs(bandwidths: List[float]) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for bw in bandwidths:
        bw_hz = float(bw) * 1e6  # 转换为Hz (e.g., 10MHz -> 10e6 Hz)
        # 🎯 使用统一的默认配置，消除硬编码
        overrides = get_default_scenario_overrides(
            bandwidth=bw_hz,
            total_bandwidth=bw_hz,
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

    def make_chart(metric: str, ylabel: str, suffix: str) -> None:
        plt.figure(figsize=(10, 6))
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
            plt.plot(
                x_positions,
                values,
                marker="o",
                linewidth=2,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        plt.xticks(x_positions, cast(List[str], labels))
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(f"Impact of {title_prefix} on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        filename = f"{chart_prefix}_vs_{suffix}.png"
        out_path = suite_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)

    make_chart("raw_cost", "Average Cost", "total_cost")
    make_chart("avg_delay", "Average Delay (s)", "delay")
    make_chart("normalized_cost", "Normalized Cost", "normalized_cost")
    make_chart("avg_throughput_mbps", "Average Throughput (Mbps)", "throughput")

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

    results = evaluate_configs(
        configs=configs,
        episodes=common_args.episodes,
        seed=common_args.seed,
        silent=common_args.silent,
        suite_path=exp_dir,
        strategies=strategy_keys,
        per_strategy_hook=metrics_enrichment_hook,
        central_resource=common_args.central_resource,
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
    common = resolve_common_args(
        args,
        default_suite_prefix="bandwidth",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
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

    print("\nAll experiments completed. Summary outputs:")
    for run in executed_runs:
        print(f"  - {run['experiment']:<12} -> {run['output_dir']}")
        print(f"      summary: {run['summary_path']}")


if __name__ == "__main__":
    main()
