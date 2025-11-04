#!/usr/bin/env python3
"""
TD3 策略多场景鲁棒性对比
==========================

目标
----
- 在多种 VEC 场景下评估策略组合的鲁棒性与适应能力
- 支持自定义场景列表与策略子集
- 生成跨场景归一化成本对比图与详细 JSON 报告
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.td3_strategy_suite.strategy_runner import (
    enrich_with_normalized_costs,
    run_strategy_suite,
    strategy_label,
)
from experiments.td3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42

DEFAULT_SCENARIOS: List[Dict[str, object]] = [
    {"key": "baseline", "label": "Baseline", "overrides": {}},
    {"key": "high_load", "label": "High Load (λ=3.0)", "overrides": {"task_arrival_rate": 3.0}},
    {"key": "low_bandwidth", "label": "Low Bandwidth (10 MHz)", "overrides": {"bandwidth": 10.0}},
    {
        "key": "large_tasks",
        "label": "Large Data (300-800KB)",
        "overrides": {"task_data_size_min_kb": 300, "task_data_size_max_kb": 800},
    },
    {
        "key": "high_mobility",
        "label": "High Mobility (30 m/s)",
        "overrides": {"vehicle_speed": 30.0},
    },
    {
        "key": "dense_network",
        "label": "Dense Network (18 vehicles)",
        "overrides": {"num_vehicles": 18},
    },
]


def parse_scenarios_argument(value: Optional[str]) -> List[Dict[str, object]]:
    if not value:
        return [dict(item) for item in DEFAULT_SCENARIOS]
    path_obj = Path(value)
    if path_obj.exists():
        data = json.loads(path_obj.read_text(encoding="utf-8"))
    else:
        data = json.loads(value)

    scenarios: List[Dict[str, object]] = []
    for item in data:
        key = str(item.get("key") or item.get("label", "")).strip()
        if not key:
            raise ValueError("Each scenario must provide a 'key' or 'label'.")
        label = str(item.get("label", key))
        overrides = dict(item.get("overrides", {}))
        scenarios.append({"key": key, "label": label, "overrides": overrides})
    return scenarios


def plot_comparison(
    scenarios: List[Dict[str, object]],
    strategy_keys: List[str],
    normalized_costs: Dict[str, Dict[str, float]],
    suite_dir: Path,
) -> Path:
    labels = [str(sc.get("label", sc["key"])) for sc in scenarios]
    scenario_keys = [str(sc["key"]) for sc in scenarios]

    plt.figure(figsize=(10, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) < len(strategy_keys):
        factor = len(strategy_keys) // max(len(color_cycle), 1) + 1
        color_cycle = (color_cycle * factor)[: len(strategy_keys)]

    for idx, strat_key in enumerate(strategy_keys):
        values = [normalized_costs[sc_key][strat_key] for sc_key in scenario_keys]
        plt.plot(range(len(labels)), values, marker="o", linewidth=2, color=color_cycle[idx], label=strategy_label(strat_key))

    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Normalized Average Cost")
    plt.xlabel("Scenario")
    plt.title("Strategy Performance Across Scenarios")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    chart_path = suite_dir / "strategy_context_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    return chart_path


def save_summary(
    suite_dir: Path,
    scenarios: List[Dict[str, object]],
    strategy_keys: List[str],
    scenario_results: Dict[str, Dict[str, Dict[str, float]]],
) -> Path:
    summary = {
        "suite_id": suite_dir.name,
        "created_at": datetime.now().isoformat(),
        "strategies": strategy_keys,
        "scenarios": scenarios,
        "results": scenario_results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TD3 strategies across multiple deployment scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="JSON file path or inline JSON describing scenarios. Defaults to built-in list.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="strategy_context",
        default_output_root="results/td3_strategy_suite",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="strategy_context",
        default_output_root="results/td3_strategy_suite",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    scenarios = parse_scenarios_argument(args.scenarios)
    suite_dir = build_suite_path(common)
    suite_dir.mkdir(parents=True, exist_ok=True)

    scenario_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    normalized_costs: Dict[str, Dict[str, float]] = {}

    for scenario in scenarios:
        sc_key = str(scenario["key"])
        sc_label = str(scenario.get("label", sc_key))
        overrides = dict(scenario.get("overrides", {}))

        print(f"\n{'='*72}\nScenario: {sc_label} ({sc_key})")
        print(f"Overrides: {json.dumps(overrides, ensure_ascii=False)}")
        print('=' * 72)

        raw_outcomes = run_strategy_suite(
            override_scenario=overrides,
            episodes=common.episodes,
            seed=common.seed,
            silent=common.silent,
            strategies=strategy_keys,
        )

        enriched = enrich_with_normalized_costs(raw_outcomes)
        scenario_results[sc_key] = enriched
        normalized_costs[sc_key] = {k: v["normalized_cost"] for k, v in enriched.items()}

        scenario_dir = suite_dir / sc_key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_dir.joinpath("scenario_summary.json").write_text(
            json.dumps(
                {
                    "scenario_key": sc_key,
                    "scenario_label": sc_label,
                    "overrides": overrides,
                    "results": enriched,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        for strat_key, metrics in enriched.items():
            print(
                f"--> {strategy_label(strat_key)} "
                f"Cost={metrics['raw_cost']:.4f} Delay={metrics['avg_delay']:.4f}s "
                f"Energy={metrics['avg_energy']:.2f}J Completion={metrics['completion_rate']:.3f}"
            )
            detail_path = scenario_dir / f"{strat_key}.json"
            detail_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    chart_path = plot_comparison(
        scenarios=scenarios,
        strategy_keys=strategy_keys,
        normalized_costs=normalized_costs,
        suite_dir=suite_dir,
    )
    summary_path = save_summary(
        suite_dir=suite_dir,
        scenarios=scenarios,
        strategy_keys=strategy_keys,
        scenario_results=scenario_results,
    )

    print("\n=== Completed Strategy Context Comparison ===")
    print(f"Suite ID       : {common.suite_id}")
    print(f"Strategies     : {format_strategy_list(common.strategies)}")
    print(f"Scenario count : {len(scenarios)}")
    print(f"Summary JSON   : {summary_path}")
    print(f"Comparison plot: {chart_path}")


if __name__ == "__main__":
    main()
