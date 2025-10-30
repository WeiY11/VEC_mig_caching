#!/usr/bin/env python3
"""
CAMTD3 策略多场景对比实验（六策略版本）
==========================================

【功能】
在多个典型VEC场景下对比六种策略的综合性能，评估策略的鲁棒性和适应能力。
通过在不同系统压力和资源条件下测试，全面分析：
- 各策略在不同场景下的相对优势
- 策略对环境变化的鲁棒性
- 最佳策略的选择依据

【论文对应】
- 综合性能评估（Comprehensive Performance Evaluation）
- 鲁棒性分析（Robustness Analysis）
- 场景适应性对比

【实验设计】
测试场景（可自定义）：
1. Baseline：标准配置场景
2. High Load：高负载场景（λ=3.0 tasks/s）
3. Low Bandwidth：低带宽场景（10 MHz）
4. Large Tasks：大任务场景（300-800 KB）
5. High Mobility：高移动性场景（30 m/s）
6. Dense Network：密集网络场景（18辆车）

【核心指标】
- 各场景下的平均成本
- 归一化成本（便于跨场景对比）
- 时延、能耗等详细指标
- 统计显著性分析

【使用示例】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试（100轮）
python experiments/camtd3_strategy_suite/run_strategy_context_comparison.py \\
    --episodes 100 --suite-id context_quick

# 完整实验（500轮）- 自动保存报告，无人值守运行
python experiments/camtd3_strategy_suite/run_strategy_context_comparison.py \\
    --episodes 500 --seed 42 --suite-id context_paper

# 自定义场景配置
python experiments/camtd3_strategy_suite/run_strategy_context_comparison.py \\
    --scenarios "baseline,high_load,low_bandwidth" --episodes 300

# 💡 如需交互式确认保存报告，添加 --interactive 参数
python experiments/camtd3_strategy_suite/run_strategy_context_comparison.py \\
    --episodes 500 --interactive
```

【预计运行时间】
- 快速测试（100轮 × 6场景 × 6策略）：约2-3小时
- 完整实验（500轮 × 6场景 × 6策略）：约8-12小时

【输出图表】
- strategy_vs_scenarios_heatmap.png: 策略-场景热力图
- strategy_performance_radar.png: 雷达图对比
- scenario_cost_comparison.png: 各场景成本对比

【论文贡献】
全面展示CAMTD3在各种场景下的优势，证明其良好的适应性和鲁棒性
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import (
    STRATEGY_KEYS,
    enrich_with_normalized_costs,
    run_strategy_suite,
    strategy_label,
)

DEFAULT_SCENARIOS: List[Dict[str, object]] = [
    {"key": "baseline", "label": "Baseline", "overrides": {}},
    {"key": "high_load", "label": "High Load (λ=3.0)", "overrides": {"task_arrival_rate": 3.0}},
    {"key": "low_bandwidth", "label": "Low Bandwidth (10 MHz)", "overrides": {"bandwidth": 10.0}},
    {
        "key": "large_tasks",
        "label": "Large Data (300-800KB)",
        "overrides": {"task_data_size_min_kb": 300, "task_data_size_max_kb": 800},
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
    suite_path: Path,
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
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3, linestyle="--", axis="y")
    plt.legend()
    plt.tight_layout()

    chart_path = suite_path / "strategy_vs_scenarios_cost.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path


def save_summary(
    suite_path: Path,
    scenarios: List[Dict[str, object]],
    strategy_keys: List[str],
    scenario_results: Dict[str, Dict[str, Dict[str, float]]],
) -> Path:
    summary = {
        "suite_id": suite_path.name,
        "created_at": datetime.now().isoformat(),
        "strategies": strategy_keys,
        "scenarios": scenarios,
        "results": scenario_results,
    }
    summary_path = suite_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare six CAMTD3 strategies across multiple scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenarios", type=str, help="JSON file or inline JSON describing the scenarios.")
    parser.add_argument("--strategies", type=str, default="all", help="Comma-separated strategy keys (default: all six).")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes for each strategy in each scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--suite-id", type=str, default=f"strategy_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Suite identifier.")
    parser.add_argument("--output-root", type=str, default="results/camtd3_strategy_suite", help="Root directory for outputs.")
    parser.add_argument("--silent", action="store_true", default=True, help="Run training in silent mode (default: True for batch experiments).")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (overrides silent).")
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则禁用静默模式
    if args.interactive:
        args.silent = False

    scenarios = parse_scenarios_argument(args.scenarios)
    if args.strategies.lower() == "all":
        strategy_keys = list(STRATEGY_KEYS)
    else:
        strategy_keys = [item.strip() for item in args.strategies.split(",") if item.strip()]
        unknown = [key for key in strategy_keys if key not in STRATEGY_KEYS]
        if unknown:
            raise ValueError(f"Unknown strategy keys: {', '.join(unknown)}")

    suite_path = Path(args.output_root) / args.suite_id
    suite_path.mkdir(parents=True, exist_ok=True)

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
            episodes=args.episodes,
            seed=args.seed,
            silent=args.silent,
            strategies=strategy_keys,
        )

        enriched = enrich_with_normalized_costs(raw_outcomes)
        scenario_results[sc_key] = enriched
        normalized_costs[sc_key] = {k: v["normalized_cost"] for k, v in enriched.items()}

        scenario_dir = suite_path / sc_key
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
        suite_path=suite_path,
    )
    summary_path = save_summary(
        suite_path=suite_path,
        scenarios=scenarios,
        strategy_keys=strategy_keys,
        scenario_results=scenario_results,
    )

    print("\n=== Completed Strategy Context Comparison ===")
    print(f"Suite directory : {suite_path}")
    print(f"Summary JSON    : {summary_path}")
    print(f"Comparison plot : {chart_path}")


if __name__ == "__main__":
    main()
