#!/usr/bin/env python3
"""
CAMTD3 策略消融实验批量运行器
==============================

功能概述
--------
- 顺序训练预设的多种策略配置（或指定子集）
- 汇总所有策略的训练指标与工件
- 计算归一化代价并输出对比图
- 生成可复现的实验目录结构，便于论文排版与复现

推荐用途
--------
- 一键获得 CAMTD3 各模块消融实验结果
- 作为批量实验脚本 `run_batch_experiments.py` 的底层工具
- 论文或报告中的策略性能对比图生成
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

from experiments.camtd3_strategy_suite.plot_strategy_comparison import (
    compute_normalized_costs,
    load_summary,
    plot_line,
)
from experiments.camtd3_strategy_suite.run_strategy_training import (
    DEFAULT_EPISODES,
    DEFAULT_SEED,
    STRATEGY_ORDER,
    run_strategy,
)
from experiments.camtd3_strategy_suite.suite_cli import (
    CommonArgs,
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    suite_path as build_suite_path,
)


def run_suite(strategies: Iterable[str], common_args: CommonArgs) -> None:
    """
    顺序执行多个策略的训练流程。
    """

    strategy_list = list(strategies)
    total = len(strategy_list)
    base_kwargs = {
        "episodes": common_args.episodes,
        "seed": common_args.seed,
        "suite_id": common_args.suite_id,
        "output_root": str(common_args.output_root),
        "silent": common_args.silent,
    }

    for index, strategy in enumerate(strategy_list, start=1):
        run_args = SimpleNamespace(**base_kwargs)
        print(f"\n=== [{index}/{total}] Running strategy: {strategy} ===")
        run_strategy(strategy, run_args)


def _plot_and_print(common_args: CommonArgs, output_name: str) -> None:
    suite_dir = build_suite_path(common_args)
    summary_path = suite_dir / "summary.json"
    summary = load_summary(summary_path)

    labels, values = compute_normalized_costs(summary)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    chart_path = suite_dir / output_name
    plot_line(labels, values, chart_path)

    print("\n=== Suite Completed ===")
    print(f"Strategies     : {format_strategy_list(common_args.strategies)}")
    print(f"Episodes       : {common_args.episodes}")
    print(f"Seed           : {common_args.seed}")
    print(f"Suite Path     : {suite_dir}")
    for label, value in zip(labels, values):
        print(f"{label:28s}: {value:.4f}")
    print(f"Chart saved to : {chart_path}")
    print(f"Summary updated: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute multiple CAMTD3 strategy runs and produce the comparison plot."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="CAMTD3_suite",
        default_output_root="results/camtd3_strategy_suite",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="camtd3_strategy_cost.png",
        help="Filename for the generated comparison chart.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting after the runs (summary.json will still be updated).",
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="CAMTD3_suite",
        default_output_root="results/camtd3_strategy_suite",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    strategies: List[str] = common.strategies or list(STRATEGY_ORDER)
    run_suite(strategies, common)

    if not args.skip_plot:
        _plot_and_print(common, args.output_name)


if __name__ == "__main__":
    main()
