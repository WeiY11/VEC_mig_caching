#!/usr/bin/env python3
"""
Run baseline and layered TD3 strategy suites in sequence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiments.td3_strategy_suite import suite_cli
from experiments.td3_strategy_suite.strategy_runner import (
    STRATEGY_GROUPS,
    evaluate_configs,
    strategies_for_group,
    strategy_group,
)

GROUP_SEQUENCE = ("baseline", "layered")


def parse_groups(value: str) -> List[str]:
    if not value or value.strip().lower() in {"all", ""}:
        return list(GROUP_SEQUENCE)
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    canonical = {name.lower(): name for name in STRATEGY_GROUPS}
    invalid = [item for item in requested if item not in canonical]
    if invalid:
        raise ValueError(
            f"Unknown groups: {', '.join(sorted(set(invalid)))}. "
            f"Available groups: {', '.join(STRATEGY_GROUPS)}"
        )
    ordered: List[str] = []
    for group in GROUP_SEQUENCE:
        if group.lower() in requested:
            ordered.append(group)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline → layered → joint TD3 strategy groups sequentially."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="baseline,layered",
        help="Comma separated list drawn from baseline,layered. Use 'all' for both groups.",
    )
    suite_cli.add_common_experiment_args(
        parser,
        default_suite_prefix="joint_vs_layered",
        default_output_root="results/strategy_groups",
        default_episodes=500,
        default_seed=42,
        allow_strategies=True,
    )

    args = parser.parse_args()
    selected_groups = parse_groups(args.groups)

    common = suite_cli.resolve_common_args(
        args,
        default_suite_prefix="joint_vs_layered",
        default_output_root="results/strategy_groups",
        default_episodes=500,
        default_seed=42,
        allow_strategies=True,
    )

    suite_dir = suite_cli.suite_path(common)
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Baseline vs Layered Strategy Suite")
    print("=" * 72)
    print(f"Groups     : {', '.join(selected_groups)}")
    print(f"Strategies : {suite_cli.format_strategy_list(common.strategies)}")
    print(f"Episodes   : {common.episodes} | Seed: {common.seed}")
    print(f"Output dir : {suite_dir}")
    print("=" * 72)

    aggregate: Dict[str, Dict[str, object]] = {"runs": []}

    default_config = {
        "key": "default",
        "label": "Default Scenario",
        "overrides": {},
    }

    for group in selected_groups:
        if common.strategies:
            group_keys = [key for key in common.strategies if strategy_group(key) == group]
        else:
            group_keys = strategies_for_group(group)

        if not group_keys:
            print(f"\n>>> Skipping {group!r}: no strategies available after filtering.")
            continue

        print(f"\n>>> Running group: {group} ({len(group_keys)} strategies)")
        group_dir = suite_dir / f"group_{group}"
        results = evaluate_configs(
            configs=[default_config],
            episodes=common.episodes,
            seed=common.seed,
            silent=common.silent,
            suite_path=group_dir,
            strategies=group_keys,
            per_strategy_hook=None,
            central_resource=common.central_resource,
        )
        aggregate["runs"].append(
            {
                "group": group,
                "strategies": group_keys,
                "output_dir": str(group_dir),
                "results": results,
            }
        )

    summary_path = suite_dir / "group_summary.json"
    aggregate["created_at"] = datetime.now().isoformat()
    aggregate["episodes"] = common.episodes
    aggregate["seed"] = common.seed
    aggregate["strategy_filter"] = common.strategies
    aggregate["selected_groups"] = selected_groups
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCompleted group runs:")
    for entry in aggregate["runs"]:
        print(f"  - {entry['group']:<8} -> {entry['output_dir']}")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
