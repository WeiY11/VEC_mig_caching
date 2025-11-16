#!/usr/bin/env python3
"""
Common helpers for CLI parsing across TD3 strategy comparison experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from experiments.td3_strategy_suite.strategy_runner import (
    STRATEGY_GROUPS,
    STRATEGY_KEYS,
    strategy_group,
)


@dataclass(frozen=True)
class CommonArgs:
    """Resolved CLI arguments shared by the comparison experiments."""

    episodes: int
    seed: int
    suite_id: str
    output_root: Path
    silent: bool
    strategies: Optional[List[str]]
    central_resource: bool = False  # 🎯 中央资源分配架构
    strategy_groups: Optional[List[str]] = None
    optimize_heuristic: bool = True  # 🎯 启发式策略优化开关


# 🎯 默认基准场景配置（消除硬编码）
DEFAULT_SCENARIO_CONFIG = {
    "num_vehicles": 12,
    "num_rsus": 4,
    "num_uavs": 2,
    "override_topology": True,
}


def get_default_scenario_overrides(**custom_overrides) -> dict:
    """获取默认场景配置，支持自定义覆盖
    
    【功能】
    统一所有实验的基准场景配置，消除硬编码问题。
    
    【参数】
    **custom_overrides: 自定义覆盖配置（如 bandwidth=20e6）
    
    【返回值】
    合并后的配置字典
    
    【使用示例】
    ```python
    # 带宽实验
    overrides = get_default_scenario_overrides(bandwidth=20e6)
    
    # 自定义车辆数
    overrides = get_default_scenario_overrides(num_vehicles=16, bandwidth=30e6)
    ```
    """
    config = dict(DEFAULT_SCENARIO_CONFIG)
    config.update(custom_overrides)
    return config


def _add_boolean_toggle(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    default: bool,
    help_enable: str,
    help_disable: str,
) -> None:
    """
    Register a `--name/--no-name` boolean toggle with a default value.

    This keeps compatibility with Python versions that do not include
    `argparse.BooleanOptionalAction`.
    """

    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_enable)
    group.add_argument(f"--no-{name}", dest=name, action="store_false", help=help_disable)
    parser.set_defaults(**{name: default})


def add_common_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    default_suite_prefix: str,
    default_output_root: str,
    default_episodes: int,
    default_seed: int,
    allow_strategies: bool = False,
    allow_interactive_alias: bool = True,
) -> None:
    """
    Append frequently used CLI arguments for comparison experiments.
    """

    parser.add_argument(
        "--episodes",
        type=int,
        help=f"Override the number of training episodes (default: {default_episodes}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Random seed applied to every run (default: {default_seed}).",
    )
    parser.add_argument(
        "--suite-id",
        type=str,
        help=(
            "Suite identifier used for output aggregation. "
            f"Default: {default_suite_prefix}_YYYYmmdd_HHMMSS"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=default_output_root,
        help=f"Root directory for experiment outputs (default: {default_output_root}).",
    )

    _add_boolean_toggle(
        parser,
        name="silent",
        default=True,
        help_enable="Run training in silent mode (default).",
        help_disable="Disable silent mode and print verbose logs.",
    )

    if allow_interactive_alias:
        parser.add_argument(
            "--interactive",
            action="store_true",
            help="Alias for --no-silent to keep backward compatibility.",
        )

    if allow_strategies:
        parser.add_argument(
            "--strategies",
            type=str,
            help=(
                "Comma separated strategy names or 'all'. "
                "Defaults to all strategies defined in STRATEGY_PRESETS."
            ),
        )
    parser.add_argument(
        "--strategy-groups",
        type=str,
        help=(
            "Comma separated strategy group names (e.g. baseline,layered,joint) or 'all'."
        ),
    )
    
    # 🎯 中央资源分配架构参数
    parser.add_argument(
        "--central-resource",
        action="store_true",
        help="启用中央资源分配架构（Phase 1决策 + Phase 2执行），对比分层模式 vs 标准模式",
        )


def parse_strategy_selection(value: Optional[str]) -> Optional[List[str]]:
    """
    Convert a comma separated string into an ordered list of strategies.
    """

    if not value:
        return None

    lowered = value.strip().lower()
    if lowered in {"all", ""}:
        return None

    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in STRATEGY_KEYS]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(sorted(set(unknown)))}")

    requested_set = set(requested)
    ordered = [strategy for strategy in STRATEGY_KEYS if strategy in requested_set]
    return ordered or None


def parse_strategy_group_selection(value: Optional[str]) -> Optional[List[str]]:
    """
    Convert a comma separated string into an ordered list of strategy groups.
    """

    if not value:
        return None

    lowered = value.strip().lower()
    if lowered in {"all", ""}:
        return None

    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    canonical = {group.lower(): group for group in STRATEGY_GROUPS}
    unknown = [item for item in requested if item not in canonical]
    if unknown:
        raise ValueError(
            f"Unknown strategy groups: {', '.join(sorted(set(unknown)))}. "
            f"Available: {', '.join(STRATEGY_GROUPS)}"
        )

    ordered: List[str] = []
    for group in STRATEGY_GROUPS:
        if group.lower() in requested:
            ordered.append(group)
    return ordered or None


def resolve_common_args(
    args: argparse.Namespace,
    *,
    default_suite_prefix: str,
    default_output_root: str,
    default_episodes: int,
    default_seed: int,
    allow_strategies: bool = False,
) -> CommonArgs:
    """
    Materialise parsed CLI arguments and fill in defaults.
    """

    episodes = args.episodes or default_episodes
    seed = args.seed if args.seed is not None else default_seed

    suite_id = args.suite_id
    if not suite_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_id = f"{default_suite_prefix}_{timestamp}"

    output_root = Path(args.output_root or default_output_root)
    silent = bool(args.silent)
    if getattr(args, "interactive", False):
        silent = False

    strategies: Optional[List[str]] = None
    if allow_strategies:
        strategies = parse_strategy_selection(getattr(args, "strategies", None))
    strategy_groups = parse_strategy_group_selection(getattr(args, "strategy_groups", None))
    if strategy_groups:
        base_keys = strategies if strategies is not None else list(STRATEGY_KEYS)
        filtered = [key for key in base_keys if strategy_group(key) in strategy_groups]
        if not filtered:
            raise ValueError(
                f"No strategies remain after applying group filter(s): {', '.join(strategy_groups)}"
            )
        strategies = filtered
    
    # 🎯 获取optimize_heuristic参数
    central_resource = getattr(args, "central_resource", False)
    optimize_heuristic = getattr(args, "optimize_heuristic", True)

    return CommonArgs(
        episodes=episodes,
        seed=seed,
        suite_id=suite_id,
        output_root=output_root,
        silent=silent,
        strategies=strategies,
        central_resource=central_resource,
        strategy_groups=strategy_groups,
        optimize_heuristic=optimize_heuristic,
    )


def suite_path(common_args: CommonArgs) -> Path:
    """
    Convenience helper to build the suite path from common arguments.
    """

    return common_args.output_root / common_args.suite_id


def format_strategy_list(strategies: Optional[Sequence[str]]) -> str:
    """
    Human readable string for a strategy selection.
    """

    if not strategies:
        return ", ".join(STRATEGY_KEYS)
    return ", ".join(strategies)


def resolve_strategy_keys(strategies: Optional[Sequence[str]]) -> List[str]:
    """
    Return the ordered list of strategy keys that should participate in an experiment.
    """

    return list(strategies) if strategies else list(STRATEGY_KEYS)
