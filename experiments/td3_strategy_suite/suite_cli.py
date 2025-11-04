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

from experiments.td3_strategy_suite.strategy_runner import STRATEGY_KEYS


@dataclass(frozen=True)
class CommonArgs:
    """Resolved CLI arguments shared by the comparison experiments."""

    episodes: int
    seed: int
    suite_id: str
    output_root: Path
    silent: bool
    strategies: Optional[List[str]]


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

    return CommonArgs(
        episodes=episodes,
        seed=seed,
        suite_id=suite_id,
        output_root=output_root,
        silent=silent,
        strategies=strategies,
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
