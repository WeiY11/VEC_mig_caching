#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3 多随机种子实验脚本

示例用法：
    python experiments/run_td3_seed_sweep.py --seeds 42 2025 3407 --episodes 200
    python experiments/run_td3_seed_sweep.py --seed-start 0 --seed-count 5 --episodes 100

脚本会循环调用 `train_single_agent.train_single_algorithm`，分别设置随机种子，
并保存每次运行的关键指标到 JSON 和 Markdown 报告中，方便论文复现实验。

【用途】
- 批量运行TD3在多个随机种子下的训练，汇总均值与置信区间，输出JSON与Markdown简报。

【运行命令】
- 指定种子列表：python experiments/run_td3_seed_sweep.py --seeds 42 2025 3407 --episodes 200
- 连续生成种子：python experiments/run_td3_seed_sweep.py --seed-start 0 --seed-count 5 --episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from train_single_agent import (
    _apply_global_seed_from_env,  # type: ignore F401 - 内部函数用于重设随机种子
    train_single_algorithm,
)

try:
    from scipy.stats import t as student_t  # type: ignore
except ImportError:  # pragma: no cover - optional依赖
    student_t = None

CONFIDENCE_LEVEL = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行TD3多随机种子实验")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        help="显式指定随机种子列表 (优先级高于 --seed-start/--seed-count)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="当未显式指定 --seeds 时，起始随机种子 (默认: 0)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=3,
        help="当未显式指定 --seeds 时，需要运行的种子数量 (默认: 3)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="每个种子的训练轮数 (默认: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/td3_seed_sweep"),
        help="实验结果输出目录 (默认: results/experiments/td3_seed_sweep)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="训练评估间隔，透传给 train_single_algorithm (默认自动)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="模型保存间隔，透传给 train_single_algorithm (默认自动)",
    )
    return parser.parse_args()


def _build_seed_list(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return args.seeds
    return [args.seed_start + idx for idx in range(args.seed_count)]


def _run_single_seed(seed: int, episodes: int, eval_interval: int | None, save_interval: int | None) -> Dict:
    previous_seed = os.environ.get("RANDOM_SEED")
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()
    try:
        return train_single_algorithm(
            "TD3",
            num_episodes=episodes,
            eval_interval=eval_interval,
            save_interval=save_interval,
            silent_mode=True  # 🔧 启用静默模式，避免用户交互阻塞批量实验
        )
    finally:
        if previous_seed is not None:
            os.environ["RANDOM_SEED"] = previous_seed
            _apply_global_seed_from_env()
        else:
            os.environ.pop("RANDOM_SEED", None)


def _extract_summary(seed: int, run_result: Dict) -> Dict:
    final_perf = run_result.get("final_performance", {})
    training_cfg = run_result.get("training_config", {})
    return {
        "seed": seed,
        "episodes": training_cfg.get("num_episodes", 0),
        "training_time_hours": training_cfg.get("training_time_hours", 0.0),
        "avg_step_reward": final_perf.get("avg_step_reward", 0.0),
        "avg_delay": final_perf.get("avg_delay", 0.0),
        "avg_completion": final_perf.get("avg_completion", 0.0),
    }


def _compute_confidence_interval(values: List[float], confidence: float = CONFIDENCE_LEVEL) -> Dict[str, float]:
    cleaned = [float(v) for v in values if v is not None]
    n = len(cleaned)
    if n == 0:
        return {"mean": 0.0, "half_width": 0.0}
    mean_val = sum(cleaned) / n
    if n == 1:
        return {"mean": mean_val, "half_width": 0.0}
    variance = sum((v - mean_val) ** 2 for v in cleaned) / (n - 1)
    std_dev = sqrt(variance)
    standard_error = std_dev / sqrt(n)
    if student_t is not None:
        critical = float(student_t.ppf((1 + confidence) / 2.0, n - 1))
    else:
        critical = float(NormalDist().inv_cdf((1 + confidence) / 2.0))
    return {"mean": mean_val, "half_width": critical * standard_error}


def _aggregate_metrics(summaries: List[Dict], confidence: float) -> Dict[str, Dict[str, float]]:
    metric_keys = {
        "avg_step_reward": "avg_step_reward",
        "avg_delay": "avg_delay",
        "avg_completion": "avg_completion",
    }
    aggregates: Dict[str, Dict[str, float]] = {}
    for metric, source_key in metric_keys.items():
        metric_values = [summary.get(source_key, 0.0) for summary in summaries]
        aggregates[metric] = _compute_confidence_interval(metric_values, confidence=confidence)
    return aggregates


def _save_results(output_dir: Path, summaries: List[Dict],
                  aggregates: Optional[Dict[str, Dict[str, float]]],
                  confidence: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"td3_seed_sweep_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        payload = {"runs": summaries}
        if aggregates is not None:
            payload["aggregated"] = {
                "confidence": confidence,
                "metrics": aggregates,
            }
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    # 同步输出Markdown简报，方便论文使用
    md_path = output_dir / f"td3_seed_sweep_summary_{timestamp}.md"
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# TD3 多随机种子实验结果\n\n")
        fp.write("| Seed | Episodes | Training Hours | Avg Step Reward | Avg Delay (s) | Completion Rate |\n")
        fp.write("| ---- | -------- | --------------- | ---------------- | ------------- | ---------------- |\n")
        for item in summaries:
            fp.write(
                f"| {item['seed']} | {item['episodes']} | {item['training_time_hours']:.3f} |"
                f" {item['avg_step_reward']:.4f} | {item['avg_delay']:.4f} | {item['avg_completion']:.2%} |\n"
            )
        if aggregates:
            fp.write("\n## 统计汇总\n\n")
            fp.write(f"置信水平：{confidence:.0%}\n\n")
            fp.write("| 指标 | 平均值 | 置信区间半宽 |\n")
            fp.write("| ---- | ------ | -------------- |\n")
            for metric, stats in aggregates.items():
                fp.write(f"| {metric} | {stats['mean']:.4f} | ±{stats['half_width']:.4f} |\n")

    print(f"✅ 结果已保存: {summary_path}")
    print(f"✅ Markdown 简报: {md_path}")


def main() -> None:
    args = parse_args()
    seeds = _build_seed_list(args)
    print("=" * 80)
    print("🚀 TD3 多随机种子实验启动")
    print(f"运行种子: {seeds}")
    print(f"每个实验训练轮数: {args.episodes}")
    print("=" * 80)

    summaries: List[Dict] = []
    for seed in seeds:
        print("-" * 60)
        print(f"▶️ 开始运行 Seed = {seed}")
        result = _run_single_seed(seed, args.episodes, args.eval_interval, args.save_interval)
        summary = _extract_summary(seed, result)
        summaries.append(summary)
        print(f"✅ Seed {seed} 运行完成: Avg Delay={summary['avg_delay']:.4f}s, Completion={summary['avg_completion']:.2%}")

    aggregates = _aggregate_metrics(summaries, confidence=CONFIDENCE_LEVEL)
    print("\n统计汇总（均值 ± 置信区间半宽）")
    for metric, stats in aggregates.items():
        print(f"- {metric}: {stats['mean']:.4f} ± {stats['half_width']:.4f} (@ {CONFIDENCE_LEVEL:.0%})")

    _save_results(args.output_dir, summaries, aggregates, CONFIDENCE_LEVEL)
    print("=" * 80)
    print("🎉 多随机种子实验全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


