#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3 多车辆数量灵敏度实验脚本

示例用法：
    python experiments/run_td3_vehicle_sweep.py --vehicles 8 12 16 --episodes 200
    python experiments/run_td3_vehicle_sweep.py --vehicle-range 8 16 4 --episodes 150

脚本会针对不同车辆数量运行 TD3 训练，
通过环境变量 `TRAINING_SCENARIO_OVERRIDES` 覆盖仿真器配置，
并将关键结果汇总保存。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from train_single_agent import _apply_global_seed_from_env, train_single_algorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行TD3不同车辆数量实验")
    parser.add_argument(
        "--vehicles",
        type=int,
        nargs="*",
        help="显式指定车辆数量列表 (优先级高于 --vehicle-range)",
    )
    parser.add_argument(
        "--vehicle-range",
        type=int,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="使用范围生成车辆数量 (含起始, 不含终止)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="每个车辆设置的训练轮次 (默认: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="实验统一随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/td3_vehicle_sweep"),
        help="实验结果输出目录 (默认: results/experiments/td3_vehicle_sweep)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="评估间隔 (透传给train_single_algorithm)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="保存间隔 (透传给train_single_algorithm)",
    )
    return parser.parse_args()


def _build_vehicle_list(args: argparse.Namespace) -> List[int]:
    if args.vehicles:
        return args.vehicles
    if args.vehicle_range:
        start, end, step = args.vehicle_range
        if step <= 0:
            raise ValueError("vehicle-range 的步长必须为正数")
        return list(range(start, end, step))
    return [8, 12, 16]


def _run_single_setting(num_vehicles: int, seed: int, episodes: int, eval_interval: int | None, save_interval: int | None) -> Dict:
    os.environ['RANDOM_SEED'] = str(seed)
    overrides = {"num_vehicles": num_vehicles, "override_topology": True}
    os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(overrides)
    _apply_global_seed_from_env()
    try:
        return train_single_algorithm(
            "TD3",
            num_episodes=episodes,
            eval_interval=eval_interval,
            save_interval=save_interval,
            silent_mode=True,  # 🔧 启用静默模式，避免用户交互阻塞批量实验
            override_scenario=overrides
        )
    finally:
        os.environ.pop('TRAINING_SCENARIO_OVERRIDES', None)


def _extract_summary(num_vehicles: int, run_result: Dict) -> Dict:
    final_perf = run_result.get("final_performance", {})
    training_cfg = run_result.get("training_config", {})
    
    # 从训练环境获取实际状态维度（如果可用）
    state_dim = "N/A"
    if "state_dim" in run_result:
        state_dim = run_result["state_dim"]
    
    return {
        "num_vehicles": num_vehicles,
        "state_dim": state_dim,
        "episodes": training_cfg.get("num_episodes", 0),
        "training_time_hours": training_cfg.get("training_time_hours", 0.0),
        "avg_step_reward": final_perf.get("avg_step_reward", 0.0),
        "avg_delay": final_perf.get("avg_delay", 0.0),
        "avg_completion": final_perf.get("avg_completion", 0.0),
    }


def _save_results(output_dir: Path, summaries: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"td3_vehicle_sweep_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2, ensure_ascii=False)

    md_path = output_dir / f"td3_vehicle_sweep_summary_{timestamp}.md"
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# TD3 不同车辆数量实验结果\n\n")
        fp.write("| Vehicles | State Dim | Episodes | Training Hours | Avg Step Reward | Avg Delay (s) | Completion Rate |\n")
        fp.write("| -------- | --------- | -------- | --------------- | ---------------- | ------------- | ---------------- |\n")
        for item in summaries:
            fp.write(
                f"| {item['num_vehicles']} | {item['state_dim']} | {item['episodes']} | {item['training_time_hours']:.3f} |"
                f" {item['avg_step_reward']:.4f} | {item['avg_delay']:.4f} | {item['avg_completion']:.2%} |\n"
            )

    print(f"✅ 结果已保存: {summary_path}")
    print(f"✅ Markdown 简报: {md_path}")


def main() -> None:
    args = parse_args()
    vehicle_list = _build_vehicle_list(args)

    print("=" * 80)
    print("🚗 TD3 不同车辆数量实验启动")
    print(f"车辆数设置: {vehicle_list}")
    print(f"统一随机种子: {args.seed}")
    print(f"每个实验训练轮次: {args.episodes}")
    print("=" * 80)

    summaries: List[Dict] = []
    for num_vehicles in vehicle_list:
        print("-" * 60)
        print(f"▶️ 开始运行 num_vehicles = {num_vehicles}")
        result = _run_single_setting(num_vehicles, args.seed, args.episodes, args.eval_interval, args.save_interval)
        summary = _extract_summary(num_vehicles, result)
        summaries.append(summary)
        print(
            f"✅ num_vehicles={num_vehicles} 完成: "
            f"Delay={summary['avg_delay']:.4f}s, Completion={summary['avg_completion']:.2%}"
        )

    _save_results(args.output_dir, summaries)
    print("=" * 80)
    print("🎉 不同车辆数量实验全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


