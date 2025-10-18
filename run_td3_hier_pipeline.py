#!/usr/bin/env python3
"""
一键运行 TD3-Hierarchical 分层策略训练 → 蒸馏 → 学生策略回放验证。

流程:
1. 多任务训练 (调用 experiments.run_hierarchical_multitask_training)
2. 教师策略补充训练并保存 (SingleAgentTrainingEnvironment)
3. 策略蒸馏 (experiments.distill_hierarchical_policy)
4. 学生策略回放验证 (collect_policy_rollout)
python run_td3_hier_pipeline.py --scenarios 2 --episodes-per-task 100 --teacher-episodes 20 --rollout-steps 600

默认参数尽量保持运行时间可控，可通过命令行覆盖。
python run_td3_hier_pipeline.py \
  --scenarios 3 \
  --episodes-per-task 200 \
  --teacher-episodes 10 \
  --rollout-steps 400 \
  --teacher-prefix results/models/single_agent/td3_hierarchical/teacher_custom \
  --student-prefix results/models/single_agent/td3_hierarchical/student_custom \
  --student-hidden-dim 256 \
  --student-high-hidden 192 \
  --student-low-hidden 320 \
  --distill-epochs 50 \
  --distill-batch-size 512 \
  --distill-rollout-steps 800 \
  --distill-repeats 8

"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import List, Dict

import numpy as np

from experiments import (
    get_default_multitask_scenarios,
    run_hierarchical_multitask_training,
    distill_hierarchical_policy,
    DistillationConfig,
)
from train_single_agent import SingleAgentTrainingEnvironment
from single_agent.td3_hierarchical import HierarchicalTD3Config, HierarchicalTD3Agent


def ensure_td3_env_defaults() -> None:
    """确保关键环境变量与分层策略尺寸匹配。"""
    defaults = {
        "TD3_HIDDEN_DIM": "256",
        "TD3_ACTOR_LR": "5e-05",
        "TD3_CRITIC_LR": "8e-05",
        "TD3_BATCH_SIZE": "128",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def train_teacher_policy(
    scenario: Dict,
    episodes: int,
    teacher_prefix: str,
    use_enhanced_cache: bool,
) -> str:
    """使用单环境补充训练教师策略，确保模型文件存在。"""
    env = SingleAgentTrainingEnvironment(
        "TD3-HIER",
        override_scenario=scenario,
        use_enhanced_cache=use_enhanced_cache,
    )
    for episode in range(episodes):
        env.run_episode(episode + 1)
    env.agent_env.save_models(teacher_prefix)
    return teacher_prefix


def evaluate_student_policy(
    scenario: Dict,
    student_prefix: str,
    rollout_steps: int,
    distill_cfg: DistillationConfig,
    use_enhanced_cache: bool,
) -> Dict[str, float]:
    """加载学生策略回放评估，返回核心指标。"""
    env = SingleAgentTrainingEnvironment(
        "TD3-HIER",
        override_scenario=scenario,
        use_enhanced_cache=use_enhanced_cache,
    )

    student_config = HierarchicalTD3Config(hidden_dim=distill_cfg.student_hidden_dim)
    student_config.high_level_hidden = distill_cfg.student_high_hidden
    student_config.low_level_hidden = distill_cfg.student_low_hidden
    student_config.actor_lr = distill_cfg.learning_rate

    env.agent_env.config = student_config
    env.agent_env.agent = HierarchicalTD3Agent(
        env.agent_env.state_dim,
        env.agent_env.action_dim,
        student_config,
        env.agent_env.num_rsus,
        env.agent_env.num_uavs,
    )
    env.agent_env.load_models(student_prefix)

    rollout = env.collect_policy_rollout(max_steps=rollout_steps, deterministic=True)
    steps = int(rollout["states"].shape[0])
    avg_reward = float(np.mean(rollout["rewards"])) if steps > 0 else np.nan
    sample_metrics = rollout["metrics"][0] if rollout["metrics"] else {}

    return {
        "rollout_steps": steps,
        "avg_reward": avg_reward,
        "sample_metrics": sample_metrics,
    }


def build_timestamp(prefix: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键运行 TD3-HIER 分层策略训练 + 蒸馏 + 回放验证",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=1,
        help="参与多任务训练的默认场景数量 (默认: 1)",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=12,
        help="run_hierarchical_multitask_training 每个场景运行的训练轮数 (默认: 12)",
    )
    parser.add_argument(
        "--teacher-episodes",
        type=int,
        default=3,
        help="额外教师补充训练的轮次 (默认: 3，保持运行时间可控)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=200,
        help="学生策略验证的回放步数 (默认: 200)",
    )
    parser.add_argument(
        "--use-enhanced-cache",
        action="store_true",
        help="启用增强缓存系统 (默认关闭以缩短运行时间)",
    )
    parser.add_argument(
        "--teacher-prefix",
        type=str,
        default=None,
        help="教师模型保存前缀 (默认自动生成)",
    )
    parser.add_argument(
        "--student-prefix",
        type=str,
        default=None,
        help="学生模型保存前缀 (默认自动生成)",
    )
    parser.add_argument(
        "--student-hidden-dim",
        type=int,
        default=256,
        help="学生 Actor trunk 隐藏维度 (默认: 256)",
    )
    parser.add_argument(
        "--student-high-hidden",
        type=int,
        default=192,
        help="学生高层策略隐藏维度 (默认: 192)",
    )
    parser.add_argument(
        "--student-low-hidden",
        type=int,
        default=320,
        help="学生低层策略隐藏维度 (默认: 320)",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=40,
        help="蒸馏训练 epochs (默认: 40)",
    )
    parser.add_argument(
        "--distill-batch-size",
        type=int,
        default=512,
        help="蒸馏批次大小 (默认: 512)",
    )
    parser.add_argument(
        "--distill-rollout-steps",
        type=int,
        default=600,
        help="蒸馏数据采集 rollout 步数 (默认: 600)",
    )
    parser.add_argument(
        "--distill-repeats",
        type=int,
        default=6,
        help="蒸馏每场景 rollout 次数 (默认: 6)",
    )

    args = parser.parse_args()

    ensure_td3_env_defaults()

    scenarios: List[Dict] = get_default_multitask_scenarios()[: max(1, args.scenarios)]
    use_enhanced = bool(args.use_enhanced_cache)

    print("==== Step 1/4: 多任务训练 ====")
    run_hierarchical_multitask_training(
        scenarios=scenarios,
        episodes_per_task=args.episodes_per_task,
        use_enhanced_cache=use_enhanced,
    )

    print("\n==== Step 2/4: 教师策略补充训练 ====")
    teacher_prefix = (
        args.teacher_prefix
        if args.teacher_prefix
        else f"results/models/single_agent/td3_hierarchical/{build_timestamp('teacher_pipeline')}"
    )
    Path(teacher_prefix).parent.mkdir(parents=True, exist_ok=True)
    train_teacher_policy(
        scenario=scenarios[0],
        episodes=args.teacher_episodes,
        teacher_prefix=teacher_prefix,
        use_enhanced_cache=use_enhanced,
    )

    print("\n==== Step 3/4: 策略蒸馏 ====")
    distill_cfg = DistillationConfig(
        rollout_steps=args.distill_rollout_steps,
        rollout_repeats=args.distill_repeats,
        batch_size=args.distill_batch_size,
        epochs=args.distill_epochs,
        student_hidden_dim=args.student_hidden_dim,
        student_high_hidden=args.student_high_hidden,
        student_low_hidden=args.student_low_hidden,
    )
    student_prefix = (
        args.student_prefix
        if args.student_prefix
        else f"results/models/single_agent/td3_hierarchical/{build_timestamp('student_pipeline')}"
    )
    Path(student_prefix).parent.mkdir(parents=True, exist_ok=True)
    distill_hierarchical_policy(
        teacher_checkpoint=teacher_prefix,
        output_path=student_prefix,
        scenarios=scenarios,
        use_enhanced_cache=use_enhanced,
        config=distill_cfg,
    )

    print("\n==== Step 4/4: 学生策略回放验证 ====")
    eval_stats = evaluate_student_policy(
        scenario=scenarios[0],
        student_prefix=student_prefix,
        rollout_steps=args.rollout_steps,
        distill_cfg=distill_cfg,
        use_enhanced_cache=use_enhanced,
    )

    print("\n===== Pipeline Summary =====")
    print(f" - Teacher model: {teacher_prefix}_td3.pth")
    print(f" - Student model: {student_prefix}_td3.pth")
    print(f" - Rollout steps: {eval_stats['rollout_steps']}")
    print(f" - Avg reward  : {eval_stats['avg_reward']:.4f}")
    sample_metrics = eval_stats.get("sample_metrics") or {}
    if sample_metrics:
        print(" - Sample metrics (first step):")
        for k, v in sample_metrics.items():
            print(f"    * {k}: {v}")
    else:
        print(" - Sample metrics: <empty>")
    print("============================")


if __name__ == "__main__":
    main()
