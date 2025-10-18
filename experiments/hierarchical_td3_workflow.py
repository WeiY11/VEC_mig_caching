"""
分层TD3（TD3-Hier）多任务训练 & 策略蒸馏工作流工具

用途：
1. run_hierarchical_multitask_training: 给定场景列表批量训练 TD3-Hier。
2. distill_hierarchical_policy: 将高容量教师策略蒸馏到更轻量学生网络。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from single_agent.td3_hierarchical import (
    HierarchicalTD3Config,
    HierarchicalTD3Agent,
    TD3HierarchicalEnvironment,
)
from train_single_agent import train_single_algorithm, SingleAgentTrainingEnvironment


def get_default_multitask_scenarios() -> List[Dict[str, Any]]:
    """针对VEC缓存/迁移特性的默认多任务场景组合。"""
    base_arrival = 1.8
    scenarios = [
        {"num_vehicles": 8, "task_arrival_rate": base_arrival * 0.7, "high_load_mode": False},
        {"num_vehicles": 12, "task_arrival_rate": base_arrival, "high_load_mode": True},
        {"num_vehicles": 16, "task_arrival_rate": base_arrival * 1.2, "bandwidth": 18, "cache_capacity": 90},
        {"num_vehicles": 20, "task_arrival_rate": base_arrival * 1.4, "bandwidth": 22, "cache_capacity": 110},
    ]
    # 添加轻载但高能效阈值压力场景
    scenarios.append({"num_vehicles": 10, "task_arrival_rate": base_arrival * 0.9, "energy_upper_tolerance": 2500.0})
    return scenarios


def run_hierarchical_multitask_training(
    scenarios: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    episodes_per_task: int = 400,
    use_enhanced_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    对一组场景依次训练 TD3-Hier，返回每个场景的训练摘要。
    """
    scenarios = list(scenarios or get_default_multitask_scenarios())
    results: List[Dict[str, Any]] = []

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n=== 多任务训练场景 {idx}/{len(scenarios)} ===")
        print(f"配置: {scenario}")
        res = train_single_algorithm(
            "TD3-HIER",
            num_episodes=episodes_per_task,
            silent_mode=True,
            override_scenario=scenario,
            use_enhanced_cache=use_enhanced_cache,
            enable_realtime_vis=False,
        )
        results.append(
            {
                "scenario": scenario,
                "training_result": res,
            }
        )
    return results


@dataclass
class DistillationConfig:
    """策略蒸馏参数配置。"""

    rollout_steps: int = 600
    rollout_repeats: int = 6
    batch_size: int = 512
    epochs: int = 40
    kl_weight: float = 0.25
    learning_rate: float = 3e-4
    student_hidden_dim: int = 256
    student_high_hidden: int = 192
    student_low_hidden: int = 320


def _collect_teacher_dataset(
    teacher_checkpoint: str,
    scenario: Dict[str, Any],
    *,
    rollout_steps: int,
    repeats: int,
    use_enhanced_cache: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """在指定场景中运行教师策略收集状态-动作对。"""
    env = SingleAgentTrainingEnvironment(
        "TD3-HIER",
        override_scenario=scenario,
        use_enhanced_cache=use_enhanced_cache,
    )
    env.agent_env.load_models(teacher_checkpoint)

    states_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []

    for _ in range(repeats):
        rollout = env.collect_policy_rollout(rollout_steps, deterministic=True)
        if rollout["states"].size == 0:
            continue
        states_list.append(rollout["states"])
        actions_list.append(rollout["actions"])

    if not states_list:
        raise RuntimeError("无法从教师策略收集到任何有效数据")

    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    return states, actions


def distill_hierarchical_policy(
    teacher_checkpoint: str,
    output_path: str,
    scenarios: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    config: Optional[DistillationConfig] = None,
    use_enhanced_cache: bool = True,
) -> str:
    """
    将教师 TD3-Hier 策略蒸馏为轻量学生模型。
    返回学生模型保存路径（不含后缀）。
    """
    distill_cfg = config or DistillationConfig()
    scenarios = list(scenarios or get_default_multitask_scenarios())

    teacher_env = TD3HierarchicalEnvironment(
        num_vehicles=scenarios[0].get("num_vehicles", 12),
        num_rsus=4,
        num_uavs=2,
    )
    teacher_env.load_models(teacher_checkpoint)

    student_config = HierarchicalTD3Config(
        hidden_dim=distill_cfg.student_hidden_dim,
        high_level_hidden=distill_cfg.student_high_hidden,
        low_level_hidden=distill_cfg.student_low_hidden,
        actor_lr=distill_cfg.learning_rate,
    )
    student_agent = HierarchicalTD3Agent(
        teacher_env.state_dim,
        teacher_env.action_dim,
        student_config,
        teacher_env.num_rsus,
        teacher_env.num_uavs,
    )

    datasets: List[TensorDataset] = []
    for scenario in scenarios:
        states, actions = _collect_teacher_dataset(
            teacher_checkpoint,
            scenario,
            rollout_steps=distill_cfg.rollout_steps,
            repeats=distill_cfg.rollout_repeats,
            use_enhanced_cache=use_enhanced_cache,
        )
        tensor_dataset = TensorDataset(
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).float(),
        )
        datasets.append(tensor_dataset)

    merged_dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(
        merged_dataset,
        batch_size=distill_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    device = student_agent.device
    student_agent.actor.train()
    optimizer = torch.optim.Adam(student_agent.actor.parameters(), lr=distill_cfg.learning_rate)
    mse_loss = nn.MSELoss()

    high_dim = 3 + teacher_env.num_rsus + teacher_env.num_uavs
    eps = 1e-6

    for epoch in range(distill_cfg.epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            student_actions = student_agent.actor(batch_states)

            mse = mse_loss(student_actions, batch_actions)

            # KL on hierarchical distributions (task/rsu/uav)
            kl_total = torch.tensor(0.0, device=device)
            teacher_task = torch.clamp(batch_actions[:, :3], eps, 1.0)
            student_task = torch.clamp(student_actions[:, :3], eps, 1.0)
            kl_total += torch.sum(teacher_task * (torch.log(teacher_task) - torch.log(student_task)), dim=-1).mean()

            if teacher_env.num_rsus > 0:
                t_rsu = torch.clamp(batch_actions[:, 3:3 + teacher_env.num_rsus], eps, 1.0)
                s_rsu = torch.clamp(student_actions[:, 3:3 + teacher_env.num_rsus], eps, 1.0)
                kl_total += torch.sum(t_rsu * (torch.log(t_rsu) - torch.log(s_rsu)), dim=-1).mean()

            if teacher_env.num_uavs > 0:
                start = 3 + teacher_env.num_rsus
                end = start + teacher_env.num_uavs
                t_uav = torch.clamp(batch_actions[:, start:end], eps, 1.0)
                s_uav = torch.clamp(student_actions[:, start:end], eps, 1.0)
                kl_total += torch.sum(t_uav * (torch.log(t_uav) - torch.log(s_uav)), dim=-1).mean()

            loss = mse + distill_cfg.kl_weight * kl_total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_agent.actor.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"[Distill] Epoch {epoch+1}/{distill_cfg.epochs} | Loss: {avg_loss:.5f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    student_agent.save_model(output_path)
    print(f"✅ 蒸馏学生模型已保存到: {output_path}_td3.pth")
    return output_path


__all__ = [
    "get_default_multitask_scenarios",
    "run_hierarchical_multitask_training",
    "distill_hierarchical_policy",
    "DistillationConfig",
]
