#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置模块 - VEC边缘计算迁移与缓存系统

【功能】
提供系统所有配置参数，包括：
- 任务配置：任务生成、分类、优先级
- 网络配置：节点数量、拓扑、通信参数
- 计算配置：CPU频率、能耗模型、资源分配
- RL配置：奖励函数权重、训练超参数
- 实验配置：训练轮次、评估间隔

【论文对应】
- 任务模型：对应论文Section 2.1 "Task Model"
- 通信模型：对应论文Section 2.2 "Communication Model"（3GPP标准）
- 能耗模型：对应论文Section 2.3 "Energy Consumption Model"
- 奖励函数：对应论文Section 3.2 "Reward Function Design"

【设计原则】
1. 所有参数基于3GPP TR 38.901/38.306标准
2. 能耗模型基于实际硬件校准（Intel NUC i7、12GHz服务器）
3. Time slot design: 0.1s = unified sync granularity
4. 12车辆高负载场景：arrival_rate = 2.5 tasks/s

【使用示例】
```python
from config.system_config import config
print(f"车辆数量: {config.num_vehicles}")
print(f"时延权重: {config.rl.reward_weight_delay}")
```
"""

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional


@dataclass(frozen=True)
class TaskProfileSpec:
    """
    任务类型配置规格
    
    【功能】定义单个任务类型的完整参数规格
    【论文对应】Section 2.1 Task Classification
    
    【属性说明】
    - task_type: 任务类型编号 (1-4)
        1: 极度时延敏感 (紧急制动)
        2: 时延敏感 (导航)
        3: 中度时延容忍 (视频处理)
        4: 时延容忍 (数据分析)
    - data_range: 数据量范围 (bytes)
    - compute_density: 计算密度 (cycles/bit)
    - max_latency_slots: 最大可容忍时延时隙数
    - latency_weight: 时延成本权重 (论文Table IV)
    """
    task_type: int
    data_range: Tuple[float, float]
    compute_density: float
    max_latency_slots: int
    latency_weight: float


@dataclass(frozen=True)
class TaskScenarioSpec:
    """
    任务场景配置规格
    
    【功能】定义具体应用场景的任务特征
    【论文对应】Section 2.1 Application Scenarios
    
    【属性说明】
    - name: 场景名称 (如 'emergency_brake', 'navigation')
    - min_deadline: 最小截止时间 (seconds)
    - max_deadline: 最大截止时间 (seconds)
    - task_type: 对应的任务类型 (1-4)
    - relax_factor: 截止时间放松因子
    - weight: 场景出现概率权重
    
    【典型场景】
    - emergency_brake: 0.18-0.22s, 权重8%, 类型1
    - navigation: 0.38-0.42s, 权重25%, 类型2
    - video_process: 0.58-0.64s, 权重20%, 类型3
    """
    name: str
    min_deadline: float
    max_deadline: float
    task_type: int
    relax_factor: float
    weight: float

class ExperimentConfig:
    """
    实验配置类
    
    【功能】控制训练和评估的全局实验参数
    【论文对应】Section 4 "Performance Evaluation"
    
    【配置说明】
    - num_episodes: 训练总轮次（默认1000，快速测试可用200）
    - num_runs: 多次运行取平均（提供统计显著性）
    - save_interval: 模型保存间隔（每100轮保存一次）
    - eval_interval: 评估间隔（每50轮评估一次）
    - log_interval: 日志记录间隔
    - max_steps_per_episode: 每轮最大步数（对应仿真时长）
    - warmup_episodes: 预热轮次（收集初始经验）
    - use_timestamp: 是否使用时间戳区分实验
    - timestamp_format: 时间戳格式（年月日_时分秒）
    
    【学术实验建议】
    - 完整实验：num_episodes=1000, num_runs=5
    - 快速验证：num_episodes=200, num_runs=3
    - 消融实验：num_episodes=500, num_runs=3
    """
    
    def __init__(self):
        self.num_episodes = 1000
        self.num_runs = 3
        self.save_interval = 100
        self.eval_interval = 50
        self.log_interval = 20  # 20 * 0.1 s ~ 2 s, matches previous cadence
        self.max_steps_per_episode = 200  # 🚀 加速优化：缩短50%步数（200×0.1s=20s）
        self.warmup_episodes = 10
        self.use_timestamp = True
        self.timestamp_format = "%Y%m%d_%H%M%S"

class RLConfig:
    """
    强化学习配置类
    
    【功能】定义所有RL算法的超参数和奖励函数权重
    【论文对应】Section 3 "Deep Reinforcement Learning Framework"
    
    【核心超参数】
    - state_dim: 状态空间维度（观测维度）
    - action_dim: 动作空间维度
    - hidden_dim: 隐藏层维度（256适合中等复杂度任务）
    - lr/actor_lr/critic_lr: 学习率（3e-4是标准值）
    - gamma: 折扣因子（0.99适合长期优化）
    - tau: 软更新系数（0.005保证稳定性）
    - batch_size: 批次大小（128平衡速度与稳定性）
    - memory_size: 经验回放缓冲区大小
    
    【TD3专用参数】
    - noise_std: 噪声标准差（0.05降低探索强度）
    - policy_delay: 策略延迟更新（2是标准TD3设置）
    - noise_clip: 噪声裁剪范围（0.3防止过度探索）
    - exploration_noise: 探索噪声（0.05适度探索）
    - policy_noise: 策略噪声（0.1平滑目标策略）
    - target_noise: 目标噪声（0.1提高鲁棒性）
    
    【奖励函数权重 - 核心优化目标】
    ⚠️ 重要：这是系统的核心优化目标！
    
    核心目标函数（🔧 修复问题6和问题10：归一化基准与优化目标对齐）：
        norm_delay = delay / latency_target (0.4s)
        norm_energy = energy / energy_target (1200J)
        Objective = ω_T × norm_delay + ω_E × norm_energy
        Reward = -(ω_T × norm_delay + ω_E × norm_energy) - 0.02 × dropped_tasks
    
    🔧 修复问题6：归一化基准现在直接使用latency_target和energy_target
    （而非旧的硬编码值0.2s和1000J，确保权重含义一致）
    
    【权重设置分析】
    典型值：delay ≈ 0.3s, energy ≈ 1000J
    归一化后：norm_delay = 0.3/0.2 = 1.5, norm_energy = 1000/1000 = 1.0
    
    如果使用原始权重 (2.4, 1.0)：
        weighted_delay = 2.4 × 1.5 = 3.6
        weighted_energy = 1.0 × 1.0 = 1.0
        → 时延贡献是能耗的3.6倍，过度偏向时延优化
    
    优化后权重 (1.5, 1.0)：
        weighted_delay = 1.5 × 1.5 = 2.25
        weighted_energy = 1.0 × 1.0 = 1.0
        → 时延贡献是能耗的2.25倍，更平衡的优化
    
    权重设置：
    - reward_weight_delay = 1.5    # 时延权重（降低以平衡归一化效果）
    - reward_weight_energy = 1.0   # 能耗权重（保持基准）
    - reward_penalty_dropped = 0.02  # 丢弃任务轻微惩罚（保证完成率约束）
    
    ⚠️ 已废弃参数（保留兼容性）：
    - reward_weight_loss = 0.0        # 已移除：data_loss是时延的衍生指标
    - reward_weight_completion = 0.0  # 已集成到dropped_penalty
    - reward_weight_cache = 0.35      # 缓存是手段，不是优化目标
    - reward_weight_migration = 0.0   # 迁移是手段，不是优化目标
    
    【优化目标阈值】
    供算法动态调整的参考目标：
    - latency_target: 目标平均时延（0.40s）
    - latency_upper_tolerance: 时延上限容忍（0.80s）
    - energy_target: 目标能耗（1200.0J）
    - energy_upper_tolerance: 能耗上限容忍（1800.0J）
    
    【论文对应】
    - 奖励函数设计：Section 3.2 "Reward Function Design"
    - 权重选择：Section 4.2 "Parameter Settings"
    - TD3参数：Section 3.3 "TD3 Algorithm Implementation"
    """
    
    def __init__(self):
        self.num_agents = 3
        self.state_dim = 20
        self.action_dim = 10
        self.hidden_dim = 256
        self.lr = 0.0001  # 🔧 0.0003 → 0.0001 (降低学习率，提高稳定性)
        self.actor_lr = 0.0001  # 🔧 0.0003 → 0.0001
        self.critic_lr = 0.0001  # 🔧 0.0003 → 0.0001
        self.gamma = 0.995  # Adjusted for 0.1 s slots (~sqrt of 0.99)
        self.tau = 0.005
        self.batch_size = 256          # 提高批次大小（128→256）提高样本效率
        self.memory_size = 200000      # 扩大replay buffer（100k→200k）增强稳定性
        self.noise_std = 0.05          # 降低噪声标准差
        self.policy_delay = 2
        self.noise_clip = 0.3           # 降低噪声裁剪
        self.exploration_noise = 0.05   # 🔧 0.1 → 0.05 (大幅降低初始噪声)
        self.policy_noise = 0.05         # 🔧 0.1 → 0.05 (降低策略噪声)
        self.target_noise = 0.05         # 🔧 0.1 → 0.05 (降低目标噪声)
        self.update_freq = 1
        self.buffer_size = 200000      # 与memory_size保持一致
        self.warmup_steps = 1000
        
        # 🆕 学习率衰减策略（优化收敛性）
        self.lr_decay_rate = 0.995     # 学习率衰减率
        self.lr_decay_interval = 100   # 每100轮衰减一次
        self.min_lr = 0.00005          # 最小学习率
        
        # 🆕 噪声衰减策略（提高后期稳定性）
        self.noise_decay = 0.99       # 🔧 0.995 → 0.99 (加快衰减，更快减少震荡)
        self.min_noise = 0.005         # 🔧 0.01 → 0.005 (降低最小噪声)
        
        # 🎯 核心奖励权重：延迟+能耗+完成率
        # 🔧 2024-12-02 v6修复：500 episode后仍无学习，增强奖励塑形
        #    问题：奖励信号太弱(-0.89~-0.90)，信噪比低，无法学习
        #    策略：增强完成率惩罚，让22%任务丢弃产生更强的梯度信号
        self.reward_weight_delay = 0.3  # 🔧 1.0 → 0.3 (降低，让完成率主导)
        self.reward_weight_energy = 0.2  # 🔧 1.0 → 0.2 (降低，让完成率主导)
        
        # 🔥 关键惩罚：大幅增强任务丢弃惩罚
        # 🔧 1.0 → 0.5 (每丢弃1个任务扣0.5分，22%丢弃率≈44任务/episode≈-22分)
        self.reward_penalty_dropped = 0.5
        self.completion_target = 0.85  # 🔧 0.88 → 0.85 (更宽松目标)
        
        # 🔥 启用完成率差距惩罚：让智能体关注完成率
        # 77%完成率 vs 85%目标 = 8% gap → 惩罚 0.4 * 0.08 / 0.2 ≈ 0.16/step
        self.reward_weight_completion_gap = 2.0  # 🔧 0.0 → 2.0 (强惩罚)
        self.reward_weight_loss_ratio = 0.0  # 保持禁用
        self.reward_weight_cache_pressure = 0.0  # 保持禁用
        self.reward_weight_cache_bonus = 0.0  # 保持禁用
        self.reward_weight_queue_overload = 0.0  # 保持禁用

        # ⚠️ 已弃用参数（保留以兼容旧代码）
        self.reward_weight_loss = 0.0      # 已移除：data_loss是时延的衡生指标
        self.reward_weight_completion = 0.0  # 已集成到dropped_penalty
        # 🚫 禁用所有辅助优化项（专注核心目标）
        self.reward_weight_cache = 0.0  # 🔧 0.1 → 0.0 (禁用缓存惩罚)
        self.reward_weight_migration = 0.0  # 🔧 0.05 → 0.0 (禁用迁移惩罚)
        self.reward_weight_joint = 0.0  # 🔧 0.02 → 0.0 (禁用联动奖励)
        self.reward_weight_remote_reject = 0.0  # 🔧 0.08 → 0.0 (禁用远程拒绝惩罚)
        self.reward_weight_offload_bonus = 0.0  # 🔧 0.05 → 0.0 (禁用卸载奖励)
        self.reward_weight_local_penalty = 0.0  # 保持禁用

        # 🎯 延时-能耗优化目标阈值（供算法动态调整）
        # 🔧 基准目标值会在训练中根据实际系统表现自动调整（最多放宽3倍）
        # 初始目标：保守估计，让系统在前几个episode后自适应放宽
        # 🚀 关键修复:能耗目标值调整至实际水平,避免奖励过低
        # 问题:原目标energy_target=3000J,但实际能耗在6000-9000J,导致norm_energy>2.0,奖励永远为负
        # 解决:调整target=7000J(实际中值),tolerance=10000J,让奖励在合理范围
        # 🔧 2024-12-02: 归一化目标再次调整（产生更明显的学习信号）
        # 当前问题：目标函数稳定但奖励震荡 → 辅助项噪声过大
        # 新策略：提高归一化目标，让更多episode产生正成本，增强梯度
        #         同时禁用所有辅助项，让核心信号清晰可见
        self.latency_target = 1.5  # 🔧 1.6 → 1.5 (降低目标，让50%episode产生成本)
        self.latency_upper_tolerance = 2.2  # 🔧 2.5 → 2.2
        # 🔧 修复：能耗目标改为Step级别 (25J/step)，而非Episode级别 (900J/episode)
        # 12车辆*5W + RSU*25W + UAV*25W ≈ 110W静态功率 -> 11J/step
        # 加上动态功耗，预计20-30J/step
        self.energy_target = 1000.0  # 🔧 200.0 → 1000.0 (Aligned with dynamic energy ~900J)
        self.energy_upper_tolerance = 2000.0  # 🔧 500.0 → 2000.0

        # 🆕 动态归一化开关
        self.use_dynamic_reward_normalization = False  # 禁用以改善收敛性（原为True）

class QueueConfig:
    """
    队列配置类
    
    【功能】定义任务队列管理参数
    【论文对应】Section 2.4 "Queue Management"
    
    【配置说明】
    - max_lifetime: 任务最大生命周期（时隙数，与0.1s时隙同步）
    - max_queue_size: 队列最大容量（任务数）
    - priority_levels: 优先级级别数（4级对应4种任务类型）
    - aging_factor: 老化因子（0.25表示每步强衰减，适合短时隙）
    
    【设计说明】
    时隙同步设计：max_lifetime = 6 × 0.1s = 0.6s最大等待时间（收紧约束）
    强衰减策略：aging_factor = 0.25确保老任务优先处理
    
    【🔧 优化修复】
    - 提高队列容量：支持高负载场景（12车辆 × 3.0 tasks/s）
    - 放宽超载阈值：允许适度排队缓冲
    """
    
    def __init__(self):
        # 🔧 调整：允许更长的排队寿命（1.0s）以缓冲高负载
        self.max_lifetime = 10
        self.max_queue_size = 100
        self.priority_levels = 4
        # Aging factor tuned for short slots (strong decay each step)
        self.aging_factor = 0.25
        # 🔧 修复：提高超载阈值，允许合理排队（1.1 → 1.5）
        self.max_load_factor = 1.5
        self.global_rho_threshold = float(os.environ.get('QUEUE_GLOBAL_RHO_THRESHOLD', '0.6'))
        self.stability_warning_ratio = float(os.environ.get('QUEUE_STABILITY_WARNING_RATIO', '0.9'))
        # 🔧 修复：大幅提高队列容量，匹配高负载场景需求
        # RSU: 20 → 50 (每个RSU需处理约180任务/episode，允许缓冲)
        # UAV: 10 → 30 (UAV也需要充足队列空间)
        # Vehicle: 20 (本地队列，实际很少使用，主要用于Phase 2调度)
        self.rsu_nominal_capacity = float(os.environ.get('QUEUE_RSU_NOMINAL_CAPACITY', '50.0'))
        self.uav_nominal_capacity = float(os.environ.get('QUEUE_UAV_NOMINAL_CAPACITY', '30.0'))
        self.vehicle_nominal_capacity = float(os.environ.get('QUEUE_VEHICLE_NOMINAL_CAPACITY', '20.0'))
        # 🔧 修复：扩大队列字节容量，匹配任务数据大小增加（3-4倍）
        # 平均任务大小：1.5 MB (原 500 KB)
        # Capacity limits (bytes) used for queue admission control
        self.vehicle_queue_capacity = float(os.environ.get('QUEUE_VEHICLE_CAPACITY', '8e8'))     # 250MB → 800MB
        self.rsu_queue_capacity = float(os.environ.get('QUEUE_RSU_CAPACITY', '5e9'))            # 1.5GB → 5GB
        self.uav_queue_capacity = float(os.environ.get('QUEUE_UAV_CAPACITY', '2e9'))            # 600MB → 2GB

class TaskConfig:
    """
    任务配置类
    
    【功能】定义任务生成和分类参数
    【论文对应】Section 2.1 "Task Model"
    
    【核心参数】
    - num_priority_levels: 优先级级别数（4级）
    - task_compute_density: 默认计算密度（120 cycles/bit）
    - arrival_rate: 任务到达率（2.5 tasks/s，12车辆高负载场景）
    
    【任务参数设计】
    - data_size_range: 数据量范围 1-50 Mbits = 0.125-6.25 MB
    - compute_cycles_range: 计算周期范围 1e8-7.5e9 cycles
    - deadline_range: 截止时间范围 0.15-0.95s（对应2-10个时隙@100ms）
    - task_output_ratio: 输出大小为输入的5%
    
    【任务类型阈值】（基于100ms时隙 - 收紧约束以充分利用精细时隙）
    - delay_thresholds:
        * extremely_sensitive: 2个时隙 = 0.2s (不变，已经很紧)
        * sensitive: 3个时隙 = 0.3s (收紧)
        * moderately_tolerant: 4个时隙 = 0.4s (收紧)
        * tolerant: 6个时隙 = 0.6s (收紧)
    
    【时延成本权重】（对应论文Table IV）
    - latency_cost_weights: {1: 1.0, 2: 0.4, 3: 0.4, 4: 0.4}
    
    【论文对应】
    - 任务分类：Section 2.1 "Task Classification"
    - 时延权重：Table IV "Latency Cost Weights"
    """
    
    def __init__(self):
        self.num_priority_levels = 4
        
        # 🎯 极限负载场景：强制降低完成率到75-85%
        # 🔧 2024-12-02 v4修复：v3仍然完成率99.97%，没有学习空间
        # 问题诊断：2000 episode后奖励仅改呙1%，任务太简单
        self.task_compute_density = 100  # 🔧 v5: 200 → 100
        self.arrival_rate = 3.5   # 🔧 v5: 10 → 3.5
        
        # 🎯 数据范围：300KB-5MB
        self.data_size_range = (300e3, 5e6)
        self.task_data_size_range = self.data_size_range

        # 计算周期配置
        self.compute_cycles_range = (300e3 * 8 * 100, 5e6 * 8 * 100)
        
        # 🔧 收紧截止时间
        self.deadline_range = (0.15, 0.50)
        # 输出比例配置
        self.task_output_ratio = 0.05  # 输出大小是输入大小的5%
        
        # 任务类型阈值
        self.delay_thresholds = {
            'extremely_sensitive': 2,
            'sensitive': 3,
            'moderately_tolerant': 4,
        }

        # Latency cost weights (aligned with task_profiles and Table IV)
        # 🔧 修正：与task_profiles中的latency_weight保持一致
        self.latency_cost_weights = {
            1: 1.0,  # 极度敏感，最高权重
            2: 0.7,  # 敏感，高权重
            3: 0.5,  # 中度容忍，中等权重
            4: 0.4,  # 容忍，低权重
        }

        # Deadline 放松参数
        self.deadline_relax_default = 1.0
        # 🔧 修复：騍松因子需要阻止任务过早报告为不需要的类型（例如：1.3⁮放斐会将因楳上升）。计算时应造阈愿。
        # 低四亚蹡：每个类枠先恰会正。描例：简回因子=1.3是削溥计帄，将保骇时閒=0.3的任务上升。
        self.deadline_relax_fallback = 1.0  # 騍松因子改为1.0（无騍松），确保任务类型冠正

        # 任务类型配置
        self.task_profiles: Dict[int, TaskProfileSpec] = {
            1: TaskProfileSpec(1, (100e3, 500e3), 80, 2, 1.0),
            2: TaskProfileSpec(2, (500e3, 2e6), 100, 3, 0.7),
            3: TaskProfileSpec(3, (2e6, 4e6), 120, 4, 0.5),
            4: TaskProfileSpec(4, (4e6, 5e6), 150, 5, 0.4),
        }
        # Backwards-compatible dictionary view for legacy code
        self.task_type_specs = {
            k: {
                'data_range': v.data_range,
                'compute_density': v.compute_density,
                'max_latency_slots': v.max_latency_slots,
                'latency_weight': v.latency_weight,
            }
            for k, v in self.task_profiles.items()
        }

        # 场景定义
        self.scenarios: List[TaskScenarioSpec] = [
            TaskScenarioSpec('emergency_brake', 0.15, 0.25, 1, 1.0, 0.25),
            TaskScenarioSpec('collision_avoid', 0.18, 0.28, 1, 1.0, 0.20),
            TaskScenarioSpec('navigation', 0.25, 0.35, 2, 1.0, 0.15),
            TaskScenarioSpec('traffic_signal', 0.28, 0.38, 2, 1.0, 0.10),
            TaskScenarioSpec('video_process', 0.35, 0.45, 3, 1.0, 0.15),
            TaskScenarioSpec('image_recognition', 0.38, 0.48, 3, 1.0, 0.10),
            TaskScenarioSpec('data_analysis', 0.42, 0.50, 4, 1.0, 0.04),
            TaskScenarioSpec('ml_training', 0.45, 0.52, 4, 1.0, 0.01),
        ]
        self._scenario_weights = [scenario.weight for scenario in self.scenarios]
        self._scenario_lookup = {scenario.name: scenario for scenario in self.scenarios}
        self.type_priority_weights = self._compute_type_priority_weights()
    
    def get_task_type(
        self,
        max_delay_slots: int,
        data_size: Optional[float] = None,
        compute_cycles: Optional[float] = None,
        compute_density: Optional[float] = None,
        time_slot: Optional[float] = None,
        system_load: Optional[float] = None,
        is_cacheable: bool = False,
    ) -> int:
        """
        基于多维特征的任务分类，兼顾时延、数据规模和计算强度。
        """
        thresholds = self._get_dynamic_delay_thresholds(system_load)
        base_type = self._determine_base_type(max_delay_slots, thresholds)
        slot_duration = self._resolve_time_slot(time_slot)

        density = compute_density
        if density is None and compute_cycles is not None and data_size:
            bits = max(float(data_size) * 8.0, 1.0)
            density = compute_cycles / bits

        task_type = self._adjust_by_processing_capacity(
            base_type, compute_cycles, max_delay_slots, slot_duration
        )
        task_type = self._iterative_adjustment(
            task_type,
            lambda t: self._adjust_by_data_size(t, data_size, is_cacheable),
        )
        task_type = self._iterative_adjustment(
            task_type,
            lambda t: self._adjust_by_compute_density(t, density, is_cacheable),
        )

        return max(1, min(4, int(task_type)))

    def _get_dynamic_delay_thresholds(self, system_load: Optional[float]) -> Dict[str, int]:
        thresholds = dict(self.delay_thresholds)
        if system_load is None:
            return thresholds
        try:
            load = float(system_load)
        except (TypeError, ValueError):
            return thresholds
        load = max(0.0, min(load, 1.5))
        scale = 1.0 + 0.35 * (load - 0.5)
        scale = max(0.6, min(scale, 1.4))
        for key, value in thresholds.items():
            thresholds[key] = max(1, int(round(value * scale)))
        return thresholds

    def _determine_base_type(self, max_delay_slots: int, thresholds: Dict[str, int]) -> int:
        if max_delay_slots <= thresholds['extremely_sensitive']:
            return 1
        if max_delay_slots <= thresholds['sensitive']:
            return 2
        if max_delay_slots <= thresholds['moderately_tolerant']:
            return 3
        return 4

    def _resolve_time_slot(self, override: Optional[float]) -> float:
        if override and override > 0:
            return float(override)
        global_cfg = globals().get('config')
        try:
            return float(getattr(getattr(global_cfg, 'network', None), 'time_slot_duration', 0.1))
        except Exception:
            return 0.1

    def _iterative_adjustment(self, task_type: int, adjust_fn) -> int:
        adjusted = task_type
        for _ in range(3):
            new_value = adjust_fn(adjusted)
            if new_value == adjusted:
                break
            adjusted = new_value
        return adjusted

    def _adjust_by_data_size(self, task_type: int, data_size: Optional[float], is_cacheable: bool) -> int:
        if data_size is None or data_size <= 0:
            return task_type
        profile = self.task_profiles.get(task_type)
        if not profile:
            return task_type
        data_min, data_max = profile.data_range
        upper_margin = 1.25 if not is_cacheable else 1.45
        lower_margin = 0.55
        if data_size > data_max * upper_margin and task_type < 4:
            return task_type + 1
        if data_size < data_min * lower_margin and task_type > 1:
            return task_type - 1
        return task_type

    def _adjust_by_compute_density(self, task_type: int, compute_density: Optional[float], is_cacheable: bool) -> int:
        if compute_density is None or compute_density <= 0:
            return task_type
        profile = self.task_profiles.get(task_type)
        if not profile or profile.compute_density <= 0:
            return task_type
        baseline = profile.compute_density
        upper_margin = 1.35 if not is_cacheable else 1.55
        lower_margin = 0.65
        if compute_density > baseline * upper_margin and task_type < 4:
            return task_type + 1
        if compute_density < baseline * lower_margin and task_type > 1:
            return task_type - 1
        return task_type

    def _adjust_by_processing_capacity(
        self,
        task_type: int,
        compute_cycles: Optional[float],
        max_delay_slots: int,
        slot_duration: float,
    ) -> int:
        if compute_cycles is None or compute_cycles <= 0 or max_delay_slots <= 0:
            return task_type
        budgets = self._estimate_processing_budgets(max_delay_slots, slot_duration)
        adjusted = task_type
        while adjusted < 4 and compute_cycles > budgets.get(adjusted, float('inf')):
            adjusted += 1
        return adjusted

    def _estimate_processing_budgets(self, max_delay_slots: int, slot_duration: float) -> Dict[int, float]:
        compute_cfg = getattr(globals().get('config'), 'compute', None)
        efficiency = 0.8
        vehicle_peak = 2.5e9
        rsu_peak = 12e9
        if compute_cfg:
            efficiency = float(getattr(compute_cfg, 'parallel_efficiency', efficiency))
            vehicle_range = getattr(compute_cfg, 'vehicle_cpu_freq_range', (vehicle_peak, vehicle_peak))
            rsu_range = getattr(compute_cfg, 'rsu_cpu_freq_range', (rsu_peak, rsu_peak))
            vehicle_peak = float(vehicle_range[1])
            rsu_peak = float(rsu_range[1])
        slot = max(slot_duration, 1e-6)
        budgets = {
            1: vehicle_peak * efficiency * slot * max(1, max_delay_slots),
            2: rsu_peak * efficiency * slot * max(1, max_delay_slots),
            3: rsu_peak * efficiency * slot * max(2, int(max_delay_slots * 1.2)),
            4: float('inf'),
        }
        return budgets


    def sample_scenario(self) -> TaskScenarioSpec:
        """
        根据配置权重随机选择任务场景
        
        【功能】使用weighted random sampling选择场景
        【返回值】TaskScenarioSpec 对象
        """
        return random.choices(self.scenarios, weights=self._scenario_weights, k=1)[0]

    def get_profile(self, task_type: int) -> TaskProfileSpec:
        """
        获取指定任务类型的配置规格
        
        【功能】返回任务类型的完整参数规格
        【参数】task_type: 任务类型编号(1-4)
        【返回值】TaskProfileSpec 对象（包含数据范围、计算密度等）
        """
        if task_type in self.task_profiles:
            return self.task_profiles[task_type]

        default_slots = int(self.delay_thresholds.get('moderately_tolerant', 3))
        latency_weight = float(self.latency_cost_weights.get(task_type, 1.0))
        return TaskProfileSpec(
            task_type,
            self.data_size_range,
            self.task_compute_density,
            default_slots,
            latency_weight,
        )

    def get_relax_factor(self, task_type: int) -> float:
        """
        获取任务类型的截止时间放松因子
        
        【功能】返回deadline relaxation factor
        【参数】task_type: 任务类型编号
        【返回值】放松因子（通常为1.0）
        """
        for scenario in self.scenarios:
            if scenario.task_type == task_type:
                return scenario.relax_factor
        return self.deadline_relax_default

    def _compute_type_priority_weights(self) -> Dict[int, float]:
        """
        计算任务类型的优先级权重
        
        【功能】聚合场景权重，导出每个任务类型的优先级权重
        【返回值】字典 {task_type: priority_weight}
        【算法】加权聚合 + 归一化
        """
        totals = defaultdict(float)
        for scenario in self.scenarios:
            profile = self.task_profiles.get(scenario.task_type)
            latency_weight = profile.latency_weight if profile else 1.0
            totals[scenario.task_type] += scenario.weight * latency_weight

        for task_type, profile in self.task_profiles.items():
            totals[task_type] = max(totals[task_type], profile.latency_weight)

        # 确保每个任务类型至少具备基线权重
        for task_type in self.task_profiles.keys():
            totals.setdefault(task_type, 1.0)

        values = list(totals.values())
        mean_val = sum(values) / len(values) if values else 1.0
        if mean_val <= 0:
            mean_val = 1.0

        priority_weights = {
            task_type: float(max(0.1, totals[task_type] / mean_val))
            for task_type in self.task_profiles.keys()
        }
        return priority_weights

    def get_latency_cost_weight(self, task_type: int) -> float:
        """获取时延成本权重（论文Table IV）"""
        return float(self.latency_cost_weights.get(task_type, 1.0))

    def get_priority_weight(self, task_type: int) -> float:
        """获取缓存的优先级权重"""
        return float(self.type_priority_weights.get(task_type, 1.0))


class ServiceConfig:
    """
    服务能力配置类
    
    【功能】定义RSU和UAV的服务能力参数
    【论文对应】Section 2.5 "Service Capacity Model"
    
    【RSU服务能力】
    - rsu_base_service: 基础服务能力（4个任务/时隙）
    - rsu_max_service: 最大服务能力（9个任务/时隙）
    - rsu_work_capacity: 工作容量（2.5个单位/时隙）
    - rsu_queue_boost_divisor: 队列加速因子（5.0）
    
    【UAV服务能力】
    - uav_base_service: 基础服务能力（3个任务/时隙）
    - uav_max_service: 最大服务能力（6个任务/时隙）
    - uav_work_capacity: 工作容量（1.7个单位/时隙）
    - uav_queue_boost_divisor: 队列加速因子（4.0）
    
    【设计说明】
    RSU服务能力 > UAV服务能力（符合实际硬件差异）
    动态服务能力 = base + (queue_length / boost_divisor)
    
    【🔧 优化修复】
    - 大幅提高处理速度：匹配高负载场景（12车辆 × 3.0 tasks/s = 36 tasks/s）
    - 优化动态提升：更快响应队列堆积
    """

    def __init__(self):
        # 🔧 修复：大幅提高RSU服务能力（匹配高负载需求）
        # base: 5 → 10 (基础处理速度翻倍)
        # max: 12 → 25 (峰值处理能力提升)
        # work_capacity: 3.5 → 6.0 (工作容量提升70%)
        self.rsu_base_service = 10
        self.rsu_max_service = 25
        self.rsu_work_capacity = 6.0  # 相当于每个时隙的工作单位
        self.rsu_queue_boost_divisor = 4.0  # 5.0 → 4.0 (更快响应队列堆积)

        # 🔧 UAV优化修正:提升UAV算力以改善系统负载均衡
        # 目标：UAV达到RSU的60-70%性能水平，避免资源浪费
        # 硬件基准：NVIDIA Jetson Xavier NX支持动态调频（Boost模式）
        self.uav_base_service = 8              # 基于5GHz的合理服务能力（+33%）
        self.uav_max_service = 16              # 峰值处理能力（+33%）
        self.uav_work_capacity = 4.5           # 工作容量（+50%）
        self.uav_queue_boost_divisor = 2.0     # 更快响应队列堆积（优化）


class StatsConfig:
    """
    统计监控配置类
    
    【功能】定义日志和统计报告参数
    
    【配置说明】
    - drop_log_interval: log drop details every 400 steps (~40 s)
    - task_report_interval: emit task report every 100 steps (~10 s)
    """

    def __init__(self):
        self.drop_log_interval = 400
        # Shorter lifetimes require finer observation granularity
        self.task_report_interval = 100

class ComputeConfig:
    """
    计算资源配置类
    
    【功能】定义CPU频率、能耗模型参数
    【论文对应】Section 2.3 "Energy Consumption Model"
    
    【能耗模型公式】（论文Equation 3-5）- 🔧 修复问题1：统一使用 f³ 模型
    车辆能耗：E_v = κ₁ · f³ + P_static · t
    RSU能耗：E_r = κ₂ · f³ + P_static · t
    UAV能耗：E_u = κ₃ · f³ + P_static · t + P_hover · t
    
    【车辆参数】（基于Intel NUC i7实际校准）
    - vehicle_kappa1 = 5.12e-31  # W/(Hz)³ - CMOS动态功耗系数（基于实际硬件校准）
    - vehicle_static_power = 8.0W  # W - 实际车载芯片静态功耗
    - vehicle_idle_power = 3.5W    # W - 空闲功耗
    - vehicle_cpu_freq_range = 8-25 GHz
    - vehicle_default_freq = 2.5 GHz
    
    【RSU参数】（基于20GHz边缘服务器校准）
    - rsu_kappa = 2.8e-31  # W/(Hz)³ - 高性能CPU功耗系数
    - rsu_static_power = 25.0W  # W - 边缘服务器静态功耗
    - rsu_cpu_freq_range = 18-22 GHz
    - rsu_default_freq = 20 GHz  # 高性能边缘计算（Intel Xeon Platinum等）
    
    【UAV参数】（基于实际UAV硬件校准）
    - uav_kappa3 = 8.89e-31  # W/(Hz)³ - 功耗受限的UAV芯片
    - uav_static_power = 2.5W  # W - 轻量化设计
    - uav_hover_power = 25.0W  # W - 悬停功耗（更合理）
    - uav_cpu_freq_range = 1.5-9 GHz
    - uav_default_freq = 2.2 GHz  # 🔑 现代无人机边缘计算芯片（平衡性能与功耗）
    
    【内存配置】
    - vehicle_memory_size = 8 GB
    - rsu_memory_size = 32 GB
    - uav_memory_size = 4 GB
    
    【论文对应】
    - 能耗模型：Section 2.3, Equations (3)-(5)
    - 3GPP参数：基于3GPP TR 38.901标准
    - 硬件校准：附录A "Hardware Calibration"
    """
    
    def __init__(self):
        # 并行处理效率
        self.parallel_efficiency = 0.8
        
        # 🔧 优化：内存访问能耗参数配置化
        self.memory_access_ratio = 0.35  # 内存访问时间占比（35%）
        self.vehicle_dram_power = 3.5    # W - 车载DRAM功耗
        self.rsu_dram_power = 8.0        # W - RSU DRAM功耗（更大容量）
        self.uav_dram_power = 2.0        # W - UAV DRAM功耗（低功耗设计）
        # 🔑 修复：车辆能耗参数 - 基于实际硬件校准
        # 🔧 问题2修复：重新校准kappa1以匹配更高的频率范围
        # 目标：1.5GHz约8W，3.0GHz约17W（包含静态功耗）
        self.vehicle_kappa1 = 1.5e-28  # W/(Hz)³ - 重新校准（1.5GHz动态功耗3W，3.0GHz动态功耗12W）
        self.vehicle_kappa2 = 2.40e-20  # 频率平方项系数（兼容性保留）
        # 🔧 问题1/7修复：静态功耗降低至合理范围，与现代车载芯片匹配
        self.vehicle_static_power = 5.0  # W（现代车载芯片基础功耗）
        self.vehicle_idle_power = 2.0   # W（待机模式功耗，静态功耗的40%）
        
        # 🔑 修复：RSU能耗参数 - 基于20GHz边缘服务器校准
        # 🎯 优化：降低kappa系数，避免高频率下能耗过高
        # 原值2.8e-31导致17.5GHz时动态功率达1500W（不合理）
        # 新值5.0e-32确保17.5GHz时动态功率约270W（合理范围）
        self.rsu_kappa = 5.0e-32  # W/(Hz)³ - 校准后的高性能CPU功耗系数
        self.rsu_kappa2 = 5.0e-32
        self.rsu_static_power = 25.0  # W (20GHz边缘服务器静态功耗)
        
        # 🔑 修复:UAV能耗参数 - 基于NVIDIA Jetson Xavier NX实际硬件
        self.uav_kappa = 8.89e-31  # 功耗受限的UAV芯片
        self.uav_kappa3 = 8.89e-31  # 修复后参数
        self.uav_static_power = 2.5  # W (轻量化芯片基础功耗)
        # 🔧 UAV优化修正:悬停功耗优化(轻量级四旋翼)
        self.uav_hover_power = 15.0  # W (轻量级四旋翼悬停功率,持续存在)
        
        # 🔧 问题2修复:CPU频率配置更新为论文要求范围
        # 参考:论文中车辆计算资源 fv ∈ [1, 2] GHz
        self.total_vehicle_compute = 18e9     # 总本地计算:18 GHz(12车辆共享,每车1.5GHz平均)
        self.total_rsu_compute = 50e9        # 总RSU计算:50 GHz(4个RSU共享,每个12.5GHz)
        # 🔧 UAV优化2025-01-13:提升UAV总算力以匹配服务能力优化
        # NVIDIA Jetson Xavier NX Boost模式：6核@2.2GHz，等效约实际可用算力5.0GHz/核
        self.total_uav_compute = 10e9        # 总UAV计算:10 GHz(2个UAV共享,每个5.0GHz)
        
        # 🔑 初始CPU频率配置（仅用于节点初始化，运行时由中央智能体动态调整）
        # 两种模式：
        # 1. 标准模式：每个节点独立固定频率（旧设计，保留兼容性）
        # 2. 中央资源池模式：初始均匀分配，运行时由智能体动态优化（新设计）
        
        # 初始分配策略(均匀分配作为baseline)
        self.vehicle_initial_freq = self.total_vehicle_compute / 12   # 1.5 GHz - 初始均分
        self.rsu_initial_freq = self.total_rsu_compute / 4            # 12.5 GHz - 初始均分
        # 🔧 UAV优化2025-01-13:提升初始频率至5.0 GHz
        self.uav_initial_freq = self.total_uav_compute / 2            # 5.0 GHz - 初始均分
        
        # 🔧 问题2修复：CPU频率范围更新为论文要求
        # 车辆支持动态调频（DVFS），范围 fv ∈ [1, 2] GHz
        self.vehicle_cpu_freq_range = (1.0e9, 2.0e9)  # 1.0-2.0 GHz（论文要求）
        self.rsu_cpu_freq_range = (self.rsu_initial_freq, self.rsu_initial_freq)
        # 🔧 UAV优化：启用动态调频（DVFS）以优化能耗
        # Jetson Xavier NX支持3.0-7.0 GHz范围调频（基于多核Boost算力）
        self.uav_cpu_freq_range = (3.0e9, 7.0e9)  # 3.0-7.0 GHz（支持DVFS）
        
        # 默认频率（用于初始化，保留兼容性）
        self.vehicle_default_freq = self.vehicle_initial_freq
        self.rsu_default_freq = self.rsu_initial_freq
        self.uav_default_freq = self.uav_initial_freq
        
        # 节点CPU频率（用于初始化）
        self.vehicle_cpu_freq = self.vehicle_default_freq
        self.rsu_cpu_freq = self.rsu_default_freq
        self.uav_cpu_freq = self.uav_default_freq
        
        # 内存配置
        self.vehicle_memory_size = 8e9  # 8 GB
        self.rsu_memory_size = 32e9  # 32 GB
        self.uav_memory_size = 4e9  # 4 GB
        
        # 🔧 问题6修复：移除冗余的uav_hover_power配置（已在上方定义）

class NetworkConfig:
    """
    网络配置类
    
    【功能】定义网络拓扑和基础参数
    【论文对应】Section 2 "System Model"
    
    【时隙配置】
    - time_slot_duration = 0.1s  # 🔧 改为100ms，更精细的控制粒度
    
    【带宽配置】（3GPP NR高性能配置）
    - bandwidth = 80 MHz  # 🔧 提升至80MHz（城市热点/边缘计算场景）
    - carrier_frequency = 2.4 GHz
    - noise_power = -174 dBm/Hz
    
    【拓扑配置】（12车辆高负载场景）
    - num_vehicles = 12  # 恢复到原始设置
    - num_rsus = 4       # 更新为4个RSU（单向双路口场景）
    - num_uavs = 2       # 恢复到原始设置，符合论文要求
    
    【区域配置】
    - area_width = 2500m  # 缩小仿真区域
    - area_height = 2500m
    - min_distance = 50m  # 节点最小间距
    
    【路径损耗】
    - path_loss_exponent = 2.0
    - coverage_radius = 1000m
    
    【论文对应】
    - 网络拓扑：Section 2, Figure 1
    - 3GPP参数：基于3GPP TR 38.901
    """
    
    def __init__(self):
        self.time_slot_duration = 0.1  # seconds - 🔧 改为100ms，更精细的控制粒度
        self.bandwidth = 100e6  # Hz - 🎯 总带宽100MHz（5G NR高带宽，匹配卸载需求）
        # 🔧 修复：载波频率应与CommunicationConfig保持一致（3.5 GHz）
        self.carrier_frequency = 3.5e9  # Hz - 3GPP NR n78频段
        self.noise_power = -174  # dBm/Hz
        self.path_loss_exponent = 2.0
        self.coverage_radius = 300  # meters - RSU覆盖半径
        # 🔧 UAV优化：增加UAV覆盖半径配置
        self.uav_coverage_radius = 500  # meters - UAV覆盖半径（高空优势）
        self.uav_altitude = 120.0       # meters - UAV飞行高度
        self.interference_threshold = 0.1
        self.handover_threshold = 0.2
        
        # 节点数量配置
        self.num_vehicles = 12  # 恢复到原始设置
        self.num_rsus = 4       # 更新为4个RSU（单向双路口场景）
        self.num_uavs = 2       # 恢复到原始设置，符合论文要求
        
        # 网络拓扑参数
        self.area_width = 2500  # meters - 缩小仿真区域
        self.area_height = 2500  # meters
        self.min_distance = 50  # meters
        
        # 连接参数
        self.max_connections_per_node = 10
        self.connection_timeout = 30  # seconds

class CommunicationConfig:
    """
    3GPP通信配置类
    
    【功能】定义符合3GPP标准的通信参数
    【论文对应】Section 2.2 "Communication Model"
    【标准】3GPP TR 38.901/38.306
    
    【发射功率】（3GPP标准）
    - vehicle_tx_power = 23.0 dBm (200mW)  # 3GPP TS 38.101
    - rsu_tx_power = 46.0 dBm (40W)        # 3GPP TS 38.104
    - uav_tx_power = 30.0 dBm (1W)         # 3GPP TR 36.777
    
    【带宽配置】（3GPP NR标准 - 高性能配置）
    - total_bandwidth = 80 MHz      # 3GPP TS 38.104（城市热点/边缘计算）
    - channel_bandwidth = 4 MHz     # 每信道带宽（提升4倍）
    - uplink_bandwidth = 40 MHz     # 上行带宽（边缘计算上行密集）
    - downlink_bandwidth = 40 MHz   # 下行带宽
    
    【传播参数】（3GPP TR 38.901）
    - carrier_frequency = 2.0 GHz   # FR1频段
    - thermal_noise_density = -174.0 dBm/Hz
    - los_threshold = 50.0m         # LoS/NLoS门限
    - shadowing_std_los = 4.0 dB    # LoS阴影衰落
    - shadowing_std_nlos = 8.0 dB   # NLoS阴影衰落
    
    【天线增益】（3GPP标准）
    - antenna_gain_rsu = 15.0 dBi
    - antenna_gain_uav = 5.0 dBi
    - antenna_gain_vehicle = 3.0 dBi
    
    【调制参数】
    - modulation_order = 4  # QPSK
    - coding_rate = 0.5
    - noise_figure = 9.0 dB
    
    【论文对应】
    - 通信模型：Section 2.2, Equations (6)-(8)
    - 3GPP参数：Table II "3GPP Communication Parameters"
    """
    
    def __init__(self):
        # 3GPP标准发射功率
        self.vehicle_tx_power = 23.0  # dBm (200mW) - 3GPP标准
        self.rsu_tx_power = 46.0      # dBm (40W) - 3GPP标准
        self.uav_tx_power = 30.0      # dBm (1W) - 3GPP标准
        
        # 🔧 问题7修复：电路功率按节点类型差异化
        self.vehicle_circuit_power = 0.35  # W - 车辆RF前端（单天线）
        self.rsu_circuit_power = 0.85      # W - 基站多天线系统
        self.uav_circuit_power = 0.25      # W - UAV轻量化设计
        self.circuit_power = 0.35          # W - 默认值（保持向后兼容）
        
        # 🔧 优化：接收功率配置（基于3GPP TS 38.306标准）
        # 降低车辆接收功率，使其与发射功率比例更合理
        self.vehicle_rx_power = 1.8  # W - 车辆接收功率（从TX 200mW的90%）
        self.rsu_rx_power = 4.5      # W - RSU接收功率（从TX 40W的11.25%）
        self.uav_rx_power = 2.2      # W - UAV接收功率（从TX 1W的220%）
        
        # 🏢 建筑物遮挡模型配置（3GPP UMi场景）
        self.enable_blockage = True          # 启用建筑物遮挡模型
        self.building_density = 0.3          # 建筑密度：0=郊区，0.5=城市，1.0=密集城区
        self.avg_building_height = 15.0      # 平均建筑高度(m)，影响地面链路遮挡
        self.blockage_attenuation = 20.0     # NLoS额外衰减(dB)，典型范围15-25dB
        
        self.noise_figure = 9.0       # dB - 3GPP标准
        
        # 🎯 总带宽池配置（中央智能体动态分配）
        self.total_bandwidth = 100e6   # 100 MHz - 5G NR高带宽（匹配卸载通信需求）
        self.channel_bandwidth = 5e6  # 5 MHz per channel
        self.uplink_bandwidth = 50e6  # 50 MHz（边缘计算上行密集，确保卸载通畅）
        self.downlink_bandwidth = 50e6  # 50 MHz
        
        # 🔧 论文对齐：RSU/UAV下行带宽配置
        # MEC服务器（RSU）下行带宽: B_ES^down = 1000 MHz
        # 🔧 UAV优化：提升UAV下行带宽以降低返回延迟
        # UAV下行带宽: B_u^down = 50 MHz（从10MHz提升5倍）
        self.rsu_downlink_bandwidth = 1000e6  # 1000 MHz (1 GHz) - 论文要求
        self.uav_downlink_bandwidth = 50e6    # 50 MHz - 优化后（原10MHz）
        
        # 🔧 修复问题1：载波频率修正为3.5 GHz（符合论文要求和3GPP NR n78频段）
        self.carrier_frequency = 3.5e9  # 3.5 GHz - 3GPP NR n78频段（论文要求3.3-3.8 GHz，典型3.5 GHz）
        self.speed_of_light = 3e8       # m/s
        self.thermal_noise_density = -174.0  # dBm/Hz - 3GPP标准
        
        # 3GPP标准天线增益
        self.antenna_gain_rsu = 15.0     # dBi
        self.antenna_gain_uav = 5.0      # dBi
        self.antenna_gain_vehicle = 3.0  # dBi
        
        # 🔧 修复问题2/3/9：完善3GPP标准路径损耗参数（从硬编码移到配置）
        self.los_threshold = 50.0        # m - 3GPP TS 38.901视距临界距离
        self.los_decay_factor = 100.0    # m - LoS概率衰减因子
        self.shadowing_std_los = 4.0     # dB - LoS阴影衰落标准差（3GPP UMi场景）
        self.shadowing_std_nlos = 7.82   # dB - NLoS阴影衰落标准差（3GPP UMi场景）
        self.min_distance = 0.5          # m - 3GPP最小距离（UMi场景为0.5米）
        
        # 🔧 修复问题5：编码效率提升至5G NR标准（Polar/LDPC编码）
        self.coding_efficiency = 0.9     # 5G NR编码效率（论文建议0.85-0.95）
        self.processing_delay = 0.001    # s - 处理时延（1ms）
        
        # 🔧 修复问题6：可配置的干扰模型参数
        self.base_interference_power = 1e-12  # W - 基础干扰功率（可调整）
        self.interference_variation = 0.1     # 干扰变化系数（简化模型）
        
        # 🔧 修复问题7：快衰落模型参数（可选启用）
        self.enable_fast_fading = False  # 是否启用快衰落（默认关闭保持简化）
        self.fast_fading_std = 1.0       # 快衰落标准差（Rayleigh/Rician）
        self.rician_k_factor = 6.0       # dB - LoS场景的莱斯K因子
        
        # 🆕 通信模型优化主开关（集成所有优化功能）
        self.use_communication_enhancements = False  # 是否启用所有通信模型优化（快衰落+系统级干扰+动态带宽）
        self.use_system_interference = False         # 是否使用系统级干扰计算
        self.use_bandwidth_allocator = False         # 是否使用动态带宽分配器
        
        # 🆕 正交信道分配（新增）
        self.use_channel_allocation = False          # 是否启用正交信道分配（减少同频干扰）
        
        # 调制参数
        self.modulation_order = 4  # QPSK
        self.coding_rate = 0.5

class MigrationConfig:
    """
    任务迁移配置类
    
    【功能】定义任务迁移策略参数
    【论文对应】Section 2.6 "Task Migration Strategy"
    
    【基础参数】
    - migration_bandwidth = 100 Mbps  # 迁移带宽
    - migration_threshold = 0.8       # 迁移触发阈值
    - migration_cost_factor = 0.1     # 迁移成本因子
    
    【负载阈值】（触发迁移的条件）
    - rsu_overload_threshold = 0.85   # RSU 85%负载触发
    - uav_overload_threshold = 0.85   # UAV 85%负载触发
    - rsu_underload_threshold = 0.3   # RSU 30%以下欠载
    
    【队列管理】
    - follow_handover_distance = 30.0m  # 车辆跟随触发的最小距离改善
    - queue_switch_diff = 3             # 目标RSU较当前RSU队列至少少3个才切换
    - rsu_queue_overload_len = 10       # 基于实际观察提高到15个任务过载阈值
    - service_jitter_ratio = 0.2        # 服务速率±20%抖动
    
    【UAV迁移参数】
    - uav_min_battery = 0.2             # 最低电量20%
    - migration_delay_threshold = 1.0s  # 迁移延迟阈值
    - max_migration_distance = 1000m    # 最大迁移距离
    
    【迁移成本权重】（多目标优化）
    - migration_alpha_comp = 0.4   # 计算成本权重
    - migration_alpha_tx = 0.3     # 传输成本权重
    - migration_alpha_lat = 0.3    # 延迟成本权重
    
    【冷却期】
    - cooldown_period = 1.0s  # 🔑 用户要求：每秒触发一次迁移决策
    
    【论文对应】
    - 迁移策略：Section 2.6, Algorithm 1
    - 成本模型：Equation (9)
    """
    
    def __init__(self):
        self.migration_bandwidth = 100e6  # bps
        self.migration_threshold = 0.8
        self.migration_cost_factor = 0.1
        
        # 🔑 调整：合理的迁移触发阈值
        # 🔧 优化v3：进一步降低到70%，更积极触发迁移，避免过载
        self.rsu_overload_threshold = 0.70   # 70%负载即触发迁移（优化）
        self.uav_overload_threshold = 0.70   # UAV同样70%
        self.rsu_underload_threshold = 0.3   # 欠载阈值保持不变
        # 队列/切换阈值（用于车辆跟随与过载切换）
        self.follow_handover_distance = 30.0  # meters，车辆跟随触发的最小距离改善
        # 🔑 最终优化：统一队列管理标准
        self.queue_switch_diff = 3            # 个，目标RSU较当前RSU队列至少少3个才切换  
        self.rsu_queue_overload_len = 10      # 个，基于实际观察提高到15个任务过载阈值
        self.service_jitter_ratio = 0.2       # 服务速率±20%抖动
        
        # UAV迁移参数
        self.uav_min_battery = 0.2  # 20%
        self.migration_delay_threshold = 1.0  # seconds
        self.max_migration_distance = 1000  # meters
        
        # 迁移成本参数
        self.migration_alpha_comp = 0.4  # 计算成本权重
        self.migration_alpha_tx = 0.3    # 传输成本权重
        self.migration_alpha_lat = 0.3   # 延迟成本权重
        
        self.migration_energy_cost = 0.1  # J per bit
        self.migration_time_penalty = 0.05  # seconds
        
        # 🔑 用户要求：每秒触发一次迁移决策
        # 🚀 优化：差异化冷却期策略
        self.cooldown_period = 0.5  # 0.5秒基础冷却期（紧急迁移）
        self.normal_cooldown = 1.0  # 1.0秒正常迁移冷却期
        self.proactive_cooldown = 2.0  # 2.0秒主动优化冷却期

class CacheConfig:
    """
    缓存配置类
    
    【功能】定义协作缓存系统参数
    【论文对应】Section 2.7 "Collaborative Caching"
    
    【缓存容量】
    - vehicle_cache_capacity = 3 GB   # 车辆缓存（增加3倍匹配任务大小）
    - rsu_cache_capacity = 30 GB      # RSU缓存（增加3倍）
    - uav_cache_capacity = 6 GB       # UAV缓存（增加3倍）
    
    【缓存策略】
    - cache_replacement_policy = 'LRU'  # 替换策略（LRU/LFU/RANDOM）
    - cache_hit_threshold = 0.8         # 缓存命中阈值
    - cache_update_interval = 1.0s      # 缓存更新间隔
    
    【预测参数】
    - prediction_window = 10            # 预测窗口（时隙数）
    - popularity_decay_factor = 0.9     # 流行度衰减因子
    - request_history_size = 100        # 请求历史大小
    
    【论文对应】
    - 缓存策略：Section 2.7, Algorithm 2
    - 流行度预测：Equation (10)
    """
    
    def __init__(self):
        # 🔧 修复：缓存容量配置，匹配任务数据大小增加（3倍）
        # 平均任务大小：1.5 MB (原 500 KB)，单任务结果：75 KB (原 25 KB)
        self.vehicle_cache_capacity = 3e9   # 3 GB (原 1 GB)
        self.rsu_cache_capacity = 30e9      # 30 GB (原 10 GB) - 边缘服务器缓存
        self.uav_cache_capacity = 6e9       # 6 GB (原 2 GB) - 轻量级UAV缓存
        
        # 🎯 P0-1优化：差异化缓存替换策略配置
        # 针对不同节点类型使用最优策略
        self.vehicle_cache_policy = 'LRU'      # 车辆移动快，LRU适合
        self.rsu_cache_policy = 'HYBRID'       # RSU使用混合策略，平衡各因素
        self.uav_cache_policy = 'LFU'          # UAV覆盖广，LFU更优
        
        # 保留兼容性（默认策略）
        self.cache_replacement_policy = 'HYBRID'  # 默认使用混合策略
        self.cache_hit_threshold = 0.8
        self.cache_update_interval = 1.0  # seconds
        
        # 缓存预测参数
        self.prediction_window = 10  # time slots
        self.popularity_decay_factor = 0.9
        self.request_history_size = 100
        
        # 🎯 P0-2优化：预测式缓存配置
        self.enable_predictive_caching = True      # 启用预测缓存
        self.prediction_horizon = 5                # 预测未来5个高需求内容
        self.prediction_threshold = 1.5            # 增长率阈值
        
        # 🎯 P2-2优化：动态容量调整配置
        self.enable_dynamic_capacity = True        # 启用动态容量调整
        self.capacity_adjust_min_ratio = 0.5       # 最小容量比例
        self.capacity_adjust_max_ratio = 1.5       # 最大容量比例
        
        # 🎯 P3-2优化：缓存预热配置
        self.enable_cache_warmup = True            # 启用缓存预热
        self.warmup_capacity_ratio = 0.3           # 预热使用30%容量
        
        # 逻辑回归参数 - 对应论文式(1)
        self.logistic_alpha0 = -2.0  # 截距
        self.logistic_alpha1 = 1.5   # 历史频率权重
        self.logistic_alpha2 = 0.8   # 请求率权重
        self.logistic_alpha3 = 0.6   # 时间因素权重
        self.logistic_alpha4 = 0.4   # 区域特征权重
        
        # 缓存命中预测开关
        self.cache_hit_prediction_enabled = True

class NormalizationConfig:
    """
    统一控制状态特征与指标归一化的配置。

    通过环境变量即可覆盖缩放范围，方便在不同拓扑/负载下快速调参。
    """

    def __init__(self):
        # 数值稳定
        self.metric_epsilon = float(os.environ.get('NORM_EPSILON', '1e-6'))
        self.distribution_smoothing = float(os.environ.get('NORM_DISTRIBUTION_SMOOTHING', '1e-5'))

        # 位置/速度尺度
        # 🔧 修复：使用场景实际最大尺寸2060.0 (场景范围: X: 0~1030m, Y: 0~2060m)
        self.vehicle_position_range = float(os.environ.get('NORM_VEHICLE_POSITION_RANGE', '2060.0'))
        self.rsu_position_range = float(os.environ.get('NORM_RSU_POSITION_RANGE', '2060.0'))
        self.uav_position_range = float(os.environ.get('NORM_UAV_POSITION_RANGE', '2060.0'))
        self.uav_altitude_range = float(os.environ.get('NORM_UAV_ALTITUDE_RANGE', '200.0'))
        self.vehicle_speed_range = float(os.environ.get('NORM_VEHICLE_SPEED_RANGE', '50.0'))

        # 队列容量（任务数量）
        self.vehicle_queue_capacity = float(os.environ.get('NORM_VEHICLE_QUEUE_CAPACITY', '20.0'))
        self.rsu_queue_capacity = float(os.environ.get('NORM_RSU_QUEUE_CAPACITY', '20.0'))
        self.uav_queue_capacity = float(os.environ.get('NORM_UAV_QUEUE_CAPACITY', '20.0'))

        # 能耗参考（焦耳）
        self.vehicle_energy_reference = float(os.environ.get('NORM_VEHICLE_ENERGY_REF', '1000.0'))
        self.rsu_energy_reference = float(os.environ.get('NORM_RSU_ENERGY_REF', '1000.0'))
        self.uav_energy_reference = float(os.environ.get('NORM_UAV_ENERGY_REF', '1000.0'))

        # 奖励归一化参考
        # 🔧 P0修复：对齐energy_normalizer与config.rl.energy_target=900J
        # 🔧 P0修复：对齐delay_normalizer与config.rl.latency_target=1.5s
        # 默认直接对齐 RL 核心目标，避免奖励归一化与目标值不一致导致的偏置
        self.delay_normalizer_value = float(os.environ.get('NORM_DELAY_NORMALIZER', '1.5'))  # 🔧 0.4 → 1.5 (对齐RLConfig)
        self.energy_normalizer_value = float(os.environ.get('NORM_ENERGY_NORMALIZER', '900.0'))  # 🔧 3500 → 900 (对齐RLConfig)

        # 全局性能参考（供奖励/指标归一化使用）
        self.delay_reference = float(os.environ.get('NORM_DELAY_REFERENCE', '1.5'))  # 🔧 0.4 → 1.5
        self.delay_upper_reference = float(os.environ.get('NORM_DELAY_UPPER_REFERENCE', '2.2'))  # 🔧 1.0 → 2.2
        self.energy_reference = float(os.environ.get('NORM_ENERGY_REFERENCE', '1000.0'))  # 🔧 200 → 1000
        self.energy_upper_reference = float(os.environ.get('NORM_ENERGY_UPPER_REFERENCE', '2000.0'))  # 🔧 500 → 2000


class SystemConfig:
    """
    系统配置容器类
    
    【功能】整合所有子配置模块，提供统一访问接口
    【使用方式】通过全局单例 config 访问所有配置
    
    【子配置模块】
    - queue: QueueConfig           # 队列管理
    - task: TaskConfig             # 任务生成
    - compute: ComputeConfig       # 计算资源
    - network: NetworkConfig       # 网络拓扑
    - communication: CommunicationConfig  # 3GPP通信
    - migration: MigrationConfig   # 任务迁移
    - cache: CacheConfig           # 协作缓存
    - service: ServiceConfig       # 服务能力
    - stats: StatsConfig           # 统计监控
    - experiment: ExperimentConfig # 实验配置
    - rl: RLConfig                 # 强化学习
    
    【使用示例】
    ```python
    from config.system_config import config
    
    # 访问网络配置
    num_vehicles = config.num_vehicles
    
    # 访问RL配置
    delay_weight = config.rl.reward_weight_delay
    
    # 访问任务配置
    arrival_rate = config.task.arrival_rate
    ```
    """
    
    def __init__(self):
        # 基本系统配置
        cuda_flag = os.environ.get('CUDA_AVAILABLE')
        if cuda_flag is not None:
            self.device = 'cuda' if cuda_flag.lower() == 'true' else 'cpu'
        else:
            try:
                import torch  # type: ignore
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                self.device = 'cpu'
        self.num_threads = int(os.environ.get('NUM_THREADS', '4'))
        self.random_seed = int(os.environ.get('RANDOM_SEED', '42'))
        
        # 🚀 12车辆高负载场景网络配置
        self.num_vehicles = 12  # 保持12车辆，通过其他方式创造高负载
        self.num_rsus = 4       # 更新为4个RSU
        self.num_uavs = 2       # 保持UAV数量
        
        # 仿真配置
        self.simulation_time = 1000
        self.time_slot = 0.1  # 🔧 改为100ms，与network.time_slot_duration一致
        
        # 性能配置
        self.enable_performance_optimization = True
        self.batch_size_optimization = True
        self.parallel_environments = 6
        
        # 子配置模块
        self.normalization = NormalizationConfig()
        self.queue = QueueConfig()
        self.task = TaskConfig()
        self.compute = ComputeConfig()
        self.network = NetworkConfig()
        self.communication = CommunicationConfig()
        self.migration = MigrationConfig()
        self.cache = CacheConfig()
        self.service = ServiceConfig()
        self.stats = StatsConfig()
        
        # 实验配置
        self.experiment = ExperimentConfig()
        
        # 强化学习配置
        self.rl = RLConfig()
        
    def get_config_dict(self) -> Dict[str, Any]:
        """
        返回系统配置字典
        
        【功能】将主要配置参数导出为字典格式
        【返回值】包含系统关键配置的字典
        【用途】用于日志记录、配置保存、实验复现
        """
        return {
            'device': self.device,
            'num_threads': self.num_threads,
            'random_seed': self.random_seed,
            'num_vehicles': self.num_vehicles,
            'num_rsus': self.num_rsus,
            'num_uavs': self.num_uavs,
            'simulation_time': self.simulation_time,
            'time_slot': self.time_slot,
            'enable_performance_optimization': self.enable_performance_optimization,
            'batch_size_optimization': self.batch_size_optimization,
            'parallel_environments': self.parallel_environments
        }
    
    def update_config(self, **kwargs):
        """
        动态更新配置参数
        
        【功能】从关键字参数更新配置属性
        【参数】kwargs - 要更新的配置参数
        【示例】config.update_config(num_vehicles=15, random_seed=123)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 全局配置实例
config = SystemConfig()


