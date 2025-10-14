#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xuance Gym环境适配器
将CompleteSystemSimulator包装成符合OpenAI Gym接口的环境

【功能】
1. 实现标准Gym接口：reset(), step(), render(), close()
2. 定义observation_space和action_space
3. 状态-动作维度转换
4. 奖励计算（使用统一奖励函数）
5. 与现有train_single_agent.py保持一致
"""

import os
import sys
import gym
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from gym import spaces

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import config
from evaluation.system_simulator import CompleteSystemSimulator
from utils.unified_reward_calculator import calculate_unified_reward


class VECGymEnv(gym.Env):
    """
    车联网边缘计算Gym环境
    适配xuance框架使用
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, env_config: Dict[str, Any] = None):
        """
        初始化环境
        
        【参数】
        - env_config: 环境配置字典（从config_adapter获取）
        """
        super(VECGymEnv, self).__init__()
        
        self.env_config = env_config or {}
        
        # 应用配置覆盖
        self._apply_config_overrides()
        
        # 网络拓扑（固定）
        self.num_vehicles = self.env_config.get('num_vehicles', 12)
        self.num_rsus = self.env_config.get('num_rsus', 4)
        self.num_uavs = self.env_config.get('num_uavs', 2)
        
        # 计算状态和动作维度
        # 状态: num_vehicles*5 + num_rsus*5 + num_uavs*5 + 8
        self.state_dim = self.num_vehicles * 5 + self.num_rsus * 5 + self.num_uavs * 5 + 8
        
        # 动作: 3(分配) + 4(RSU) + 2(UAV) + 7(控制) = 16
        self.action_dim = 16
        
        # 定义Gym空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # 创建系统仿真器
        self.simulator = None
        self._create_simulator()
        
        # 当前状态和步数
        self.current_state = None
        self.current_step = 0
        self.max_steps = self.env_config.get('max_steps_per_episode', 100)
        self.max_episode_steps = self.max_steps  # xuance需要这个属性
        
        # 统计信息
        self.episode_count = 0
        self.total_reward = 0.0
        
        print(f"✓ VECGymEnv 初始化完成:")
        print(f"  拓扑: {self.num_vehicles}车 + {self.num_rsus}RSU + {self.num_uavs}UAV")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作维度: {self.action_dim}")
        print(f"  每轮最大步数: {self.max_steps}")
    
    def _apply_config_overrides(self):
        """应用环境变量配置覆盖"""
        import json
        
        # 设置随机种子
        seed = self.env_config.get('random_seed', 42)
        os.environ['RANDOM_SEED'] = str(seed)
        
        # 覆盖车辆数
        if self.env_config.get('num_vehicles', 12) != 12:
            overrides = {"num_vehicles": self.env_config['num_vehicles']}
            os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(overrides)
    
    def _create_simulator(self):
        """创建系统仿真器"""
        simulator_config = {
            'num_vehicles': self.num_vehicles,
            'num_rsus': self.num_rsus,
            'num_uavs': self.num_uavs,
            'simulation_time': self.env_config.get('simulation_time', 1000),
            'time_slot': self.env_config.get('time_slot', 0.2),
            'task_arrival_rate': self.env_config.get('task_arrival_rate', 2.5),
            'cache_capacity': self.env_config.get('cache_capacity', 80),
            'computation_capacity': self.env_config.get('computation_capacity', 800),
            'bandwidth': self.env_config.get('bandwidth', 15),
            'transmission_power': self.env_config.get('transmission_power', 0.15),
            'computation_power': self.env_config.get('computation_power', 1.2),
            'override_topology': True,  # 使用配置而不是system_config
        }
        
        self.simulator = CompleteSystemSimulator(simulator_config)
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        【返回】
        - 初始状态向量
        - info字典（xuance需要）
        """
        # 重新创建仿真器（确保干净的初始状态）
        self._create_simulator()
        
        # 获取初始状态
        self.current_state = self._get_state()
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_count += 1
        
        # xuance需要返回info
        info = {
            'episode': self.episode_count,
            'step': self.current_step
        }
        
        return self.current_state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步
        
        【参数】
        - action: 动作向量（16维，范围[-1, 1]）
        
        【返回】（gymnasium接口）
        - next_state: 下一状态
        - reward: 奖励
        - terminated: 是否因完成而结束
        - truncated: 是否因超时而结束
        - info: 附加信息
        """
        # 确保动作在有效范围内
        action = np.clip(action, -1.0, 1.0)
        
        # 将动作向量转换为动作字典
        actions_dict = self._build_actions_from_vector(action)
        
        # 执行动作（仿真一个时隙）
        # CompleteSystemSimulator使用run_simulation_step
        self.simulator.run_simulation_step(self.current_step, actions_dict)
        
        # 获取系统指标
        system_metrics = self.simulator.get_system_state()
        
        # 计算奖励（使用统一奖励函数）
        reward = calculate_unified_reward(system_metrics, algorithm='general')
        
        # 获取下一状态
        next_state = self._get_state()
        
        # 更新步数
        self.current_step += 1
        self.total_reward += reward
        
        # 判断是否结束（gymnasium接口：分为terminated和truncated）
        terminated = False  # 自然结束条件（如果有）
        truncated = self.current_step >= self.max_steps  # 超时截断
        
        # 构建info
        info = {
            'system_metrics': system_metrics,
            'episode_step': self.current_step,
            'episode_reward': self.total_reward,
        }
        
        self.current_state = next_state
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态向量
        
        【返回】
        - 状态向量（归一化）
        """
        state_list = []
        
        # 车辆状态（每个5维：x, y, speed, cpu_load, queue_length）
        for vehicle in self.simulator.vehicles:
            state_list.extend([
                vehicle['position'][0] / 1000.0,  # x坐标归一化
                vehicle['position'][1] / 1000.0,  # y坐标
                vehicle['velocity'] / 30.0,       # 速度（假设最大30m/s）
                vehicle.get('cpu_load', 0.0),     # CPU负载（已归一化）
                vehicle.get('queue_length', 0) / 100.0,  # 队列长度
            ])
        
        # RSU状态（每个5维）
        for rsu in self.simulator.rsus:
            state_list.extend([
                rsu['position'][0] / 1000.0,
                rsu['position'][1] / 1000.0,
                0.0,  # RSU无速度
                rsu.get('cpu_load', 0.0),
                rsu.get('queue_length', 0) / 100.0,
            ])
        
        # UAV状态（每个5维）
        for uav in self.simulator.uavs:
            state_list.extend([
                uav['position'][0] / 1000.0,
                uav['position'][1] / 1000.0,
                uav.get('altitude', 100.0) / 200.0,  # 高度归一化
                uav.get('cpu_load', 0.0),
                uav.get('queue_length', 0) / 100.0,
            ])
        
        # 全局状态（8维）
        # CompleteSystemSimulator使用get_system_state而不是get_system_metrics
        system_metrics = self.simulator.get_system_state()
        global_state = [
            system_metrics.get('avg_task_delay', 0.0) / 10.0,  # 平均时延
            system_metrics.get('total_energy_consumption', 0.0) / 1000.0,  # 能耗
            system_metrics.get('task_completion_rate', 0.0),  # 完成率
            system_metrics.get('cache_hit_rate', 0.0),  # 缓存命中率
            system_metrics.get('avg_queue_length', 0.0) / 50.0,  # 平均队列
            system_metrics.get('system_utilization', 0.0),  # 系统利用率
            float(self.current_step) / self.max_steps,  # 进度
            float(self.simulator.stats['total_tasks']) / 1000.0,  # 任务数
        ]
        state_list.extend(global_state)
        
        # 转换为numpy数组
        state = np.array(state_list, dtype=np.float32)
        
        # 确保维度正确
        if len(state) != self.state_dim:
            # 填充或截断
            if len(state) < self.state_dim:
                state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
            else:
                state = state[:self.state_dim]
        
        return state
    
    def _build_actions_from_vector(self, action_vector: np.ndarray) -> Dict:
        """
        将动作向量转换为动作字典
        
        【参数】
        - action_vector: 16维动作向量
        
        【返回】
        - 动作字典
        """
        # 动作结构：
        # [0:3] 任务分配（local, RSU, UAV）
        # [3:7] RSU选择（4个RSU）
        # [7:9] UAV选择（2个UAV）
        # [9:16] 控制参数（7维）
        
        actions = {
            'vehicle_agent': action_vector.tolist(),  # 完整的16维动作
            'rsu_agent': np.zeros(self.num_rsus * 2).tolist(),  # RSU内部动作（占位）
            'uav_agent': np.zeros(self.num_uavs * 2).tolist(),  # UAV内部动作（占位）
        }
        
        return actions
    
    def render(self, mode='human'):
        """渲染环境（可选）"""
        if mode == 'human':
            metrics = self.simulator.get_system_state()
            print(f"\nStep {self.current_step}/{self.max_steps}")
            print(f"  Delay: {metrics.get('avg_task_delay', 0):.3f}s")
            print(f"  Energy: {metrics.get('total_energy_consumption', 0):.1f}J")
            print(f"  Completion: {metrics.get('task_completion_rate', 0):.2%}")
    
    def close(self):
        """关闭环境"""
        self.simulator = None
    
    def seed(self, seed: int = None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
            os.environ['RANDOM_SEED'] = str(seed)
        return [seed]


if __name__ == "__main__":
    # 测试环境
    print("="*80)
    print("VECGymEnv 测试")
    print("="*80)
    
    from config_adapter import create_xuance_config
    
    # 创建配置
    test_config = create_xuance_config('TD3', num_episodes=10, seed=42, num_vehicles=12)
    
    # 创建环境
    env = VECGymEnv(test_config['env_config'])
    
    # 测试reset
    state = env.reset()
    print(f"\n初始状态形状: {state.shape}")
    print(f"状态范围: [{state.min():.3f}, {state.max():.3f}]")
    
    # 测试step
    print("\n测试10步随机动作:")
    for i in range(10):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        print(f"  Step {i+1}: Reward={reward:.3f}, "
              f"Delay={info['system_metrics'].get('avg_task_delay', 0):.3f}s, "
              f"Done={done}")
        
        if done:
            break
    
    # 关闭环境
    env.close()
    
    print("\n" + "="*80)
    print("VECGymEnv 测试完成！")
    print("="*80)

