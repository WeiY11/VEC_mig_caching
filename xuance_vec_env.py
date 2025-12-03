#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEC Environment for XuanCe Framework
VEC边缘计算系统 - XuanCe环境封装

本模块提供标准的Gymnasium环境接口，使VEC模拟器可以与XuanCe框架无缝集成。

使用方式：
    from xuance_vec_env import VECEnv, register_vec_env
    
    # 注册环境
    register_vec_env()
    
    # 使用xuance
    import xuance
    runner = xuance.get_runner(method='td3', env='VEC', env_id='VEC-v1')
    runner.run()
"""

from __future__ import annotations

import copy
import os
import random
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# 导入VEC系统组件
from config import config as global_config
from config.unified_config import get_config, UnifiedConfig

# 尝试导入xuance
try:
    from xuance.environment.single_agent_env import REGISTRY_ENV
    XUANCE_AVAILABLE = True
except ImportError:
    XUANCE_AVAILABLE = False
    REGISTRY_ENV = {}


class VECEnv(gym.Env):
    """
    VEC边缘计算环境 - 完整的Gymnasium接口实现
    
    该环境封装了VEC模拟器，提供：
    - 标准的观测空间和动作空间定义
    - 完整的step/reset/render接口
    - 与xuance框架的无缝集成
    
    观测空间（state_dim维）：
        - 车辆状态（位置、速度、队列、能耗、连接）
        - RSU状态（队列、能耗、缓存、负载）
        - UAV状态（位置、高度、队列、能耗、电池）
        - 全局状态（任务到达率、平均延迟、平均能耗等）
    
    动作空间（action_dim维）：
        - 带宽分配
        - 计算资源分配
        - 缓存决策
        - 卸载决策
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        config: Optional[Union[Namespace, Dict, UnifiedConfig]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        初始化VEC环境
        
        Args:
            config: 配置对象，支持Namespace、Dict或UnifiedConfig
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # 解析配置
        self._parse_config(config)
        
        # 延迟导入以避免循环依赖
        from train_single_agent import SingleAgentTrainingEnvironment
        
        # 创建训练环境
        self.training_env = SingleAgentTrainingEnvironment(
            algorithm=self._algorithm,
            override_scenario=self._scenario_overrides,
            use_enhanced_cache=self._use_enhanced_cache,
            disable_migration=self._disable_migration,
        )
        
        # 获取状态和动作维度
        self.state_dim = int(self.training_env.agent_env.state_dim)
        self.action_dim = int(getattr(self.training_env.agent_env, "action_dim", 18))
        
        # 定义观测空间和动作空间
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # 内部状态
        self._current_step = 0
        self._episode_reward = 0.0
        self._last_state: Optional[np.ndarray] = None
        self._latest_metrics: Dict[str, float] = {}
        self._episode_count = 0
        
        # xuance要求的属性
        self.max_episode_steps = self._max_episode_steps
        
        # 设置随机种子
        if self._seed is not None:
            self._apply_seed(self._seed)
    
    def _parse_config(self, config):
        """解析配置"""
        # 默认值
        self._max_episode_steps = 200
        self._seed = 42
        self._scenario_overrides = {}
        self._use_enhanced_cache = True
        self._disable_migration = False
        self._algorithm = "TD3"
        
        if config is None:
            return
        
        # 如果是Dict，转换为Namespace
        if isinstance(config, dict):
            # 检查是否有vec_config子配置
            vec_config = config.get('vec_config', {})
            self._max_episode_steps = vec_config.get('max_episode_steps', 
                                                      config.get('max_episode_steps', 200))
            self._seed = config.get('env_seed', config.get('seed', 42))
            self._use_enhanced_cache = vec_config.get('use_enhanced_cache', True)
            self._disable_migration = vec_config.get('disable_migration', False)
            self._algorithm = config.get('base_algorithm', 'TD3')
            
            # 构建场景覆盖
            self._scenario_overrides = {
                'num_vehicles': vec_config.get('num_vehicles', 12),
                'num_rsus': vec_config.get('num_rsus', 4),
                'num_uavs': vec_config.get('num_uavs', 2),
            }
            if 'arrival_rate' in vec_config:
                self._scenario_overrides['arrival_rate'] = vec_config['arrival_rate']
                
        elif isinstance(config, UnifiedConfig):
            self._max_episode_steps = config.experiment.max_steps_per_episode
            self._seed = config.experiment.random_seed
            self._scenario_overrides = {
                'num_vehicles': config.network.num_vehicles,
                'num_rsus': config.network.num_rsus,
                'num_uavs': config.network.num_uavs,
            }
            
        else:  # Namespace
            self._max_episode_steps = getattr(config, 'max_episode_steps',
                                              getattr(config, 'horizon_size', 200))
            self._seed = getattr(config, 'env_seed', getattr(config, 'seed', 42))
            self._use_enhanced_cache = getattr(config, 'use_enhanced_cache', True)
            self._disable_migration = getattr(config, 'disable_migration', False)
            self._algorithm = getattr(config, 'base_algorithm', 'TD3')
            
            vec_config = getattr(config, 'vec_config', None)
            if vec_config:
                if isinstance(vec_config, dict):
                    self._scenario_overrides = {
                        'num_vehicles': vec_config.get('num_vehicles', 12),
                        'num_rsus': vec_config.get('num_rsus', 4),
                        'num_uavs': vec_config.get('num_uavs', 2),
                    }
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        if seed is not None:
            self._apply_seed(seed)
        
        self._current_step = 0
        self._episode_reward = 0.0
        self._latest_metrics = {}
        self._episode_count += 1
        
        # 重置训练环境
        state = self.training_env.reset_environment()
        self._last_state = state.astype(np.float32, copy=True)
        
        info = {
            "episode": self._episode_count,
            "step": 0,
        }
        
        return self._last_state.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作向量
        
        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 处理动作
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[:min(self.action_dim, action_array.size)] = action_array[:self.action_dim]
            action_array = padded
        
        # 构建动作字典
        actions_dict = self.training_env._build_actions_from_vector(action_array)
        
        # 执行步骤
        next_state, reward, _, info_details = self.training_env.step(
            action_array, self._last_state, actions_dict
        )
        
        self._current_step += 1
        self._episode_reward += float(reward)
        self._last_state = next_state.astype(np.float32, copy=True)
        
        # 提取系统指标
        system_metrics = info_details.get("system_metrics", {}) or {}
        self._latest_metrics = {
            "avg_delay": float(system_metrics.get("avg_task_delay", 0.0)),
            "total_energy": float(system_metrics.get("total_energy_consumption", 0.0)),
            "task_completion_rate": float(system_metrics.get("task_completion_rate", 0.0)),
            "cache_hit_rate": float(system_metrics.get("cache_hit_rate", 0.0)),
        }
        
        # 检查终止条件
        terminated = False
        truncated = self._current_step >= self._max_episode_steps
        
        info: Dict[str, Any] = {
            "step": self._current_step,
            "episode_reward": self._episode_reward,
            "system_metrics": system_metrics,
        }
        
        if truncated:
            info["final_metrics"] = self._latest_metrics.copy()
            info["episode_length"] = self._current_step
        
        return self._last_state.copy(), float(reward), terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """渲染环境（当前未实现）"""
        if self.render_mode == "rgb_array":
            # 返回一个占位图像
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'training_env'):
            simulator = getattr(self.training_env, "simulator", None)
            if simulator and hasattr(simulator, "close"):
                simulator.close()
    
    def _apply_seed(self, seed: int):
        """设置随机种子"""
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """获取当前episode的指标"""
        return {
            "episode_reward": self._episode_reward,
            "episode_length": self._current_step,
            **self._latest_metrics,
        }


def register_vec_env():
    """
    注册VEC环境到xuance和gymnasium
    """
    # 注册到gymnasium
    try:
        gym.register(
            id="VEC-v1",
            entry_point="xuance_vec_env:VECEnv",
            max_episode_steps=200,
        )
    except Exception:
        pass  # 可能已经注册
    
    # 注册到xuance
    if XUANCE_AVAILABLE and "VEC" not in REGISTRY_ENV:
        REGISTRY_ENV["VEC"] = VECEnv
        print("[OK] VEC环境已注册到XuanCe框架")


def make_vec_env(config: Optional[Union[Namespace, Dict]] = None) -> VECEnv:
    """
    创建VEC环境的便捷函数
    
    Args:
        config: 配置对象
    
    Returns:
        VECEnv实例
    """
    return VECEnv(config=config)


# 自动注册
register_vec_env()


if __name__ == "__main__":
    # 测试环境
    print("测试VEC环境...")
    
    env = VECEnv()
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    obs, info = env.reset()
    print(f"初始观测形状: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✅ 环境测试通过!")
