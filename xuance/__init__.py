# -*- coding: utf-8 -*-
"""
XuanCe风格训练模块 - VEC边缘计算系统

提供基于XuanCe框架的统一训练接口，支持：
- 多种DRL算法 (TD3, SAC, PPO, DDPG, DQN等)
- 对比方案Baselines (Local, Heuristic, SA等)
- YAML配置文件管理
- 命令行参数覆盖
"""

from .vec_env import VECEnv, register_vec_env

__all__ = ['VECEnv', 'register_vec_env']
