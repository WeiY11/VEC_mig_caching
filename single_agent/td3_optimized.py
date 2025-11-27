"""
优化的TD3算法实现 - 修复关键问题
基于分析报告的改进版本

主要改进：
1. 重构状态空间设计
2. 重新设计动作空间
3. 优化奖励函数
4. 修复环境交互
5. 调整超参数配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from config import config


@dataclass
class OptimizedTD3Config:
    """优化的TD3配置 - 修复版本"""
    # 网络结构
    hidden_dim: int = 256        # 适当增加网络容量以学习复杂策略
    actor_lr: float = 3e-4       # 提高学习率，加速收敛
    critic_lr: float = 3e-4      # 提高学习率，加速收敛
    
    # 训练参数
    batch_size: int = 256        # 适中批次大小
    buffer_size: int = 500000    # 适中缓冲区
    tau: float = 0.005           # 标准软更新速率
    gamma: float = 0.99          # 标准折扣因子
    
    # TD3特有参数
    policy_delay: int = 2        # 标准策略延迟
    target_noise: float = 0.2    # 标准目标噪声
    noise_clip: float = 0.5      # 标准噪声裁剪
    
    # 探索参数
    exploration_noise: float = 0.25   # 适中的初始探索
    noise_decay: float = 0.9998       # 更缓慢的衰减速度
    min_noise: float = 0.08           # 保留更多探索
    
    # 训练控制
    warmup_steps: int = 10000        # 减少预热步数，约50个episode
    update_freq: int = 1             # 每步都更新
    
    # 正则化参数
    weight_decay: float = 1e-5       # 更小的L2正则化
    grad_clip: float = 1.0           # 适度梯度裁剪


class VECActionSpace:
    """VEC系统动作空间定义"""
    
    def __init__(self):
        # 动作维度定义
        self.vehicle_actions = 5    # 本地处理比例、卸载目标选择等
        self.rsu_actions = 8        # 计算资源分配、缓存策略、迁移决策等
        self.uav_actions = 6        # 计算资源分配、移动策略等
        
        self.num_vehicles = config.network.num_vehicles  # 12
        self.num_rsus = config.network.num_rsus          # 6  
        self.num_uavs = config.network.num_uavs          # 2
        
        self.total_dim = (
            self.num_vehicles * self.vehicle_actions +  # 12 * 5 = 60
            self.num_rsus * self.rsu_actions +          # 6 * 8 = 48
            self.num_uavs * self.uav_actions            # 2 * 6 = 12
        )  # 总计：120维
    
    def decompose_action(self, action: np.ndarray) -> Dict:
        """将全局动作分解为具体决策"""
        actions = {}
        idx = 0
        
        # 车辆动作
        for i in range(self.num_vehicles):
            vehicle_action = action[idx:idx+self.vehicle_actions]
            actions[f'vehicle_{i}'] = {
                'local_processing_ratio': np.clip(vehicle_action[0], 0, 1),
                'offload_target_rsu': np.argmax(vehicle_action[1:4]) if len(vehicle_action) > 3 else 0,
                'offload_target_uav': int(vehicle_action[4] > 0) if len(vehicle_action) > 4 else 0,
            }
            idx += self.vehicle_actions
        
        # RSU动作
        for i in range(self.num_rsus):
            rsu_action = action[idx:idx+self.rsu_actions]
            actions[f'rsu_{i}'] = {
                'cpu_allocation': np.clip(rsu_action[0], 0.5, 1.0),
                'cache_policy': np.argmax(rsu_action[1:4]),  # LRU/LFU/FIFO
                'migration_threshold': np.clip(rsu_action[4], 0.5, 0.9),
                'bandwidth_allocation': np.clip(rsu_action[5:8], 0.1, 1.0),
            }
            idx += self.rsu_actions
        
        # UAV动作
        for i in range(self.num_uavs):
            uav_action = action[idx:idx+self.uav_actions]
            actions[f'uav_{i}'] = {
                'cpu_allocation': np.clip(uav_action[0], 0.3, 1.0),
                'power_management': np.clip(uav_action[1], 0.5, 1.0),
                'service_priority': np.clip(uav_action[2:6], 0, 1),
            }
            idx += self.uav_actions
        
        return actions


class VECStateSpace:
    """VEC系统状态空间定义"""
    
    def __init__(self, system_config=None):
        # 如果提供了配置，使用提供的配置，否则使用默认配置
        if system_config is not None:
            self.num_vehicles = system_config.network.num_vehicles
            self.num_rsus = system_config.network.num_rsus
            self.num_uavs = system_config.network.num_uavs
        else:
            self.num_vehicles = config.network.num_vehicles
            self.num_rsus = config.network.num_rsus
            self.num_uavs = config.network.num_uavs
        
        # 状态维度计算
        self.vehicle_state_dim = 7  # 位置x,y + 速度x,y + 队列利用率 + 🔧CPU容量 + 🔧当前任务负载
        self.rsu_state_dim = 7      # CPU利用率 + 队列利用率 + 缓存利用率 + 能耗 + 🔧资源容量 + 🔧平均距离 + 🔧缓存命中率
        self.uav_state_dim = 6      # CPU利用率 + 队列利用率 + 电池电量 + 能耗 + 🔧资源容量 + 🔧平均距离
        self.global_state_dim = 16  # 全局系统指标（基础8维 + 任务类型8维）
        
        self.total_dim = (
            self.num_vehicles * self.vehicle_state_dim +  # 12 * 7 = 84
            self.num_rsus * self.rsu_state_dim +          # 4 * 7 = 28
            self.num_uavs * self.uav_state_dim +          # 2 * 6 = 12
            self.global_state_dim                         # 16
        )  # 总计：140维 (🔧优化后+34维)
    
    def encode_state(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """构建符合论文的VEC系统状态向量"""
        state_components = []
        
        # 1. 车辆状态 (12车辆 × 7维 = 84维)
        for i in range(self.num_vehicles):
            vehicle_id = f'vehicle_{i}'
            if vehicle_id in node_states:
                vehicle = node_states[vehicle_id]
                # 🔧 新增：车辆CPU容量和当前任务负载，让智能体知道本地计算能力
                # 🐞 修复：node_states是numpy数组，不是对象，直接使用索引访问
                if isinstance(vehicle, np.ndarray):
                    # 从 train_single_agent.py: [pos_x, pos_y, velocity, queue_len, energy]
                    # 需要添加 cpu_capacity 和 task_load
                    vehicle_state = [
                        vehicle[0],  # position_x (normalized)
                        vehicle[1],  # position_y (normalized)
                        0.0,         # velocity_x (TODO: 从 velocity计算)
                        0.0,         # velocity_y 
                        vehicle[3],  # queue_utilization
                        1.5e9 / 20e9,  # 🔧 CPU容量 (1.5GHz/20GHz = 0.075)
                        vehicle[3],  # 🔧 任务负载（使用queue_utilization）
                    ]
                else:
                    # 如果是对象类型（兼容旧版本）
                    cpu_freq = getattr(vehicle, 'cpu_frequency', 1.5e9)
                    queue_len = getattr(vehicle, 'queue_length', 0)
                    vehicle_state = [
                        getattr(vehicle.position, 'x', 0.0) / 2000.0,
                        getattr(vehicle.position, 'y', 0.0) / 2000.0,
                        getattr(vehicle, 'velocity_x', 0.0) / 30.0,
                        getattr(vehicle, 'velocity_y', 0.0) / 30.0,
                        getattr(vehicle, 'queue_utilization', 0.5),
                        cpu_freq / 20e9,
                        min(queue_len / 20.0, 1.0),
                    ]
            else:
                # 默认状态
                vehicle_state = [0.5, 0.5, 0.0, 0.0, 0.5, 0.075, 0.5]
            state_components.extend(vehicle_state)
        
        # 2. RSU状态 (按配置数量 × 7维)
        for i in range(self.num_rsus):
            rsu_id = f'rsu_{i}'
            if rsu_id in node_states:
                rsu = node_states[rsu_id]
                # 🐞 修复：node_states是numpy数组，不是对象
                if isinstance(rsu, np.ndarray):
                    # 从 train_single_agent.py: [pos_x, pos_y, cache_util, queue_len, energy, cpu_freq_norm]
                    # 需要添加 avg_distance 和 cache_hit_rate
                    rsu_state = [
                        rsu[3],      # queue_utilization (CPU利用率用队列代替)
                        rsu[3],      # queue_utilization
                        rsu[2],      # cache_utilization
                        rsu[4],      # energy_consumption (normalized)
                        rsu[5] if len(rsu) > 5 else 0.625,  # 🔧 CPU容量 (12.5GHz/20GHz)
                        0.5,         # 🔧 平均距离（默认，无法计算）
                        0.5,         # 🔧 缓存命中率（默认）
                    ]
                else:
                    # 如果是对象类型（兼容旧版本）
                    cpu_freq = getattr(rsu, 'cpu_frequency', 12.5e9)
                    cache_hit_rate = getattr(rsu, 'recent_cache_hit_rate', 0.5)
                    rsu_pos = getattr(rsu, 'position', None)
                    avg_distance = 0.5
                    if rsu_pos:
                        distances = []
                        for j in range(self.num_vehicles):
                            v_id = f'vehicle_{j}'
                            if v_id in node_states:
                                v_pos = getattr(node_states[v_id], 'position', None)
                                if v_pos:
                                    dist = ((rsu_pos.x - v_pos.x)**2 + (rsu_pos.y - v_pos.y)**2)**0.5
                                    distances.append(dist)
                        if distances:
                            avg_distance = min(sum(distances) / len(distances) / 1000.0, 1.0)
                    
                    rsu_state = [
                        getattr(rsu, 'cpu_utilization', 0.5),
                        getattr(rsu, 'queue_utilization', 0.5),
                        getattr(rsu, 'cache_utilization', 0.5),
                        getattr(rsu, 'energy_consumption', 500.0) / 1000.0,
                        cpu_freq / 20e9,
                        avg_distance,
                        cache_hit_rate,
                    ]
            else:
                rsu_state = [0.5, 0.5, 0.5, 0.5, 0.625, 0.5, 0.5]
            state_components.extend(rsu_state)
        
        # 3. UAV状态 (按配置数量 × 6维)
        for i in range(self.num_uavs):
            uav_id = f'uav_{i}'
            if uav_id in node_states:
                uav = node_states[uav_id]
                # 🐞 修复：node_states是numpy数组，不是对象
                if isinstance(uav, np.ndarray):
                    # 从 train_single_agent.py: [pos_x, pos_y, pos_z, cache_util, energy, cpu_freq_norm]
                    # 需要添加 avg_distance
                    uav_state = [
                        uav[3] if len(uav) > 3 else 0.5,  # queue_utilization (CPU利用率用缓存代替)
                        uav[3] if len(uav) > 3 else 0.5,  # queue_utilization
                        0.8,         # battery_level (默认)
                        uav[4] if len(uav) > 4 else 0.5,  # energy_consumption (normalized)
                        uav[5] if len(uav) > 5 else 0.25, # 🔧 CPU容量 (5GHz/20GHz)
                        0.5,         # 🔧 平均距离（默认，无法计算）
                    ]
                else:
                    # 如果是对象类型（兼容旧版本）
                    cpu_freq = getattr(uav, 'cpu_frequency', 5.0e9)
                    uav_pos = getattr(uav, 'position', None)
                    avg_distance = 0.5
                    if uav_pos:
                        distances = []
                        for j in range(self.num_vehicles):
                            v_id = f'vehicle_{j}'
                            if v_id in node_states:
                                v_pos = getattr(node_states[v_id], 'position', None)
                                if v_pos:
                                    dist = ((uav_pos.x - v_pos.x)**2 + (uav_pos.y - v_pos.y)**2)**0.5
                                    distances.append(dist)
                        if distances:
                            avg_distance = min(sum(distances) / len(distances) / 1000.0, 1.0)
                    
                    uav_state = [
                        getattr(uav, 'cpu_utilization', 0.5),
                        getattr(uav, 'queue_utilization', 0.5),
                        getattr(uav, 'battery_level', 0.8),
                        getattr(uav, 'energy_consumption', 50.0) / 100.0,
                        cpu_freq / 20e9,
                        avg_distance,
                    ]
            else:
                uav_state = [0.5, 0.5, 0.8, 0.5, 0.25, 0.5]
            state_components.extend(uav_state)
        
        # 4. 全局系统状态 (8维)
        global_state = [
            system_metrics.get('avg_task_delay', 1.0) / 2.0,
            system_metrics.get('total_energy_consumption', 2500.0) / 5000.0,
            system_metrics.get('data_loss_rate', 0.1),
            system_metrics.get('task_completion_rate', 0.8),
            system_metrics.get('cache_hit_rate', 0.6),
            system_metrics.get('migration_success_rate', 0.0),
            system_metrics.get('network_utilization', 0.5),
            system_metrics.get('load_balance_index', 0.5),
        ]
        def _extract_metric(key: str) -> List[float]:
            values = system_metrics.get(key, [])
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif not isinstance(values, (list, tuple)):
                values = []
            values = [float(np.clip(v, 0.0, 1.0)) for v in values[:4]]
            if len(values) < 4:
                values.extend([0.0] * (4 - len(values)))
            return values
        global_state.extend(_extract_metric('task_type_queue_distribution'))
        global_state.extend(_extract_metric('task_type_deadline_remaining'))
        state_components.extend(global_state)
        
        # 转换为numpy数组并检查NaN值
        state_vector = np.array(state_components, dtype=np.float32)
        
        # 检查并处理NaN值
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            print(f"警告: 状态向量包含NaN或Inf值，使用默认值替换")
            state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        
        return state_vector


class OptimizedTD3Actor(nn.Module):
    """优化的TD3 Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, max_action: float = 1.0):
        super(OptimizedTD3Actor, self).__init__()
        
        self.max_action = max_action
        
        # 更深的网络结构
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # 最后一层使用较小的权重初始化
        nn.init.uniform_(self.network[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[-2].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(state)


class OptimizedTD3Critic(nn.Module):
    """优化的TD3 Twin Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(OptimizedTD3Critic, self).__init__()
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2网络
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for network in [self.q1_network, self.q2_network]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
            
            # 最后一层使用较小的权重初始化
            nn.init.uniform_(network[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(network[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 返回两个Q值"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回Q1值"""
        sa = torch.cat([state, action], dim=1)
        return self.q1_network(sa)


class OptimizedTD3Environment:
    """优化的TD3训练环境 - 带奖励稳定机制"""
    
    def __init__(self, system_config=None):
        self.config = OptimizedTD3Config()
        
        # 状态和动作空间
        self.state_space = VECStateSpace(system_config)
        self.action_space = VECActionSpace()
        
        # 环境配置
        self.state_dim = self.state_space.total_dim    # 106维 (🔧修复后+6维)
        self.action_dim = self.action_space.total_dim  # 120维
        
        # 奖励稳定机制
        self.reward_history = deque(maxlen=100)  # 奖励历史
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_smoothing = 0.9  # 平滑系数
        
        # 创建智能体
        from single_agent.td3_optimized_agent import OptimizedTD3Agent
        self.agent = OptimizedTD3Agent(self.state_dim, self.action_dim, self.config)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        self.prev_metrics = None
        
        print(f"✓ 优化TD3环境初始化完成")
        print(f"✓ 状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"✓ 奖励稳定机制已启用")
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """构建状态向量"""
        return self.state_space.encode_state(node_states, system_metrics)
    
    def decompose_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """分解动作"""
        return self.action_space.decompose_action(action)
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        """获取动作"""
        global_action = self.agent.select_action(state, training)
        return self.decompose_action(global_action)
    
    def calculate_reward(self, system_metrics: Dict, prev_metrics: Optional[Dict] = None) -> float:
        """计算奖励 - 基于成本的负奖励 + 卸载激励（引导RSU/UAV卸载）"""
        try:
            # 提取原始指标
            delay = max(system_metrics.get('avg_task_delay', 2.0), 0.1)
            energy = max(system_metrics.get('total_energy_consumption', 600.0), 100.0)
            completion = np.clip(system_metrics.get('task_completion_rate', 0.95), 0.0, 1.0)
            cache_hit = np.clip(system_metrics.get('cache_hit_rate', 0.85), 0.0, 1.0)
            data_loss = system_metrics.get('data_loss_rate', 0.0)
            
            # 🔧 新增：提取卸载比例（引导RSU/UAV卸载）
            rsu_ratio = system_metrics.get('rsu_offload_ratio', 0.0)
            uav_ratio = system_metrics.get('uav_offload_ratio', 0.0)
            local_ratio = system_metrics.get('local_offload_ratio', 1.0)
            
            # 🔧 关键修复：归一化使用合理的基准值
            # 延迟归一化: 以2.5s为基准（12车辆场景的合理延迟）
            delay_norm = delay / 2.5
            
            # 能耗归一化: 以10000J为基准（12车*800J/车 + 余量）
            energy_norm = energy / 10000.0
            
            # 完成率惩罚: 低于98%时额外惩罚
            completion_penalty = max(0, (0.98 - completion)) * 3.0
            
            # 数据丢失惩罚: 按实际比例惩罚
            loss_penalty = data_loss * 2.0
            
            # 缓存命中率奖励: 高命中率减少成本
            cache_bonus = (cache_hit - 0.5) * 0.15  # 超过50%开始有奖励
            
            # 🎉 卸载奖励机制优化：明确区分RSU/UAV,解决偏向UAV/本地的问题
            # 
            # 【核心问题】原设计虽然RSU系数高(25.0),但智能体仍偏向UAV/本地,原因:
            # 1. RSU卸载的累积延迟(上传+队列+处理+下载)可能超过本地/UAV
            # 2. RSU队列容易满载导致拒绝,智能体学到"RSU不可靠"
            # 3. 奖励信号传递效率低,需要更强的RSU偏好引导
            # 
            # 【修复策略】
            # - 极致强化RSU奖励:从25.0提升到50.0 (翻倍)
            # - 降低UAV奖励:从3.0降低到1.5 (避免与RSU竞争)
            # - 增强本地处理惩罚:从10.0提升到15.0 (强制卸载)
            # - 添加RSU优先奖励:额外给予RSU使用率超过50%的奖励
            
            # RSU卸载奖励：每1%获得0.50奖励(极致强化)
            rsu_bonus = rsu_ratio * 50.0  # 50%占比→25.0奖励, 60%→30.0奖励
            
            # RSU优先额外奖励：当RSU占比>50%时,每超1%额外+0.20奖励
            rsu_priority_bonus = max(0, rsu_ratio - 0.5) * 20.0  # 引导>50%占比
            
            # UAV卸载奖励：每1%获得0.15奖励(降低避免竞争)
            uav_bonus = uav_ratio * 1.5  # 50%占比→0.75奖励
            
            # 本地处理惩罚：每1%扣除0.15(强化惩罚)
            local_penalty = local_ratio * 15.0  # 50%占比→扣7.5
            
            # 计算总成本（归一化后的加权和）
            cost = (
                1.2 * delay_norm +           # 延迟成本
                0.8 * energy_norm +          # 能耗成本
                completion_penalty +         # 完成率惩罚
                loss_penalty                 # 数据丢失惩罚
            )
            
            # 🎯 奖励 = -成本 + 全部奖励（极致强化RSU卸载信号）
            # 目标: 让RSU卸载奖励能够压倒延迟/能耗成本,明确优于UAV/本地
            # 
            # 【期望分布对比】
            # ❌ 错误分布(Local 70%, UAV 20%, RSU 10%):
            #    奖励 = -3.0 + 0.15(缓存) + 5.0(RSU) + 0.0(优先) + 0.3(UAV) - 10.5(本地) = -8.05
            # 
            # ✅ 目标分布(RSU 60%, UAV 20%, Local 20%):
            #    奖励 = -3.0 + 0.15(缓存) + 30.0(RSU) + 2.0(优先) + 0.3(UAV) - 3.0(本地) = +26.45
            # 
            # 🎯 最优分布(RSU 70%, UAV 15%, Local 15%):
            #    奖励 = -3.0 + 0.15(缓存) + 35.0(RSU) + 4.0(优先) + 0.225(UAV) - 2.25(本地) = +34.125
            # 
            # 差距: 最优(+34.1) vs 错误(-8.0) = 42.1差距,信号极强!
            reward = -cost + cache_bonus + rsu_bonus + rsu_priority_bonus + uav_bonus - local_penalty
            
            # 裁剪到合理范围（现在可以是很大的正值）
            # 🔧 修复:扩大上限到50.0以容纳RSU极致奖励 (最优分布可达+34)
            reward = np.clip(reward, -10.0, 50.0)  # 扩大范围以容纳更高奖励
            
            return float(reward)
            
        except Exception as e:
            print(f"⚠️ 奖励计算错误: {e}")
            return -2.5  # 默认惩罚值
    
    def train_step(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                   next_state: np.ndarray, done: bool) -> Dict:
        """执行一步训练"""
        # 确保动作是numpy数组
        if isinstance(action, int):
            action = np.array([action], dtype=np.float32)
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 存储经验
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # 更新网络
        training_info = self.agent.update()
        
        self.step_count += 1
        
        return training_info
    
    def save_models(self, filepath: str):
        """保存模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        self.agent.save_model(filepath)
        print(f"✓ 优化TD3模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
        print(f"✓ 优化TD3模型已加载: {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        # 将deque转换为list以支持切片操作
        actor_losses_list = list(self.agent.actor_losses) if self.agent.actor_losses else []
        critic_losses_list = list(self.agent.critic_losses) if self.agent.critic_losses else []
        
        return {
            'actor_loss_avg': float(np.mean(actor_losses_list[-100:])) if actor_losses_list else 0.0,
            'critic_loss_avg': float(np.mean(critic_losses_list[-100:])) if critic_losses_list else 0.0,
            'exploration_noise': self.agent.exploration_noise,
            'buffer_size': len(self.agent.replay_buffer),
            'step_count': self.step_count,
            'update_count': self.agent.update_count,
            'policy_delay': self.config.policy_delay,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
