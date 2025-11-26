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
    actor_lr: float = 1e-4       # 降低学习率，避免过快收敛（从3e-4到1e-4）
    critic_lr: float = 1e-4      # 降低学习率，避免过快收敛
    
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
    exploration_noise: float = 0.3   # 增加初始探索（从0.2到0.3）
    noise_decay: float = 0.9990      # 加快衰减速度（从0.9995到0.9990）
    min_noise: float = 0.02          # 保留最小探索（从0.05降到0.02)
    
    # 训练控制
    warmup_steps: int = 20000        # 增加预热步数,约100个episode
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
        self.vehicle_state_dim = 5  # 位置x,y + 速度x,y + 队列利用率
        self.rsu_state_dim = 5      # CPU利用率 + 队列利用率 + 缓存利用率 + 能耗 + 🔧资源容量
        self.uav_state_dim = 5      # CPU利用率 + 队列利用率 + 电池电量 + 能耗 + 🔧资源容量
        self.global_state_dim = 16  # 全局系统指标（基础8维 + 任务类型8维）
        
        self.total_dim = (
            self.num_vehicles * self.vehicle_state_dim +  # 12 * 5 = 60
            self.num_rsus * self.rsu_state_dim +          # 4 * 5 = 20
            self.num_uavs * self.uav_state_dim +          # 2 * 5 = 10
            self.global_state_dim                         # 16
        )  # 总计：106维 (🔧修复前100维)
    
    def encode_state(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """构建符合论文的VEC系统状态向量"""
        state_components = []
        
        # 1. 车辆状态 (12车辆 × 5维 = 60维)
        for i in range(self.num_vehicles):
            vehicle_id = f'vehicle_{i}'
            if vehicle_id in node_states:
                vehicle = node_states[vehicle_id]
                vehicle_state = [
                    getattr(vehicle.position, 'x', 0.0) / 2000.0,  # 归一化位置
                    getattr(vehicle.position, 'y', 0.0) / 2000.0,
                    getattr(vehicle, 'velocity_x', 0.0) / 30.0,    # 归一化速度
                    getattr(vehicle, 'velocity_y', 0.0) / 30.0,
                    getattr(vehicle, 'queue_utilization', 0.5),    # 队列利用率
                ]
            else:
                # 默认状态
                vehicle_state = [0.5, 0.5, 0.0, 0.0, 0.5]
            state_components.extend(vehicle_state)
        
        # 2. RSU状态 (按配置数量 × 5维)
        for i in range(self.num_rsus):
            rsu_id = f'rsu_{i}'
            if rsu_id in node_states:
                rsu = node_states[rsu_id]
                # 🔧 修复：添加cpu_frequency让智能体知道RSU容量优势
                cpu_freq = getattr(rsu, 'cpu_frequency', 12.5e9)  # 默认12.5 GHz
                rsu_state = [
                    getattr(rsu, 'cpu_utilization', 0.5),         # CPU利用率
                    getattr(rsu, 'queue_utilization', 0.5),       # 队列利用率
                    getattr(rsu, 'cache_utilization', 0.5),       # 缓存利用率
                    getattr(rsu, 'energy_consumption', 500.0) / 1000.0,  # 归一化能耗
                    cpu_freq / 20e9,  # 🔧 新增：归一化CPU容量（20GHz最大值）
                ]
            else:
                rsu_state = [0.5, 0.5, 0.5, 0.5, 0.625]  # 默认12.5/20=0.625
            state_components.extend(rsu_state)
        
        # 3. UAV状态 (按配置数量 × 5维)
        for i in range(self.num_uavs):
            uav_id = f'uav_{i}'
            if uav_id in node_states:
                uav = node_states[uav_id]
                # 🔧 修复：添加cpu_frequency让智能体知道UAV容量较弱
                cpu_freq = getattr(uav, 'cpu_frequency', 5.0e9)  # 默认5.0 GHz
                uav_state = [
                    getattr(uav, 'cpu_utilization', 0.5),
                    getattr(uav, 'queue_utilization', 0.5),
                    getattr(uav, 'battery_level', 0.8),           # 电池电量
                    getattr(uav, 'energy_consumption', 50.0) / 100.0,
                    cpu_freq / 20e9,  # 🔧 新增：归一化CPU容量（20GHz最大值）
                ]
            else:
                uav_state = [0.5, 0.5, 0.8, 0.5, 0.25]  # 默认5.0/20=0.25
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
        """计算奖励 - 修复版本(基于成本的负奖励)"""
        try:
            # 提取原始指标
            delay = max(system_metrics.get('avg_task_delay', 2.0), 0.1)
            energy = max(system_metrics.get('total_energy_consumption', 600.0), 100.0)
            completion = np.clip(system_metrics.get('task_completion_rate', 0.95), 0.0, 1.0)
            cache_hit = np.clip(system_metrics.get('cache_hit_rate', 0.85), 0.0, 1.0)
            data_loss = system_metrics.get('data_loss_rate', 0.0)
            
            # 使用实际值范围进行归一化(基于观察到的训练数据)
            # 延迟: 0.5-4.0s -> 归一化到[0, 1]
            delay_norm = np.clip((delay - 0.5) / 3.5, 0.0, 1.0)
            
            # 能耗: 500-3000J -> 归一化到[0, 1] (修正为实际范围)
            energy_norm = np.clip((energy - 500.0) / 2500.0, 0.0, 1.0)
            
            # 数据损失率: 0-0.6 -> 归一化到[0, 1]
            loss_norm = np.clip(data_loss / 0.6, 0.0, 1.0)
            
            # 完成率惩罚: 低于95%时给予额外惩罚
            completion_penalty = 0.0
            if completion < 0.95:
                completion_penalty = (0.95 - completion) * 5.0  # 每降低1%惩罚0.05
            
            # 缓存命中率: 0.0-0.8 -> 归一化到[0, 1]
            cache_norm = np.clip(cache_hit / 0.8, 0.0, 1.0)
            cache_bonus = (cache_norm - 0.5) * 0.2  # 超过40%才有奖励,否则惩罚
            
            # 计算总成本(全是惩罚项)
            cost = (
                1.5 * delay_norm +           # 延迟成本
                1.0 * energy_norm +          # 能耗成本
                2.0 * loss_norm +            # 数据损失成本(最重要)
                completion_penalty           # 完成率惩罚
            )
            
            # 奖励 = -成本 + 小额缓存奖励
            reward = -cost + cache_bonus
            
            # 裁剪到合理范围(永远是负数或接近0)
            reward = np.clip(reward, -5.0, 0.1)
            
            # 除以每个episode步数,得到per-step奖励
            reward = reward / 200.0  # max_steps_per_episode=200
            
            return float(reward)
            
        except Exception as e:
            print(f"⚠️ 奖励计算错误: {e}")
            return -0.025  # 默认惩罚值
    
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
