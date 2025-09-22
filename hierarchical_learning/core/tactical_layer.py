"""
战术层（Tactical Layer）实现
使用MATD3算法进行多智能体协调决策

主要功能：
1. 接收战略层的高层指导
2. 协调多个RSU和UAV之间的资源分配
3. 决定具体的任务分配和资源调度策略
4. 为执行层提供具体的执行指令
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from algorithms.matd3 import MATD3Environment, MATD3Agent
from hierarchical_learning.core.base_layer import BaseLayer
from config import config as global_config


class TacticalLayer(BaseLayer):
    """战术层 - 使用MATD3算法进行多智能体协调决策"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 战术层配置
        self.num_rsus = config.get('num_rsus', 5)
        self.num_uavs = config.get('num_uavs', 3)  # UAV位置固定，但可以调整服务策略
        
        # 战术层状态维度：包括战略指导 + 局部详细信息
        self.tactical_state_dim = config.get('tactical_state_dim', 50)
        
        # 战术层动作维度：具体的资源分配和任务调度决策
        self.tactical_action_dim = config.get('tactical_action_dim', 10)
        
        # 初始化MATD3环境
        self.matd3_env = MATD3Environment()
        
        # 创建智能体：RSU智能体 + UAV智能体
        self.agents = {}
        
        # RSU智能体 - 优化配置（暂时备份原始配置并应用优化）
        for i in range(self.num_rsus):
            agent_id = f"rsu_{i}"
            
            # 备份原始配置
            original_config = {
                'actor_lr': global_config.rl.actor_lr,
                'critic_lr': global_config.rl.critic_lr,
                'gamma': global_config.rl.gamma,
                'tau': global_config.rl.tau,
                'batch_size': global_config.rl.batch_size,
                'hidden_dim': global_config.rl.hidden_dim
            }
            
            # 临时应用优化配置
            global_config.rl.actor_lr = config.get('tactical_actor_lr', 8e-5)
            global_config.rl.critic_lr = config.get('tactical_critic_lr', 1e-4)
            global_config.rl.gamma = config.get('tactical_gamma', 0.995)
            global_config.rl.tau = config.get('tactical_tau', 0.003)
            global_config.rl.batch_size = config.get('tactical_batch_size', 64)
            global_config.rl.hidden_dim = config.get('tactical_hidden_dim', 384)
            
            # 创建智能体
            self.agents[agent_id] = MATD3Agent(
                agent_id=agent_id,
                state_dim=self.tactical_state_dim,
                action_dim=self.tactical_action_dim
            )
            
            # 恢复原始配置
            for key, value in original_config.items():
                setattr(global_config.rl, key, value)
        
        # UAV智能体（位置固定，但可以调整服务策略）- 使用相同的优化配置
        for i in range(self.num_uavs):
            agent_id = f"uav_{i}"
            
            # 备份原始配置
            original_config = {
                'actor_lr': global_config.rl.actor_lr,
                'critic_lr': global_config.rl.critic_lr,
                'gamma': global_config.rl.gamma,
                'tau': global_config.rl.tau,
                'batch_size': global_config.rl.batch_size,
                'hidden_dim': global_config.rl.hidden_dim
            }
            
            # 临时应用优化配置
            global_config.rl.actor_lr = config.get('tactical_actor_lr', 8e-5)
            global_config.rl.critic_lr = config.get('tactical_critic_lr', 1e-4)
            global_config.rl.gamma = config.get('tactical_gamma', 0.995)
            global_config.rl.tau = config.get('tactical_tau', 0.003)
            global_config.rl.batch_size = config.get('tactical_batch_size', 64)
            global_config.rl.hidden_dim = config.get('tactical_hidden_dim', 384)
            
            # 创建智能体
            self.agents[agent_id] = MATD3Agent(
                agent_id=agent_id,
                state_dim=self.tactical_state_dim,
                action_dim=self.tactical_action_dim
            )
            
            # 恢复原始配置
            for key, value in original_config.items():
                setattr(global_config.rl, key, value)
        
        # 战略指导缓存
        self.strategic_guidance = {
            'offloading_priority': 0.5,
            'caching_priority': 0.5,
            'resource_allocation_strategy': 0.5,
            'system_priority': 0.5
        }
        
        # 协调历史
        self.coordination_history = []
        
    def process_state(self, raw_state: Dict) -> Dict[str, np.ndarray]:
        """
        处理原始状态，结合战略指导生成各智能体的战术状态
        
        Args:
            raw_state: 包含环境状态和战略指导的字典
            
        Returns:
            tactical_states: 各智能体的战术状态字典
        """
        tactical_states = {}
        
        # 更新战略指导
        if 'strategic_guidance' in raw_state:
            self.strategic_guidance.update(raw_state['strategic_guidance'])
        
        # 为每个RSU生成战术状态
        for i in range(self.num_rsus):
            agent_id = f"rsu_{i}"
            tactical_states[agent_id] = self._generate_rsu_tactical_state(
                raw_state, i, self.strategic_guidance
            )
        
        # 为每个UAV生成战术状态
        for i in range(self.num_uavs):
            agent_id = f"uav_{i}"
            tactical_states[agent_id] = self._generate_uav_tactical_state(
                raw_state, i, self.strategic_guidance
            )
        
        return tactical_states
    
    def _generate_rsu_tactical_state(self, raw_state: Dict, rsu_id: int, 
                                   strategic_guidance: Dict) -> np.ndarray:
        """为RSU生成战术状态向量"""
        tactical_features = []
        
        # 1. 战略指导信息（4维）
        tactical_features.extend([
            strategic_guidance['offloading_priority'],
            strategic_guidance['caching_priority'],
            strategic_guidance['resource_allocation_strategy'],
            strategic_guidance['system_priority']
        ])
        
        # 2. 当前RSU的详细状态（10维）
        if 'rsus' in raw_state and rsu_id < len(raw_state['rsus']):
            rsu = raw_state['rsus'][rsu_id]
            tactical_features.extend([
                rsu.get('cpu_usage', 0.0),
                rsu.get('memory_usage', 0.0),
                rsu.get('network_load', 0.0),
                rsu.get('available_compute', 0.0) / 1000.0,  # 归一化
                rsu.get('queue_length', 0.0) / 10.0,  # 归一化
                rsu.get('energy_consumption', 0.0) / 100.0,  # 归一化
                rsu.get('coverage_vehicles', 0.0) / 20.0,  # 归一化
                rsu.get('success_rate', 0.0),
                rsu.get('avg_latency', 0.0) / 100.0,  # 归一化
                rsu.get('bandwidth_utilization', 0.0)
            ])
        else:
            tactical_features.extend([0.0] * 10)
        
        # 3. 邻近RSU状态（简化，5维）
        neighbor_features = []
        if 'rsus' in raw_state:
            rsus = raw_state['rsus']
            for j, rsu in enumerate(rsus):
                if j != rsu_id:  # 排除自己
                    neighbor_features.extend([
                        rsu.get('cpu_usage', 0.0),
                        rsu.get('available_compute', 0.0) / 1000.0
                    ])
                if len(neighbor_features) >= 10:  # 最多5个邻居
                    break
        
        while len(neighbor_features) < 10:
            neighbor_features.append(0.0)
        tactical_features.extend(neighbor_features[:10])
        
        # 4. UAV状态信息（9维）
        if 'uavs' in raw_state:
            uavs = raw_state['uavs']
            for i, uav in enumerate(uavs[:3]):  # 最多3个UAV
                tactical_features.extend([
                    uav.get('cpu_usage', 0.0),
                    uav.get('available_compute', 0.0) / 500.0,  # 归一化
                    uav.get('coverage_efficiency', 0.0)
                ])
            
            # 填充不足的UAV信息
            while len(tactical_features) < 4 + 10 + 10 + 9:
                tactical_features.append(0.0)
        else:
            tactical_features.extend([0.0] * 9)
        
        # 5. 车辆需求信息（12维）
        if 'vehicles' in raw_state:
            vehicles = raw_state['vehicles']
            
            # 计算该RSU覆盖范围内的车辆统计
            covered_vehicles = [v for v in vehicles if v.get('serving_rsu', -1) == rsu_id]
            
            if covered_vehicles:
                tactical_features.extend([
                    len(covered_vehicles) / 20.0,  # 覆盖车辆数量
                    np.mean([v.get('compute_demand', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('latency_requirement', 0) for v in covered_vehicles]) / 100.0,
                    np.mean([v.get('priority', 0) for v in covered_vehicles]),
                    np.sum([v.get('data_size', 0) for v in covered_vehicles]) / 10000.0,
                    np.mean([v.get('velocity', 0) for v in covered_vehicles]) / 30.0
                ])
            else:
                tactical_features.extend([0.0] * 6)
            
            # 全局车辆统计
            tactical_features.extend([
                len(vehicles) / 100.0,  # 总车辆数
                np.mean([v.get('compute_demand', 0) for v in vehicles]) / 1000.0,
                np.mean([v.get('latency_requirement', 0) for v in vehicles]) / 100.0,
                np.mean([v.get('priority', 0) for v in vehicles]),
                np.sum([v.get('data_size', 0) for v in vehicles]) / 50000.0,
                np.mean([v.get('velocity', 0) for v in vehicles]) / 30.0
            ])
        else:
            tactical_features.extend([0.0] * 12)
        
        # 6. 系统性能指标（5维）
        if 'system_metrics' in raw_state:
            metrics = raw_state['system_metrics']
            tactical_features.extend([
                metrics.get('avg_latency', 0.0) / 100.0,
                metrics.get('energy_efficiency', 0.0),
                metrics.get('success_rate', 0.0),
                metrics.get('network_utilization', 0.0),
                metrics.get('load_balance_index', 0.0)
            ])
        else:
            tactical_features.extend([0.0] * 5)
        
        # 确保特征向量长度正确
        while len(tactical_features) < self.tactical_state_dim:
            tactical_features.append(0.0)
        
        tactical_features = tactical_features[:self.tactical_state_dim]
        
        return np.array(tactical_features, dtype=np.float32)
    
    def _generate_uav_tactical_state(self, raw_state: Dict, uav_id: int, 
                                   strategic_guidance: Dict) -> np.ndarray:
        """为UAV生成战术状态向量（考虑固定位置特性）"""
        tactical_features = []
        
        # 1. 战略指导信息（4维）
        tactical_features.extend([
            strategic_guidance['offloading_priority'],
            strategic_guidance['caching_priority'],
            strategic_guidance['resource_allocation_strategy'],
            strategic_guidance['system_priority']
        ])
        
        # 2. 当前UAV的详细状态（8维）- 不包括位置信息，因为位置固定
        if 'uavs' in raw_state and uav_id < len(raw_state['uavs']):
            uav = raw_state['uavs'][uav_id]
            tactical_features.extend([
                uav.get('cpu_usage', 0.0),
                uav.get('memory_usage', 0.0),
                uav.get('available_compute', 0.0) / 500.0,  # 归一化
                uav.get('energy_level', 1.0),  # 能量水平
                uav.get('coverage_efficiency', 0.0),
                uav.get('served_vehicles', 0.0) / 15.0,  # 归一化
                uav.get('queue_length', 0.0) / 5.0,  # 归一化
                uav.get('communication_load', 0.0)
            ])
        else:
            tactical_features.extend([0.0] * 8)
        
        # 3. 其他UAV状态（6维）
        if 'uavs' in raw_state:
            uavs = raw_state['uavs']
            other_uav_features = []
            for j, uav in enumerate(uavs):
                if j != uav_id:  # 排除自己
                    other_uav_features.extend([
                        uav.get('cpu_usage', 0.0),
                        uav.get('available_compute', 0.0) / 500.0
                    ])
                if len(other_uav_features) >= 6:  # 最多3个其他UAV
                    break
            
            while len(other_uav_features) < 6:
                other_uav_features.append(0.0)
            tactical_features.extend(other_uav_features[:6])
        else:
            tactical_features.extend([0.0] * 6)
        
        # 4. RSU协作信息（15维）
        if 'rsus' in raw_state:
            rsus = raw_state['rsus']
            rsu_features = []
            for rsu in rsus[:5]:  # 最多5个RSU
                rsu_features.extend([
                    rsu.get('cpu_usage', 0.0),
                    rsu.get('available_compute', 0.0) / 1000.0,
                    rsu.get('network_load', 0.0)
                ])
            
            while len(rsu_features) < 15:
                rsu_features.append(0.0)
            tactical_features.extend(rsu_features[:15])
        else:
            tactical_features.extend([0.0] * 15)
        
        # 5. 覆盖区域车辆信息（12维）
        if 'vehicles' in raw_state:
            vehicles = raw_state['vehicles']
            
            # UAV覆盖范围内的车辆（假设UAV有更大的覆盖范围）
            covered_vehicles = [v for v in vehicles if v.get('in_uav_coverage', False)]
            
            if covered_vehicles:
                tactical_features.extend([
                    len(covered_vehicles) / 30.0,  # UAV覆盖更多车辆
                    np.mean([v.get('compute_demand', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('latency_requirement', 0) for v in covered_vehicles]) / 100.0,
                    np.mean([v.get('priority', 0) for v in covered_vehicles]),
                    np.sum([v.get('data_size', 0) for v in covered_vehicles]) / 15000.0,
                    np.mean([v.get('velocity', 0) for v in covered_vehicles]) / 30.0
                ])
            else:
                tactical_features.extend([0.0] * 6)
            
            # 全局车辆统计
            tactical_features.extend([
                len(vehicles) / 100.0,
                np.mean([v.get('compute_demand', 0) for v in vehicles]) / 1000.0,
                np.mean([v.get('latency_requirement', 0) for v in vehicles]) / 100.0,
                np.mean([v.get('priority', 0) for v in vehicles]),
                np.sum([v.get('data_size', 0) for v in vehicles]) / 50000.0,
                np.mean([v.get('velocity', 0) for v in vehicles]) / 30.0
            ])
        else:
            tactical_features.extend([0.0] * 12)
        
        # 6. 系统性能指标（5维）
        if 'system_metrics' in raw_state:
            metrics = raw_state['system_metrics']
            tactical_features.extend([
                metrics.get('avg_latency', 0.0) / 100.0,
                metrics.get('energy_efficiency', 0.0),
                metrics.get('success_rate', 0.0),
                metrics.get('network_utilization', 0.0),
                metrics.get('load_balance_index', 0.0)
            ])
        else:
            tactical_features.extend([0.0] * 5)
        
        # 确保特征向量长度正确
        while len(tactical_features) < self.tactical_state_dim:
            tactical_features.append(0.0)
        
        tactical_features = tactical_features[:self.tactical_state_dim]
        
        return np.array(tactical_features, dtype=np.float32)
    
    def get_action(self, processed_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        根据各智能体的战术状态生成协调动作
        
        Args:
            processed_states: 各智能体的战术状态字典
            
        Returns:
            tactical_actions: 各智能体的战术动作字典
        """
        tactical_actions = {}
        
        # 为每个智能体生成动作
        for agent_id, state in processed_states.items():
            if agent_id in self.agents:
                action = self.agents[agent_id].select_action(state, add_noise=True)
                tactical_actions[agent_id] = self._post_process_tactical_action(action, agent_id)
        
        # 记录协调历史
        self.coordination_history.append({
            'states': {k: v.copy() for k, v in processed_states.items()},
            'actions': {k: v.copy() for k, v in tactical_actions.items()},
            'timestamp': len(self.coordination_history)
        })
        
        return tactical_actions
    
    def _post_process_tactical_action(self, raw_action: np.ndarray, agent_id: str) -> np.ndarray:
        """
        对原始动作进行后处理，确保符合战术决策的语义
        
        Args:
            raw_action: 原始动作向量
            agent_id: 智能体ID
            
        Returns:
            processed_action: 处理后的战术动作
        """
        processed_action = raw_action.copy()
        
        if agent_id.startswith('rsu'):
            # RSU动作后处理
            # [任务接受率, 计算资源分配, 缓存策略, 协作意愿, 优先级调整, ...]
            processed_action[0] = torch.sigmoid(torch.tensor(raw_action[0])).item()  # 任务接受率 [0,1]
            processed_action[1] = torch.sigmoid(torch.tensor(raw_action[1])).item()  # 计算资源分配 [0,1]
            processed_action[2] = torch.sigmoid(torch.tensor(raw_action[2])).item()  # 缓存策略 [0,1]
            processed_action[3] = torch.sigmoid(torch.tensor(raw_action[3])).item()  # 协作意愿 [0,1]
            
        elif agent_id.startswith('uav'):
            # UAV动作后处理（考虑位置固定）
            # [服务策略, 计算资源分配, 协作模式, 覆盖优化, 能耗管理, ...]
            processed_action[0] = torch.sigmoid(torch.tensor(raw_action[0])).item()  # 服务策略 [0,1]
            processed_action[1] = torch.sigmoid(torch.tensor(raw_action[1])).item()  # 计算资源分配 [0,1]
            processed_action[2] = torch.sigmoid(torch.tensor(raw_action[2])).item()  # 协作模式 [0,1]
            processed_action[3] = torch.sigmoid(torch.tensor(raw_action[3])).item()  # 覆盖优化 [0,1]
            # 注意：不包括位置调整，因为UAV位置固定
        
        # 其余动作维度保持原样或进行适当的范围限制
        for i in range(4, len(processed_action)):
            processed_action[i] = np.clip(processed_action[i], -1.0, 1.0)
        
        return processed_action
    
    def train(self, replay_buffer=None) -> Dict[str, float]:
        """
        训练战术层MATD3模型
        
        Args:
            replay_buffer: 经验回放缓冲区（可选）
            
        Returns:
            training_stats: 训练统计信息
        """
        training_stats = {}
        
        # 训练每个智能体
        for agent_id, agent in self.agents.items():
            batch_size = getattr(agent, 'optimized_batch_size', getattr(agent, 'batch_size', 32))
            if len(agent.replay_buffer) >= batch_size:
                agent_stats = agent.train()
                training_stats[agent_id] = agent_stats
        
        return training_stats
    
    def store_experience(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                        rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                        dones: Dict[str, bool]):
        """存储多智能体经验"""
        for agent_id in self.agents.keys():
            if agent_id in states and agent_id in actions and agent_id in rewards:
                self.agents[agent_id].store_transition(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_states[agent_id],
                    dones[agent_id]
                )
    
    def save_model(self, path: str):
        """保存战术层模型"""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}_{agent_id}"
            agent.save_model(agent_path)
    
    def load_model(self, path: str):
        """加载战术层模型"""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}_{agent_id}"
            try:
                agent.load_model(agent_path)
            except FileNotFoundError:
                print(f"Warning: Model file not found for agent {agent_id}")
    
    def update_strategic_guidance(self, strategic_guidance: Dict[str, float]):
        """更新来自战略层的指导信息"""
        self.strategic_guidance.update(strategic_guidance)
    
    def get_tactical_instructions(self) -> Dict[str, Dict]:
        """
        获取当前的战术指令，供执行层使用
        
        Returns:
            instructions: 战术指令字典
        """
        if len(self.coordination_history) == 0:
            return {}
        
        latest_actions = self.coordination_history[-1]['actions']
        instructions = {}
        
        for agent_id, action in latest_actions.items():
            if agent_id.startswith('rsu'):
                instructions[agent_id] = {
                    'task_acceptance_rate': action[0],
                    'compute_allocation': action[1],
                    'caching_strategy': action[2],
                    'cooperation_willingness': action[3],
                    'priority_adjustment': action[4] if len(action) > 4 else 0.0
                }
            elif agent_id.startswith('uav'):
                instructions[agent_id] = {
                    'service_strategy': action[0],
                    'compute_allocation': action[1],
                    'cooperation_mode': action[2],
                    'coverage_optimization': action[3],
                    'energy_management': action[4] if len(action) > 4 else 0.0
                }
        
        return instructions
    
    def get_layer_stats(self) -> Dict[str, float]:
        """获取战术层统计信息"""
        if len(self.coordination_history) == 0:
            return {}
        
        stats = {
            'coordination_count': len(self.coordination_history),
            'num_agents': len(self.agents),
            'num_rsus': self.num_rsus,
            'num_uavs': self.num_uavs
        }
        
        # 添加各智能体的训练统计
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'replay_buffer'):
                stats[f'{agent_id}_buffer_size'] = len(agent.replay_buffer)
        
        return stats