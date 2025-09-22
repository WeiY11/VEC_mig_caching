"""
执行层（Operational Layer）实现
使用TD3/DDPG算法进行底层动作执行

主要功能：
1. 接收战术层的具体指令
2. 执行底层控制动作（功率控制、资源分配等）
3. 与环境直接交互
4. 提供精确的连续控制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from single_agent.td3 import TD3Environment, TD3Agent, TD3Config
from single_agent.ddpg import DDPGEnvironment, DDPGAgent, DDPGConfig
from hierarchical_learning.core.base_layer import BaseLayer


class OperationalLayer(BaseLayer):
    """执行层 - 使用TD3/DDPG算法进行底层动作执行"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 执行层配置
        self.algorithm = config.get('operational_algorithm', 'TD3')  # 'TD3' 或 'DDPG'
        self.num_rsus = config.get('num_rsus', 5)
        self.num_uavs = config.get('num_uavs', 3)
        self.num_vehicles = config.get('num_vehicles', 50)
        
        # 执行层状态维度：包括战术指令 + 底层环境状态
        self.operational_state_dim = config.get('operational_state_dim', 60)
        
        # 执行层动作维度：具体的控制动作
        self.operational_action_dim = config.get('operational_action_dim', 15)
        
        # 创建执行智能体
        self.agents = {}
        
        if self.algorithm == 'TD3':
            # 使用TD3算法 - 全面优化
            td3_config = TD3Config()
            td3_config.batch_size = config.get('batch_size', 64)        # 降低批次大小，加速更新
            td3_config.hidden_dim = config.get('hidden_dim', 384)       # 增大网络容量
            td3_config.actor_lr = config.get('actor_lr', 6e-5)          # 优化学习率
            td3_config.critic_lr = config.get('critic_lr', 8e-5)        # 稍高的critic学习率
            td3_config.gamma = config.get('gamma', 0.99)                # 适中的折扣因子（执行层关注短期）
            td3_config.tau = config.get('tau', 0.005)                   # 适中的软更新率
            td3_config.policy_noise = config.get('policy_noise', 0.2)   # 适中的策略噪声
            td3_config.noise_clip = config.get('noise_clip', 0.5)       # 适中的噪声裁剪
            td3_config.policy_delay = config.get('policy_delay', 2)     # 策略延迟更新
            td3_config.buffer_size = config.get('buffer_size', 80000)   # 增大缓冲区
            td3_config.warmup_steps = config.get('warmup_steps', 800)   # 降低预热步数
            
            # 为每个RSU创建TD3智能体
            for i in range(self.num_rsus):
                agent_id = f"rsu_{i}"
                self.agents[agent_id] = TD3Agent(
                    state_dim=self.operational_state_dim,
                    action_dim=self.operational_action_dim,
                    config=td3_config
                )
            
            # 为每个UAV创建TD3智能体（考虑位置固定）
            for i in range(self.num_uavs):
                agent_id = f"uav_{i}"
                # UAV的动作维度可能不同（不包括位置控制）
                uav_action_dim = self.operational_action_dim - 3  # 减去位置控制维度
                self.agents[agent_id] = TD3Agent(
                    state_dim=self.operational_state_dim,
                    action_dim=uav_action_dim,
                    config=td3_config
                )
                
        else:  # DDPG
            # 使用DDPG算法
            ddpg_config = DDPGConfig()
            ddpg_config.batch_size = config.get('batch_size', 128)
            ddpg_config.hidden_dim = config.get('hidden_dim', 256)
            ddpg_config.actor_lr = config.get('actor_lr', 1e-4)
            ddpg_config.critic_lr = config.get('critic_lr', 3e-4)
            
            # 为每个RSU创建DDPG智能体
            for i in range(self.num_rsus):
                agent_id = f"rsu_{i}"
                self.agents[agent_id] = DDPGAgent(
                    state_dim=self.operational_state_dim,
                    action_dim=self.operational_action_dim,
                    config=ddpg_config
                )
            
            # 为每个UAV创建DDPG智能体
            for i in range(self.num_uavs):
                agent_id = f"uav_{i}"
                uav_action_dim = self.operational_action_dim - 3
                self.agents[agent_id] = DDPGAgent(
                    state_dim=self.operational_state_dim,
                    action_dim=uav_action_dim,
                    config=ddpg_config
                )
        
        # 战术指令缓存
        self.tactical_instructions = {}
        
        # 执行历史
        self.execution_history = []
        
        # 环境交互接口
        self.env_interface = self._create_env_interface()
        
    def _create_env_interface(self):
        """创建环境交互接口"""
        if self.algorithm == 'TD3':
            return TD3Environment()
        else:
            return DDPGEnvironment()
    
    def process_state(self, raw_state: Dict) -> Dict[str, np.ndarray]:
        """
        处理原始状态，结合战术指令生成各智能体的执行状态
        
        Args:
            raw_state: 包含环境状态和战术指令的字典
            
        Returns:
            operational_states: 各智能体的执行状态字典
        """
        operational_states = {}
        
        # 更新战术指令
        if 'tactical_instructions' in raw_state:
            self.tactical_instructions.update(raw_state['tactical_instructions'])
        
        # 为每个RSU生成执行状态
        for i in range(self.num_rsus):
            agent_id = f"rsu_{i}"
            operational_states[agent_id] = self._generate_rsu_operational_state(
                raw_state, i, self.tactical_instructions.get(agent_id, {})
            )
        
        # 为每个UAV生成执行状态
        for i in range(self.num_uavs):
            agent_id = f"uav_{i}"
            operational_states[agent_id] = self._generate_uav_operational_state(
                raw_state, i, self.tactical_instructions.get(agent_id, {})
            )
        
        return operational_states
    
    def _generate_rsu_operational_state(self, raw_state: Dict, rsu_id: int, 
                                      tactical_instruction: Dict) -> np.ndarray:
        """为RSU生成执行状态向量"""
        operational_features = []
        
        # 1. 战术指令信息（5维）
        operational_features.extend([
            tactical_instruction.get('task_acceptance_rate', 0.5),
            tactical_instruction.get('compute_allocation', 0.5),
            tactical_instruction.get('caching_strategy', 0.5),
            tactical_instruction.get('cooperation_willingness', 0.5),
            tactical_instruction.get('priority_adjustment', 0.0)
        ])
        
        # 2. RSU硬件状态（10维）
        if 'rsus' in raw_state and rsu_id < len(raw_state['rsus']):
            rsu = raw_state['rsus'][rsu_id]
            operational_features.extend([
                rsu.get('cpu_usage', 0.0),
                rsu.get('memory_usage', 0.0),
                rsu.get('storage_usage', 0.0),
                rsu.get('network_bandwidth_usage', 0.0),
                rsu.get('power_consumption', 0.0) / 1000.0,  # 归一化
                rsu.get('temperature', 25.0) / 100.0,  # 归一化
                rsu.get('available_compute', 0.0) / 1000.0,
                rsu.get('available_memory', 0.0) / 1000.0,
                rsu.get('available_storage', 0.0) / 1000.0,
                rsu.get('available_bandwidth', 0.0) / 100.0
            ])
        else:
            operational_features.extend([0.0] * 10)
        
        # 3. 当前服务的车辆详细信息（15维）
        if 'vehicles' in raw_state:
            vehicles = raw_state['vehicles']
            served_vehicles = [v for v in vehicles if v.get('serving_rsu', -1) == rsu_id]
            
            if served_vehicles:
                # 车辆统计信息
                operational_features.extend([
                    len(served_vehicles) / 20.0,  # 服务车辆数量
                    np.mean([v.get('distance_to_rsu', 0) for v in served_vehicles]) / 1000.0,
                    np.mean([v.get('signal_strength', 0) for v in served_vehicles]),
                    np.mean([v.get('data_rate', 0) for v in served_vehicles]) / 100.0,
                    np.mean([v.get('latency', 0) for v in served_vehicles]) / 100.0
                ])
                
                # 任务特征
                operational_features.extend([
                    np.mean([v.get('compute_demand', 0) for v in served_vehicles]) / 1000.0,
                    np.mean([v.get('data_size', 0) for v in served_vehicles]) / 1000.0,
                    np.mean([v.get('deadline', 0) for v in served_vehicles]) / 100.0,
                    np.mean([v.get('priority', 0) for v in served_vehicles]),
                    np.sum([v.get('energy_cost', 0) for v in served_vehicles]) / 1000.0
                ])
                
                # 移动性信息
                operational_features.extend([
                    np.mean([v.get('velocity', 0) for v in served_vehicles]) / 30.0,
                    np.mean([v.get('direction', 0) for v in served_vehicles]) / 360.0,
                    np.std([v.get('velocity', 0) for v in served_vehicles]) / 10.0,
                    np.mean([v.get('predicted_stay_time', 0) for v in served_vehicles]) / 60.0,
                    np.mean([v.get('handover_probability', 0) for v in served_vehicles])
                ])
            else:
                operational_features.extend([0.0] * 15)
        else:
            operational_features.extend([0.0] * 15)
        
        # 4. 网络状态（10维）
        if 'network_state' in raw_state:
            network = raw_state['network_state']
            operational_features.extend([
                network.get('channel_quality', 0.0),
                network.get('interference_level', 0.0),
                network.get('congestion_level', 0.0),
                network.get('packet_loss_rate', 0.0),
                network.get('throughput', 0.0) / 1000.0,
                network.get('jitter', 0.0) / 10.0,
                network.get('rtt', 0.0) / 100.0,
                network.get('bandwidth_utilization', 0.0),
                network.get('error_rate', 0.0),
                network.get('retransmission_rate', 0.0)
            ])
        else:
            operational_features.extend([0.0] * 10)
        
        # 5. 邻居RSU协作信息（10维）
        if 'rsus' in raw_state:
            rsus = raw_state['rsus']
            neighbor_features = []
            for j, rsu in enumerate(rsus):
                if j != rsu_id and len(neighbor_features) < 10:
                    neighbor_features.extend([
                        rsu.get('cpu_usage', 0.0),
                        rsu.get('available_compute', 0.0) / 1000.0
                    ])
            
            while len(neighbor_features) < 10:
                neighbor_features.append(0.0)
            operational_features.extend(neighbor_features[:10])
        else:
            operational_features.extend([0.0] * 10)
        
        # 6. 系统实时指标（10维）
        if 'real_time_metrics' in raw_state:
            metrics = raw_state['real_time_metrics']
            operational_features.extend([
                metrics.get('current_latency', 0.0) / 100.0,
                metrics.get('current_throughput', 0.0) / 1000.0,
                metrics.get('current_energy', 0.0) / 1000.0,
                metrics.get('current_success_rate', 0.0),
                metrics.get('current_queue_length', 0.0) / 20.0,
                metrics.get('current_load', 0.0),
                metrics.get('current_efficiency', 0.0),
                metrics.get('current_cost', 0.0) / 100.0,
                metrics.get('current_reliability', 0.0),
                metrics.get('current_availability', 0.0)
            ])
        else:
            operational_features.extend([0.0] * 10)
        
        # 确保特征向量长度正确
        while len(operational_features) < self.operational_state_dim:
            operational_features.append(0.0)
        
        operational_features = operational_features[:self.operational_state_dim]
        
        return np.array(operational_features, dtype=np.float32)
    
    def _generate_uav_operational_state(self, raw_state: Dict, uav_id: int, 
                                      tactical_instruction: Dict) -> np.ndarray:
        """为UAV生成执行状态向量（考虑位置固定特性）"""
        operational_features = []
        
        # 1. 战术指令信息（5维）- 不包括位置控制
        operational_features.extend([
            tactical_instruction.get('service_strategy', 0.5),
            tactical_instruction.get('compute_allocation', 0.5),
            tactical_instruction.get('cooperation_mode', 0.5),
            tactical_instruction.get('coverage_optimization', 0.5),
            tactical_instruction.get('energy_management', 0.5)
        ])
        
        # 2. UAV硬件状态（8维）- 不包括位置信息
        if 'uavs' in raw_state and uav_id < len(raw_state['uavs']):
            uav = raw_state['uavs'][uav_id]
            operational_features.extend([
                uav.get('cpu_usage', 0.0),
                uav.get('memory_usage', 0.0),
                uav.get('available_compute', 0.0) / 500.0,
                uav.get('energy_level', 1.0),
                uav.get('communication_power', 0.0) / 100.0,
                uav.get('computation_power', 0.0) / 100.0,
                uav.get('antenna_gain', 0.0) / 10.0,
                uav.get('signal_processing_capability', 0.0)
            ])
        else:
            operational_features.extend([0.0] * 8)
        
        # 3. 覆盖区域车辆详细信息（18维）
        if 'vehicles' in raw_state:
            vehicles = raw_state['vehicles']
            covered_vehicles = [v for v in vehicles if v.get('in_uav_coverage', False)]
            
            if covered_vehicles:
                # 车辆统计信息
                operational_features.extend([
                    len(covered_vehicles) / 30.0,  # UAV覆盖更多车辆
                    np.mean([v.get('distance_to_uav', 0) for v in covered_vehicles]) / 2000.0,
                    np.mean([v.get('elevation_angle', 0) for v in covered_vehicles]) / 90.0,
                    np.mean([v.get('signal_strength', 0) for v in covered_vehicles]),
                    np.mean([v.get('data_rate', 0) for v in covered_vehicles]) / 100.0,
                    np.mean([v.get('latency', 0) for v in covered_vehicles]) / 100.0
                ])
                
                # 任务特征
                operational_features.extend([
                    np.mean([v.get('compute_demand', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('data_size', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('deadline', 0) for v in covered_vehicles]) / 100.0,
                    np.mean([v.get('priority', 0) for v in covered_vehicles]),
                    np.sum([v.get('energy_cost', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('qos_requirement', 0) for v in covered_vehicles])
                ])
                
                # 移动性和预测信息
                operational_features.extend([
                    np.mean([v.get('velocity', 0) for v in covered_vehicles]) / 30.0,
                    np.mean([v.get('direction', 0) for v in covered_vehicles]) / 360.0,
                    np.std([v.get('velocity', 0) for v in covered_vehicles]) / 10.0,
                    np.mean([v.get('predicted_trajectory_x', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('predicted_trajectory_y', 0) for v in covered_vehicles]) / 1000.0,
                    np.mean([v.get('coverage_duration', 0) for v in covered_vehicles]) / 60.0
                ])
            else:
                operational_features.extend([0.0] * 18)
        else:
            operational_features.extend([0.0] * 18)
        
        # 4. 与RSU协作信息（15维）
        if 'rsus' in raw_state:
            rsus = raw_state['rsus']
            rsu_features = []
            for rsu in rsus[:5]:  # 最多5个RSU
                rsu_features.extend([
                    rsu.get('cpu_usage', 0.0),
                    rsu.get('available_compute', 0.0) / 1000.0,
                    rsu.get('cooperation_willingness', 0.5)
                ])
            
            while len(rsu_features) < 15:
                rsu_features.append(0.0)
            operational_features.extend(rsu_features[:15])
        else:
            operational_features.extend([0.0] * 15)
        
        # 5. 其他UAV协作信息（6维）
        if 'uavs' in raw_state:
            uavs = raw_state['uavs']
            other_uav_features = []
            for j, uav in enumerate(uavs):
                if j != uav_id and len(other_uav_features) < 6:
                    other_uav_features.extend([
                        uav.get('cpu_usage', 0.0),
                        uav.get('available_compute', 0.0) / 500.0
                    ])
            
            while len(other_uav_features) < 6:
                other_uav_features.append(0.0)
            operational_features.extend(other_uav_features[:6])
        else:
            operational_features.extend([0.0] * 6)
        
        # 6. 环境和网络状态（8维）
        if 'environment_state' in raw_state:
            env = raw_state['environment_state']
            operational_features.extend([
                env.get('weather_condition', 0.5),  # 影响UAV通信
                env.get('atmospheric_attenuation', 0.0),
                env.get('interference_level', 0.0),
                env.get('channel_quality', 0.0),
                env.get('los_probability', 0.8),  # 视距概率
                env.get('path_loss', 0.0) / 100.0,
                env.get('doppler_shift', 0.0) / 1000.0,
                env.get('multipath_fading', 0.0)
            ])
        else:
            operational_features.extend([0.0] * 8)
        
        # 确保特征向量长度正确
        while len(operational_features) < self.operational_state_dim:
            operational_features.append(0.0)
        
        operational_features = operational_features[:self.operational_state_dim]
        
        return np.array(operational_features, dtype=np.float32)
    
    def get_action(self, processed_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        根据各智能体的执行状态生成底层控制动作
        
        Args:
            processed_states: 各智能体的执行状态字典
            
        Returns:
            operational_actions: 各智能体的执行动作字典
        """
        operational_actions = {}
        
        # 为每个智能体生成动作
        for agent_id, state in processed_states.items():
            if agent_id in self.agents:
                action = self.agents[agent_id].select_action(state, training=True)
                operational_actions[agent_id] = self._post_process_operational_action(action, agent_id)
        
        # 记录执行历史
        self.execution_history.append({
            'states': {k: v.copy() for k, v in processed_states.items()},
            'actions': {k: v.copy() for k, v in operational_actions.items()},
            'timestamp': len(self.execution_history)
        })
        
        return operational_actions
    
    def _post_process_operational_action(self, raw_action: np.ndarray, agent_id: str) -> np.ndarray:
        """
        对原始动作进行后处理，确保符合物理约束和控制语义
        
        Args:
            raw_action: 原始动作向量
            agent_id: 智能体ID
            
        Returns:
            processed_action: 处理后的执行动作
        """
        processed_action = raw_action.copy()
        
        if agent_id.startswith('rsu'):
            # RSU执行动作后处理
            # [传输功率, CPU频率, 内存分配, 带宽分配, 缓存更新率, 队列优先级, 
            #  协作传输功率, 数据压缩率, 编码方案, 调度策略, 能耗控制, 散热控制, ...]
            
            # 传输功率控制 [0, 1] -> [0, 最大功率]
            processed_action[0] = torch.sigmoid(torch.tensor(raw_action[0])).item()
            
            # CPU频率控制 [0, 1]
            processed_action[1] = torch.sigmoid(torch.tensor(raw_action[1])).item()
            
            # 内存分配 [0, 1]
            processed_action[2] = torch.sigmoid(torch.tensor(raw_action[2])).item()
            
            # 带宽分配 [0, 1]
            processed_action[3] = torch.sigmoid(torch.tensor(raw_action[3])).item()
            
            # 缓存更新率 [0, 1]
            processed_action[4] = torch.sigmoid(torch.tensor(raw_action[4])).item()
            
            # 其余控制参数
            for i in range(5, min(len(processed_action), 15)):
                processed_action[i] = torch.sigmoid(torch.tensor(raw_action[i])).item()
                
        elif agent_id.startswith('uav'):
            # UAV执行动作后处理（不包括位置控制）
            # [传输功率, 波束成形, 天线方向, 服务带宽, 计算资源分配, 
            #  协作模式, 覆盖策略, 能耗管理, 信号处理, 编码策略, 调度算法, ...]
            
            # 传输功率控制 [0, 1]
            processed_action[0] = torch.sigmoid(torch.tensor(raw_action[0])).item()
            
            # 波束成形参数 [-1, 1] -> 角度控制
            processed_action[1] = np.clip(raw_action[1], -1.0, 1.0)
            
            # 天线方向 [0, 1] -> [0, 360度]
            processed_action[2] = torch.sigmoid(torch.tensor(raw_action[2])).item()
            
            # 服务带宽分配 [0, 1]
            processed_action[3] = torch.sigmoid(torch.tensor(raw_action[3])).item()
            
            # 计算资源分配 [0, 1]
            processed_action[4] = torch.sigmoid(torch.tensor(raw_action[4])).item()
            
            # 其余控制参数
            for i in range(5, len(processed_action)):
                if i < 8:  # 连续控制参数
                    processed_action[i] = torch.sigmoid(torch.tensor(raw_action[i])).item()
                else:  # 其他参数
                    processed_action[i] = np.clip(raw_action[i], -1.0, 1.0)
        
        return processed_action
    
    def train(self, replay_buffer=None) -> Dict[str, float]:
        """
        训练执行层模型
        
        Args:
            replay_buffer: 经验回放缓冲区（可选）
            
        Returns:
            training_stats: 训练统计信息
        """
        training_stats = {}
        
        # 训练每个智能体
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) >= agent.config.batch_size:
                agent_stats = agent.update()
                training_stats[agent_id] = agent_stats
        
        return training_stats
    
    def store_experience(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                        rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                        dones: Dict[str, bool]):
        """存储执行经验"""
        for agent_id in self.agents.keys():
            if agent_id in states and agent_id in actions and agent_id in rewards:
                self.agents[agent_id].store_experience(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_states[agent_id],
                    dones[agent_id]
                )
    
    def save_model(self, path: str):
        """保存执行层模型"""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}_{agent_id}"
            agent.save_model(agent_path)
    
    def load_model(self, path: str):
        """加载执行层模型"""
        for agent_id, agent in self.agents.items():
            agent_path = f"{path}_{agent_id}"
            try:
                agent.load_model(agent_path)
            except FileNotFoundError:
                print(f"Warning: Model file not found for agent {agent_id}")
    
    def update_tactical_instructions(self, tactical_instructions: Dict[str, Dict]):
        """更新来自战术层的指令"""
        self.tactical_instructions.update(tactical_instructions)
    
    def get_control_commands(self) -> Dict[str, Dict]:
        """
        获取当前的控制命令，用于环境执行
        
        Returns:
            commands: 控制命令字典
        """
        if len(self.execution_history) == 0:
            return {}
        
        latest_actions = self.execution_history[-1]['actions']
        commands = {}
        
        for agent_id, action in latest_actions.items():
            if agent_id.startswith('rsu'):
                commands[agent_id] = {
                    'transmission_power': action[0],
                    'cpu_frequency': action[1],
                    'memory_allocation': action[2],
                    'bandwidth_allocation': action[3],
                    'cache_update_rate': action[4],
                    'queue_priority': action[5] if len(action) > 5 else 0.5,
                    'cooperation_power': action[6] if len(action) > 6 else 0.5,
                    'compression_rate': action[7] if len(action) > 7 else 0.5,
                    'encoding_scheme': action[8] if len(action) > 8 else 0.5,
                    'scheduling_policy': action[9] if len(action) > 9 else 0.5
                }
            elif agent_id.startswith('uav'):
                commands[agent_id] = {
                    'transmission_power': action[0],
                    'beamforming': action[1],
                    'antenna_direction': action[2],
                    'service_bandwidth': action[3],
                    'compute_allocation': action[4],
                    'cooperation_mode': action[5] if len(action) > 5 else 0.5,
                    'coverage_strategy': action[6] if len(action) > 6 else 0.5,
                    'energy_management': action[7] if len(action) > 7 else 0.5,
                    'signal_processing': action[8] if len(action) > 8 else 0.5
                }
        
        return commands
    
    def get_layer_stats(self) -> Dict[str, float]:
        """获取执行层统计信息"""
        if len(self.execution_history) == 0:
            return {}
        
        stats = {
            'execution_count': len(self.execution_history),
            'num_agents': len(self.agents),
            'algorithm': self.algorithm,
            'num_rsus': self.num_rsus,
            'num_uavs': self.num_uavs
        }
        
        # 添加各智能体的训练统计
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'replay_buffer'):
                stats[f'{agent_id}_buffer_size'] = len(agent.replay_buffer)
        
        return stats