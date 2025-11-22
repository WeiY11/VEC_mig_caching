"""
分层训练环境（Hierarchical Training Environment）
集成战略层、战术层和执行层的完整训练环境

主要功能：
1. 协调三层架构的信息流
2. 管理分层决策过程
3. 处理固定无人机位置约束
4. 提供统一的训练接口
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hierarchical_learning.core.strategic_layer import StrategicLayer
from hierarchical_learning.core.tactical_layer import TacticalLayer
from hierarchical_learning.core.operational_layer import OperationalLayer
from config import config


class HierarchicalEnvironment:
    """分层训练环境 - 集成三层架构的完整环境"""
    
    def __init__(self, env_config: Dict):
        """
        初始化分层环境
        
        Args:
            env_config: 环境配置字典
        """
        self.config = env_config
        
        # 环境基本参数
        self.num_rsus = env_config.get('num_rsus', 4)
        self.num_uavs = env_config.get('num_uavs', 2)
        self.num_vehicles = env_config.get('num_vehicles', 50)
        self.area_size = env_config.get('area_size', (2000, 2000))  # 区域大小(m)
        
        # 固定UAV位置（根据用户要求）
        self.fixed_uav_positions = self._initialize_fixed_uav_positions()
        
        # 初始化三层架构
        self.strategic_layer = StrategicLayer(env_config.get('strategic_config', {}))
        self.tactical_layer = TacticalLayer(env_config.get('tactical_config', {}))
        self.operational_layer = OperationalLayer(env_config.get('operational_config', {}))
        
        # 环境状态
        self.current_state = {}
        self.episode_step = 0
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        self.time_slot = float(env_config.get('time_slot', getattr(config, 'time_slot', 0.1)))
        self.rsu_service_rate = float(env_config.get('rsu_service_rate', 5.0))  # tasks per second
        self.uav_service_rate = float(env_config.get('uav_service_rate', 3.0))  # tasks per second
        queue_cfg = getattr(config, 'queue', None)
        default_max_queue = getattr(queue_cfg, 'max_queue_size', 100) if queue_cfg is not None else 100
        self.max_queue_length = float(env_config.get('max_queue_length', default_max_queue))
        
        # 性能指标
        self.performance_metrics = {
            'total_latency': 0.0,
            'total_energy': 0.0,
            'success_rate': 0.0,
            'throughput': 0.0,
            'cost_efficiency': 0.0
        }
        
        # 训练统计
        self.training_stats = {
            'strategic_updates': 0,
            'tactical_updates': 0,
            'operational_updates': 0,
            'total_episodes': 0,
            'total_steps': 0
        }
        
        # 分层决策历史
        self.decision_history = {
            'strategic': [],
            'tactical': [],
            'operational': []
        }
        
        # 分层协调机制
        self.strategic_guidance = {
            'resource_allocation': {},
            'priority_weights': {},
            'global_objectives': {}
        }
        
        self.tactical_instructions = {
            'scheduling_policy': {},
            'load_balancing': {},
            'migration_strategy': {}
        }
        
        # 协调参数
        self.coordination_weights = {
            'strategic_weight': 0.4,
            'tactical_weight': 0.35,
            'operational_weight': 0.25
        }
        
        # 初始化环境
        self.reset()
        
        # 缓存最近一次分层动作，供经验存储使用
        self.last_actions = {
            'strategic': None,
            'tactical': {},
            'operational': {}
        }
    
    def _initialize_fixed_uav_positions(self) -> List[Tuple[float, float, float]]:
        """初始化固定的UAV位置"""
        positions = []
        
        # 根据区域大小和UAV数量，均匀分布UAV位置
        area_width, area_height = self.area_size
        
        if self.num_uavs == 1:
            # 单个UAV放在中心
            positions.append((area_width/2, area_height/2, 100.0))  # 高度100m
        elif self.num_uavs == 2:
            # 两个UAV分别放在1/3和2/3位置
            positions.append((area_width/3, area_height/2, 100.0))
            positions.append((2*area_width/3, area_height/2, 100.0))
        elif self.num_uavs == 3:
            # 三个UAV形成三角形分布
            positions.append((area_width/4, area_height/4, 100.0))
            positions.append((3*area_width/4, area_height/4, 100.0))
            positions.append((area_width/2, 3*area_height/4, 100.0))
        else:
            # 更多UAV时，网格分布
            rows = int(np.sqrt(self.num_uavs))
            cols = int(np.ceil(self.num_uavs / rows))
            
            for i in range(self.num_uavs):
                row = i // cols
                col = i % cols
                x = (col + 1) * area_width / (cols + 1)
                y = (row + 1) * area_height / (rows + 1)
                positions.append((x, y, 100.0))
        
        return positions
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境到初始状态
        
        Returns:
            initial_states: 各层的初始状态
        """
        self.episode_step = 0
        
        # 重置性能指标
        self.performance_metrics = {
            'total_latency': 0.0,
            'total_energy': 0.0,
            'success_rate': 1.0,
            'throughput': 0.0,
            'cost_efficiency': 1.0
        }
        
        # 生成初始环境状态
        self.current_state = self._generate_initial_state()
        
        # 获取各层的初始状态
        initial_states = self._get_hierarchical_states()
        
        return initial_states
    
    def _generate_initial_state(self) -> Dict:
        """生成初始环境状态"""
        state = {}
        
        # RSU状态
        state['rsus'] = []
        for i in range(self.num_rsus):
            rsu_state = {
                'id': i,
                'position': (
                    np.random.uniform(0, self.area_size[0]),
                    np.random.uniform(0, self.area_size[1])
                ),
                'cpu_usage': np.random.uniform(0.1, 0.3),
                'memory_usage': np.random.uniform(0.1, 0.3),
                'storage_usage': np.random.uniform(0.1, 0.5),
                'network_bandwidth_usage': np.random.uniform(0.1, 0.4),
                'available_compute': np.random.uniform(800, 1200),  # GFLOPS
                'available_memory': np.random.uniform(16, 32),  # GB
                'available_storage': np.random.uniform(500, 1000),  # GB
                'available_bandwidth': np.random.uniform(50, 100),  # Mbps
                'power_consumption': np.random.uniform(100, 200),  # W
                'temperature': np.random.uniform(20, 30),  # °C
                'queue_length': 0,
                'served_vehicles': 0,
                'success_rate': 1.0,
                'avg_latency': 0.0,
                'energy_consumption': 0.0,
                'coverage_vehicles': 0
            }
            state['rsus'].append(rsu_state)
        
        # UAV状态（位置固定）
        state['uavs'] = []
        for i in range(self.num_uavs):
            uav_state = {
                'id': i,
                'position': self.fixed_uav_positions[i],  # 固定位置
                'cpu_usage': np.random.uniform(0.1, 0.3),
                'memory_usage': np.random.uniform(0.1, 0.3),
                'available_compute': np.random.uniform(400, 600),  # GFLOPS
                'energy_level': np.random.uniform(0.8, 1.0),
                'communication_power': np.random.uniform(10, 20),  # W
                'computation_power': np.random.uniform(50, 100),  # W
                'antenna_gain': np.random.uniform(5, 10),  # dBi
                'signal_processing_capability': np.random.uniform(0.7, 1.0),
                'coverage_efficiency': np.random.uniform(0.7, 0.9),
                'served_vehicles': 0,
                'queue_length': 0,
                'communication_load': 0.0,
                'in_uav_coverage': False
            }
            state['uavs'].append(uav_state)
        
        # 车辆状态
        state['vehicles'] = []
        for i in range(self.num_vehicles):
            vehicle_state = {
                'id': i,
                'position': (
                    np.random.uniform(0, self.area_size[0]),
                    np.random.uniform(0, self.area_size[1])
                ),
                'velocity': np.random.uniform(10, 30),  # m/s
                'direction': np.random.uniform(0, 360),  # degrees
                'compute_demand': np.random.uniform(100, 1000),  # MFLOPS
                'data_size': np.random.uniform(1, 10),  # MB
                'latency_requirement': np.random.uniform(10, 100),  # ms
                'deadline': np.random.uniform(50, 200),  # ms
                'priority': np.random.uniform(0.1, 1.0),
                'energy_cost': np.random.uniform(1, 10),  # J
                'qos_requirement': np.random.uniform(0.8, 1.0),
                'serving_rsu': -1,
                'in_uav_coverage': False,
                'distance_to_rsu': 0.0,
                'distance_to_uav': 0.0,
                'signal_strength': 0.0,
                'data_rate': 0.0,
                'latency': 0.0,
                'predicted_stay_time': 0.0,
                'handover_probability': 0.0,
                'predicted_trajectory_x': 0.0,
                'predicted_trajectory_y': 0.0,
                'coverage_duration': 0.0,
                'elevation_angle': 0.0
            }
            state['vehicles'].append(vehicle_state)
        
        # 重置服务计数器和队列长度
        for rsu in state['rsus']:
            rsu['served_vehicles'] = 0
            rsu['coverage_vehicles'] = 0
            rsu['queue_length'] = 0
        
        for uav in state['uavs']:
            uav['served_vehicles'] = 0
            uav['queue_length'] = 0
        
        # 重置车辆状态
        for vehicle in state['vehicles']:
            vehicle['serving_rsu'] = -1
            vehicle['in_uav_coverage'] = False
            vehicle['latency'] = 0.0
            vehicle['signal_strength'] = 0.0
        
        # 更新车辆与RSU/UAV的关系
        self._update_vehicle_associations(state)
        
        # 网络状态
        state['network_state'] = {
            'channel_quality': np.random.uniform(0.6, 0.9),
            'interference_level': np.random.uniform(0.1, 0.3),
            'congestion_level': np.random.uniform(0.1, 0.4),
            'packet_loss_rate': np.random.uniform(0.001, 0.01),
            'throughput': np.random.uniform(500, 1000),  # Mbps
            'jitter': np.random.uniform(1, 5),  # ms
            'rtt': np.random.uniform(10, 50),  # ms
            'bandwidth_utilization': np.random.uniform(0.3, 0.7),
            'error_rate': np.random.uniform(0.001, 0.005),
            'retransmission_rate': np.random.uniform(0.01, 0.05)
        }
        
        # 环境状态
        state['environment_state'] = {
            'weather_condition': np.random.uniform(0.7, 1.0),  # 1.0为最佳天气
            'atmospheric_attenuation': np.random.uniform(0.1, 0.3),
            'interference_level': np.random.uniform(0.1, 0.3),
            'channel_quality': np.random.uniform(0.6, 0.9),
            'los_probability': np.random.uniform(0.7, 0.9),
            'path_loss': np.random.uniform(80, 120),  # dB
            'doppler_shift': np.random.uniform(0, 100),  # Hz
            'multipath_fading': np.random.uniform(0.1, 0.3)
        }
        
        # 系统指标
        state['system_metrics'] = {
            'avg_latency': np.random.uniform(20, 50),
            'energy_efficiency': np.random.uniform(0.6, 0.8),
            'success_rate': np.random.uniform(0.8, 0.95),
            'network_utilization': np.random.uniform(0.4, 0.7),
            'load_balance_index': np.random.uniform(0.5, 0.8)
        }
        
        # 实时指标
        state['real_time_metrics'] = {
            'current_latency': np.random.uniform(15, 40),
            'current_throughput': np.random.uniform(400, 800),
            'current_energy': np.random.uniform(200, 500),
            'current_success_rate': np.random.uniform(0.85, 0.98),
            'current_queue_length': np.random.uniform(0, 10),
            'current_load': np.random.uniform(0.3, 0.7),
            'current_efficiency': np.random.uniform(0.6, 0.9),
            'current_cost': np.random.uniform(10, 50),
            'current_reliability': np.random.uniform(0.8, 0.95),
            'current_availability': np.random.uniform(0.9, 0.99)
        }
        
        return state
    
    def _update_vehicle_associations(self, state: Dict):
        """Update vehicle associations with RSUs and UAVs"""
        for vehicle in state['vehicles']:
            vehicle['serving_rsu'] = -1
            vehicle['in_uav_coverage'] = False
            vehicle['latency'] = 0.0
            vehicle['signal_strength'] = 0.0
            vehicle['data_rate'] = 0.0
            vehicle['distance_to_rsu'] = 0.0
            vehicle['distance_to_uav'] = 0.0

            vehicle_pos = vehicle['position']

            # ???????RSU
            min_rsu_distance = float('inf')
            closest_rsu = -1

            for rsu in state['rsus']:
                rsu_pos = rsu['position']
                distance = np.sqrt((vehicle_pos[0] - rsu_pos[0])**2 +
                                   (vehicle_pos[1] - rsu_pos[1])**2)

                if distance < min_rsu_distance and distance <= 500:  # RSU????500m
                    min_rsu_distance = distance
                    closest_rsu = rsu['id']

            if closest_rsu != -1:
                rsu = state['rsus'][closest_rsu]
                vehicle['serving_rsu'] = closest_rsu
                vehicle['distance_to_rsu'] = min_rsu_distance
                vehicle['signal_strength'] = max(0.1, 1.0 - min_rsu_distance / 500.0)
                vehicle['data_rate'] = vehicle['signal_strength'] * 100  # Mbps

                propagation_delay = min_rsu_distance / 3e8 * 1000  # ????(ms)
                processing_delay = rsu['cpu_usage'] * 20 + np.random.uniform(1, 5)  # ????(ms)
                queuing_delay = rsu['queue_length'] * 2  # ????(ms)
                transmission_delay = vehicle['data_size'] / max(vehicle['data_rate'], 1e-6) * 8  # ????(ms)

                vehicle['latency'] = propagation_delay + processing_delay + queuing_delay + transmission_delay

                rsu['served_vehicles'] += 1
                rsu['coverage_vehicles'] = rsu.get('coverage_vehicles', 0) + 1
                rsu['queue_length'] = min(self.max_queue_length, rsu.get('queue_length', 0.0) + 1)

            # ?????RSU???UAV
            connected_to_uav = False
            for uav in state['uavs']:
                uav_pos = uav['position']
                distance_2d = np.sqrt((vehicle_pos[0] - uav_pos[0])**2 +
                                      (vehicle_pos[1] - uav_pos[1])**2)
                distance_3d = np.sqrt(distance_2d**2 + uav_pos[2]**2)

                if distance_2d <= 1000:  # UAV????1000m
                    vehicle['in_uav_coverage'] = True
                    vehicle['distance_to_uav'] = distance_3d
                    vehicle['elevation_angle'] = np.arctan(uav_pos[2] / max(distance_2d, 1e-6)) * 180 / np.pi

                    if vehicle['serving_rsu'] == -1 and not connected_to_uav:
                        propagation_delay = distance_3d / 3e8 * 1000  # ????(ms)
                        processing_delay = uav['cpu_usage'] * 25 + np.random.uniform(2, 8)  # ????(ms)
                        queuing_delay = uav['queue_length'] * 3  # ????(ms)
                        air_interface_delay = 5 + np.random.uniform(1, 3)  # ??????(ms)

                        signal_quality = max(0.3, 1.0 - distance_2d / 1000.0)
                        uav_data_rate = signal_quality * 80  # Mbps
                        transmission_delay = vehicle['data_size'] / max(uav_data_rate, 1e-6) * 8  # ????(ms)

                        vehicle['latency'] = propagation_delay + processing_delay + queuing_delay + air_interface_delay + transmission_delay
                        uav['queue_length'] = min(self.max_queue_length, uav.get('queue_length', 0.0) + 1)
                        uav['served_vehicles'] += 1
                        connected_to_uav = True
                    break

    def _get_hierarchical_states(self) -> Dict[str, Dict[str, np.ndarray]]:
        """获取各层的状态表示"""
        hierarchical_states = {}
        
        # 战略层状态
        strategic_raw_state = {
            'system_metrics': self.current_state['system_metrics'],
            'rsus': self.current_state['rsus'],
            'uavs': self.current_state['uavs'],
            'vehicles': self.current_state['vehicles'],
            'network_state': self.current_state['network_state'],
            'environment_state': self.current_state['environment_state']
        }
        hierarchical_states['strategic'] = self.strategic_layer.process_state(strategic_raw_state)
        
        # 战术层状态（包含战略指导）
        strategic_guidance = self.strategic_layer.get_strategic_guidance()
        tactical_raw_state = {
            'strategic_guidance': strategic_guidance,
            'rsus': self.current_state['rsus'],
            'uavs': self.current_state['uavs'],
            'vehicles': self.current_state['vehicles'],
            'system_metrics': self.current_state['system_metrics']
        }
        hierarchical_states['tactical'] = self.tactical_layer.process_state(tactical_raw_state)
        
        # 执行层状态（包含战术指令）
        tactical_instructions = self.tactical_layer.get_tactical_instructions()
        operational_raw_state = {
            'tactical_instructions': tactical_instructions,
            'rsus': self.current_state['rsus'],
            'uavs': self.current_state['uavs'],
            'vehicles': self.current_state['vehicles'],
            'network_state': self.current_state['network_state'],
            'environment_state': self.current_state['environment_state'],
            'real_time_metrics': self.current_state['real_time_metrics']
        }
        hierarchical_states['operational'] = self.operational_layer.process_state(operational_raw_state)

        return hierarchical_states

    def _service_queue(self, node: Dict, service_rate: float) -> None:
        """Process queued tasks based on node service rate."""
        queue_length = float(node.get('queue_length', 0.0))
        processed = service_rate * self.time_slot
        node['queue_length'] = max(0.0, queue_length - processed)

    def _reset_step_counters(self) -> None:
        """Reset per-step counters and decay queues before new assignments."""
        for rsu in self.current_state.get('rsus', []):
            rsu['served_vehicles'] = 0
            rsu['coverage_vehicles'] = 0
            self._service_queue(rsu, self.rsu_service_rate)
            rsu['queue_length'] = min(self.max_queue_length, rsu.get('queue_length', 0.0))

        for uav in self.current_state.get('uavs', []):
            uav['served_vehicles'] = 0
            self._service_queue(uav, self.uav_service_rate)
            uav['queue_length'] = min(self.max_queue_length, uav.get('queue_length', 0.0))

    def step(self, actions: Optional[Dict] = None) -> Tuple[Dict[str, Dict[str, np.ndarray]],
                                                           Dict[str, float], bool, Dict]:
        """
        执行一步环境交互
        
        Args:
            actions: 外部提供的动作（可选，用于测试）
            
        Returns:
            next_states: 下一步的各层状态
            rewards: 各层的奖励
            done: 是否结束
            info: 额外信息
        """
        self.episode_step += 1
        self._reset_step_counters()
        
        # 获取当前状态
        current_hierarchical_states = self._get_hierarchical_states()
        
        # 分层决策协调过程
        if actions is None:
            # 1. 战略层决策 - 全局资源分配和长期规划
            strategic_actions = self.strategic_layer.get_action(current_hierarchical_states['strategic'])
            self._update_strategic_guidance(strategic_actions)
            
            # 2. 战术层决策 - 基于战略指导的中期调度
            tactical_actions = self.tactical_layer.get_action(current_hierarchical_states['tactical'])
            self._update_tactical_instructions(tactical_actions)
            
            # 3. 执行层决策 - 基于战术指令的实时执行
            operational_actions = self.operational_layer.get_action(current_hierarchical_states['operational'])
            
            # 4. 分层协调和冲突解决
            coordinated_actions = self._coordinate_hierarchical_decisions(
                strategic_actions, tactical_actions, operational_actions
            )
            
            # 5. 信息反馈和学习更新
            self._update_hierarchical_feedback(coordinated_actions)
            operational_actions = self.operational_layer.get_action(current_hierarchical_states['operational'])
        else:
            # 使用外部提供的动作
            strategic_actions = actions.get('strategic', {})
            tactical_actions = actions.get('tactical', {})
            operational_actions = actions.get('operational', {})
        
        # 缓存最近动作
        self.last_actions['strategic'] = strategic_actions
        self.last_actions['tactical'] = tactical_actions
        self.last_actions['operational'] = operational_actions
        
        # 记录决策历史
        self.decision_history['strategic'].append(strategic_actions)
        self.decision_history['tactical'].append(tactical_actions)
        self.decision_history['operational'].append(operational_actions)
        
        # 执行动作并更新环境状态
        self._execute_actions(strategic_actions, tactical_actions, operational_actions)
        
        # 计算奖励
        rewards = self._calculate_hierarchical_rewards()
        
        # 获取下一步状态
        next_hierarchical_states = self._get_hierarchical_states()
        
        # 检查是否结束
        done = self.episode_step >= self.max_episode_steps
        
        # 构建信息字典
        info = {
            'episode_step': self.episode_step,
            'performance_metrics': self.performance_metrics.copy(),
            'training_stats': self.training_stats.copy(),
            'strategic_guidance': self.strategic_layer.get_strategic_guidance(),
            'tactical_instructions': self.tactical_layer.get_tactical_instructions(),
            'control_commands': self.operational_layer.get_control_commands()
        }
        
        return next_hierarchical_states, rewards, done, info
    
    def _execute_actions(self, strategic_actions: Dict, tactical_actions: Dict, 
                        operational_actions: Dict):
        """???????????????????"""
        
        
        # ??????????
        self._update_vehicle_mobility()
        
        # ??????????????
        control_commands = self.operational_layer.get_control_commands()
        self._apply_control_commands(control_commands)
        
        # ??????????????
        self._update_vehicle_associations(self.current_state)
        
        # ?????????????
        self._update_performance_metrics()
    def _update_vehicle_mobility(self):
        """更新车辆移动状态"""
        for vehicle in self.current_state['vehicles']:
            # 简单的移动模型
            velocity = vehicle['velocity']
            direction = vehicle['direction']
            # Update displacement
            dt = self.time_slot  # time step (s)
            dx = velocity * np.cos(np.radians(direction)) * dt
            dy = velocity * np.sin(np.radians(direction)) * dt
            
            # 更新位置
            new_x = vehicle['position'][0] + dx
            new_y = vehicle['position'][1] + dy
            
            # 边界处理
            new_x = np.clip(new_x, 0, self.area_size[0])
            new_y = np.clip(new_y, 0, self.area_size[1])
            
            vehicle['position'] = (new_x, new_y)
            
            # 随机改变方向（模拟真实交通）
            if np.random.random() < 0.1:  # 10%概率改变方向
                vehicle['direction'] = np.random.uniform(0, 360)
    
    def _apply_control_commands(self, control_commands: Dict):
        """应用控制命令到环境"""
        for agent_id, commands in control_commands.items():
            if agent_id.startswith('rsu'):
                rsu_id = int(agent_id.split('_')[1])
                if rsu_id < len(self.current_state['rsus']):
                    rsu = self.current_state['rsus'][rsu_id]
                    
                    # 应用RSU控制命令
                    rsu['cpu_usage'] = min(1.0, rsu['cpu_usage'] + 
                                         (commands.get('cpu_frequency', 0.5) - 0.5) * 0.1)
                    rsu['memory_usage'] = min(1.0, rsu['memory_usage'] + 
                                            (commands.get('memory_allocation', 0.5) - 0.5) * 0.1)
                    rsu['power_consumption'] = rsu['power_consumption'] * (1 + 
                                             (commands.get('transmission_power', 0.5) - 0.5) * 0.2)
            
            elif agent_id.startswith('uav'):
                uav_id = int(agent_id.split('_')[1])
                if uav_id < len(self.current_state['uavs']):
                    uav = self.current_state['uavs'][uav_id]
                    
                    # 应用UAV控制命令（不包括位置控制）
                    uav['cpu_usage'] = min(1.0, uav['cpu_usage'] + 
                                         (commands.get('compute_allocation', 0.5) - 0.5) * 0.1)
                    uav['communication_power'] = uav['communication_power'] * (1 + 
                                                (commands.get('transmission_power', 0.5) - 0.5) * 0.2)
                    uav['energy_level'] = max(0.0, uav['energy_level'] - 0.001)  # 能量消耗
    
    def _update_performance_metrics(self):
        """更新系统性能指标"""
        # 计算平均延迟
        total_latency = 0.0
        served_vehicles = 0
        
        for vehicle in self.current_state['vehicles']:
            if vehicle['serving_rsu'] != -1 or vehicle['in_uav_coverage']:
                total_latency += vehicle.get('latency', 0.0)
                served_vehicles += 1
        
        if served_vehicles > 0:
            self.performance_metrics['total_latency'] = total_latency / served_vehicles
        
        # 计算总能耗（改进的动态能耗模型）
        total_energy = 0.0
        
        # RSU能耗：基础功耗 + 负载相关功耗
        for rsu in self.current_state['rsus']:
            base_power = rsu.get('power_consumption', 100.0)  # 基础功耗
            load_factor = rsu.get('cpu_usage', 0.0)
            dynamic_power = base_power * (0.3 + 0.7 * load_factor)  # 动态功耗
            
            # 通信功耗（基于服务的车辆数）
            comm_power = rsu.get('served_vehicles', 0) * 2.0  # 每个车辆2W通信功耗
            
            rsu_total_power = dynamic_power + comm_power
            total_energy += rsu_total_power
        
        # UAV能耗：悬停功耗 + 通信功耗 + 计算功耗
        for uav in self.current_state['uavs']:
            # 悬停功耗（无人机维持位置的功耗）
            hover_power = 150.0  # 基础悬停功耗(W)
            
            # 通信功耗（基于覆盖的车辆数和距离）
            comm_power = uav.get('communication_power', 20.0)
            served_vehicles = uav.get('served_vehicles', 0)
            comm_load_power = served_vehicles * 3.0  # 每个车辆3W通信功耗
            
            # 计算功耗（基于CPU使用率）
            cpu_usage = uav.get('cpu_usage', 0.0)
            comp_power = 50.0 + cpu_usage * 100.0  # 计算功耗
            
            uav_total_power = hover_power + comm_power + comm_load_power + comp_power
            total_energy += uav_total_power
        
        self.performance_metrics['total_energy'] = total_energy
        
        # 计算成功率（改进的QoS满足率）
        if self.num_vehicles > 0:
            successful_vehicles = 0
            for vehicle in self.current_state['vehicles']:
                if vehicle['serving_rsu'] != -1 or vehicle['in_uav_coverage']:
                    # 检查QoS要求是否满足
                    latency_satisfied = vehicle['latency'] <= vehicle['latency_requirement']
                    deadline_satisfied = vehicle['latency'] <= vehicle['deadline']
                    qos_satisfied = vehicle.get('signal_strength', 0.0) >= vehicle['qos_requirement']
                    
                    if latency_satisfied and deadline_satisfied and qos_satisfied:
                        successful_vehicles += 1
            
            self.performance_metrics['success_rate'] = successful_vehicles / self.num_vehicles
        else:
            self.performance_metrics['success_rate'] = 0.0
        
        # 更新系统指标
        self.current_state['system_metrics']['avg_latency'] = self.performance_metrics['total_latency']
        self.current_state['system_metrics']['success_rate'] = self.performance_metrics['success_rate']
    
    def _calculate_hierarchical_rewards(self) -> Dict[str, float]:
        """计算各层的奖励 - 精细化的分层奖励设计"""
        rewards = {}
        
        # 获取原始性能指标
        raw_latency = self.performance_metrics['total_latency']
        raw_energy = self.performance_metrics['total_energy']
        success_rate = self.performance_metrics['success_rate']
        failure_rate = 1.0 - success_rate
        
        # 改进的归一化策略（使用更合理的基准值）
        normalized_latency = raw_latency / 50.0  # 50ms作为目标延迟
        normalized_energy = raw_energy / 2000.0  # 2000W作为目标总功耗
        
        # 计算基础目标函数值
        objective_function = (
            config.rl.reward_weight_delay * normalized_latency +      # ω_T * T
            config.rl.reward_weight_energy * normalized_energy +      # ω_E * E  
            config.rl.reward_weight_loss * failure_rate               # ω_D * D
        )
        
        # 基础奖励（负的目标函数值）
        base_reward = -objective_function
        
        # 战略层奖励：全局优化 + 长期性能
        # 1. 系统平衡性奖励
        balance_score = self._calculate_system_balance()
        # 2. 长期性能稳定性
        stability_bonus = self._calculate_performance_stability()
        # 3. 资源利用效率
        efficiency_bonus = self._calculate_resource_efficiency()
        
        strategic_reward = (
            base_reward * 1.0 +          # 基础性能
            balance_score * 0.2 +        # 系统平衡
            stability_bonus * 0.15 +     # 性能稳定性
            efficiency_bonus * 0.1       # 资源效率
        )
        rewards['strategic'] = strategic_reward
        
        # 战术层奖励：协调优化 + 负载均衡
        # 1. 负载均衡效果
        load_balance_score = self._calculate_load_balance()
        # 2. 覆盖优化效果
        coverage_score = self._calculate_coverage_optimization()
        # 3. QoS满足程度
        qos_score = self._calculate_qos_satisfaction()
        
        tactical_reward = (
            base_reward * 0.8 +          # 基础性能（权重稍低）
            load_balance_score * 0.3 +   # 负载均衡
            coverage_score * 0.2 +       # 覆盖优化
            qos_score * 0.25             # QoS满足
        )
        rewards['tactical'] = tactical_reward
        
        # 执行层奖励：实时性能 + 控制精度
        # 1. 延迟控制效果
        latency_control_score = self._calculate_latency_control()
        # 2. 能耗控制效果
        energy_control_score = self._calculate_energy_control()
        # 3. 实时响应性
        responsiveness_score = self._calculate_responsiveness()
        
        operational_reward = (
            base_reward * 0.9 +                # 基础性能
            latency_control_score * 0.4 +      # 延迟控制（执行层关注重点）
            energy_control_score * 0.2 +       # 能耗控制
            responsiveness_score * 0.2          # 实时响应
        )
        rewards['operational'] = operational_reward
        
        return rewards
    
    def _calculate_system_balance(self) -> float:
        """计算系统平衡性分数"""
        # RSU负载平衡
        rsu_loads = [rsu.get('cpu_usage', 0.0) for rsu in self.current_state.get('rsus', [])]
        rsu_balance = 1.0 - np.var(rsu_loads) if rsu_loads else 0.0
        
        # UAV负载平衡
        uav_loads = [uav.get('cpu_usage', 0.0) for uav in self.current_state.get('uavs', [])]
        uav_balance = 1.0 - np.var(uav_loads) if uav_loads else 0.0
        
        # 服务分布平衡
        rsu_services = [rsu.get('served_vehicles', 0) for rsu in self.current_state.get('rsus', [])]
        service_balance = 1.0 - (np.var(rsu_services) / (np.mean(rsu_services) + 1e-6)) if rsu_services else 0.0
        
        return (rsu_balance + uav_balance + service_balance) / 3.0
    
    def _calculate_performance_stability(self) -> float:
        """计算性能稳定性分数"""
        # 基于成功率的稳定性（越接近1越好）
        success_stability = 1.0 - abs(1.0 - self.performance_metrics['success_rate'])
        
        # 延迟稳定性（变化越小越好）
        vehicles = self.current_state.get('vehicles', [])
        latencies = [v.get('latency', 0.0) for v in vehicles if v.get('latency', 0.0) > 0]
        if latencies and len(latencies) > 1:
            latency_stability = 1.0 - min(1.0, np.std(latencies) / (np.mean(latencies) + 1e-6))
        else:
            latency_stability = 1.0
        
        return (success_stability + latency_stability) / 2.0
    
    def _calculate_resource_efficiency(self) -> float:
        """计算资源利用效率"""
        # CPU利用率效率（适中利用率最好）
        rsu_cpu_usage = [rsu.get('cpu_usage', 0.0) for rsu in self.current_state.get('rsus', [])]
        uav_cpu_usage = [uav.get('cpu_usage', 0.0) for uav in self.current_state.get('uavs', [])]
        
        all_cpu_usage = rsu_cpu_usage + uav_cpu_usage
        if all_cpu_usage:
            # 理想利用率在0.6-0.8之间
            efficiency_scores = [1.0 - abs(usage - 0.7) for usage in all_cpu_usage]
            cpu_efficiency = np.mean(efficiency_scores)
        else:
            cpu_efficiency = 0.0
        
        # 服务效率（服务车辆数与资源利用的比率）
        total_served = sum([rsu.get('served_vehicles', 0) for rsu in self.current_state.get('rsus', [])])
        total_served += sum([uav.get('served_vehicles', 0) for uav in self.current_state.get('uavs', [])])
        
        if total_served > 0 and self.num_vehicles > 0:
            service_efficiency = min(1.0, total_served / self.num_vehicles)
        else:
            service_efficiency = 0.0
        
        return (cpu_efficiency + service_efficiency) / 2.0
    
    def _calculate_load_balance(self) -> float:
        """计算负载均衡分数"""
        rsu_loads = [rsu.get('served_vehicles', 0) for rsu in self.current_state.get('rsus', [])]
        if rsu_loads and len(rsu_loads) > 1:
            mean_load = np.mean(rsu_loads)
            if mean_load > 0:
                cv = np.std(rsu_loads) / mean_load  # 变异系数
                return max(0.0, 1.0 - cv)  # 变异系数越小，均衡性越好
        return 1.0
    
    def _calculate_coverage_optimization(self) -> float:
        """计算覆盖优化分数"""
        total_vehicles = len(self.current_state.get('vehicles', []))
        covered_vehicles = 0
        
        for vehicle in self.current_state.get('vehicles', []):
            if vehicle.get('serving_rsu', -1) != -1 or vehicle.get('in_uav_coverage', False):
                covered_vehicles += 1
        
        coverage_rate = covered_vehicles / total_vehicles if total_vehicles > 0 else 0.0
        
        # 覆盖质量（信号强度）
        signal_strengths = [v.get('signal_strength', 0.0) for v in self.current_state.get('vehicles', []) 
                           if v.get('signal_strength', 0.0) > 0]
        avg_signal_quality = np.mean(signal_strengths) if signal_strengths else 0.0
        
        return (coverage_rate + avg_signal_quality) / 2.0
    
    def _calculate_qos_satisfaction(self) -> float:
        """计算QoS满足度分数"""
        vehicles = self.current_state.get('vehicles', [])
        if not vehicles:
            return 0.0
        
        qos_satisfied = 0
        for vehicle in vehicles:
            if vehicle.get('serving_rsu', -1) != -1 or vehicle.get('in_uav_coverage', False):
                latency_ok = vehicle.get('latency', float('inf')) <= vehicle.get('latency_requirement', 0)
                deadline_ok = vehicle.get('latency', float('inf')) <= vehicle.get('deadline', 0)
                signal_ok = vehicle.get('signal_strength', 0.0) >= vehicle.get('qos_requirement', 1.0)
                
                if latency_ok and deadline_ok and signal_ok:
                    qos_satisfied += 1
        
        return qos_satisfied / len(vehicles)
    
    def _calculate_latency_control(self) -> float:
        """计算延迟控制效果"""
        vehicles = self.current_state.get('vehicles', [])
        if not vehicles:
            return 0.0
        
        latency_scores = []
        for vehicle in vehicles:
            actual_latency = vehicle.get('latency', 0.0)
            required_latency = vehicle.get('latency_requirement', 50.0)
            
            if actual_latency <= required_latency:
                # 满足要求，分数基于超额程度
                score = 1.0 - (actual_latency / required_latency) * 0.5
            else:
                # 不满足要求，给予惩罚
                score = max(0.0, 1.0 - (actual_latency / required_latency - 1.0))
            
            latency_scores.append(score)
        
        return np.mean(latency_scores)
    
    def _calculate_energy_control(self) -> float:
        """计算能耗控制效果"""
        # 评估当前能耗相对于理想能耗的表现
        current_energy = self.performance_metrics['total_energy']
        ideal_energy = 1500.0  # 理想总功耗
        
        if current_energy <= ideal_energy:
            return 1.0 - (current_energy / ideal_energy) * 0.3  # 低于理想值给予奖励
        else:
            return max(0.0, 1.0 - (current_energy / ideal_energy - 1.0) * 2)  # 高于理想值给予惩罚
    
    def _calculate_responsiveness(self) -> float:
        """计算实时响应性分数"""
        # 基于排队延迟的响应性评估
        rsu_queues = [rsu.get('queue_length', 0) for rsu in self.current_state.get('rsus', [])]
        uav_queues = [uav.get('queue_length', 0) for uav in self.current_state.get('uavs', [])]
        
        all_queues = rsu_queues + uav_queues
        if all_queues:
            avg_queue = np.mean(all_queues)
            max_acceptable_queue = 5.0
            responsiveness = max(0.0, 1.0 - avg_queue / max_acceptable_queue)
        else:
            responsiveness = 1.0
        
        return responsiveness
    
    def train_step(self) -> Dict[str, Dict]:
        """执行一步训练"""
        training_results = {}
        
        # 训练战略层
        if hasattr(self.strategic_layer, 'sac_agent') and hasattr(self.strategic_layer.sac_agent, 'replay_buffer'):
            if len(self.strategic_layer.sac_agent.replay_buffer) >= 32:
                strategic_stats = self.strategic_layer.train()
                training_results['strategic'] = strategic_stats
                self.training_stats['strategic_updates'] += 1
        
        # 训练战术层（仅在有有效更新时计数）
        tactical_stats = self.tactical_layer.train()
        if isinstance(tactical_stats, dict) and any(
            isinstance(s, dict) and (('loss' in s) or ('critic_loss' in s) or ('actor_loss' in s))
            for s in tactical_stats.values()
        ):
            training_results['tactical'] = tactical_stats
            self.training_stats['tactical_updates'] += 1
        
        # 训练执行层（仅在有有效更新时计数）
        operational_stats = self.operational_layer.train()
        if isinstance(operational_stats, dict) and any(
            isinstance(s, dict) and (('loss' in s) or ('critic_loss' in s) or ('actor_loss' in s))
            for s in operational_stats.values()
        ):
            training_results['operational'] = operational_stats
            self.training_stats['operational_updates'] += 1
        
        self.training_stats['total_steps'] += 1
        
        return training_results
    
    def store_experience(self, states: Dict, actions: Dict, rewards: Dict, 
                        next_states: Dict, dones: Dict):
        """存储分层经验"""
        # 若未显式传入动作，使用最近一次动作
        if not actions:
            actions = self.last_actions
            if actions is None:
                actions = {'strategic': None, 'tactical': {}, 'operational': {}}
        
        # 存储战略层经验
        if 'strategic' in states and actions.get('strategic') is not None:
            self.strategic_layer.store_experience(
                states['strategic'], actions['strategic'], 
                rewards.get('strategic', 0.0), next_states['strategic'], 
                dones.get('strategic', False)
            )
        
        # 存储战术层经验
        if 'tactical' in states:
            tactical_states = states['tactical']
            tactical_next_states = next_states.get('tactical', {})
            tactical_actions = actions.get('tactical', {}) or {}
            # 将层级奖励广播到每个战术智能体
            tactical_reward_value = rewards.get('tactical', 0.0)
            tactical_rewards = {agent_id: tactical_reward_value for agent_id in tactical_states.keys()}
            # dones 广播
            tactical_done_flag = dones.get('tactical', False)
            tactical_dones = {agent_id: tactical_done_flag for agent_id in tactical_states.keys()}
            self.tactical_layer.store_experience(
                tactical_states, tactical_actions, tactical_rewards, tactical_next_states, tactical_dones
            )
        
        # 存储执行层经验
        if 'operational' in states:
            operational_states = states['operational']
            operational_next_states = next_states.get('operational', {})
            operational_actions = actions.get('operational', {}) or {}
            operational_reward_value = rewards.get('operational', 0.0)
            operational_rewards = {agent_id: operational_reward_value for agent_id in operational_states.keys()}
            operational_done_flag = dones.get('operational', False)
            operational_dones = {agent_id: operational_done_flag for agent_id in operational_states.keys()}
            self.operational_layer.store_experience(
                operational_states, operational_actions, operational_rewards, operational_next_states, operational_dones
            )

    def get_last_actions(self) -> Dict:
        """获取最近一次分层动作副本"""
        return {
            'strategic': None if self.last_actions['strategic'] is None else self.last_actions['strategic'],
            'tactical': {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.last_actions.get('tactical', {}).items()},
            'operational': {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.last_actions.get('operational', {}).items()},
        }
    
    def _update_strategic_guidance(self, strategic_actions):
        """更新战略指导信息"""
        # 检查输入类型，如果是numpy数组则转换为字典格式
        if isinstance(strategic_actions, np.ndarray):
            # 战略层动作数组格式：[计算卸载权重, 内容缓存权重, 资源分配策略, 优先级设置]
            strategic_dict = self._convert_strategic_array_to_dict(strategic_actions)
        elif isinstance(strategic_actions, dict):
            strategic_dict = strategic_actions
        else:
            # 如果类型不匹配，使用默认值
            strategic_dict = {}
        
        # 资源分配指导 - 基于数组的策略值动态计算
        allocation_strategy = strategic_dict.get('allocation_strategy', 0.5)
        self.strategic_guidance['resource_allocation'] = {
            'rsu_allocation': self._generate_rsu_allocation(allocation_strategy),
            'uav_allocation': self._generate_uav_allocation(allocation_strategy),
            'bandwidth_allocation': self._generate_bandwidth_allocation(allocation_strategy)
        }
        
        # 优先级权重 - 基于卸载和缓存权重计算
        offloading_weight = strategic_dict.get('offloading_weight', 0.5)
        caching_weight = strategic_dict.get('caching_weight', 0.5)
        priority_setting = strategic_dict.get('priority_setting', 0.5)
        
        self.strategic_guidance['priority_weights'] = {
            'latency_priority': 0.2 + offloading_weight * 0.3,  # 卸载权重影响延迟优先级
            'energy_priority': 0.2 + (1 - offloading_weight) * 0.2,  # 非卸载时更关注能耗
            'reliability_priority': 0.2 + caching_weight * 0.2,  # 缓存权重影响可靠性
            'cost_priority': 0.4 - priority_setting * 0.2  # 优先级设置影响成本考虑
        }
        
        # 全局目标 - 基于权重动态调整
        base_latency = 50.0
        base_efficiency = 0.8
        base_success_rate = 0.95
        
        self.strategic_guidance['global_objectives'] = {
            'target_latency': base_latency * (1.5 - offloading_weight),  # 卸载权重高时目标延迟更低
            'target_energy_efficiency': base_efficiency + (1 - offloading_weight) * 0.15,
            'target_success_rate': base_success_rate + caching_weight * 0.04
        }
    
    def _convert_strategic_array_to_dict(self, strategic_array: np.ndarray) -> Dict:
        """将战略层动作数组转换为字典格式"""
        # 确保数组长度至少为4
        if len(strategic_array) < 4:
            strategic_array = np.pad(strategic_array, (0, 4 - len(strategic_array)), 'constant', constant_values=0.5)
        
        return {
            'offloading_weight': float(strategic_array[0]),    # 计算卸载权重
            'caching_weight': float(strategic_array[1]),       # 内容缓存权重
            'allocation_strategy': float(strategic_array[2]),  # 资源分配策略
            'priority_setting': float(strategic_array[3])      # 优先级设置
        }
    
    def _generate_rsu_allocation(self, allocation_strategy: float) -> Dict:
        """基于分配策略生成RSU资源分配"""
        num_rsus = self.num_rsus
        
        # 策略值影响分配的均匀性 vs 集中性
        if allocation_strategy > 0.7:
            # 高策略值：更集中的分配，优先给负载高的RSU
            allocation_type = 'concentrated'
            weights = [0.3, 0.25, 0.2, 0.15, 0.08, 0.02][:num_rsus]
        elif allocation_strategy < 0.3:
            # 低策略值：更均匀的分配
            allocation_type = 'balanced'
            weights = [1.0/num_rsus] * num_rsus
        else:
            # 中等策略值：适度集中
            allocation_type = 'moderate'
            weights = [0.25, 0.2, 0.18, 0.15, 0.12, 0.1][:num_rsus]
        
        return {
            'type': allocation_type,
            'weights': weights,
            'total_rsus': num_rsus
        }
    
    def _generate_uav_allocation(self, allocation_strategy: float) -> Dict:
        """基于分配策略生成UAV资源分配"""
        num_uavs = self.num_uavs
        
        # UAV分配策略
        if allocation_strategy > 0.6:
            # 高策略值：UAV更多承担计算任务
            allocation_focus = 'computation_heavy'
            computation_ratio = 0.7
        else:
            # 低策略值：UAV更多用于通信中继
            allocation_focus = 'communication_relay'
            computation_ratio = 0.3
        
        return {
            'focus': allocation_focus,
            'computation_ratio': computation_ratio,
            'communication_ratio': 1.0 - computation_ratio,
            'total_uavs': num_uavs
        }
    
    def _generate_bandwidth_allocation(self, allocation_strategy: float) -> Dict:
        """基于分配策略生成带宽分配"""
        # 带宽分配策略
        if allocation_strategy > 0.7:
            # 高策略值：优先保障高优先级任务
            allocation_policy = 'priority_based'
            high_priority_ratio = 0.6
            medium_priority_ratio = 0.3
            low_priority_ratio = 0.1
        elif allocation_strategy < 0.3:
            # 低策略值：公平分配
            allocation_policy = 'fair_share'
            high_priority_ratio = 0.4
            medium_priority_ratio = 0.4
            low_priority_ratio = 0.2
        else:
            # 中等策略值：平衡分配
            allocation_policy = 'balanced'
            high_priority_ratio = 0.5
            medium_priority_ratio = 0.35
            low_priority_ratio = 0.15
        
        return {
            'policy': allocation_policy,
            'high_priority_ratio': high_priority_ratio,
            'medium_priority_ratio': medium_priority_ratio,
            'low_priority_ratio': low_priority_ratio
        }
    
    def _update_tactical_instructions(self, tactical_actions: Dict):
        """更新战术指令信息"""
        # 调度策略
        self.tactical_instructions['scheduling_policy'] = {
            'task_scheduling': tactical_actions.get('task_scheduling', 'fifo'),
            'resource_scheduling': tactical_actions.get('resource_scheduling', 'round_robin'),
            'priority_scheduling': tactical_actions.get('priority_scheduling', False)
        }
        
        # 负载均衡
        self.tactical_instructions['load_balancing'] = {
            'load_threshold': tactical_actions.get('load_threshold', 0.8),
            'balancing_strategy': tactical_actions.get('balancing_strategy', 'least_loaded'),
            'migration_enabled': tactical_actions.get('migration_enabled', True)
        }
        
        # 迁移策略
        self.tactical_instructions['migration_strategy'] = {
            'migration_trigger': tactical_actions.get('migration_trigger', 'overload'),
            'migration_target': tactical_actions.get('migration_target', 'nearest'),
            'migration_cost_threshold': tactical_actions.get('migration_cost_threshold', 0.5)
        }
    
    def _coordinate_hierarchical_decisions(self, strategic_actions, 
                                         tactical_actions: Dict, 
                                         operational_actions: Dict) -> Dict:
        """简化的分层决策协调（暂时简化以快速验证优化效果）"""
        coordinated_actions = {}
        
        # 将strategic_actions转换为字典格式（如果是numpy数组）
        if isinstance(strategic_actions, np.ndarray):
            strategic_dict = self._convert_strategic_array_to_dict(strategic_actions)
        else:
            strategic_dict = strategic_actions if isinstance(strategic_actions, dict) else {}
        
        # 简化的资源分配协调
        strategic_resources = strategic_dict.get('resource_allocation', {})
        tactical_resources = tactical_actions.get('resource_usage', {})
        operational_resources = operational_actions.get('immediate_resources', {})
        
        # 简化权重融合资源决策
        coordinated_actions['resource_allocation'] = self._weighted_resource_fusion(
            strategic_resources, tactical_resources, operational_resources
        )
        
        # 简化任务调度协调
        coordinated_actions['task_scheduling'] = self._coordinate_task_scheduling(
            strategic_dict, tactical_actions, operational_actions
        )
        
        # 简化迁移决策协调
        coordinated_actions['migration_decisions'] = self._coordinate_migration_decisions(
            strategic_dict, tactical_actions, operational_actions
        )
        
        return coordinated_actions
    
    def _weighted_resource_fusion(self, strategic_res: Dict, tactical_res: Dict, operational_res: Dict) -> Dict:
        """加权融合资源分配决策"""
        fused_resources = {}
        
        # 获取协调权重
        w_s = self.coordination_weights['strategic_weight']
        w_t = self.coordination_weights['tactical_weight']
        w_o = self.coordination_weights['operational_weight']
        
        # 融合RSU资源分配
        fused_resources['rsu_allocation'] = {}
        for rsu_id in range(self.num_rsus):
            strategic_alloc = strategic_res.get('rsu_allocation', {}).get(rsu_id, 0.5)
            tactical_alloc = tactical_res.get('rsu_allocation', {}).get(rsu_id, 0.5)
            operational_alloc = operational_res.get('rsu_allocation', {}).get(rsu_id, 0.5)
            
            fused_resources['rsu_allocation'][rsu_id] = (
                w_s * strategic_alloc + w_t * tactical_alloc + w_o * operational_alloc
            )
        
        # 融合UAV资源分配
        fused_resources['uav_allocation'] = {}
        for uav_id in range(self.num_uavs):
            strategic_alloc = strategic_res.get('uav_allocation', {}).get(uav_id, 0.5)
            tactical_alloc = tactical_res.get('uav_allocation', {}).get(uav_id, 0.5)
            operational_alloc = operational_res.get('uav_allocation', {}).get(uav_id, 0.5)
            
            fused_resources['uav_allocation'][uav_id] = (
                w_s * strategic_alloc + w_t * tactical_alloc + w_o * operational_alloc
            )
        
        return fused_resources
    
    def _coordinate_task_scheduling(self, strategic_actions: Dict, tactical_actions: Dict, operational_actions: Dict) -> Dict:
        """协调任务调度决策"""
        # 基于战略优先级和战术策略的任务调度
        scheduling_policy = {
            'priority_weights': self.strategic_guidance['priority_weights'],
            'scheduling_strategy': self.tactical_instructions['scheduling_policy'],
            'immediate_adjustments': operational_actions.get('scheduling_adjustments', {})
        }
        
        return scheduling_policy
    
    def _coordinate_migration_decisions(self, strategic_actions: Dict, tactical_actions: Dict, operational_actions: Dict) -> Dict:
        """协调迁移决策"""
        # 综合考虑各层的迁移建议
        migration_decisions = {
            'strategic_migration': strategic_actions.get('migration_plan', {}),
            'tactical_migration': tactical_actions.get('migration_actions', {}),
            'operational_migration': operational_actions.get('immediate_migration', {})
        }
        
        # 冲突解决：优先级 战略 > 战术 > 执行
        final_migration = {}
        
        # 首先应用战略层的长期迁移计划
        if migration_decisions['strategic_migration']:
            final_migration.update(migration_decisions['strategic_migration'])
        
        # 然后应用战术层的中期调整
        if migration_decisions['tactical_migration']:
            for key, value in migration_decisions['tactical_migration'].items():
                if key not in final_migration:
                    final_migration[key] = value
        
        # 最后应用执行层的紧急调整
        if migration_decisions['operational_migration']:
            for key, value in migration_decisions['operational_migration'].items():
                if key not in final_migration and value.get('urgent', False):
                    final_migration[key] = value
        
        return final_migration
    
    def _update_hierarchical_feedback(self, coordinated_actions: Dict):
        """更新分层反馈信息"""
        # 记录协调决策历史
        self.decision_history['strategic'].append(self.strategic_guidance.copy())
        self.decision_history['tactical'].append(self.tactical_instructions.copy())
        self.decision_history['operational'].append(coordinated_actions.copy())
        
        # 限制历史记录长度
        max_history = 100
        for layer in self.decision_history:
            if len(self.decision_history[layer]) > max_history:
                self.decision_history[layer] = self.decision_history[layer][-max_history:]
        
        # 更新协调权重（基于性能反馈）
        self._adapt_coordination_weights()
    
    def _adapt_coordination_weights(self):
        """基于性能反馈自适应调整协调权重"""
        # 简化的自适应机制
        current_performance = self.performance_metrics
        
        # 如果延迟过高，增加执行层权重
        if current_performance['total_latency'] > 100.0:
            self.coordination_weights['operational_weight'] = min(0.4, 
                self.coordination_weights['operational_weight'] + 0.05)
            self.coordination_weights['strategic_weight'] = max(0.2,
                self.coordination_weights['strategic_weight'] - 0.025)
            self.coordination_weights['tactical_weight'] = max(0.2,
                self.coordination_weights['tactical_weight'] - 0.025)
        
        # 如果能效过低，增加战略层权重
        elif current_performance.get('energy_efficiency', 0.0) < 0.6:
            self.coordination_weights['strategic_weight'] = min(0.5,
                self.coordination_weights['strategic_weight'] + 0.05)
            self.coordination_weights['operational_weight'] = max(0.15,
                self.coordination_weights['operational_weight'] - 0.025)
            self.coordination_weights['tactical_weight'] = max(0.15,
                self.coordination_weights['tactical_weight'] - 0.025)
        
        # 归一化权重
        total_weight = sum(self.coordination_weights.values())
        for key in self.coordination_weights:
            self.coordination_weights[key] /= total_weight

    def save_models(self, save_path: str):
        """保存所有层的模型"""
        self.strategic_layer.save_model(f"{save_path}_strategic")
        self.tactical_layer.save_model(f"{save_path}_tactical")
        self.operational_layer.save_model(f"{save_path}_operational")
    
    def load_models(self, load_path: str):
        """加载所有层的模型"""
        self.strategic_layer.load_model(f"{load_path}_strategic")
        self.tactical_layer.load_model(f"{load_path}_tactical")
        self.operational_layer.load_model(f"{load_path}_operational")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        stats = self.training_stats.copy()
        stats.update({
            'strategic_stats': self.strategic_layer.get_layer_stats(),
            'tactical_stats': self.tactical_layer.get_layer_stats(),
            'operational_stats': self.operational_layer.get_layer_stats(),
            'performance_metrics': self.performance_metrics.copy(),
            'episode_step': self.episode_step,
            'fixed_uav_positions': self.fixed_uav_positions
        })
        return stats
    
    def render(self, mode='human'):
        """可视化环境状态（可选实现）"""
        if mode == 'human':
            print(f"Episode Step: {self.episode_step}")
            print(f"Performance Metrics: {self.performance_metrics}")
            print(f"Training Stats: {self.training_stats}")
    
    def close(self):
        """关闭环境"""
        pass
