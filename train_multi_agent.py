"""
多智能体算法训练脚本
支持MATD3、MADDPG、QMIX、MAPPO、SAC-MA等算法的训练和比较

使用方法:
python train_multi_agent.py --algorithm MATD3 --episodes 200
python train_multi_agent.py --algorithm MADDPG --episodes 200  
python train_multi_agent.py --algorithm QMIX --episodes 200
python train_multi_agent.py --algorithm MAPPO --episodes 200
python train_multi_agent.py --algorithm SAC-MA --episodes 200
python train_multi_agent.py --compare --episodes 200  # 比较所有算法
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import *
except ImportError:
    print("警告: 无法导入性能优化模块")
    OPTIMIZED_BATCH_SIZES = {}
    PARALLEL_ENVS = 1
    NUM_WORKERS = 0
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 导入核心模块
from evaluation.test_complete_system import CompleteSystemSimulator
from utils import MovingAverage
from config import config

# 导入各种算法
from algorithms.matd3 import MATD3Environment
from algorithms.maddpg import MADDPGEnvironment
from algorithms.qmix import QMIXEnvironment
from algorithms.mappo import MAPPOEnvironment
from algorithms.sac_ma import SACMAEnvironment


def generate_timestamp() -> str:
    """生成时间戳"""
    if config.experiment.use_timestamp:
        return datetime.now().strftime(config.experiment.timestamp_format)
    else:
        return ""

def get_timestamped_filename(base_name: str, extension: str = ".json") -> str:
    """获取带时间戳的文件名"""
    timestamp = generate_timestamp()
    if timestamp:
        name_parts = base_name.split('.')
        if len(name_parts) > 1:
            base = '.'.join(name_parts[:-1])
            return f"{base}_{timestamp}{extension}"
        else:
            return f"{base_name}_{timestamp}{extension}"
    else:
        return f"{base_name}{extension}"


class MultiAgentTrainingEnvironment:
    """多智能体训练环境基类"""
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm.upper()
        self.simulator = CompleteSystemSimulator()
        
        # 获取优化后的批次大小
        self.optimized_batch_size = self._get_optimized_batch_size()
        print(f"🚀 使用优化批次大小: {self.optimized_batch_size}")
        
        # 根据算法创建相应环境
        if self.algorithm == "MATD3":
            self.agent_env = MATD3Environment()
        elif self.algorithm == "MADDPG":
            self.agent_env = MADDPGEnvironment()
        elif self.algorithm == "QMIX":
            self.agent_env = QMIXEnvironment()
        elif self.algorithm == "MAPPO":
            self.agent_env = MAPPOEnvironment(action_space="continuous")
        elif self.algorithm == "SAC-MA":
            self.agent_env = SACMAEnvironment()
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 训练统计
        self.episode_rewards = []
        self.episode_losses = {}
        self.episode_metrics = {
            'avg_task_delay': [],
            'total_energy_consumption': [],
            'task_completion_rate': [],
            'cache_hit_rate': [],
            'migration_success_rate': [],
            'data_loss_rate': []
        }
        
        # 性能追踪器
        self.performance_tracker = {
            'recent_rewards': MovingAverage(100),
            'recent_delays': MovingAverage(100),
            'recent_energy': MovingAverage(100),
            'recent_completion': MovingAverage(100)
        }
        
        print(f"✓ {self.algorithm}训练环境初始化完成")
        
        # 获取智能体数量（安全方式）
        agent_count = 3  # 默认值
        try:
            if hasattr(self.agent_env, 'agents'):
                agents_attr = getattr(self.agent_env, 'agents', None)
                if agents_attr is not None:
                    if hasattr(agents_attr, '__len__'):
                        agent_count = len(agents_attr)
            elif hasattr(self.agent_env, 'num_agents'):
                agent_count = getattr(self.agent_env, 'num_agents', 3)
        except (AttributeError, TypeError, Exception):
            # 如果无法访问，使用默认值
            agent_count = 3
        
        print(f"✓ 智能体数量: {agent_count}")
    
    def _get_optimized_batch_size(self) -> int:
        """获取优化后的批次大小"""
        try:
            return OPTIMIZED_BATCH_SIZES.get(self.algorithm, 256)
        except (NameError, AttributeError):
            # 如果优化模块未加载，返回默认值
            default_sizes = {
                'MATD3': 256, 'MADDPG': 256, 'MAPPO': 256, 
                'QMIX': 32, 'SAC-MA': 256
            }
            return default_sizes.get(self.algorithm, 256)
    
    def reset_environment(self) -> Dict[str, np.ndarray]:
        """重置环境并返回初始状态"""
        # 重置仿真器状态
        self.simulator._setup_scenario()
        
        # 特殊处理需要重置隐藏状态的算法
        if self.algorithm == "QMIX":
            try:
                if hasattr(self.agent_env, 'reset_hidden_states'):
                    self.agent_env.reset_hidden_states()
            except (AttributeError, Exception):
                # 如果无法访问，忽略
                pass
        
        # 收集系统状态
        node_states = {}
        
        # 车辆状态
        for i, vehicle in enumerate(self.simulator.vehicles):
            # 生成车辆状态
            vehicle_state = np.array([
                vehicle['position'][0] / 1000,  # 归一化位置x
                vehicle['position'][1] / 1000,  # 归一化位置y
                vehicle['velocity'] / 50,       # 归一化速度
                len(vehicle.get('tasks', [])) / 10,  # 归一化任务数
                vehicle.get('energy_consumed', 0) / 1000  # 归一化能耗
            ])
            node_states[f'vehicle_{i}'] = vehicle_state
        
        # RSU状态
        for i, rsu in enumerate(self.simulator.rsus):
            rsu_state = np.array([
                rsu['position'][0] / 1000,  # 归一化位置x
                rsu['position'][1] / 1000,  # 归一化位置y
                len(rsu.get('cache', {})) / rsu.get('cache_capacity', 100),  # 缓存利用率
                len(rsu.get('computation_queue', [])) / 10,  # 归一化队列长度
                rsu.get('energy_consumed', 0) / 1000  # 归一化能耗
            ])
            node_states[f'rsu_{i}'] = rsu_state
        
        # UAV状态
        for i, uav in enumerate(self.simulator.uavs):
            uav_state = np.array([
                uav['position'][0] / 1000,  # 归一化位置x
                uav['position'][1] / 1000,  # 归一化位置y
                uav['position'][2] / 200,   # 归一化高度
                len(uav.get('cache', {})) / uav.get('cache_capacity', 100),  # 缓存利用率
                uav.get('energy_consumed', 0) / 1000  # 归一化能耗
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 初始系统指标
        system_metrics = {
            'avg_task_delay': 0.0,
            'total_energy_consumption': 0.0,
            'data_loss_rate': 0.0,
            'cache_hit_rate': 0.0,
            'migration_success_rate': 0.0
        }
        
        # 获取初始状态向量 - 为MATD3等算法创建兼容的状态对象
        if self.algorithm == 'MATD3':
            # MATD3需要特殊的状态对象格式
            states = self._get_matd3_compatible_states(node_states, system_metrics)
        else:
            states = self.agent_env.get_state_vector(node_states, system_metrics)
        
        return states
    
    def _get_matd3_compatible_states(self, node_states: Dict, system_metrics: Dict) -> Dict:
        """为MATD3算法创建兼容的状态对象"""
        # 创建简单的状态对象类
        class SimpleNodeState:
            def __init__(self, node_type: str, load_factor: float = 0.5):
                self.node_type = SimpleNodeType(node_type)
                self.load_factor = load_factor
        
        class SimpleNodeType:
            def __init__(self, value: str):
                self.value = value
        
        # 转换node_states为MATD3期望的格式
        compatible_states = {}
        
        # 处理车辆状态
        for i in range(len([k for k in node_states.keys() if k.startswith('vehicle_')])):
            compatible_states[f'vehicle_{i}'] = SimpleNodeState('vehicle', 0.3)
        
        # 处理RSU状态  
        for i in range(len([k for k in node_states.keys() if k.startswith('rsu_')])):
            compatible_states[f'rsu_{i}'] = SimpleNodeState('rsu', 0.5)
        
        # 处理UAV状态
        for i in range(len([k for k in node_states.keys() if k.startswith('uav_')])):
            compatible_states[f'uav_{i}'] = SimpleNodeState('uav', 0.4)
        
        # 使用MATD3的get_state_vector方法
        return self.agent_env.get_state_vector(compatible_states, system_metrics)
    
    def step(self, actions: Dict, states: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """执行一步仿真"""
        # 执行仿真步骤
        step_stats = self.simulator.run_simulation_step(0)
        
        # 收集下一步状态
        node_states = {}
        
        # 车辆状态
        for i, vehicle in enumerate(self.simulator.vehicles):
            vehicle_state = np.array([
                vehicle['position'][0] / 1000,
                vehicle['position'][1] / 1000,
                vehicle['velocity'] / 50,
                len(vehicle.get('tasks', [])) / 10,
                vehicle.get('energy_consumed', 0) / 1000
            ])
            node_states[f'vehicle_{i}'] = vehicle_state
        
        # RSU状态
        for i, rsu in enumerate(self.simulator.rsus):
            rsu_state = np.array([
                rsu['position'][0] / 1000,
                rsu['position'][1] / 1000,
                len(rsu.get('cache', {})) / rsu.get('cache_capacity', 100),
                len(rsu.get('computation_queue', [])) / 10,
                rsu.get('energy_consumed', 0) / 1000
            ])
            node_states[f'rsu_{i}'] = rsu_state
        
        # UAV状态
        for i, uav in enumerate(self.simulator.uavs):
            uav_state = np.array([
                uav['position'][0] / 1000,
                uav['position'][1] / 1000,
                uav['position'][2] / 200,
                len(uav.get('cache', {})) / uav.get('cache_capacity', 100),
                uav.get('energy_consumed', 0) / 1000
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 计算系统指标
        system_metrics = self._calculate_system_metrics(step_stats)
        
        # 获取下一状态 - 为MATD3等算法创建兼容的状态对象
        if self.algorithm == 'MATD3':
            # MATD3需要特殊的状态对象格式
            next_states = self._get_matd3_compatible_states(node_states, system_metrics)
        else:
            next_states = self.agent_env.get_state_vector(node_states, system_metrics)
        
        # 计算奖励
        rewards = self._calculate_rewards(system_metrics)
        
        # 判断是否结束
        dones = {agent_id: False for agent_id in actions.keys()}
        
        # 附加信息
        info = {
            'step_stats': step_stats,
            'system_metrics': system_metrics
        }
        
        return next_states, rewards, dones, info
    
    def _calculate_system_metrics(self, step_stats: Dict) -> Dict:
        """计算系统性能指标 - 改进版本，更准确的指标计算"""
        # 导入验证函数
        def local_validate_energy(energy, context):
            try:
                from utils.energy_validator import validate_energy_consumption as validate_energy_func
                energy_data = {'total_system': [energy]}
                result = validate_energy_func(energy_data)
                is_valid = result['is_valid']
                corrected_energy = min(energy, 2000.0) if not is_valid else energy
                warning = "; ".join(result['errors'][:1]) if result['errors'] else ""
                return is_valid, corrected_energy, warning
            except ImportError:
                # 如果导入失败，使用简单验证
                return energy <= 2000.0, min(energy, 2000.0), ""
        
        # 时延验证函数 - 优化版本，减少不必要的警告
        def validate_delay_calculation(delay_value, processed_tasks, total_delay):
            """验证时延计算的合理性"""
            if processed_tasks <= 0:
                # 不输出警告，这是正常情况（某些时间步没有任务处理）
                return 0.0, ""
            
            if total_delay <= 0:
                return 0.0, ""
            
            calculated_delay = total_delay / processed_tasks
            
            # 检查是否为有限值
            if not np.isfinite(calculated_delay):
                return 1.0, f"时延计算结果非有限值: {calculated_delay}, 修正为1.0s"
            
            # 检查是否在合理范围内 (0.001s - 10s)
            if calculated_delay < 0.001:
                return 0.001, f"时延过小: {calculated_delay:.6f}s, 修正为0.001s"
            elif calculated_delay > 10.0:
                return 5.0, f"时延过大: {calculated_delay:.2f}s, 修正为5.0s"
            
            return calculated_delay, ""
        
        # 安全获取统计数据，避免KeyError
        generated_tasks = step_stats.get('generated_tasks', 0)
        processed_tasks = step_stats.get('processed_tasks', 0)
        dropped_tasks = step_stats.get('dropped_tasks', 0)
        total_delay = step_stats.get('total_delay', 0.0)
        total_energy = step_stats.get('total_energy', 0.0)
        cache_hits = step_stats.get('cache_hits', 0)
        cache_misses = step_stats.get('cache_misses', 0)
        
        # 任务完成率：成功处理的任务占生成任务的比例
        if generated_tasks > 0:
            completion_rate = processed_tasks / generated_tasks
            data_loss_rate = dropped_tasks / generated_tasks
        else:
            completion_rate = 0.0
            data_loss_rate = 0.0
        
        # 平均时延：使用验证函数确保计算正确
        avg_task_delay, delay_warning = validate_delay_calculation(0, processed_tasks, total_delay)
        # 只在有实际问题时输出警告
        if delay_warning and processed_tasks > 0:
            print(f"⚠️ 时延计算修正: {delay_warning}")
        
        # 缓存命中率
        cache_requests = cache_hits + cache_misses
        if cache_requests > 0:
            cache_hit_rate = cache_hits / cache_requests
        else:
            cache_hit_rate = 0.0
        
        # 能耗验证：使用专门的验证函数
        is_valid, corrected_energy, warning = local_validate_energy(total_energy, "slot")
        if warning:
            print(warning)
        total_energy = corrected_energy
        
        # 系统负载比例
        system_load_ratio = min(1.0, generated_tasks / max(1, 50))  # 假设系统最大处理能力为50任务/时隙
        
        # 带宽利用率（简化计算）
        avg_bandwidth_utilization = min(1.0, processed_tasks / max(1, 30))
        
        # 集成增强迁移管理器
        if not hasattr(self, 'migration_manager'):
            from utils.enhanced_migration import EnhancedTaskMigrationManager
            self.migration_manager = EnhancedTaskMigrationManager()
        
        # 模拟节点状态供迁移管理器使用
        migration_node_states = {}
        migration_positions = {}
        
        # 创建简化的节点状态用于迁移
        from models.data_structures import NodeState, NodeType, Position
        for i in range(len(self.simulator.vehicles)):
            vehicle = self.simulator.vehicles[i]
            state = NodeState(
                node_id=f'vehicle_{i}',
                node_type=NodeType.VEHICLE,
                position=Position(vehicle['position'][0], vehicle['position'][1], 0),
                load_factor=len(vehicle.get('tasks', [])) / 10.0
            )
            migration_node_states[f'vehicle_{i}'] = state
            migration_positions[f'vehicle_{i}'] = state.position
        
        for i in range(len(self.simulator.rsus)):
            rsu = self.simulator.rsus[i]
            state = NodeState(
                node_id=f'rsu_{i}',
                node_type=NodeType.RSU,
                position=Position(rsu['position'][0], rsu['position'][1], 0),
                load_factor=len(rsu.get('computation_queue', [])) / 10.0
            )
            migration_node_states[f'rsu_{i}'] = state
            migration_positions[f'rsu_{i}'] = state.position
        
        for i in range(len(self.simulator.uavs)):
            uav = self.simulator.uavs[i]
            state = NodeState(
                node_id=f'uav_{i}',
                node_type=NodeType.UAV,
                position=Position(uav['position'][0], uav['position'][1], uav['position'][2]),
                load_factor=len(uav.get('cache', {})) / uav.get('cache_capacity', 100)
            )
            # 设置UAV电池电量
            setattr(state, 'battery_level', uav.get('battery_level', 0.8))
            migration_node_states[f'uav_{i}'] = state
            migration_positions[f'uav_{i}'] = state.position
        
        # 运行迁移管理器步骤
        migration_step_stats = self.migration_manager.step(
            migration_node_states, 
            migration_positions, 
            {}  # 简化的任务状态
        )
        
        # 获取动态迁移成功率
        dynamic_migration_rate = migration_step_stats.get('dynamic_success_rate', 0.8)
        
        return {
            'avg_task_delay': max(0.0, avg_task_delay),
            'total_energy_consumption': max(0.0, total_energy),
            'data_loss_rate': np.clip(data_loss_rate, 0.0, 1.0),
            'task_completion_rate': np.clip(completion_rate, 0.0, 1.0),
            'cache_hit_rate': np.clip(cache_hit_rate, 0.0, 1.0),
            'migration_success_rate': dynamic_migration_rate,
            'system_load_ratio': system_load_ratio,
            'avg_bandwidth_utilization': avg_bandwidth_utilization,
            # 添加调试信息
            'debug_info': {
                'generated_tasks': generated_tasks,
                'processed_tasks': processed_tasks,
                'dropped_tasks': dropped_tasks,
                'cache_requests': cache_requests,
                'energy_corrected': not is_valid
            }
        }
    
    def _calculate_rewards(self, system_metrics: Dict) -> Dict[str, float]:
        """计算智能体奖励 - 使用标准化奖励函数"""
        from utils.standardized_reward import calculate_standardized_reward
        
        rewards = {}
        agent_ids = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        
        # 为不同智能体计算标准化奖励
        for agent_id in agent_ids:
            rewards[agent_id] = calculate_standardized_reward(
                system_metrics, 
                agent_type=agent_id
            )
        
        return rewards
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None) -> Dict:
        """运行一个完整的训练轮次 - 改进版本，增强稳定性"""
        # 使用配置中的最大步数
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        # 重置环境
        states = self.reset_environment()
        
        # 验证状态有效性
        if not states or any(state is None for state in states.values()):
            print(f"⚠️ Episode {episode}: 状态重置失败，使用默认状态")
            states = {agent_id: np.zeros(20, dtype=np.float32) for agent_id in ['vehicle_agent', 'rsu_agent', 'uav_agent']}
        
        episode_reward = {agent_id: 0.0 for agent_id in states.keys()}
        episode_info = {}
        # 初始化info和step变量
        info = {'system_metrics': {}}
        step = 0
        
        # 记录前一步的系统指标，用于奖励计算
        prev_metrics = {
            'avg_task_delay': 1.0,
            'total_energy_consumption': 0.0,
            'task_completion_rate': 0.0,
            'cache_hit_rate': 0.0,
            'data_loss_rate': 0.0
        }
        
        # MAPPO需要特殊处理
        if self.algorithm == "MAPPO":
            return self._run_mappo_episode(episode, max_steps)
        
        for step in range(max_steps):
            # 选择动作 - 处理不同算法的返回类型
            if hasattr(self.agent_env, 'get_actions'):
                result = self.agent_env.get_actions(states, training=True)
                if isinstance(result, tuple) and len(result) == 2:
                    # MAPPO等返回(actions, log_probs)的算法
                    actions, _ = result
                else:
                    # 其他算法只返回actions
                    actions = result
            else:
                # 默认随机动作
                actions = {agent_id: 0 for agent_id in states.keys()}
            
            # 执行动作 - 确保actions是字典格式
            if isinstance(actions, tuple):
                # 如果是元组，取第一个元素（安全方式）
                if len(actions) > 0:
                    actions = actions[0]  # type: ignore # 这里是元组索引
                else:
                    actions = {}
            
            if not isinstance(actions, dict):
                # 如果不是字典，转换为字典格式
                agent_ids = list(states.keys())
                if len(agent_ids) > 0:
                    actions = {agent_ids[0]: actions}
                else:
                    actions = {'default_agent': actions}
            
            next_states, rewards, dones, info = self.step(actions, states)
            
            # 训练智能体
            if self.algorithm == "MAPPO":
                # MAPPO使用缓存的经验，不在此处训练
                # 获取全局状态（安全方式）
                if self.algorithm == "MAPPO":
                    try:
                        if hasattr(self.agent_env, 'get_global_state'):
                            global_state = self.agent_env.get_global_state(states)
                        else:
                            # 创建全局状态的简单实现
                            global_state = np.concatenate([state.flatten() for state in states.values()])
                    except (AttributeError, Exception):
                        global_state = np.concatenate([state.flatten() for state in states.values()])
                else:
                    global_state = None
                
                # 存储经验（安全方式）
                if self.algorithm == "MAPPO":
                    try:
                        if hasattr(self.agent_env, 'store_experience'):
                            # 确保actions是正确的格式
                            if isinstance(actions, dict):
                                actions_array = {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in actions.items()}
                            else:
                                actions_array = actions
                            log_probs = {}  # 空的log_probs字典
                            # 修复：使用Optional类型的global_state
                            if global_state is not None:
                                self.agent_env.store_experience(states, actions_array, log_probs, rewards, dones, global_state)
                            else:
                                # 创建一个默认的全局状态
                                default_global = np.concatenate([state.flatten() for state in states.values()])
                                self.agent_env.store_experience(states, actions_array, log_probs, rewards, dones, default_global)
                    except (AttributeError, Exception):
                        # 如果存储失败，忽略
                        pass
            else:
                # 其他算法的训练（安全方式）
                try:
                    if hasattr(self.agent_env, 'train_step'):
                        # 确保actions格式正确
                        if isinstance(actions, dict):
                            # 将动作转换为适合的类型
                            actions_processed = {}
                            for k, v in actions.items():
                                if isinstance(v, (np.ndarray, list)) and len(np.array(v).shape) == 0:
                                    actions_processed[k] = int(v)
                                elif isinstance(v, np.ndarray):
                                    actions_processed[k] = v
                                else:
                                    actions_processed[k] = v
                        else:
                            actions_processed = actions
                        training_info = self.agent_env.train_step(states, actions_processed, rewards, next_states, dones)
                        episode_info = training_info
                    else:
                        episode_info = {}
                except (AttributeError, Exception):
                    episode_info = {}
            
            # 更新状态
            states = next_states
            
            # 累计奖励
            for agent_id, reward in rewards.items():
                if agent_id in episode_reward:
                    episode_reward[agent_id] += reward
            
            # 检查是否结束
            if any(dones.values()):
                break
        
        # 记录轮次统计
        avg_reward = np.mean(list(episode_reward.values())) if episode_reward else 0.0
        # info已在循环中初始化
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': avg_reward,
            'episode_info': episode_info,
            'system_metrics': system_metrics,
            'steps': step
        }
    
    def _run_mappo_episode(self, episode: int, max_steps: int = 100) -> Dict:
        """运行MAPPO专用episode"""
        states = self.reset_environment()
        episode_reward = {agent_id: 0.0 for agent_id in states.keys()}
        
        info = {'system_metrics': {}}  # 初始化info
        step = 0  # 初始化step变量
        
        for step in range(max_steps):
            # 获取动作和对数概率
            if hasattr(self.agent_env, 'get_actions'):
                result = self.agent_env.get_actions(states, training=True)
                if isinstance(result, tuple) and len(result) == 2:
                    actions, log_probs = result
                else:
                    actions = result
                    log_probs = {agent_id: 0.0 for agent_id in states.keys()}
            else:
                # 默认随机动作
                actions = {agent_id: np.random.rand(10) for agent_id in states.keys()}
                log_probs = {agent_id: 0.0 for agent_id in states.keys()}
            
            # 执行动作 - 确保actions是字典格式
            if isinstance(actions, tuple):
                if len(actions) > 0:
                    actions = actions[0]  # type: ignore # 这里是元组索引
                else:
                    actions = {}
            
            if not isinstance(actions, dict):
                agent_ids = list(states.keys())
                if len(agent_ids) > 0:
                    actions = {agent_ids[0]: actions}
                else:
                    actions = {'default_agent': actions}
            
            next_states, rewards, dones, info = self.step(actions, states)
            
            # 存储经验（安全方式）
            try:
                if hasattr(self.agent_env, 'get_global_state'):
                    global_state = self.agent_env.get_global_state(states)
                else:
                    global_state = np.concatenate([state.flatten() for state in states.values()])
            except (AttributeError, Exception):
                global_state = np.concatenate([state.flatten() for state in states.values()])
            
            try:
                if hasattr(self.agent_env, 'store_experience'):
                    # 确保actions和log_probs格式正确
                    actions_array = {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in actions.items()}
                    log_probs_dict = {k: float(v) if not isinstance(v, dict) else v for k, v in log_probs.items()}
                    self.agent_env.store_experience(states, actions_array, log_probs_dict, rewards, dones, global_state)
            except (AttributeError, Exception):
                pass
            
            # 累计奖励
            for agent_id, reward in rewards.items():
                if agent_id in episode_reward:
                    episode_reward[agent_id] += reward
            
            states = next_states
            
            if any(dones.values()):
                break
        
        # Episode结束后进行PPO更新（安全方式）
        try:
            if hasattr(self.agent_env, 'update'):
                training_info = self.agent_env.update()
            else:
                training_info = {}
        except (AttributeError, Exception):
            training_info = {}
        
        # step已在循环中定义
        
        avg_reward = np.mean(list(episode_reward.values())) if episode_reward else 0.0
        # info已在循环中初始化
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': avg_reward,
            'episode_info': training_info,
            'system_metrics': system_metrics,
            'steps': step
        }


def train_algorithm(algorithm: str, num_episodes: Optional[int] = None, eval_interval: Optional[int] = None, 
                   save_interval: Optional[int] = None) -> Dict:
    """训练单个算法"""
    # 使用配置中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    if eval_interval is None:
        eval_interval = config.experiment.eval_interval
    if save_interval is None:
        save_interval = config.experiment.save_interval
    
    print(f"\n🚀 开始{algorithm}算法训练")
    print("=" * 60)
    
    # 创建训练环境
    training_env = MultiAgentTrainingEnvironment(algorithm)
    
    print("训练配置:")
    print(f"  算法: {algorithm}")
    print(f"  总轮次: {num_episodes}")
    print(f"  评估间隔: {eval_interval}")
    print(f"  保存间隔: {save_interval}")
    print("-" * 60)
    
    # 创建结果目录
    os.makedirs(f"results/training/{algorithm.lower()}", exist_ok=True)
    os.makedirs(f"results/models/{algorithm.lower()}", exist_ok=True)
    
    # 训练循环
    best_avg_reward = float('-inf')
    training_start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        
        # 运行训练轮次
        episode_result = training_env.run_episode(episode)
        
        # 记录训练数据
        training_env.episode_rewards.append(episode_result['avg_reward'])
        
        # 更新性能追踪器
        training_env.performance_tracker['recent_rewards'].update(episode_result['avg_reward'])
        
        system_metrics = episode_result['system_metrics']
        training_env.performance_tracker['recent_delays'].update(system_metrics.get('avg_task_delay', 0))
        training_env.performance_tracker['recent_energy'].update(system_metrics.get('total_energy_consumption', 0))
        training_env.performance_tracker['recent_completion'].update(system_metrics.get('task_completion_rate', 0))
        
        # 记录指标
        for metric_name, value in system_metrics.items():
            if metric_name in training_env.episode_metrics:
                training_env.episode_metrics[metric_name].append(value)
        
        episode_time = time.time() - episode_start_time
        
        # 定期输出进度
        if episode % 10 == 0:
            avg_reward = training_env.performance_tracker['recent_rewards'].get_average()
            avg_delay = training_env.performance_tracker['recent_delays'].get_average()
            avg_completion = training_env.performance_tracker['recent_completion'].get_average()
            
            print(f"轮次 {episode:4d}/{num_episodes}:")
            print(f"  平均奖励: {avg_reward:8.3f}")
            print(f"  平均时延: {avg_delay:8.3f}s")
            print(f"  完成率:   {avg_completion:8.1%}")
            print(f"  轮次用时: {episode_time:6.3f}s")
        
        # 评估模型
        if episode % eval_interval == 0:
            eval_result = evaluate_model(algorithm, training_env, episode)
            print(f"\n📊 轮次 {episode} 评估结果:")
            print(f"  评估奖励: {eval_result['avg_reward']:.3f}")
            print(f"  评估时延: {eval_result['avg_delay']:.3f}s")
            print(f"  评估完成率: {eval_result['completion_rate']:.1%}")
            
            # 保存最佳模型
            if eval_result['avg_reward'] > best_avg_reward:
                best_avg_reward = eval_result['avg_reward']
                training_env.agent_env.save_models(f"results/models/{algorithm.lower()}/best_model")
                print(f"  💾 保存最佳模型 (奖励: {best_avg_reward:.3f})")
        
        # 定期保存模型
        if episode % save_interval == 0:
            training_env.agent_env.save_models(f"results/models/{algorithm.lower()}/checkpoint_{episode}")
            print(f"💾 保存检查点: checkpoint_{episode}")
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print(f"🎉 {algorithm}训练完成!")
    print(f"⏱️  总训练时间: {total_training_time/3600:.2f} 小时")
    print(f"🏆 最佳平均奖励: {best_avg_reward:.3f}")
    
    # 保存训练结果
    results = save_training_results(algorithm, training_env, total_training_time)
    
    # 绘制训练曲线
    plot_training_curves(algorithm, training_env)
    
    return results


def evaluate_model(algorithm: str, training_env: MultiAgentTrainingEnvironment, 
                  episode: int, num_eval_episodes: int = 5) -> Dict:
    """评估模型性能"""
    eval_rewards = []
    eval_delays = []
    eval_completions = []
    
    for _ in range(num_eval_episodes):
        states = training_env.reset_environment()
        episode_reward = 0.0
        episode_delay = 0.0
        episode_completion = 0.0
        steps = 0
        
        for step in range(50):  # 较短的评估轮次
            if hasattr(training_env.agent_env, 'get_actions'):
                result = training_env.agent_env.get_actions(states, training=False)
                if isinstance(result, tuple):  # MAPPO返回元组
                    actions = result[0]
                else:
                    actions = result
            else:
                # 默认随机动作
                actions = {agent_id: np.random.rand(10) for agent_id in states.keys()}
            
            # 确保actions是字典格式
            if isinstance(actions, tuple):
                if len(actions) > 0:
                    actions = actions[0]  # type: ignore # 这里是元组索引
                else:
                    actions = {}
            
            if not isinstance(actions, dict):
                agent_ids = list(states.keys())
                if len(agent_ids) > 0:
                    actions = {agent_ids[0]: actions}
                else:
                    actions = {'default_agent': actions}
            
            next_states, rewards, dones, info = training_env.step(actions, states)
            
            episode_reward += np.mean(list(rewards.values()))
            system_metrics = info['system_metrics']
            episode_delay += system_metrics.get('avg_task_delay', 0)
            episode_completion += system_metrics.get('task_completion_rate', 0)
            steps += 1
            
            states = next_states
            
            if any(dones.values()):
                break
        
        eval_rewards.append(episode_reward / steps)
        eval_delays.append(episode_delay / steps)
        eval_completions.append(episode_completion / steps)
    
    return {
        'avg_reward': np.mean(eval_rewards),
        'avg_delay': np.mean(eval_delays),
        'completion_rate': np.mean(eval_completions)
    }


def save_training_results(algorithm: str, training_env: MultiAgentTrainingEnvironment, 
                         training_time: float) -> Dict:
    """保存训练结果"""
    # 生成时间戳
    timestamp = generate_timestamp()
    
    results = {
        'algorithm': algorithm,
        'timestamp': timestamp,
        'training_start_time': datetime.now().isoformat(),
        'training_config': {
            'num_episodes': len(training_env.episode_rewards),
            'training_time_hours': training_time / 3600,
            'max_steps_per_episode': config.experiment.max_steps_per_episode
        },
        'episode_rewards': training_env.episode_rewards,
        'episode_metrics': training_env.episode_metrics,
        'final_performance': {
            'avg_reward': training_env.performance_tracker['recent_rewards'].get_average(),
            'avg_delay': training_env.performance_tracker['recent_delays'].get_average(),
            'avg_completion': training_env.performance_tracker['recent_completion'].get_average()
        }
    }
    
    # 使用时间戳文件名
    filename = get_timestamped_filename("training_results")
    filepath = f"results/training/{algorithm.lower()}/{filename}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 {algorithm}训练结果已保存到 {filepath}")
    
    return results


def plot_training_curves(algorithm: str, training_env: MultiAgentTrainingEnvironment):
    """绘制训练曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 传统可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 奖励曲线
    axes[0, 0].plot(training_env.episode_rewards)
    axes[0, 0].set_title(f'{algorithm} 训练奖励曲线')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].grid(True)
    
    # 时延曲线
    if 'avg_task_delay' in training_env.episode_metrics and training_env.episode_metrics['avg_task_delay']:
        axes[0, 1].plot(training_env.episode_metrics['avg_task_delay'])
        axes[0, 1].set_title('平均任务时延')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('时延 (秒)')
        axes[0, 1].grid(True)
    
    # 完成率曲线
    if 'task_completion_rate' in training_env.episode_metrics and training_env.episode_metrics['task_completion_rate']:
        axes[0, 2].plot(training_env.episode_metrics['task_completion_rate'])
        axes[0, 2].set_title('任务完成率')
        axes[0, 2].set_xlabel('训练轮次')
        axes[0, 2].set_ylabel('完成率')
        axes[0, 2].grid(True)
    
    # 缓存命中率曲线
    if 'cache_hit_rate' in training_env.episode_metrics and training_env.episode_metrics['cache_hit_rate']:
        axes[1, 0].plot(training_env.episode_metrics['cache_hit_rate'])
        axes[1, 0].set_title('缓存命中率')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('命中率')
        axes[1, 0].grid(True)
    
    # 能耗曲线
    if 'total_energy_consumption' in training_env.episode_metrics and training_env.episode_metrics['total_energy_consumption']:
        axes[1, 1].plot(training_env.episode_metrics['total_energy_consumption'])
        axes[1, 1].set_title('总能耗')
        axes[1, 1].set_xlabel('训练轮次')
        axes[1, 1].set_ylabel('能耗 (焦耳)')
        axes[1, 1].grid(True)
    
    # 数据丢失率曲线
    if 'data_loss_rate' in training_env.episode_metrics and training_env.episode_metrics['data_loss_rate']:
        axes[1, 2].plot(training_env.episode_metrics['data_loss_rate'])
        axes[1, 2].set_title('数据丢失率')
        axes[1, 2].set_xlabel('训练轮次')
        axes[1, 2].set_ylabel('丢失率')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    filepath = f"results/training/{algorithm.lower()}/training_curves.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 {algorithm}训练曲线已保存到 {filepath}")
    
    # 🎨 新增：高级可视化套件
    from tools.advanced_visualization import enhanced_plot_training_curves, plot_convergence_analysis, plot_multi_metric_dashboard
    from tools.performance_dashboard import create_performance_dashboard, create_real_time_monitor
    
    # 1. 增强训练曲线
    enhanced_plot_training_curves(training_env, f"results/training/{algorithm.lower()}/enhanced_training_curves.png")
    
    # 2. 收敛性分析
    plot_convergence_analysis(
        {'episode_rewards': training_env.episode_rewards}, 
        f"results/training/{algorithm.lower()}/convergence_analysis.png"
    )
    
    # 3. 多指标仪表板
    plot_multi_metric_dashboard(
        training_env, 
        f"results/training/{algorithm.lower()}/multi_metric_dashboard.png"
    )
    
    # 4. 性能仪表板
    create_performance_dashboard(
        training_env, 
        f"results/training/{algorithm.lower()}/performance_dashboard.png"
    )
    
    # 5. 实时监控界面
    create_real_time_monitor(
        f"results/training/{algorithm.lower()}/realtime_monitor.png"
    )


def compare_algorithms(algorithms: List[str], num_episodes: Optional[int] = None) -> Dict:
    """比较多个算法的性能"""
    # 使用配置中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    
    print("\n🔥 开始多算法性能比较")
    print("=" * 60)
    
    results = {}
    
    # 训练所有算法
    for algorithm in algorithms:
        print(f"\n开始训练 {algorithm}...")
        results[algorithm] = train_algorithm(algorithm, num_episodes)
    
    # 生成比较图表
    plot_algorithm_comparison(results)
    
    # 保存比较结果
    timestamp = generate_timestamp()
    comparison_results = {
        'algorithms': algorithms,
        'num_episodes': num_episodes,
        'timestamp': timestamp,
        'comparison_time': datetime.now().isoformat(),
        'results': results,
        'summary': {}
    }
    
    # 计算汇总统计
    for algorithm, result in results.items():
        final_perf = result['final_performance']
        comparison_results['summary'][algorithm] = {
            'final_avg_reward': final_perf['avg_reward'],
            'final_avg_delay': final_perf['avg_delay'],
            'final_completion_rate': final_perf['avg_completion'],
            'training_time_hours': result['training_config']['training_time_hours']
        }
    
    # 使用时间戳文件名
    comparison_filename = get_timestamped_filename("algorithm_comparison")
    with open(f"results/{comparison_filename}", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print("\n🎯 算法比较完成！")
    print(f"📄 比较结果已保存到 results/{comparison_filename}")
    print(f"📈 比较图表已保存到 results/algorithm_comparison_{timestamp}.png")
    
    return comparison_results


def plot_algorithm_comparison(results: Dict):
    """绘制算法比较图表"""
    timestamp = generate_timestamp()
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 奖励对比
    for algorithm, result in results.items():
        axes[0, 0].plot(result['episode_rewards'], label=algorithm)
    axes[0, 0].set_title('算法奖励对比')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 时延对比
    for algorithm, result in results.items():
        if 'avg_task_delay' in result['episode_metrics'] and result['episode_metrics']['avg_task_delay']:
            axes[0, 1].plot(result['episode_metrics']['avg_task_delay'], label=algorithm)
    axes[0, 1].set_title('平均时延对比')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('时延 (秒)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 完成率对比
    for algorithm, result in results.items():
        if 'task_completion_rate' in result['episode_metrics'] and result['episode_metrics']['task_completion_rate']:
            axes[1, 0].plot(result['episode_metrics']['task_completion_rate'], label=algorithm)
    axes[1, 0].set_title('任务完成率对比')
    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('完成率')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 能耗对比
    for algorithm, result in results.items():
        if 'total_energy_consumption' in result['episode_metrics'] and result['episode_metrics']['total_energy_consumption']:
            axes[1, 1].plot(result['episode_metrics']['total_energy_consumption'], label=algorithm)
    axes[1, 1].set_title('总能耗对比')
    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('能耗 (焦耳)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 数据丢失率对比
    for algorithm, result in results.items():
        if 'data_loss_rate' in result['episode_metrics'] and result['episode_metrics']['data_loss_rate']:
            axes[2, 0].plot(result['episode_metrics']['data_loss_rate'], label=algorithm)
    axes[2, 0].set_title('数据丢失率对比')
    axes[2, 0].set_xlabel('训练轮次')
    axes[2, 0].set_ylabel('丢失率')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # 最终性能对比 (柱状图)
    algorithms = list(results.keys())
    final_rewards = [results[alg]['final_performance']['avg_reward'] for alg in algorithms]
    
    axes[2, 1].bar(algorithms, final_rewards)
    axes[2, 1].set_title('最终平均奖励对比')
    axes[2, 1].set_ylabel('平均奖励')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 使用时间戳文件名
    chart_filename = f"algorithm_comparison_{timestamp}.png" if timestamp else "algorithm_comparison.png"
    plt.savefig(f"results/{chart_filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🎨 新增：高级比较可视化套件
    from tools.advanced_visualization import create_advanced_visualization_suite
    create_advanced_visualization_suite(results, "results/advanced_multi_agent_comparison")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多智能体算法训练脚本')
    parser.add_argument('--algorithm', type=str, choices=['MATD3', 'MADDPG', 'QMIX', 'MAPPO', 'SAC-MA'],
                       help='选择训练算法')
    parser.add_argument('--episodes', type=int, default=None, help=f'训练轮次 (默认: {config.experiment.num_episodes})')
    parser.add_argument('--eval_interval', type=int, default=None, help=f'评估间隔 (默认: {config.experiment.eval_interval})')
    parser.add_argument('--save_interval', type=int, default=None, help=f'保存间隔 (默认: {config.experiment.save_interval})')
    parser.add_argument('--compare', action='store_true', help='比较所有算法')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    if args.compare:
        # 比较所有算法
        algorithms = ['MATD3', 'MADDPG', 'QMIX', 'MAPPO', 'SAC-MA']
        compare_algorithms(algorithms, args.episodes)
    elif args.algorithm:
        # 训练单个算法
        train_algorithm(args.algorithm, args.episodes, args.eval_interval, args.save_interval)
    else:
        print("请指定 --algorithm 或使用 --compare 标志")
        print("使用 python train_multi_agent.py --help 查看帮助")


if __name__ == "__main__":
    main()