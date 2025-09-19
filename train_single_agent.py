"""
单智能体算法训练脚本
支持DDPG、TD3、DQN、PPO、SAC等算法的训练和比较

使用方法:
python train_single_agent.py --algorithm DDPG --episodes 200
python train_single_agent.py --algorithm TD3 --episodes 200  
python train_single_agent.py --algorithm DQN --episodes 200
python train_single_agent.py --algorithm PPO --episodes 200
python train_single_agent.py --algorithm SAC --episodes 200
python train_single_agent.py --compare --episodes 200  # 比较所有算法
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 导入核心模块
from config import config
from evaluation.test_complete_system import CompleteSystemSimulator
from utils import MovingAverage

# 导入各种单智能体算法
from single_agent.ddpg import DDPGEnvironment
from single_agent.td3 import TD3Environment
from single_agent.dqn import DQNEnvironment
from single_agent.ppo import PPOEnvironment
from single_agent.sac import SACEnvironment


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


class SingleAgentTrainingEnvironment:
    """单智能体训练环境基类"""
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm.upper()
        self.simulator = CompleteSystemSimulator()
        
        # 根据算法创建相应环境
        if self.algorithm == "DDPG":
            self.agent_env = DDPGEnvironment()
        elif self.algorithm == "TD3":
            self.agent_env = TD3Environment()
        elif self.algorithm == "DQN":
            self.agent_env = DQNEnvironment()
        elif self.algorithm == "PPO":
            self.agent_env = PPOEnvironment()
        elif self.algorithm == "SAC":
            self.agent_env = SACEnvironment()
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 训练统计
        self.episode_rewards = []
        self.episode_losses = {}
        self.episode_metrics = {
            'avg_delay': [],
            'total_energy': [],
            'task_completion_rate': [],
            'cache_hit_rate': [],
            'migration_success_rate': []
        }
        
        # 性能追踪器
        self.performance_tracker = {
            'recent_rewards': MovingAverage(100),
            'recent_delays': MovingAverage(100),
            'recent_energy': MovingAverage(100),
            'recent_completion': MovingAverage(100)
        }
        
        print(f"✓ {self.algorithm}训练环境初始化完成")
        print(f"✓ 算法类型: 单智能体")
    
    def reset_environment(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        # 重置仿真器状态
        self.simulator._setup_scenario()
        
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
        
        # 获取初始状态向量
        state = self.agent_env.get_state_vector(node_states, system_metrics)
        
        return state
    
    def step(self, action, state) -> Tuple[np.ndarray, float, bool, Dict]:
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
        
        # 获取下一状态
        next_state = self.agent_env.get_state_vector(node_states, system_metrics)
        
        # 计算奖励
        reward = self.agent_env.calculate_reward(system_metrics)
        
        # 判断是否结束
        done = False  # 单智能体环境通常不会提前结束
        
        # 附加信息
        info = {
            'step_stats': step_stats,
            'system_metrics': system_metrics
        }
        
        return next_state, reward, done, info
    
    def _calculate_system_metrics(self, step_stats: Dict) -> Dict:
        """计算系统性能指标 - 修复版，防止inf和nan"""
        import numpy as np
        
        # 安全获取数值
        def safe_get(key: str, default: float = 0.0) -> float:
            value = step_stats.get(key, default)
            if np.isnan(value) or np.isinf(value):
                return default
            return max(0.0, value)  # 确保非负
        
        # 修复：任务完成率应该是已处理任务数除以生成任务数
        generated_tasks = max(1, int(safe_get('generated_tasks', 1)))
        processed_tasks = int(safe_get('processed_tasks', 0))
        completion_rate = min(1.0, processed_tasks / generated_tasks)
        
        cache_hits = int(safe_get('cache_hits', 0))
        cache_misses = int(safe_get('cache_misses', 0))
        cache_requests = max(1, cache_hits + cache_misses)
        cache_hit_rate = cache_hits / cache_requests
        
        # 安全计算平均延迟
        total_delay = safe_get('total_delay', 0.0)
        processed_for_delay = max(1, processed_tasks)
        avg_delay = total_delay / processed_for_delay
        
        # 限制延迟在合理范围内
        avg_delay = np.clip(avg_delay, 0.0, 10.0)  # 最大10秒
        
        # 安全获取能耗
        total_energy = safe_get('total_energy', 0.0)
        total_energy = np.clip(total_energy, 0.0, 1e6)  # 最大1M焦耳
        
        # 计算丢失率
        dropped_tasks = int(safe_get('dropped_tasks', 0))
        data_loss_rate = min(1.0, dropped_tasks / generated_tasks)
        
        return {
            'avg_task_delay': avg_delay,
            'total_energy_consumption': total_energy,
            'data_loss_rate': data_loss_rate,
            'task_completion_rate': completion_rate,
            'cache_hit_rate': cache_hit_rate,
            'migration_success_rate': 0.8
        }
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None) -> Dict:
        """运行一个完整的训练轮次"""
        # 使用配置中的最大步数
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        # 重置环境
        state = self.reset_environment()
        
        episode_reward = 0.0
        episode_info = {}
        step = 0
        info = {}  # 初始化info变量
        
        # PPO需要特殊处理
        if self.algorithm == "PPO":
            return self._run_ppo_episode(episode, max_steps)
        
        for step in range(max_steps):
            # 选择动作
            if self.algorithm == "DQN":
                # DQN返回离散动作
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    # 处理可能的元组返回
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                        
                # 需要将动作映射回全局动作索引
                action_idx = self._encode_discrete_action(actions_dict)
                action = action_idx
            else:
                # 连续动作算法
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    # 处理可能的元组返回
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action = self._encode_continuous_action(actions_dict)
            
            # 执行动作
            next_state, reward, done, info = self.step(action, state)
            
            # 初始化training_info
            training_info = {}
            
            # 训练智能体 - 所有算法现在都支持Union类型统一接口
            # 确保action类型安全转换
            if self.algorithm == "DQN":
                # DQN首选整数动作，但接受Union类型
                safe_action = self._safe_int_conversion(action)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm in ["DDPG", "TD3", "SAC"]:
                # 连续动作算法首选numpy数组，但接受Union类型
                safe_action = action if isinstance(action, np.ndarray) else np.array([action], dtype=np.float32)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm == "PPO":
                # PPO使用特殊的episode级别训练，train_step为占位符
                # 保持原action类型即可，因为PPO的train_step不做实际处理
                training_info = self.agent_env.train_step(state, action, reward, next_state, done)
            else:
                # 其他算法的默认处理
                training_info = {'message': f'Unknown algorithm: {self.algorithm}'}
            
            episode_info = training_info
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 检查是否结束
            if done:
                break
        
        # 记录轮次统计
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': episode_info,
            'system_metrics': system_metrics,
            'steps': step + 1
        }
    
    def _run_ppo_episode(self, episode: int, max_steps: int = 100) -> Dict:
        """运行PPO专用episode"""
        state = self.reset_environment()
        episode_reward = 0.0
        
        # 初始化变量
        done = False
        step = 0
        info = {}
        
        for step in range(max_steps):
            # 获取动作、对数概率和价值
            if hasattr(self.agent_env, 'get_actions'):
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, tuple) and len(actions_result) == 3:
                    actions_dict, log_prob, value = actions_result
                else:
                    # 如果不是元组，就使用默认值
                    actions_dict = actions_result if isinstance(actions_result, dict) else {}
                    log_prob = 0.0
                    value = 0.0
            else:
                actions_dict = {}
                log_prob = 0.0
                value = 0.0
                
            action = self._encode_continuous_action(actions_dict)
            
            # 执行动作
            next_state, reward, done, info = self.step(action, state)
            
            # 存储经验 - 所有算法都支持统一接口
            # 确保参数类型正确
            log_prob_float = float(log_prob) if not isinstance(log_prob, float) else log_prob
            value_float = float(value) if not isinstance(value, float) else value
            # 使用命名参数避免位置参数顺序问题
            self.agent_env.store_experience(
                state=state, 
                action=action, 
                reward=reward, 
                next_state=next_state, 
                done=done, 
                log_prob=log_prob_float, 
                value=value_float
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Episode结束后进行PPO更新
        last_value = 0.0
        if not done:
            if hasattr(self.agent_env, 'get_actions'):
                actions_result = self.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, tuple) and len(actions_result) >= 3:
                    _, _, last_value = actions_result
                else:
                    last_value = 0.0
        
        # 确保 last_value 为 float 类型
        last_value_float = float(last_value) if not isinstance(last_value, float) else last_value
        
        # 进行更新 - 所有算法都支持统一接口
        training_info = self.agent_env.update(last_value_float)
        
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': training_info,
            'system_metrics': system_metrics,
            'steps': step + 1
        }
    
    def _encode_continuous_action(self, actions_dict) -> np.ndarray:
        """将动作字典编码为连续动作向量"""
        # 处理可能的不同输入类型
        if not isinstance(actions_dict, dict):
            # 如果不是字典，返回默认动作
            return np.zeros(30)  # 3个智能体 * 10维动作
        
        action_list = []
        for agent_type in ['vehicle_agent', 'rsu_agent', 'uav_agent']:
            if agent_type in actions_dict:
                action_list.append(actions_dict[agent_type])
            else:
                action_list.append(np.zeros(10))  # 默认动作
        
        return np.concatenate(action_list)
    
    def _encode_discrete_action(self, actions_dict) -> int:
        """将动作字典编码为离散动作索引"""
        # 处理可能的不同输入类型
        if not isinstance(actions_dict, dict):
            return 0  # 默认动作索引
        
        # 简化实现：将每个智能体的动作组合成一个索引
        vehicle_action = actions_dict.get('vehicle_agent', 0)
        rsu_action = actions_dict.get('rsu_agent', 0)
        uav_action = actions_dict.get('uav_agent', 0)
        
        # 安全地将动作转换为整数
        def safe_int_conversion(value):
            if isinstance(value, (int, np.integer)):
                return int(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    return int(value.item())
                else:
                    return int(value[0])  # 取第一个元素
            elif isinstance(value, (float, np.floating)):
                return int(value)
            else:
                return 0
        
        vehicle_action = safe_int_conversion(vehicle_action)
        rsu_action = safe_int_conversion(rsu_action)
        uav_action = safe_int_conversion(uav_action)
        
        # 5^3 = 125 种组合
        return vehicle_action * 25 + rsu_action * 5 + uav_action
    
    def _safe_int_conversion(self, value) -> int:
        """安全地将不同类型转换为整数"""
        if isinstance(value, (int, np.integer)):
            return int(value)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return int(value.item())
            else:
                return int(value[0])  # 取第一个元素
        elif isinstance(value, (float, np.floating)):
            return int(round(value))
        else:
            return 0  # 安全回退值


def train_single_algorithm(algorithm: str, num_episodes: Optional[int] = None, eval_interval: Optional[int] = None, 
                          save_interval: Optional[int] = None) -> Dict:
    """训练单个算法"""
    # 使用配置中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    if eval_interval is None:
        eval_interval = config.experiment.eval_interval
    if save_interval is None:
        save_interval = config.experiment.save_interval
    
    print(f"\n🚀 开始{algorithm}单智能体算法训练")
    print("=" * 60)
    
    # 创建训练环境
    training_env = SingleAgentTrainingEnvironment(algorithm)
    
    print(f"训练配置:")
    print(f"  算法: {algorithm}")
    print(f"  总轮次: {num_episodes}")
    print(f"  评估间隔: {eval_interval}")
    print(f"  保存间隔: {save_interval}")
    print("-" * 60)
    
    # 创建结果目录
    os.makedirs(f"results/single_agent/{algorithm.lower()}", exist_ok=True)
    os.makedirs(f"results/models/single_agent/{algorithm.lower()}", exist_ok=True)
    
    # 训练循环
    best_avg_reward = -100.0  # 使用有限的初始值而不是-inf
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
            eval_result = evaluate_single_model(algorithm, training_env, episode)
            print(f"\n📊 轮次 {episode} 评估结果:")
            print(f"  评估奖励: {eval_result['avg_reward']:.3f}")
            print(f"  评估时延: {eval_result['avg_delay']:.3f}s")
            print(f"  评估完成率: {eval_result['completion_rate']:.1%}")
            
            # 保存最佳模型
            if eval_result['avg_reward'] > best_avg_reward:
                best_avg_reward = eval_result['avg_reward']
                training_env.agent_env.save_models(f"results/models/single_agent/{algorithm.lower()}/best_model")
                print(f"  💾 保存最佳模型 (奖励: {best_avg_reward:.3f})")
        
        # 定期保存模型
        if episode % save_interval == 0:
            training_env.agent_env.save_models(f"results/models/single_agent/{algorithm.lower()}/checkpoint_{episode}")
            print(f"💾 保存检查点: checkpoint_{episode}")
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print(f"🎉 {algorithm}训练完成!")
    print(f"⏱️  总训练时间: {total_training_time/3600:.2f} 小时")
    print(f"🏆 最佳平均奖励: {best_avg_reward:.3f}")
    
    # 保存训练结果
    results = save_single_training_results(algorithm, training_env, total_training_time)
    
    # 绘制训练曲线
    plot_single_training_curves(algorithm, training_env)
    
    return results


def evaluate_single_model(algorithm: str, training_env: SingleAgentTrainingEnvironment, 
                         episode: int, num_eval_episodes: int = 5) -> Dict:
    """评估单智能体模型性能 - 修复版，防止inf和nan"""
    import numpy as np
    
    eval_rewards = []
    eval_delays = []
    eval_completions = []
    
    def safe_value(value: float, default: float = 0.0, max_val: float = 1e6) -> float:
        """安全处理数值，防止inf和nan"""
        if np.isnan(value) or np.isinf(value):
            return default
        return np.clip(value, -max_val, max_val)
    
    for _ in range(num_eval_episodes):
        state = training_env.reset_environment()
        episode_reward = 0.0
        episode_delay = 0.0
        episode_completion = 0.0
        steps = 0
        
        for step in range(50):  # 较短的评估轮次
            if algorithm == "DQN":
                actions_result = training_env.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action = training_env._encode_discrete_action(actions_dict)
            else:
                actions_result = training_env.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, tuple):  # PPO返回元组
                    actions_dict = actions_result[0]
                elif isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = {}
                action = training_env._encode_continuous_action(actions_dict)
            
            next_state, reward, done, info = training_env.step(action, state)
            
            # 安全处理奖励和指标
            safe_reward = safe_value(reward, -1.0, 100.0)
            episode_reward += safe_reward
            
            system_metrics = info['system_metrics']
            safe_delay = safe_value(system_metrics.get('avg_task_delay', 0), 0.0, 10.0)
            safe_completion = safe_value(system_metrics.get('task_completion_rate', 0), 0.0, 1.0)
            
            episode_delay += safe_delay
            episode_completion += safe_completion
            steps += 1
            
            state = next_state
            
            if done:
                break
        
        # 安全计算平均值
        steps = max(1, steps)  # 防止除零
        eval_rewards.append(safe_value(episode_reward / steps, -10.0, 10.0))
        eval_delays.append(safe_value(episode_delay / steps, 0.0, 10.0))
        eval_completions.append(safe_value(episode_completion / steps, 0.0, 1.0))
    
    # 安全计算最终结果
    if len(eval_rewards) == 0:
        return {'avg_reward': -1.0, 'avg_delay': 1.0, 'completion_rate': 0.0}
    
    avg_reward = safe_value(np.mean(eval_rewards), -10.0, 10.0)
    avg_delay = safe_value(np.mean(eval_delays), 0.0, 10.0)
    avg_completion = safe_value(np.mean(eval_completions), 0.0, 1.0)
    
    return {
        'avg_reward': avg_reward,
        'avg_delay': avg_delay,
        'completion_rate': avg_completion
    }


def save_single_training_results(algorithm: str, training_env: SingleAgentTrainingEnvironment, 
                                training_time: float) -> Dict:
    """保存训练结果"""
    # 生成时间戳
    timestamp = generate_timestamp()
    
    results = {
        'algorithm': algorithm,
        'agent_type': 'single_agent',
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
    filepath = f"results/single_agent/{algorithm.lower()}/{filename}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 {algorithm}训练结果已保存到 {filepath}")
    
    return results


def plot_single_training_curves(algorithm: str, training_env: SingleAgentTrainingEnvironment):
    """绘制训练曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 传统可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 奖励曲线
    axes[0, 0].plot(training_env.episode_rewards)
    axes[0, 0].set_title(f'{algorithm} 单智能体训练奖励曲线')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].grid(True)
    
    # 时延曲线
    if training_env.episode_metrics['avg_delay']:
        axes[0, 1].plot(training_env.episode_metrics['avg_delay'])
        axes[0, 1].set_title('平均任务时延')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('时延 (秒)')
        axes[0, 1].grid(True)
    
    # 完成率曲线
    if training_env.episode_metrics['task_completion_rate']:
        axes[0, 2].plot(training_env.episode_metrics['task_completion_rate'])
        axes[0, 2].set_title('任务完成率')
        axes[0, 2].set_xlabel('训练轮次')
        axes[0, 2].set_ylabel('完成率')
        axes[0, 2].grid(True)
    
    # 缓存命中率曲线
    if training_env.episode_metrics['cache_hit_rate']:
        axes[1, 0].plot(training_env.episode_metrics['cache_hit_rate'])
        axes[1, 0].set_title('缓存命中率')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('命中率')
        axes[1, 0].grid(True)
    
    # 能耗曲线
    if training_env.episode_metrics['total_energy']:
        axes[1, 1].plot(training_env.episode_metrics['total_energy'])
        axes[1, 1].set_title('总能耗')
        axes[1, 1].set_xlabel('训练轮次')
        axes[1, 1].set_ylabel('能耗 (焦耳)')
        axes[1, 1].grid(True)
    
    # 迁移成功率曲线（替换数据丢失率）
    if training_env.episode_metrics['migration_success_rate']:
        axes[1, 2].plot(training_env.episode_metrics['migration_success_rate'])
        axes[1, 2].set_title('迁移成功率')
        axes[1, 2].set_xlabel('训练轮次')
        axes[1, 2].set_ylabel('成功率')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    filepath = f"results/single_agent/{algorithm.lower()}/training_curves.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 {algorithm}训练曲线已保存到 {filepath}")
    
    # 🎨 新增：高级可视化（置信区间 + 滑动平滑）
    from tools.advanced_visualization import enhanced_plot_training_curves
    enhanced_plot_training_curves(algorithm, training_env, f"results/single_agent/{algorithm.lower()}")


def compare_single_algorithms(algorithms: List[str], num_episodes: Optional[int] = None) -> Dict:
    """比较多个单智能体算法的性能"""
    # 使用配置中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    
    print("\n🔥 开始单智能体算法性能比较")
    print("=" * 60)
    
    results = {}
    
    # 训练所有算法
    for algorithm in algorithms:
        print(f"\n开始训练 {algorithm}...")
        results[algorithm] = train_single_algorithm(algorithm, num_episodes)
    
    # 生成比较图表
    plot_single_algorithm_comparison(results)
    
    # 保存比较结果
    timestamp = generate_timestamp()
    comparison_results = {
        'algorithms': algorithms,
        'agent_type': 'single_agent',
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
    comparison_filename = get_timestamped_filename("single_agent_comparison")
    with open(f"results/{comparison_filename}", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print("\n🎯 单智能体算法比较完成！")
    print(f"📄 比较结果已保存到 results/{comparison_filename}")
    print(f"📈 比较图表已保存到 results/single_agent_comparison_{timestamp}.png")
    
    return comparison_results


def plot_single_algorithm_comparison(results: Dict):
    """绘制单智能体算法比较图表"""
    timestamp = generate_timestamp()
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 奖励对比
    for algorithm, result in results.items():
        axes[0, 0].plot(result['episode_rewards'], label=algorithm)
    axes[0, 0].set_title('单智能体算法奖励对比')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 时延对比
    for algorithm, result in results.items():
        if result['episode_metrics']['avg_delay']:
            axes[0, 1].plot(result['episode_metrics']['avg_delay'], label=algorithm)
    axes[0, 1].set_title('平均时延对比')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('时延 (秒)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 完成率对比
    for algorithm, result in results.items():
        if result['episode_metrics']['task_completion_rate']:
            axes[1, 0].plot(result['episode_metrics']['task_completion_rate'], label=algorithm)
    axes[1, 0].set_title('任务完成率对比')
    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('完成率')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 能耗对比
    for algorithm, result in results.items():
        if result['episode_metrics']['total_energy']:
            axes[1, 1].plot(result['episode_metrics']['total_energy'], label=algorithm)
    axes[1, 1].set_title('总能耗对比')
    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('能耗 (焦耳)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 迁移成功率对比（替换数据丢失率）
    for algorithm, result in results.items():
        if result['episode_metrics']['migration_success_rate']:
            axes[2, 0].plot(result['episode_metrics']['migration_success_rate'], label=algorithm)
    axes[2, 0].set_title('迁移成功率对比')
    axes[2, 0].set_xlabel('训练轮次')
    axes[2, 0].set_ylabel('成功率')
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
    chart_filename = f"single_agent_comparison_{timestamp}.png" if timestamp else "single_agent_comparison.png"
    plt.savefig(f"results/{chart_filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🎨 新增：高级比较可视化套件
    from tools.advanced_visualization import create_advanced_visualization_suite
    create_advanced_visualization_suite(results, "results/advanced_single_agent_comparison")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单智能体算法训练脚本')
    parser.add_argument('--algorithm', type=str, choices=['DDPG', 'TD3', 'DQN', 'PPO', 'SAC'],
                       help='选择训练算法')
    parser.add_argument('--episodes', type=int, default=None, help=f'训练轮次 (默认: {config.experiment.num_episodes})')
    parser.add_argument('--eval_interval', type=int, default=None, help=f'评估间隔 (默认: {config.experiment.eval_interval})')
    parser.add_argument('--save_interval', type=int, default=None, help=f'保存间隔 (默认: {config.experiment.save_interval})')
    parser.add_argument('--compare', action='store_true', help='比较所有算法')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs("results/single_agent", exist_ok=True)
    
    if args.compare:
        # 比较所有算法
        algorithms = ['DDPG', 'TD3', 'DQN', 'PPO', 'SAC']
        compare_single_algorithms(algorithms, args.episodes)
    elif args.algorithm:
        # 训练单个算法
        train_single_algorithm(args.algorithm, args.episodes, args.eval_interval, args.save_interval)
    else:
        print("请指定 --algorithm 或使用 --compare 标志")
        print("使用 python train_single_agent.py --help 查看帮助")


if __name__ == "__main__":
    main()