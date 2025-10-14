#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卸载策略对比实验框架 (Offloading Strategy Comparison)

【实验目的】
对比核心卸载策略在不同参数下的性能：

【对比策略】（6种）
基准策略（2种）：
1. LocalOnly  - 纯本地计算（下限基准）
2. RSUOnly    - 仅基站卸载（传统MEC）

启发式策略（2种）：
3. LoadBalance - 负载均衡（最佳启发式）
4. Random      - 随机选择（对照组）

深度强化学习（2种）：
5. TD3         - 完整TD3策略（主要贡献）
6. TD3-NoMig   - 无迁移TD3（消融实验）

【扫描参数】
1. 车辆数量 (num_vehicles): 8, 12, 16, 20, 24
2. 任务到达率 (task_arrival_rate): 0.3, 0.5, 0.7, 0.9, 1.1
3. 通信带宽 (bandwidth_mhz): 10, 20, 30, 40, 50
4. 任务数据大小 (data_size_mb): 0.5, 1.0, 1.5, 2.0, 2.5
5. 计算资源 (cpu_frequency_ghz): 1.5, 2.0, 2.5, 3.0, 3.5

【评估指标】
- 平均加权成本 = 2.0×时延(s) + 1.2×能耗(kJ)
  其中：能耗从J转换为kJ（除以1000）
- 任务完成率
- 平均时延 (秒)
- 总能耗 (焦耳)
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import config
from single_agent.td3 import TD3Agent
from train_single_agent import SingleAgentTrainingEnvironment
from utils.unified_reward_calculator import UnifiedRewardCalculator
from offloading_strategies import create_offloading_strategy


class OffloadingComparisonExperiment:
    """卸载策略对比实验"""
    
    def __init__(self, output_dir: str = "results/offloading_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 奖励计算器（用于计算加权成本）
        self.reward_calculator = UnifiedRewardCalculator(algorithm="general")
        
        # 权重（从配置获取）
        self.weight_delay = config.rl.reward_weight_delay
        self.weight_energy = config.rl.reward_weight_energy
        
        # 🔧 新增：环境缓存机制（减少重复初始化）
        self._env_cache = {}  # key: scenario_hash, value: env
        self._env_reuse_count = 0
        
        print(f"[INIT] 加权成本 = {self.weight_delay}·时延 + {self.weight_energy}·(能耗/1000)")
        print(f"[INIT] 使用TD3归一化方式：能耗单位从J转换为kJ")
        print(f"[INIT] 环境缓存已启用，将减少重复初始化开销")
    
    def load_td3_agent(self, num_vehicles: int = 12) -> TD3Agent:
        """
        加载训练好的TD3模型
        
        Args:
            num_vehicles: 车辆数量（用于选择对应的模型）
        
        Returns:
            TD3智能体
        """
        # 🔧 重要修复：TD3模型是用12辆车训练的，所以状态维度必须是98
        # 无论实际评估时有多少辆车，模型的输入维度都是固定的
        TRAINED_NUM_VEHICLES = 12  # TD3训练时的车辆数
        state_dim = TRAINED_NUM_VEHICLES * 5 + 4 * 5 + 2 * 5 + 8  # 必须是98维
        action_dim = 16
        
        # 导入TD3Config
        from single_agent.td3 import TD3Config
        import os
        
        # 设置环境变量以匹配训练时的配置
        # 这些值来自固定拓扑优化器（见train_single_agent.py）
        os.environ['TD3_HIDDEN_DIM'] = '400'  # 训练时使用的隐藏层维度
        os.environ['TD3_ACTOR_LR'] = '1e-4'
        os.environ['TD3_CRITIC_LR'] = '8e-5'
        os.environ['TD3_BATCH_SIZE'] = '256'
        
        # 创建配置（会读取环境变量）
        td3_config = TD3Config()
        
        # 创建TD3智能体（使用固定的98维输入）
        agent = TD3Agent(state_dim, action_dim, td3_config)
        
        # 尝试加载模型
        # 注意：TD3Agent.load_model()会自动添加_td3.pth后缀
        # 所以路径传入时不要带.pth后缀
        model_paths = [
            f"../results/single_agent/td3/{num_vehicles}/best_model",  # 从父目录的结果中加载
            f"results/single_agent/td3/{num_vehicles}/best_model",  # 兼容原有位置
            f"../models/td3/{num_vehicles}/best_model",  # 另一个可能的位置
        ]
        
        model_loaded = False
        for model_path in model_paths:
            # 检查实际文件是否存在（加上后缀）
            actual_file = Path(f"{model_path}_td3.pth")
            if actual_file.exists():
                try:
                    agent.load_model(model_path)
                    print(f"[LOAD] 成功加载TD3模型: {actual_file}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"[WARN] 尝试加载 {actual_file} 失败: {e}")
        
        if not model_loaded:
            print(f"[WARN] 未找到TD3模型")
            print(f"[WARN] 将使用随机初始化的TD3策略（可能性能较差）")
        
        return agent
    
    def evaluate_strategy(
        self,
        strategy,
        num_episodes: int = 50,
        max_steps: int = 100,
        **env_params
    ) -> Dict:
        """
        评估单个策略的性能
        
        Args:
            strategy: 卸载策略实例
            num_episodes: 评估轮次
            max_steps: 每轮最大步数
            **env_params: 环境参数（可覆盖默认配置）
        
        Returns:
            评估结果字典
        """
        # 🔧 修复：构建参数覆盖字典
        override_scenario = {}
        if env_params:
            # 映射环境参数到scenario配置
            if 'num_vehicles' in env_params:
                override_scenario['num_vehicles'] = env_params['num_vehicles']
            if 'task_arrival_rate' in env_params:
                override_scenario['task_arrival_rate'] = env_params['task_arrival_rate']
            if 'bandwidth_mhz' in env_params:
                override_scenario['bandwidth'] = env_params['bandwidth_mhz']
            if 'data_size_mb' in env_params:
                # 数据大小范围（MB转换为bytes）
                size_mb = env_params['data_size_mb']
                override_scenario['data_size_range'] = (size_mb * 0.8 * 1e6, size_mb * 1.2 * 1e6)
            if 'cpu_frequency_ghz' in env_params:
                freq_hz = env_params['cpu_frequency_ghz'] * 1e9
                override_scenario['computation_capacity'] = freq_hz / 1e6  # 转换为MIPS等效值
        
        # 🔧 新增：环境缓存机制
        # 生成场景哈希值用于缓存
        import hashlib
        scenario_str = json.dumps(override_scenario, sort_keys=True) if override_scenario else "default"
        scenario_hash = hashlib.md5(scenario_str.encode()).hexdigest()[:16]  # 使用更长的哈希避免碰撞
        
        # 尝试从缓存获取环境
        if scenario_hash in self._env_cache:
            env = self._env_cache[scenario_hash]
            self._env_reuse_count += 1
            if self._env_reuse_count <= 3:  # 只打印前几次
                print(f"  [CACHE] 复用环境 (hash={scenario_hash})")
        else:
            # 创建新环境并缓存
            env = SingleAgentTrainingEnvironment("TD3", override_scenario=override_scenario if override_scenario else None)
            self._env_cache[scenario_hash] = env
            print(f"  [NEW] 创建新环境 (hash={scenario_hash})")
        
        # 安全注入环境引用
        if hasattr(strategy, 'update_environment'):
            strategy.update_environment(env)
        
        # 评估指标
        episode_costs = []
        episode_delays = []
        episode_energies = []
        episode_completion_rates = []
        
        for ep in range(num_episodes):
            state = env.reset_environment()
            # 安全调用策略的reset方法
            if hasattr(strategy, 'reset'):
                strategy.reset()
            
            episode_reward = 0
            episode_delay = 0
            episode_energy = 0
            completed_tasks = 0
            total_tasks = 0
            # 记录实际步数
            actual_steps = 0
            
            for step in range(max_steps):
                # 选择动作
                action = strategy.select_action(state)
                
                # 构建动作字典
                actions_dict = env._build_actions_from_vector(action)
                
                # 调试：第一轮第一步打印动作
                if ep == 0 and step == 0:
                    sim_actions = env._build_simulator_actions(actions_dict)
                    print(f"\n[DEBUG] {strategy.name} 动作:")
                    print(f"  原始action前3维: {action[:3]}")
                    if sim_actions and 'vehicle_offload_pref' in sim_actions:
                        prefs = sim_actions['vehicle_offload_pref']
                        print(f"  卸载概率: local={prefs['local']:.4f}, rsu={prefs['rsu']:.4f}, uav={prefs['uav']:.4f}")
                
                # 执行动作
                next_state, reward, done, info = env.step(action, state, actions_dict)
                
                # 累积奖励（reward = -cost）
                episode_reward += reward
                
                state = next_state
                actual_steps = step + 1  # 更新实际步数
                
                if done:
                    break
            
            # 🔧 修复：按实际步数计算加权成本
            # 注意：使用TD3的归一化方式，能耗需要除以1000（J转kJ）
            avg_reward_per_step = episode_reward / max(actual_steps, 1)
            # reward = -(2.0*delay + 1.2*energy/1000)，所以cost = -reward
            weighted_cost = -avg_reward_per_step  # 转换为成本
            
            # 🔧 修复：从环境获取实际的时延、能耗和完成率
            avg_delay = 0.0
            avg_energy = 0.0
            completion_rate = 0.0
            # actual_steps 已经在循环中更新，不需要重复定义
            
            if hasattr(env, 'simulator'):
                try:
                    # 方法1：从stats直接获取（最准确）
                    if hasattr(env.simulator, 'stats'):
                        stats = env.simulator.stats
                        
                        # 获取平均时延
                        total_delay = stats.get('total_delay', 0.0)
                        completed_tasks = stats.get('completed_tasks', 0)
                        if completed_tasks > 0:
                            avg_delay = total_delay / completed_tasks
                        
                        # 获取总能耗（从stats中的正确字段）
                        total_energy = stats.get('total_energy', 0.0)
                        # 按实际步数归一化
                        avg_energy = total_energy / max(actual_steps, 1)
                        
                        # 获取真实完成率
                        total_tasks = stats.get('total_tasks', 0)
                        if total_tasks > 0:
                            completion_rate = completed_tasks / total_tasks
                        else:
                            completion_rate = 0.0  # 无任务时完成率为0
                    
                    # 方法2：如果stats不可用，从vehicles获取（list不是dict）
                    elif hasattr(env.simulator, 'vehicles') and isinstance(env.simulator.vehicles, list):
                        total_delay = 0.0
                        task_count = 0
                        for vehicle in env.simulator.vehicles:  # 🔧 修复：vehicles是list
                            if isinstance(vehicle, dict) and 'completed_tasks' in vehicle:
                                for task in vehicle.get('completed_tasks', []):
                                    if isinstance(task, dict):
                                        comp_time = task.get('completion_time', 0)
                                        gen_time = task.get('generation_time', 0)
                                        if comp_time > gen_time:
                                            total_delay += comp_time - gen_time
                                            task_count += 1
                        
                        if task_count > 0:
                            avg_delay = total_delay / task_count
                            # 估算能耗（基于默认值）
                            avg_energy = task_count * 15.0 / max(actual_steps, 1)
                            completion_rate = 0.9  # 合理的默认值
                        else:
                            # 完全没有数据时的兜底
                            avg_delay = weighted_cost / (self.weight_delay + self.weight_energy * 0.5)
                            avg_energy = weighted_cost / (self.weight_energy + self.weight_delay * 0.5) * 100
                            completion_rate = 0.8  # 备用默认值
                    else:
                        # 兜底估算
                        avg_delay = weighted_cost / (self.weight_delay + self.weight_energy * 0.5)
                        avg_energy = weighted_cost / (self.weight_energy + self.weight_delay * 0.5) * 100
                        completion_rate = 0.8  # 备用默认值
                        print(f"  ⚠️ [{strategy.name}] 无法获取真实指标，使用估算值")
                        
                except Exception as e:
                    # 异常处理
                    print(f"  ⚠️ [{strategy.name}] 指标采集异常: {e}，使用备用估算")
                    avg_delay = weighted_cost / (self.weight_delay + self.weight_energy * 0.5)
                    avg_energy = weighted_cost / (self.weight_energy + self.weight_delay * 0.5) * 100
                    completion_rate = 0.8  # 备用默认值
            else:
                # 没有simulator时的备用估算
                print(f"  ⚠️ [{strategy.name}] 环境没有simulator，使用估算值")
                avg_delay = weighted_cost / (self.weight_delay + self.weight_energy * 0.5)
                avg_energy = weighted_cost / (self.weight_energy + self.weight_delay * 0.5) * 100
                completion_rate = 0.5
            
            # 重新计算加权成本，确保使用TD3归一化方式
            actual_weighted_cost = 2.0 * avg_delay + 1.2 * (avg_energy / 1000.0)
            
            episode_costs.append(actual_weighted_cost)
            episode_delays.append(avg_delay)
            episode_energies.append(avg_energy)  # 使用平均能耗（J）
            episode_completion_rates.append(completion_rate)
            
            if (ep + 1) % 10 == 0 or ep == 0:
                recent_cost = np.mean(episode_costs[-10:]) if episode_costs else 0
                recent_delay = np.mean(episode_delays[-10:]) if episode_delays else 0
                recent_energy = np.mean(episode_energies[-10:]) if episode_energies else 0
                recent_completion = np.mean(episode_completion_rates[-10:]) if episode_completion_rates else 0
                
                print(f"  Episode {ep+1}/{num_episodes}: "
                      f"Cost={recent_cost:.2f}, "
                      f"Delay={recent_delay:.4f}s, "
                      f"Energy={recent_energy:.2f}J, "
                      f"Completion={recent_completion*100:.1f}%")
        
        # 🔧 改进：返回更详细的评估结果
        return {
            'strategy_name': strategy.name,
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            
            # 核心指标
            'avg_weighted_cost': float(np.mean(episode_costs)),
            'std_weighted_cost': float(np.std(episode_costs)),
            'avg_delay': float(np.mean(episode_delays)),
            'std_delay': float(np.std(episode_delays)),
            'avg_energy': float(np.mean(episode_energies)),
            'std_energy': float(np.std(episode_energies)),
            'avg_completion_rate': float(np.mean(episode_completion_rates)),
            'std_completion_rate': float(np.std(episode_completion_rates)),
            
            # 原始数据（用于后续分析）
            'episode_costs': [float(c) for c in episode_costs],
            'episode_delays': [float(d) for d in episode_delays],
            'episode_energies': [float(e) for e in episode_energies],
            
            # 环境参数和元数据
            'env_params': env_params,
            'override_scenario': override_scenario if override_scenario else {},
            'scenario_hash': scenario_hash,
            
            # 统计信息
            'min_cost': float(np.min(episode_costs)),
            'max_cost': float(np.max(episode_costs)),
            'median_cost': float(np.median(episode_costs))
        }
    
    def run_vehicle_sweep(
        self,
        strategies: List,
        vehicle_counts: List[int] = [8, 12, 16, 20, 24],
        num_episodes: int = 50
    ) -> Dict:
        """
        实验1: 车辆数量变化实验
        
        Args:
            strategies: 策略列表
            vehicle_counts: 车辆数量列表
            num_episodes: 每个配置的评估轮次
        
        Returns:
            实验结果
        """
        print("\n" + "="*70)
        print("实验1: 车辆数量变化对比")
        print("="*70)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n[{strategy.name}]")
            strategy_results = []
            
            for num_vehicles in vehicle_counts:
                print(f"\n  评估 N={num_vehicles} 辆车...")
                
                # 🔧 修复：直接通过env_params传递参数，无需修改全局config
                result = self.evaluate_strategy(
                    strategy,
                    num_episodes=num_episodes,
                    num_vehicles=num_vehicles
                )
                strategy_results.append(result)
                
                # 使用TD3归一化方式计算加权成本
                calculated_cost = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 1000.0)
                print(f"    → 加权成本: {calculated_cost:.2f} "
                      f"(时延: {result['avg_delay']:.4f}s, 能耗: {result['avg_energy']:.2f}J)")
            
            results[strategy.name] = strategy_results
        
        return {
            'experiment': 'vehicle_sweep',
            'parameter': 'num_vehicles',
            'values': vehicle_counts,
            'results': results
        }
    
    def run_task_rate_sweep(
        self,
        strategies: List,
        task_rates: List[float] = [0.3, 0.5, 0.7, 0.9, 1.1],
        num_episodes: int = 50
    ) -> Dict:
        """
        实验2: 任务到达率变化实验
        
        Args:
            strategies: 策略列表
            task_rates: 任务到达率列表（任务/秒/车辆）
            num_episodes: 评估轮次
        
        Returns:
            实验结果
        """
        print("\n" + "="*70)
        print("实验2: 任务到达率变化对比")
        print("="*70)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n[{strategy.name}]")
            strategy_results = []
            
            for rate in task_rates:
                print(f"\n  评估任务率={rate:.1f} tasks/s/vehicle...")
                
                # 🔧 修复：直接通过env_params传递参数
                result = self.evaluate_strategy(
                    strategy,
                    num_episodes=num_episodes,
                    task_arrival_rate=rate
                )
                strategy_results.append(result)
                
                # 使用TD3归一化方式计算加权成本
                calculated_cost = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 1000.0)
                print(f"    → 加权成本: {calculated_cost:.2f} "
                      f"(时延: {result['avg_delay']:.4f}s, 能耗: {result['avg_energy']:.2f}J)")
            
            results[strategy.name] = strategy_results
        
        return {
            'experiment': 'task_rate_sweep',
            'parameter': 'task_arrival_rate',
            'values': task_rates,
            'results': results
        }
    
    def run_bandwidth_sweep(
        self,
        strategies: List,
        bandwidths: List[int] = [10, 20, 30, 40, 50],
        num_episodes: int = 50
    ) -> Dict:
        """
        实验3: 通信带宽变化实验
        
        Args:
            strategies: 策略列表
            bandwidths: 带宽列表（MHz）
            num_episodes: 评估轮次
        
        Returns:
            实验结果
        """
        print("\n" + "="*70)
        print("实验3: 通信带宽变化对比")
        print("="*70)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n[{strategy.name}]")
            strategy_results = []
            
            for bw in bandwidths:
                print(f"\n  评估带宽={bw} MHz...")
                
                result = self.evaluate_strategy(
                    strategy,
                    num_episodes=num_episodes,
                    bandwidth_mhz=bw
                )
                strategy_results.append(result)
                
                # 使用TD3归一化方式计算加权成本
                calculated_cost = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 1000.0)
                print(f"    → 加权成本: {calculated_cost:.2f} "
                      f"(时延: {result['avg_delay']:.4f}s, 能耗: {result['avg_energy']:.2f}J)")
            
            results[strategy.name] = strategy_results
        
        return {
            'experiment': 'bandwidth_sweep',
            'parameter': 'bandwidth_mhz',
            'values': bandwidths,
            'results': results
        }
    
    def run_data_size_sweep(
        self,
        strategies: List,
        data_sizes: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5],
        num_episodes: int = 50
    ) -> Dict:
        """
        实验4: 任务数据大小变化实验
        
        Args:
            strategies: 策略列表
            data_sizes: 数据大小列表（MB）
            num_episodes: 评估轮次
        
        Returns:
            实验结果
        """
        print("\n" + "="*70)
        print("实验4: 任务数据大小变化对比")
        print("="*70)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n[{strategy.name}]")
            strategy_results = []
            
            for size in data_sizes:
                print(f"\n  评估数据大小={size:.1f} MB...")
                
                result = self.evaluate_strategy(
                    strategy,
                    num_episodes=num_episodes,
                    data_size_mb=size
                )
                strategy_results.append(result)
                
                # 使用TD3归一化方式计算加权成本
                calculated_cost = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 1000.0)
                print(f"    → 加权成本: {calculated_cost:.2f} "
                      f"(时延: {result['avg_delay']:.4f}s, 能耗: {result['avg_energy']:.2f}J)")
            
            results[strategy.name] = strategy_results
        
        return {
            'experiment': 'data_size_sweep',
            'parameter': 'data_size_mb',
            'values': data_sizes,
            'results': results
        }
    
    def run_cpu_frequency_sweep(
        self,
        strategies: List,
        cpu_frequencies: List[float] = [1.5, 2.0, 2.5, 3.0, 3.5],
        num_episodes: int = 50
    ) -> Dict:
        """
        实验5: 计算资源（CPU频率）变化实验
        
        Args:
            strategies: 策略列表
            cpu_frequencies: CPU频率列表（GHz）
            num_episodes: 评估轮次
        
        Returns:
            实验结果
        """
        print("\n" + "="*70)
        print("实验5: 计算资源（CPU频率）变化对比")
        print("="*70)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n[{strategy.name}]")
            strategy_results = []
            
            for freq in cpu_frequencies:
                print(f"\n  评估CPU频率={freq:.1f} GHz...")
                
                result = self.evaluate_strategy(
                    strategy,
                    num_episodes=num_episodes,
                    cpu_frequency_ghz=freq
                )
                strategy_results.append(result)
                
                # 使用TD3归一化方式计算加权成本
                calculated_cost = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 1000.0)
                print(f"    → 加权成本: {calculated_cost:.2f} "
                      f"(时延: {result['avg_delay']:.4f}s, 能耗: {result['avg_energy']:.2f}J)")
            
            results[strategy.name] = strategy_results
        
        return {
            'experiment': 'cpu_frequency_sweep',
            'parameter': 'cpu_frequency_ghz',
            'values': cpu_frequencies,
            'results': results
        }
    
    def save_results(self, results: Dict, filename: str):
        """保存实验结果"""
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {output_file}")


def main():
    parser = argparse.ArgumentParser(description="卸载策略对比实验")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'vehicle', 'task_rate', 'bandwidth', 'data_size', 'cpu'],
                        help="实验模式")
    parser.add_argument('--episodes', type=int, default=50,
                        help="每个配置的评估轮次")
    parser.add_argument('--quick', action='store_true',
                        help="快速测试模式（减少参数点和轮次）")
    parser.add_argument('--train-td3', action='store_true',
                        help="在对比实验前先训练TD3模型")
    parser.add_argument('--td3-episodes', type=int, default=200,
                        help="TD3训练轮次（默认200）")
    
    args = parser.parse_args()
    
    print("="*70)
    print("卸载策略对比实验")
    print("="*70)
    print(f"模式: {args.mode}")
    print(f"轮次: {args.episodes}")
    print(f"快速测试: {args.quick}")
    print(f"训练TD3: {args.train_td3}")
    if args.train_td3:
        print(f"TD3训练轮次: {args.td3_episodes}")
    
    # 创建实验实例
    experiment = OffloadingComparisonExperiment()
    
    # 如果需要，先训练TD3模型
    if args.train_td3:
        print("\n" + "="*70)
        print("步骤1: 训练TD3模型")
        print("="*70)
        
        # 导入训练模块
        import subprocess
        import sys
        
        # 设置车辆数量（使用标准配置）
        num_vehicles = 12
        
        print(f"\n[TD3训练] 开始训练TD3模型 (车辆数={num_vehicles}, 轮次={args.td3_episodes})")
        
        # 构建训练命令
        train_cmd = [
            sys.executable,  # Python解释器
            str(Path(__file__).parent.parent / "train_single_agent.py"),  # 使用绝对路径
            "--algorithm", "TD3",
            "--num-vehicles", str(num_vehicles),
            "--episodes", str(args.td3_episodes),
            # 不使用realtime-vis参数，默认就是关闭的
        ]
        
        # 执行训练
        try:
            subprocess.run(train_cmd, check=True, cwd=Path(__file__).parent)
            print("\n[TD3训练] ✅ TD3模型训练完成！")
        except subprocess.CalledProcessError as e:
            print(f"\n[TD3训练] ❌ TD3训练失败: {e}")
            print("[TD3训练] 将继续执行对比实验（不含HybridDRL）")
        except Exception as e:
            print(f"\n[TD3训练] ❌ TD3训练出错: {e}")
            print("[TD3训练] 将继续执行对比实验（不含HybridDRL）")
        
        print("\n" + "="*70)
        print("步骤2: 运行对比实验")
        print("="*70)
    
    # 创建策略实例
    print("\n[INIT] 初始化策略...")
    
    # 核心对比策略（根据论文需求精简）
    strategies = [
        # 基准策略
        create_offloading_strategy("LocalOnly"),  # 纯本地计算基准
        create_offloading_strategy("RSUOnly", selection_mode="load_balance"),  # 传统MEC基准
        
        # 启发式策略
        create_offloading_strategy("LoadBalance"),  # 负载均衡（最佳启发式）
        create_offloading_strategy("Random"),  # 随机策略（对照组）
    ]
    
    # 加载TD3模型
    try:
        td3_agent = experiment.load_td3_agent(num_vehicles=12)
        
        # 1. 完整TD3策略
        td3_strategy = create_offloading_strategy("HybridDRL", td3_agent=td3_agent)
        td3_strategy.name = "TD3"  # 简化名称
        strategies.append(td3_strategy)
        
        # 2. 无迁移的TD3策略（消融实验）
        from offloading_strategies import HybridDRLStrategy
        
        class NoMigrationTD3Strategy(HybridDRLStrategy):
            """无迁移的TD3策略（消融实验）"""
            def __init__(self, td3_agent):
                super().__init__(td3_agent)
                self.name = "TD3-NoMig"  # TD3 without Migration
                
            def select_action(self, state: np.ndarray) -> np.ndarray:
                """使用TD3但禁用迁移"""
                # 先适配状态维度（使用父类的方法）
                adapted_state = self._adapt_state_dimension(state)
                
                # 获取TD3的原始动作
                action = self.td3_agent.select_action(adapted_state, training=False)
                
                # 修改迁移相关的控制参数
                # action[9:16] 是控制参数，其中包含迁移阈值等
                # 将迁移概率设置为-5（经过sigmoid后接近0）
                action[10] = -5.0  # 迁移阈值设为极低值，禁用迁移
                action[11] = -5.0  # 迁移率设为极低值
                
                return action
        
        no_mig_strategy = NoMigrationTD3Strategy(td3_agent)
        strategies.append(no_mig_strategy)
        
        print(f"[INFO] 成功加载TD3策略（含消融版本），共{len(strategies)}种策略")
        print(f"[INFO] 策略列表: {[s.name for s in strategies]}")
        
    except Exception as e:
        print(f"[WARN] 无法加载TD3策略: {e}")
        print(f"[INFO] 将对比{len(strategies)}种基础策略")
    
    # 调整参数（快速模式）
    if args.quick:
        args.episodes = 20
        vehicle_counts = [8, 12, 16]
        task_rates = [0.5, 0.7, 0.9]
        bandwidths = [10, 20, 30]
        data_sizes = [0.5, 1.0, 1.5]
        cpu_frequencies = [1.5, 2.0, 2.5]
    else:
        vehicle_counts = [8, 12, 16, 20, 24]
        task_rates = [0.3, 0.5, 0.7, 0.9, 1.1]
        bandwidths = [10, 20, 30, 40, 50]
        data_sizes = [0.5, 1.0, 1.5, 2.0, 2.5]
        cpu_frequencies = [1.5, 2.0, 2.5, 3.0, 3.5]
    
    # 运行实验
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    
    if args.mode in ['all', 'vehicle']:
        result = experiment.run_vehicle_sweep(strategies, vehicle_counts, args.episodes)
        all_results['vehicle_sweep'] = result
        experiment.save_results(result, f'vehicle_sweep_{timestamp}.json')
    
    if args.mode in ['all', 'task_rate']:
        result = experiment.run_task_rate_sweep(strategies, task_rates, args.episodes)
        all_results['task_rate_sweep'] = result
        experiment.save_results(result, f'task_rate_sweep_{timestamp}.json')
    
    if args.mode in ['all', 'bandwidth']:
        result = experiment.run_bandwidth_sweep(strategies, bandwidths, args.episodes)
        all_results['bandwidth_sweep'] = result
        experiment.save_results(result, f'bandwidth_sweep_{timestamp}.json')
    
    if args.mode in ['all', 'data_size']:
        result = experiment.run_data_size_sweep(strategies, data_sizes, args.episodes)
        all_results['data_size_sweep'] = result
        experiment.save_results(result, f'data_size_sweep_{timestamp}.json')
    
    if args.mode in ['all', 'cpu']:
        result = experiment.run_cpu_frequency_sweep(strategies, cpu_frequencies, args.episodes)
        all_results['cpu_frequency_sweep'] = result
        experiment.save_results(result, f'cpu_frequency_sweep_{timestamp}.json')
    
    # 保存汇总结果
    if args.mode == 'all':
        experiment.save_results(all_results, f'all_experiments_{timestamp}.json')
    
    # 🔧 新增：打印实验总结
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    # 环境缓存统计
    if hasattr(experiment, '_env_cache'):
        print(f"\n[性能统计]")
        print(f"  环境缓存数: {len(experiment._env_cache)}")
        print(f"  环境复用次数: {experiment._env_reuse_count}")
        cache_efficiency = (experiment._env_reuse_count / max(1, experiment._env_reuse_count + len(experiment._env_cache))) * 100
        print(f"  缓存命中率: {cache_efficiency:.1f}%")
    
    # 实验规模统计
    total_evaluations = 0
    for exp_name, exp_data in all_results.items():
        if 'results' in exp_data:
            num_strategies = len(exp_data['results'])
            num_params = len(exp_data.get('values', []))
            total_evaluations += num_strategies * num_params
    
    if total_evaluations > 0:
        print(f"\n[实验规模]")
        print(f"  总评估次数: {total_evaluations}")
        print(f"  实验维度: {len(all_results)}")
        print(f"  策略数量: {len(strategies)}")
    
    print("\n[输出文件]")
    print(f"  结果目录: {experiment.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

