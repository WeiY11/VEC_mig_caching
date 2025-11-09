#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 跨场景评估工具 - Cross-Scenario Evaluation Tool

【功能】
使用训练好的agent在不同场景下进行泛化能力测试，包括：
1. 车辆密度变化（8/12/16/20/24辆）
2. 任务到达率变化（低/中/高负载）
3. 任务类型分布变化（紧急任务比例）
4. 网络条件变化（信道质量）
5. RSU/UAV数量变化

【使用方法】
# 1. 评估单个模型在多场景下
python evaluate_cross_scenario.py --model results/models/single_agent/td3/best_model_td3.pth --algorithm TD3

# 2. 评估多个算法对比
python evaluate_cross_scenario.py --compare --algorithms TD3 SAC DDPG --scenario-set all

# 3. 自定义场景
python evaluate_cross_scenario.py --model results/models/single_agent/td3/best_model_td3.pth \
    --algorithm TD3 --num-vehicles 20 --arrival-rate 3.5 --eval-episodes 20

# 4. 泛化能力分析（训练场景 vs 测试场景）
python evaluate_cross_scenario.py --model results/models/single_agent/td3/best_model_td3.pth \
    --algorithm TD3 --generalization-test

【输出】
- 各场景下的性能指标（时延、能耗、完成率）
- 泛化能力分析图表
- 性能对比雷达图
- 详细的JSON结果文件
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from config.system_config import config
from evaluation.system_simulator import CompleteSystemSimulator


# ========== 场景定义 ==========

class ScenarioConfig:
    """场景配置类"""
    
    def __init__(self, name: str, description: str, params: Dict[str, Any]):
        self.name = name
        self.description = description
        self.params = params
    
    def apply_to_config(self):
        """应用场景参数到全局配置"""
        for key, value in self.params.items():
            if '.' in key:
                # 处理嵌套属性，如 'rl.reward_weight_delay'
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, key, value)


# 预定义场景集
SCENARIO_SETS = {
    # ========== 1. 车辆密度变化 ==========
    "vehicle_density": [
        ScenarioConfig(
            name="low_density",
            description="低密度场景（8辆车）",
            params={'num_vehicles': 8, 'task_arrival_rate': 2.0}
        ),
        ScenarioConfig(
            name="medium_density", 
            description="中密度场景（12辆车，训练场景）",
            params={'num_vehicles': 12, 'task_arrival_rate': 2.5}
        ),
        ScenarioConfig(
            name="high_density",
            description="高密度场景（16辆车）",
            params={'num_vehicles': 16, 'task_arrival_rate': 3.0}
        ),
        ScenarioConfig(
            name="extreme_density",
            description="极高密度场景（24辆车）",
            params={'num_vehicles': 24, 'task_arrival_rate': 4.0}
        ),
    ],
    
    # ========== 2. 任务负载变化 ==========
    "task_load": [
        ScenarioConfig(
            name="light_load",
            description="轻负载（到达率1.5 tasks/s）",
            params={'task_arrival_rate': 1.5}
        ),
        ScenarioConfig(
            name="normal_load",
            description="正常负载（到达率2.5 tasks/s，训练场景）",
            params={'task_arrival_rate': 2.5}
        ),
        ScenarioConfig(
            name="heavy_load",
            description="重负载（到达率3.5 tasks/s）",
            params={'task_arrival_rate': 3.5}
        ),
        ScenarioConfig(
            name="extreme_load",
            description="极端负载（到达率5.0 tasks/s）",
            params={'task_arrival_rate': 5.0}
        ),
    ],
    
    # ========== 3. 基础设施变化 ==========
    "infrastructure": [
        ScenarioConfig(
            name="limited_rsu",
            description="有限RSU（4个）",
            params={'num_rsus': 4}
        ),
        ScenarioConfig(
            name="standard_rsu",
            description="标准RSU（6个，训练场景）",
            params={'num_rsus': 6}
        ),
        ScenarioConfig(
            name="abundant_rsu",
            description="充足RSU（8个）",
            params={'num_rsus': 8}
        ),
        ScenarioConfig(
            name="with_uav",
            description="增加UAV支持（2个UAV）",
            params={'num_uavs': 2}
        ),
    ],
    
    # ========== 4. 网络条件变化 ==========
    "network_condition": [
        ScenarioConfig(
            name="poor_channel",
            description="差信道条件（高噪声）",
            params={'noise_power_dbm': -164}  # 提高10dB噪声
        ),
        ScenarioConfig(
            name="normal_channel",
            description="正常信道条件（训练场景）",
            params={'noise_power_dbm': -174}
        ),
        ScenarioConfig(
            name="good_channel",
            description="好信道条件（低噪声）",
            params={'noise_power_dbm': -184}  # 降低10dB噪声
        ),
    ],
    
    # ========== 5. 任务类型分布变化 ==========
    "task_distribution": [
        ScenarioConfig(
            name="high_urgency",
            description="高紧急任务比例（40%类型1）",
            params={'emergency_task_ratio': 0.40}
        ),
        ScenarioConfig(
            name="normal_mix",
            description="正常混合分布（训练场景）",
            params={'emergency_task_ratio': 0.15}
        ),
        ScenarioConfig(
            name="low_urgency",
            description="低紧急任务比例（5%类型1）",
            params={'emergency_task_ratio': 0.05}
        ),
    ],
}


# ========== Agent加载器 ==========

def load_trained_agent(algorithm: str, model_path: str, state_dim: int, action_dim: int):
    """
    加载训练好的agent
    
    【参数】
    - algorithm: 算法名称（TD3/SAC/DDPG/PPO/DQN）
    - model_path: 模型文件路径
    - state_dim: 状态空间维度
    - action_dim: 动作空间维度
    
    【返回】
    - agent: 加载好的智能体
    """
    algorithm = algorithm.upper()
    
    # 导入对应的agent类
    if algorithm == 'TD3':
        from single_agent.td3 import TD3Agent
        agent = TD3Agent(state_dim, action_dim, config.rl)
    elif algorithm == 'SAC':
        from single_agent.sac import SACAgent
        agent = SACAgent(state_dim, action_dim, config.rl)
    elif algorithm == 'DDPG':
        from single_agent.ddpg import DDPGAgent
        agent = DDPGAgent(state_dim, action_dim, config.rl)
    elif algorithm == 'PPO':
        from single_agent.ppo import PPOAgent
        agent = PPOAgent(state_dim, action_dim, config.rl)
    elif algorithm == 'DQN':
        from single_agent.dqn import DQNAgent
        # DQN的action_dim是离散动作数量
        agent = DQNAgent(state_dim, action_dim, config.rl)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 加载模型参数
    if os.path.exists(model_path):
        # 移除文件后缀（如_td3.pth），因为load_model会自动添加
        base_path = model_path.replace(f'_{algorithm.lower()}.pth', '')
        agent.load_model(base_path)
        print(f"✓ 成功加载模型: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 设置为评估模式（不探索）
    agent.actor.eval()
    
    return agent


def evaluate_agent_in_scenario(agent, algorithm: str, scenario: ScenarioConfig, 
                               num_episodes: int = 20) -> Dict[str, Any]:
    """
    在指定场景下评估agent性能
    
    【参数】
    - agent: 训练好的智能体
    - algorithm: 算法名称
    - scenario: 场景配置
    - num_episodes: 评估轮次
    
    【返回】
    - results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"📊 评估场景: {scenario.name}")
    print(f"   描述: {scenario.description}")
    print(f"   参数: {scenario.params}")
    print(f"{'='*60}\n")
    
    # 应用场景配置
    original_config = {}
    for key, value in scenario.params.items():
        if '.' not in key and hasattr(config, key):
            original_config[key] = getattr(config, key)
    
    scenario.apply_to_config()
    
    # 创建环境（使用对应算法的Environment类）
    if algorithm.upper() == 'TD3':
        from single_agent.td3 import TD3Environment
        env = TD3Environment()
    elif algorithm.upper() == 'SAC':
        from single_agent.sac import SACEnvironment
        env = SACEnvironment()
    elif algorithm.upper() == 'DDPG':
        from single_agent.ddpg import DDPGEnvironment
        env = DDPGEnvironment()
    elif algorithm.upper() == 'PPO':
        from single_agent.ppo import PPOEnvironment
        env = PPOEnvironment()
    elif algorithm.upper() == 'DQN':
        from single_agent.dqn import DQNEnvironment
        env = DQNEnvironment()
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 收集评估指标
    episode_rewards = []
    episode_delays = []
    episode_energies = []
    episode_completion_rates = []
    episode_cache_hit_rates = []
    episode_migration_success_rates = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        step_count = 0
        
        while not done and step_count < config.experiment.max_steps_per_episode:
            # Agent选择动作（不添加噪声）
            with torch.no_grad():
                if algorithm.upper() in ['TD3', 'SAC', 'DDPG']:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    action = agent.actor(state_tensor).cpu().numpy()[0]
                elif algorithm.upper() == 'PPO':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    action, _, _ = agent.actor(state_tensor)
                    action = action.cpu().numpy()[0]
                elif algorithm.upper() == 'DQN':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_values = agent.q_network(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            step_count += 1
        
        # 收集episode统计
        episode_rewards.append(episode_reward)
        
        # 从仿真器获取性能指标
        metrics = env.simulator.get_metrics()
        episode_delays.append(metrics.get('avg_task_delay', 0.0))
        episode_energies.append(metrics.get('total_energy_consumption', 0.0))
        episode_completion_rates.append(metrics.get('task_completion_rate', 0.0))
        episode_cache_hit_rates.append(metrics.get('cache_hit_rate', 0.0))
        episode_migration_success_rates.append(metrics.get('migration_success_rate', 0.0))
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Delay={episode_delays[-1]:.3f}s, "
                  f"Completion={episode_completion_rates[-1]:.2%}")
    
    # 计算平均结果
    results = {
        'scenario_name': scenario.name,
        'scenario_description': scenario.description,
        'scenario_params': scenario.params,
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_delay': float(np.mean(episode_delays)),
        'std_delay': float(np.std(episode_delays)),
        'avg_energy': float(np.mean(episode_energies)),
        'std_energy': float(np.std(episode_energies)),
        'avg_completion_rate': float(np.mean(episode_completion_rates)),
        'std_completion_rate': float(np.std(episode_completion_rates)),
        'avg_cache_hit_rate': float(np.mean(episode_cache_hit_rates)),
        'avg_migration_success_rate': float(np.mean(episode_migration_success_rates)),
    }
    
    # 恢复原始配置
    for key, value in original_config.items():
        setattr(config, key, value)
    
    print(f"\n✓ 场景评估完成:")
    print(f"  平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  平均时延: {results['avg_delay']:.3f}s ± {results['std_delay']:.3f}s")
    print(f"  平均能耗: {results['avg_energy']:.2f} ± {results['std_energy']:.2f}")
    print(f"  完成率: {results['avg_completion_rate']:.2%} ± {results['std_completion_rate']:.2%}")
    
    return results


# ========== 泛化能力测试 ==========

def run_generalization_test(agent, algorithm: str, num_episodes: int = 20) -> Dict:
    """
    运行泛化能力测试（在所有预定义场景下评估）
    
    【功能】
    测试模型在训练场景之外的泛化能力，包括：
    1. 车辆密度泛化
    2. 任务负载泛化
    3. 基础设施变化适应
    4. 网络条件鲁棒性
    5. 任务分布变化适应
    """
    print(f"\n{'='*80}")
    print(f"🎯 开始泛化能力测试")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for set_name, scenarios in SCENARIO_SETS.items():
        print(f"\n📦 测试场景集: {set_name}")
        print(f"{'='*60}")
        
        set_results = []
        for scenario in scenarios:
            result = evaluate_agent_in_scenario(agent, algorithm, scenario, num_episodes)
            set_results.append(result)
        
        all_results[set_name] = set_results
    
    return all_results


# ========== 多算法对比 ==========

def compare_algorithms_cross_scenario(algorithms: List[str], scenario_set: str,
                                      num_episodes: int = 20) -> Dict:
    """
    对比多个算法在不同场景下的性能
    
    【参数】
    - algorithms: 算法列表，如['TD3', 'SAC', 'DDPG']
    - scenario_set: 场景集名称，如'vehicle_density', 'task_load', 'all'
    - num_episodes: 每个场景的评估轮次
    """
    print(f"\n{'='*80}")
    print(f"🔬 多算法跨场景对比")
    print(f"算法: {', '.join(algorithms)}")
    print(f"场景集: {scenario_set}")
    print(f"{'='*80}\n")
    
    # 确定要测试的场景
    if scenario_set == 'all':
        test_scenarios = []
        for scenarios in SCENARIO_SETS.values():
            test_scenarios.extend(scenarios)
    elif scenario_set in SCENARIO_SETS:
        test_scenarios = SCENARIO_SETS[scenario_set]
    else:
        raise ValueError(f"未知场景集: {scenario_set}")
    
    # 为每个算法加载模型并评估
    comparison_results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"📊 评估算法: {algorithm}")
        print(f"{'='*60}")
        
        # 查找最佳模型
        model_path = f"results/models/single_agent/{algorithm.lower()}/best_model_{algorithm.lower()}.pth"
        
        if not os.path.exists(model_path):
            print(f"⚠️ 未找到{algorithm}的最佳模型，跳过")
            continue
        
        # 加载agent
        # 需要获取state_dim和action_dim（这里使用默认值）
        from single_agent.td3 import TD3Environment
        temp_env = TD3Environment()
        state_dim = temp_env.get_state_dim()
        action_dim = temp_env.get_action_dim()
        
        try:
            agent = load_trained_agent(algorithm, model_path, state_dim, action_dim)
        except Exception as e:
            print(f"⚠️ 加载{algorithm}模型失败: {e}")
            continue
        
        # 在所有场景下评估
        algorithm_results = []
        for scenario in test_scenarios:
            result = evaluate_agent_in_scenario(agent, algorithm, scenario, num_episodes)
            algorithm_results.append(result)
        
        comparison_results[algorithm] = algorithm_results
    
    return comparison_results


# ========== 结果可视化 ==========

def visualize_cross_scenario_results(results: Dict, save_dir: str):
    """
    可视化跨场景评估结果
    
    【生成图表】
    1. 各场景性能对比柱状图
    2. 泛化能力雷达图
    3. 性能分布箱线图
    4. 场景敏感性分析
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果是泛化测试结果（包含多个场景集）
    if isinstance(next(iter(results.values())), list) and \
       isinstance(next(iter(results.values()))[0], dict) and \
       'scenario_name' in next(iter(results.values()))[0]:
        
        # 1. 各场景集的性能对比
        for set_name, set_results in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'场景集: {set_name}', fontsize=16, fontweight='bold')
            
            scenario_names = [r['scenario_name'] for r in set_results]
            
            # 时延对比
            ax = axes[0, 0]
            delays = [r['avg_delay'] for r in set_results]
            delay_stds = [r['std_delay'] for r in set_results]
            ax.bar(scenario_names, delays, yerr=delay_stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_ylabel('平均时延 (s)')
            ax.set_title('时延对比')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # 能耗对比
            ax = axes[0, 1]
            energies = [r['avg_energy'] for r in set_results]
            energy_stds = [r['std_energy'] for r in set_results]
            ax.bar(scenario_names, energies, yerr=energy_stds, capsize=5, alpha=0.7, color='coral')
            ax.set_ylabel('总能耗 (J)')
            ax.set_title('能耗对比')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # 完成率对比
            ax = axes[1, 0]
            completion = [r['avg_completion_rate'] * 100 for r in set_results]
            completion_stds = [r['std_completion_rate'] * 100 for r in set_results]
            ax.bar(scenario_names, completion, yerr=completion_stds, capsize=5, alpha=0.7, color='seagreen')
            ax.set_ylabel('完成率 (%)')
            ax.set_title('任务完成率对比')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 105])
            
            # 奖励对比
            ax = axes[1, 1]
            rewards = [r['avg_reward'] for r in set_results]
            reward_stds = [r['std_reward'] for r in set_results]
            ax.bar(scenario_names, rewards, yerr=reward_stds, capsize=5, alpha=0.7, color='mediumpurple')
            ax.set_ylabel('平均奖励')
            ax.set_title('奖励对比')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'scenario_comparison_{set_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 保存图表: scenario_comparison_{set_name}.png")


def save_results_to_file(results: Dict, filepath: str):
    """保存结果到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存到: {filepath}")


# ========== 命令行接口 ==========

def parse_args():
    parser = argparse.ArgumentParser(description='跨场景评估工具')
    
    # 模型参数
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--algorithm', type=str, default='TD3',
                       choices=['TD3', 'SAC', 'DDPG', 'PPO', 'DQN'],
                       help='算法名称')
    
    # 评估模式
    parser.add_argument('--generalization-test', action='store_true',
                       help='运行完整泛化能力测试')
    parser.add_argument('--compare', action='store_true',
                       help='多算法对比模式')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['TD3', 'SAC', 'DDPG'],
                       help='对比的算法列表')
    
    # 场景参数
    parser.add_argument('--scenario-set', type=str, default='vehicle_density',
                       choices=list(SCENARIO_SETS.keys()) + ['all'],
                       help='场景集名称')
    parser.add_argument('--num-vehicles', type=int, help='自定义车辆数')
    parser.add_argument('--arrival-rate', type=float, help='自定义任务到达率')
    
    # 评估参数
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='每个场景的评估轮次')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='results/cross_scenario',
                       help='结果保存目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🚀 跨场景评估工具")
    print(f"{'='*80}\n")
    
    # ========== 模式1: 多算法对比 ==========
    if args.compare:
        results = compare_algorithms_cross_scenario(
            algorithms=args.algorithms,
            scenario_set=args.scenario_set,
            num_episodes=args.eval_episodes
        )
        
        # 保存结果
        result_file = os.path.join(output_dir, 'algorithm_comparison.json')
        save_results_to_file(results, result_file)
        
        # 可视化
        visualize_cross_scenario_results(results, output_dir)
    
    # ========== 模式2: 泛化能力测试 ==========
    elif args.generalization_test:
        if not args.model:
            print("❌ 错误: 泛化测试需要指定--model参数")
            return
        
        # 加载agent
        from single_agent.td3 import TD3Environment
        temp_env = TD3Environment()
        state_dim = temp_env.get_state_dim()
        action_dim = temp_env.get_action_dim()
        
        agent = load_trained_agent(args.algorithm, args.model, state_dim, action_dim)
        
        # 运行泛化测试
        results = run_generalization_test(agent, args.algorithm, args.eval_episodes)
        
        # 保存结果
        result_file = os.path.join(output_dir, f'generalization_test_{args.algorithm.lower()}.json')
        save_results_to_file(results, result_file)
        
        # 可视化
        visualize_cross_scenario_results(results, output_dir)
    
    # ========== 模式3: 单场景评估 ==========
    else:
        if not args.model:
            print("❌ 错误: 需要指定--model参数")
            return
        
        # 加载agent
        from single_agent.td3 import TD3Environment
        temp_env = TD3Environment()
        state_dim = temp_env.get_state_dim()
        action_dim = temp_env.get_action_dim()
        
        agent = load_trained_agent(args.algorithm, args.model, state_dim, action_dim)
        
        # 自定义场景
        if args.num_vehicles or args.arrival_rate:
            params = {}
            if args.num_vehicles:
                params['num_vehicles'] = args.num_vehicles
            if args.arrival_rate:
                params['task_arrival_rate'] = args.arrival_rate
            
            scenario = ScenarioConfig(
                name="custom_scenario",
                description="自定义场景",
                params=params
            )
            
            result = evaluate_agent_in_scenario(agent, args.algorithm, scenario, args.eval_episodes)
            
            # 保存结果
            result_file = os.path.join(output_dir, 'custom_scenario_result.json')
            save_results_to_file(result, result_file)
        
        # 使用预定义场景集
        else:
            scenarios = SCENARIO_SETS[args.scenario_set]
            results = []
            
            for scenario in scenarios:
                result = evaluate_agent_in_scenario(agent, args.algorithm, scenario, args.eval_episodes)
                results.append(result)
            
            # 保存结果
            result_file = os.path.join(output_dir, f'{args.scenario_set}_results.json')
            save_results_to_file({'scenarios': results}, result_file)
            
            # 可视化
            visualize_cross_scenario_results({args.scenario_set: results}, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ 评估完成！结果保存在: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()




