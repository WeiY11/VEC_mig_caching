#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline对比实验主脚本
真实训练所有算法，不使用模拟数据

【对比算法】
DRL: TD3, DDPG, SAC, PPO, DQN（真实训练）
Baseline: Random, Greedy, RoundRobin, LocalFirst, NearestNode（策略执行）

【使用】
快速测试: python run_baseline_comparison.py --episodes 50 --quick
标准对比: python run_baseline_comparison.py --episodes 200
完整对比: python run_baseline_comparison.py --episodes 500 --full
单独算法: python run_baseline_comparison.py --algorithm TD3 --episodes 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# 导入项目模块
from config import config
from train_single_agent import SingleAgentTrainingEnvironment
from evaluation.system_simulator import CompleteSystemSimulator

# 优先使用改进的baseline算法，如果不存在则使用旧版
try:
    from baseline_comparison.improved_baseline_algorithms import create_baseline_algorithm
    print("[INFO] 使用改进的Baseline算法（16维动作）")
except ImportError:
    from baseline_comparison.baseline_algorithms import create_baseline_algorithm
    print("[WARN] 使用旧版Baseline算法（18维动作）")


class BaselineComparisonExperiment:
    """
    Baseline对比实验执行器
    
    【职责】
    1. 真实训练所有DRL算法
    2. 真实运行所有启发式算法
    3. 公平对比和分析
    """
    
    def __init__(self, save_dir: str = None, num_vehicles: int = 12):
        """初始化实验环境"""
        if save_dir is None:
            self.save_dir = Path(__file__).parent / "results"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建分析目录
        self.analysis_dir = Path(__file__).parent / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # 实验结果存储
        self.results = {}
        
        # 网络拓扑配置（固定RSU和UAV）
        self.num_vehicles = num_vehicles
        self.num_rsus = 4  # 固定
        self.num_uavs = 2  # 固定
        
        print("="*80)
        print("Baseline对比实验环境初始化")
        print("="*80)
        print(f"结果保存目录: {self.save_dir}")
        print(f"分析保存目录: {self.analysis_dir}")
        print(f"网络拓扑: {self.num_vehicles}辆车 + {self.num_rsus}个RSU + {self.num_uavs}个UAV (固定)")
        print("="*80)
    
    def run_drl_algorithm(self, algorithm: str, num_episodes: int, random_seed: int = 42) -> Dict:
        """
        运行DRL算法（真实训练）
        
        【参数】
        - algorithm: 算法名称（TD3, DDPG, SAC, PPO, DQN）
        - num_episodes: 训练轮次
        - random_seed: 随机种子
        """
        print(f"\n{'='*80}")
        print(f"真实训练 DRL算法: {algorithm}")
        print(f"{'='*80}")
        
        # 设置随机种子
        np.random.seed(random_seed)
        os.environ['RANDOM_SEED'] = str(random_seed)
        
        # 设置车辆数（保持固定拓扑）
        if self.num_vehicles != 12:  # 如果不是默认值，需要覆盖
            overrides = {"num_vehicles": self.num_vehicles}
            os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(overrides)
        
        # 创建训练环境
        training_env = SingleAgentTrainingEnvironment(algorithm)
        
        # 训练统计
        episode_rewards = []
        episode_delays = []
        episode_energies = []
        episode_completion_rates = []
        
        start_time = time.time()
        
        # 训练循环
        for episode in range(1, num_episodes + 1):
            episode_result = training_env.run_episode(episode)
            
            # 收集指标
            episode_rewards.append(episode_result['avg_reward'])
            
            system_metrics = episode_result.get('system_metrics', {})
            episode_delays.append(system_metrics.get('avg_task_delay', 0))
            episode_energies.append(system_metrics.get('total_energy_consumption', 0))
            episode_completion_rates.append(system_metrics.get('task_completion_rate', 0))
            
            # 打印进度
            if episode % 20 == 0 or episode == num_episodes:
                print(f"  Episode {episode}/{num_episodes}: "
                      f"Reward={episode_rewards[-1]:.3f}, "
                      f"Delay={episode_delays[-1]:.3f}s, "
                      f"Energy={episode_energies[-1]:.1f}J, "
                      f"Completion={episode_completion_rates[-1]:.1%}")
        
        training_time = time.time() - start_time
        
        # 计算稳定期性能（后50%）
        stable_start = num_episodes // 2
        
        result = {
            'algorithm': algorithm,
            'algorithm_type': 'DRL',
            'num_episodes': num_episodes,
            'random_seed': random_seed,
            'training_time': training_time,
            
            # 性能指标（稳定期平均）
            'avg_delay': float(np.mean(episode_delays[stable_start:])),
            'std_delay': float(np.std(episode_delays[stable_start:])),
            'avg_energy': float(np.mean(episode_energies[stable_start:])),
            'std_energy': float(np.std(episode_energies[stable_start:])),
            'avg_completion_rate': float(np.mean(episode_completion_rates[stable_start:])),
            'std_completion_rate': float(np.std(episode_completion_rates[stable_start:])),
            
            # 收敛性指标
            'initial_reward': float(np.mean(episode_rewards[:10])),
            'final_reward': float(np.mean(episode_rewards[-10:])),
            'improvement': float(np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10])),
            
            # 完整历史数据
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_delays': [float(d) for d in episode_delays],
            'episode_energies': [float(e) for e in episode_energies],
            'episode_completion_rates': [float(c) for c in episode_completion_rates]
        }
        
        print(f"\n{'='*80}")
        print(f"{algorithm} 训练完成")
        print(f"{'='*80}")
        print(f"  平均时延: {result['avg_delay']:.3f}±{result['std_delay']:.3f}s")
        print(f"  平均能耗: {result['avg_energy']:.1f}±{result['std_energy']:.1f}J")
        print(f"  任务完成率: {result['avg_completion_rate']:.2%}")
        print(f"  性能改善: {result['improvement']:.1f}")
        print(f"  训练耗时: {training_time:.1f}秒")
        print(f"{'='*80}\n")
        
        return result
    
    def run_baseline_algorithm(self, algorithm: str, num_episodes: int, random_seed: int = 42) -> Dict:
        """
        运行启发式Baseline算法（策略执行）
        
        【参数】
        - algorithm: 算法名称（Random, Greedy, RoundRobin, LocalFirst, NearestNode）
        - num_episodes: 运行轮次
        - random_seed: 随机种子
        """
        print(f"\n{'='*80}")
        print(f"运行 Baseline算法: {algorithm}")
        print(f"{'='*80}")
        
        # 设置随机种子
        np.random.seed(random_seed)
        os.environ['RANDOM_SEED'] = str(random_seed)
        
        # 设置车辆数（保持固定拓扑）
        if self.num_vehicles != 12:
            overrides = {"num_vehicles": self.num_vehicles}
            os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(overrides)

        # 创建环境（使用TD3环境以确保一致性）
        env = SingleAgentTrainingEnvironment("TD3")
        baseline_algo = create_baseline_algorithm(algorithm)
        baseline_algo.update_environment(env)

        episode_rewards = []
        episode_delays = []
        episode_energies = []
        episode_completion_rates = []

        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            state = env.reset_environment()
            baseline_algo.reset()

            episode_reward = 0.0
            episode_steps = 0
            info = {}

            for step in range(config.experiment.max_steps_per_episode):
                action_vec = baseline_algo.select_action(state)
                actions_dict = env._build_actions_from_vector(action_vec)
                next_state, reward, done, info = env.step(action_vec, state, actions_dict)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if done:
                    break

            episode_metrics = info.get('system_metrics', {})
            episode_rewards.append(episode_reward / max(1, episode_steps))
            episode_delays.append(episode_metrics.get('avg_task_delay', 0))
            episode_energies.append(episode_metrics.get('total_energy_consumption', 0))
            episode_completion_rates.append(episode_metrics.get('task_completion_rate', 0))

            if episode % 20 == 0 or episode == num_episodes:
                print(f"  Episode {episode}/{num_episodes}: "
                      f"Reward={episode_rewards[-1]:.3f}, "
                      f"Delay={episode_delays[-1]:.3f}s, "
                      f"Energy={episode_energies[-1]:.1f}J, "
                      f"Completion={episode_completion_rates[-1]:.1%}")

        execution_time = time.time() - start_time

        stable_start = num_episodes // 2

        result = {
            'algorithm': algorithm,
            'algorithm_type': 'Baseline',
            'num_episodes': num_episodes,
            'random_seed': random_seed,
            'execution_time': execution_time,
            'avg_delay': float(np.mean(episode_delays[stable_start:])),
            'std_delay': float(np.std(episode_delays[stable_start:])),
            'avg_energy': float(np.mean(episode_energies[stable_start:])),
            'std_energy': float(np.std(episode_energies[stable_start:])),
            'avg_completion_rate': float(np.mean(episode_completion_rates[stable_start:])),
            'std_completion_rate': float(np.std(episode_completion_rates[stable_start:])),
            'initial_reward': float(np.mean(episode_rewards[:10])) if len(episode_rewards) >= 10 else float(np.mean(episode_rewards)),
            'final_reward': float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else float(np.mean(episode_rewards)),
            'improvement': 0.0,
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_delays': [float(d) for d in episode_delays],
            'episode_energies': [float(e) for e in episode_energies],
            'episode_completion_rates': [float(c) for c in episode_completion_rates]
        }

        print(f"\n{'='*80}")
        print(f"{algorithm} 运行完成")
        print(f"{'='*80}")
        print(f"  平均时延: {result['avg_delay']:.3f}±{result['std_delay']:.3f}s")
        print(f"  平均能耗: {result['avg_energy']:.1f}±{result['std_energy']:.1f}J")
        print(f"  任务完成率: {result['avg_completion_rate']:.2%}")
        print(f"  运行耗时: {execution_time:.1f}秒")
        print(f"{'='*80}\n")

        return result
    
    def run_all_algorithms(self, num_episodes: int = 200, random_seed: int = 42):
        """
        运行所有对比算法
        
        【参数】
        - num_episodes: 训练/运行轮次
        - random_seed: 随机种子
        """
        # 定义所有算法
        drl_algorithms = ['TD3', 'DDPG', 'SAC', 'PPO', 'DQN']
        baseline_algorithms = ['Random', 'Greedy', 'RoundRobin', 'LocalFirst', 'NearestNode']
        
        all_algorithms = drl_algorithms + baseline_algorithms
        
        print("\n" + "="*80)
        print("开始运行所有Baseline对比实验")
        print("="*80)
        print(f"  DRL算法: {len(drl_algorithms)} 个（需训练）")
        print(f"  Baseline算法: {len(baseline_algorithms)} 个（策略执行）")
        print(f"  每算法轮次: {num_episodes}")
        print(f"  预计总耗时: ~{len(drl_algorithms) * num_episodes * 2 / 60:.0f}分钟（DRL）+ ~{len(baseline_algorithms) * 5:.0f}分钟（Baseline）")
        print("="*80)
        
        total_start = time.time()
        
        # 1. 运行DRL算法（需要训练）
        print("\n" + "="*80)
        print("第一阶段：真实训练DRL算法")
        print("="*80)
        
        for i, algo_name in enumerate(drl_algorithms, 1):
            print(f"\n[{i}/{len(drl_algorithms)}] 训练 {algo_name}...")
            
            result = self.run_drl_algorithm(algo_name, num_episodes, random_seed)
            self.results[algo_name] = result
            
            # 保存单个算法结果
            algo_dir = self.save_dir / algo_name
            algo_dir.mkdir(exist_ok=True, parents=True)
            result_file = algo_dir / f"result_{algo_name}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # 打印进度
            elapsed = time.time() - total_start
            estimated_total = elapsed / i * len(drl_algorithms)
            remaining = estimated_total - elapsed
            print(f"进度: DRL {i}/{len(drl_algorithms)}, "
                  f"已用时: {elapsed/60:.1f}分钟, "
                  f"剩余: {remaining/60:.1f}分钟")
        
        # 2. 运行Baseline算法（无训练）
        print("\n" + "="*80)
        print("第二阶段：运行启发式Baseline算法")
        print("="*80)
        
        for i, algo_name in enumerate(baseline_algorithms, 1):
            print(f"\n[{i}/{len(baseline_algorithms)}] 运行 {algo_name}...")
            
            result = self.run_baseline_algorithm(algo_name, num_episodes, random_seed)
            self.results[algo_name] = result
            
            # 保存单个算法结果
            algo_dir = self.save_dir / algo_name
            algo_dir.mkdir(exist_ok=True, parents=True)
            result_file = algo_dir / f"result_{algo_name}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*80)
        print("所有算法运行完成!")
        print("="*80)
        print(f"  总耗时: {total_time/60:.1f}分钟")
        print(f"  算法数: {len(self.results)}")
        print("="*80)
    
    def run_single_algorithm(self, algorithm: str, num_episodes: int, random_seed: int = 42):
        """运行单个算法"""
        drl_list = ['TD3', 'DDPG', 'SAC', 'PPO', 'DQN']
        
        if algorithm in drl_list:
            result = self.run_drl_algorithm(algorithm, num_episodes, random_seed)
        else:
            result = self.run_baseline_algorithm(algorithm, num_episodes, random_seed)
        
        self.results[algorithm] = result
        
        # 保存结果
        algo_dir = self.save_dir / algorithm
        algo_dir.mkdir(exist_ok=True, parents=True)
        result_file = algo_dir / f"result_{algorithm}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def analyze_results(self):
        """分析对比结果"""
        if not self.results:
            print("警告: 没有实验结果可分析")
            return
        
        print("\n" + "="*80)
        print("Baseline对比结果分析")
        print("="*80)
        
        # 分组统计
        drl_results = {k: v for k, v in self.results.items() if v['algorithm_type'] == 'DRL'}
        baseline_results = {k: v for k, v in self.results.items() if v['algorithm_type'] == 'Baseline'}
        
        # 找到最佳性能（TD3作为参考）
        if 'TD3' in self.results:
            td3_result = self.results['TD3']
            
            print(f"\nTD3性能（参考）:")
            print(f"  时延: {td3_result['avg_delay']:.3f}s")
            print(f"  能耗: {td3_result['avg_energy']:.1f}J")
            print(f"  完成率: {td3_result['avg_completion_rate']:.2%}")
            
            print(f"\n{'算法':<15} {'类型':<10} {'时延(s)':<12} {'时延提升':<12} {'能耗(J)':<12} {'能耗提升':<12} {'完成率':<10}")
            print("-"*100)
            
            # 按时延排序
            sorted_results = sorted(self.results.items(), key=lambda x: x[1]['avg_delay'])
            
            for algo_name, result in sorted_results:
                algo_type = result['algorithm_type']
                delay = result['avg_delay']
                energy = result['avg_energy']
                completion = result['avg_completion_rate']
                
                # 计算相对TD3的提升
                delay_improve = (delay - td3_result['avg_delay']) / td3_result['avg_delay'] * 100
                energy_improve = (energy - td3_result['avg_energy']) / td3_result['avg_energy'] * 100
                
                marker = "⭐" if algo_name == 'TD3' else ""
                
                print(f"{algo_name:<15} {algo_type:<10} {delay:<12.3f} {delay_improve:>+10.1f}% "
                      f"{energy:<12.1f} {energy_improve:>+10.1f}% {completion*100:<10.1f} {marker}")
        
        print("\n" + "="*80)
        
        # 计算复合指标和统计显著性
        self._calculate_composite_metrics()
    
    def _calculate_composite_metrics(self):
        """
        计算复合指标（目标函数值）和统计显著性检验
        
        目标函数: J = ω_T × 时延 + ω_E × 能耗
        其中: ω_T = 2.0, ω_E = 1.2
        """
        if not self.results or 'TD3' not in self.results:
            return
        
        print("\n" + "="*80)
        print("复合指标分析（目标函数）")
        print("="*80)
        print(f"目标函数: J = 2.0 × Delay + 1.2 × Energy (越小越好)")
        print("="*80)
        
        # 计算每个算法的目标函数值
        objective_values = {}
        for algo_name, result in self.results.items():
            delay = result['avg_delay']
            energy = result['avg_energy']
            # 目标函数：J = ω_T × delay + ω_E × (energy/600)
            # 能耗归一化以匹配时延的数量级
            objective = 2.0 * delay + 1.2 * (energy / 600.0)
            objective_values[algo_name] = objective
            result['objective_value'] = objective
        
        # 按目标函数值排序
        sorted_by_objective = sorted(objective_values.items(), key=lambda x: x[1])
        
        print(f"\n{'算法':<15} {'类型':<10} {'目标函数值':<15} {'相对TD3':<15}")
        print("-"*60)
        
        td3_objective = objective_values.get('TD3', 1.0)
        
        for algo_name, obj_value in sorted_by_objective:
            algo_type = self.results[algo_name]['algorithm_type']
            relative_perf = (obj_value - td3_objective) / td3_objective * 100
            marker = "⭐" if algo_name == 'TD3' else ""
            
            print(f"{algo_name:<15} {algo_type:<10} {obj_value:<15.4f} {relative_perf:>+13.1f}% {marker}")
        
        # 统计显著性检验（仅对有历史数据的算法）
        self._perform_statistical_test()
    
    def _perform_statistical_test(self):
        """执行统计显著性检验（t-test）"""
        try:
            from scipy import stats
        except ImportError:
            print("\n[WARN] scipy未安装，跳过统计检验")
            return
        
        if 'TD3' not in self.results:
            return
        
        print("\n" + "="*80)
        print("统计显著性检验（vs TD3）")
        print("="*80)
        
        td3_delays = self.results['TD3']['episode_delays']
        td3_energies = self.results['TD3']['episode_energies']
        
        # 使用稳定期数据
        stable_start = len(td3_delays) // 2
        td3_delays_stable = td3_delays[stable_start:]
        td3_energies_stable = td3_energies[stable_start:]
        
        print(f"\n{'算法':<15} {'时延p-value':<15} {'能耗p-value':<15} {'显著性':<10}")
        print("-"*60)
        
        for algo_name, result in self.results.items():
            if algo_name == 'TD3':
                continue
            
            algo_delays = result['episode_delays'][stable_start:]
            algo_energies = result['episode_energies'][stable_start:]
            
            # t检验（双侧）
            _, p_delay = stats.ttest_ind(td3_delays_stable, algo_delays)
            _, p_energy = stats.ttest_ind(td3_energies_stable, algo_energies)
            
            # 判断显著性（p < 0.05为显著）
            sig_delay = "***" if p_delay < 0.001 else ("**" if p_delay < 0.01 else ("*" if p_delay < 0.05 else "n.s."))
            sig_energy = "***" if p_energy < 0.001 else ("**" if p_energy < 0.01 else ("*" if p_energy < 0.05 else "n.s."))
            significance = f"{sig_delay}/{sig_energy}"
            
            print(f"{algo_name:<15} {p_delay:<15.6f} {p_energy:<15.6f} {significance:<10}")
        
        print("\n说明: *** p<0.001, ** p<0.01, * p<0.05, n.s. 不显著")
        print("="*80)
    
    def generate_plots(self):
        """生成对比图表"""
        if not self.results:
            print("警告: 没有结果可绘图")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("警告: 缺少matplotlib")
            return
        
        print("\n生成对比图表...")
        
        # 提取数据并分组
        drl_names = []
        drl_delays = []
        drl_energies = []
        drl_completions = []
        
        baseline_names = []
        baseline_delays = []
        baseline_energies = []
        baseline_completions = []
        
        for algo_name, result in self.results.items():
            if result['algorithm_type'] == 'DRL':
                drl_names.append(algo_name)
                drl_delays.append(result['avg_delay'])
                drl_energies.append(result['avg_energy'])
                drl_completions.append(result['avg_completion_rate'] * 100)
            else:
                baseline_names.append(algo_name)
                baseline_delays.append(result['avg_delay'])
                baseline_energies.append(result['avg_energy'])
                baseline_completions.append(result['avg_completion_rate'] * 100)
        
        # 创建对比图（3个指标）
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 时延对比
        x_drl = np.arange(len(drl_names))
        x_baseline = np.arange(len(baseline_names)) + len(drl_names) + 0.5
        
        axes[0].bar(x_drl, drl_delays, width=0.6, label='DRL算法', color='skyblue', edgecolor='navy', alpha=0.8)
        axes[0].bar(x_baseline, baseline_delays, width=0.6, label='启发式算法', color='lightcoral', edgecolor='darkred', alpha=0.8)
        axes[0].set_title('平均任务时延对比', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('时延 (秒)', fontsize=12)
        axes[0].set_xticks(list(x_drl) + list(x_baseline))
        axes[0].set_xticklabels(drl_names + baseline_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axvline(x=len(drl_names)-0.25, color='gray', linestyle='--', alpha=0.5)
        
        # 能耗对比
        axes[1].bar(x_drl, drl_energies, width=0.6, label='DRL算法', color='skyblue', edgecolor='navy', alpha=0.8)
        axes[1].bar(x_baseline, baseline_energies, width=0.6, label='启发式算法', color='lightcoral', edgecolor='darkred', alpha=0.8)
        axes[1].set_title('系统总能耗对比', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('能耗 (焦耳)', fontsize=12)
        axes[1].set_xticks(list(x_drl) + list(x_baseline))
        axes[1].set_xticklabels(drl_names + baseline_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axvline(x=len(drl_names)-0.25, color='gray', linestyle='--', alpha=0.5)
        
        # 完成率对比
        axes[2].bar(x_drl, drl_completions, width=0.6, label='DRL算法', color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        axes[2].bar(x_baseline, baseline_completions, width=0.6, label='启发式算法', color='lightyellow', edgecolor='orange', alpha=0.8)
        axes[2].set_title('任务完成率对比', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('完成率 (%)', fontsize=12)
        axes[2].set_xticks(list(x_drl) + list(x_baseline))
        axes[2].set_xticklabels(drl_names + baseline_names, rotation=45, ha='right')
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].axvline(x=len(drl_names)-0.25, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = self.analysis_dir / 'performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 性能对比图: {plot_path}")
        plt.close()
        
        # 生成复合指标对比图
        self._generate_objective_comparison()
        
        # 生成收敛曲线（仅DRL）
        self._generate_convergence_curves()
    
    def _generate_objective_comparison(self):
        """生成目标函数对比图（论文核心指标）"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        if not self.results or 'objective_value' not in list(self.results.values())[0]:
            return
        
        print("\n生成目标函数对比图...")
        
        # 提取数据
        drl_names = []
        drl_objectives = []
        baseline_names = []
        baseline_objectives = []
        
        for algo_name, result in self.results.items():
            obj_val = result.get('objective_value', 0)
            if result['algorithm_type'] == 'DRL':
                drl_names.append(algo_name)
                drl_objectives.append(obj_val)
            else:
                baseline_names.append(algo_name)
                baseline_objectives.append(obj_val)
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x_drl = np.arange(len(drl_names))
        x_baseline = np.arange(len(baseline_names)) + len(drl_names) + 0.5
        
        # 绘制柱状图
        bars1 = ax.bar(x_drl, drl_objectives, width=0.6, label='DRL Algorithms', 
                       color='#2E86AB', edgecolor='navy', alpha=0.85)
        bars2 = ax.bar(x_baseline, baseline_objectives, width=0.6, label='Heuristic Algorithms', 
                       color='#C73E1D', edgecolor='darkred', alpha=0.85)
        
        # 突出显示TD3
        if 'TD3' in drl_names:
            td3_idx = drl_names.index('TD3')
            bars1[td3_idx].set_color('#F18F01')
            bars1[td3_idx].set_edgecolor('darkorange')
            bars1[td3_idx].set_linewidth(3)
        
        ax.set_title('Objective Function Comparison (J = 2.0×Delay + 1.2×Energy)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Objective Value (Lower is Better)', fontsize=12)
        ax.set_xticks(list(x_drl) + list(x_baseline))
        ax.set_xticklabels(drl_names + baseline_names, rotation=45, ha='right')
        ax.legend(frameon=False, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axvline(x=len(drl_names)-0.25, color='gray', linestyle='--', alpha=0.5)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, rotation=0)
        
        plt.tight_layout()
        plot_path = self.analysis_dir / 'objective_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 目标函数对比图: {plot_path}")
        plt.close()
    
    def _generate_convergence_curves(self):
        """生成DRL算法收敛曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        drl_results = {k: v for k, v in self.results.items() if v['algorithm_type'] == 'DRL'}
        
        if not drl_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 滑动平均函数
        def smooth(data, window=20):
            if len(data) < window:
                return data
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window + 1)
                smoothed.append(np.mean(data[start:i+1]))
            return smoothed
        
        for algo_name, result in drl_results.items():
            # 时延收敛
            delays_smooth = smooth(result['episode_delays'])
            axes[0, 0].plot(delays_smooth, label=algo_name, alpha=0.85, linewidth=2)
            
            # 能耗收敛
            energies_smooth = smooth(result['episode_energies'])
            axes[0, 1].plot(energies_smooth, label=algo_name, alpha=0.85, linewidth=2)
            
            # 完成率收敛
            completions_smooth = smooth(result['episode_completion_rates'])
            axes[1, 0].plot(completions_smooth, label=algo_name, alpha=0.85, linewidth=2)
            
            # 奖励收敛
            rewards_smooth = smooth(result['episode_rewards'])
            axes[1, 1].plot(rewards_smooth, label=algo_name, alpha=0.85, linewidth=2)
        
        axes[0, 0].set_title('时延收敛曲线 (滑动平均)', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Episode', fontsize=10)
        axes[0, 0].set_ylabel('平均时延 (s)', fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_title('能耗收敛曲线 (滑动平均)', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Episode', fontsize=10)
        axes[0, 1].set_ylabel('总能耗 (J)', fontsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_title('完成率收敛曲线 (滑动平均)', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Episode', fontsize=10)
        axes[1, 0].set_ylabel('完成率', fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].set_title('奖励收敛曲线 (滑动平均, 窗口=20)', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Episode', fontsize=10)
        axes[1, 1].set_ylabel('平均奖励', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        curve_path = self.analysis_dir / 'convergence_curves.png'
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 收敛曲线图: {curve_path}")
        plt.close()
        
        # 生成训练过程数据点图（可选，用于展示训练细节）
        self._generate_training_scatter_plots()
    
    def _generate_training_scatter_plots(self):
        """
        生成训练过程数据点图（每个episode一个点）
        用途: 展示训练过程的波动和收敛细节
        注意: 这不是参数扫描对比图，而是单次实验的详细视图
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        drl_results = {k: v for k, v in self.results.items() if v['algorithm_type'] == 'DRL'}
        
        if not drl_results:
            return
        
        print("\n生成离散折线图...")
        
        # 定义marker样式
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D', 
                 '#17BEBB', '#9B59B6', '#E67E22']
        
        # 创建2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle('Discrete Line Plots with Data Points', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 降采样函数（避免数据点过密）
        def downsample(data, max_points=50):
            """智能降采样，保留关键数据点"""
            n = len(data)
            if n <= max_points:
                return list(range(n)), data
            
            # 计算采样间隔
            step = n // max_points
            indices = list(range(0, n, step))
            
            # 确保包含最后一个点
            if indices[-1] != n - 1:
                indices.append(n - 1)
            
            sampled_data = [data[i] for i in indices]
            return indices, sampled_data
        
        for idx, (algo_name, result) in enumerate(drl_results.items()):
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            
            # 图1: 时延（离散点）
            episodes_delay, delays_sampled = downsample(result['episode_delays'])
            axes[0, 0].plot(episodes_delay, delays_sampled, 
                          label=algo_name, marker=marker, markersize=6,
                          color=color, linewidth=1.5, alpha=0.8,
                          markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
            
            # 图2: 能耗（离散点）
            episodes_energy, energies_sampled = downsample(result['episode_energies'])
            axes[0, 1].plot(episodes_energy, energies_sampled,
                          label=algo_name, marker=marker, markersize=6,
                          color=color, linewidth=1.5, alpha=0.8,
                          markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
            
            # 图3: 完成率（离散点）
            episodes_comp, completions_sampled = downsample(result['episode_completion_rates'])
            axes[1, 0].plot(episodes_comp, completions_sampled,
                          label=algo_name, marker=marker, markersize=6,
                          color=color, linewidth=1.5, alpha=0.8,
                          markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
            
            # 图4: 奖励（离散点）
            episodes_reward, rewards_sampled = downsample(result['episode_rewards'])
            axes[1, 1].plot(episodes_reward, rewards_sampled,
                          label=algo_name, marker=marker, markersize=6,
                          color=color, linewidth=1.5, alpha=0.8,
                          markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
        
        # 配置子图
        axes[0, 0].set_title('Average Task Delay', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Episode', fontsize=11)
        axes[0, 0].set_ylabel('Delay (seconds)', fontsize=11)
        axes[0, 0].legend(fontsize=9, framealpha=0.9, loc='best')
        axes[0, 0].grid(alpha=0.3, linestyle='--')
        
        axes[0, 1].set_title('Total Energy Consumption', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Episode', fontsize=11)
        axes[0, 1].set_ylabel('Energy (Joules)', fontsize=11)
        axes[0, 1].legend(fontsize=9, framealpha=0.9, loc='best')
        axes[0, 1].grid(alpha=0.3, linestyle='--')
        
        axes[1, 0].set_title('Task Completion Rate', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Episode', fontsize=11)
        axes[1, 0].set_ylabel('Completion Rate', fontsize=11)
        axes[1, 0].legend(fontsize=9, framealpha=0.9, loc='best')
        axes[1, 0].grid(alpha=0.3, linestyle='--')
        
        axes[1, 1].set_title('Episode Reward', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Episode', fontsize=11)
        axes[1, 1].set_ylabel('Average Reward', fontsize=11)
        axes[1, 1].legend(fontsize=9, framealpha=0.9, loc='best')
        axes[1, 1].grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        scatter_plot_path = self.analysis_dir / 'training_scatter_plots.png'
        plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 训练数据点图: {scatter_plot_path}")
        plt.close()
    
    def save_all_results(self):
        """保存所有结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存汇总JSON
        summary_file = self.save_dir / f"comparison_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果汇总已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Baseline对比实验（真实训练）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  快速测试:   python run_baseline_comparison.py --quick
  标准对比:   python run_baseline_comparison.py --episodes 200
  完整实验:   python run_baseline_comparison.py --full
  单独算法:   python run_baseline_comparison.py --algorithm TD3 --episodes 100
  指定车辆数: python run_baseline_comparison.py --episodes 200 --num-vehicles 16
  多seed运行: python run_baseline_comparison.py --episodes 200 --multi-seed 3
        """
    )
    
    parser.add_argument('--episodes', type=int, default=200, help='训练/运行轮次（默认:200）')
    parser.add_argument('--algorithm', type=str, default=None, 
                       help='单独运行某个算法（TD3/DDPG/SAC/PPO/DQN/Random/Greedy/RoundRobin/LocalFirst/NearestNode）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认:42）')
    parser.add_argument('--num-vehicles', type=int, default=12, 
                       help='车辆数量（默认:12，固定拓扑4 RSU+2 UAV）')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（50轮）')
    parser.add_argument('--full', action='store_true', help='完整实验模式（500轮）')
    parser.add_argument('--multi-seed', type=int, default=1, 
                       help='多seed运行次数（用于统计可靠性，默认:1）')
    parser.add_argument('--only-drl', action='store_true', help='只运行DRL算法（跳过启发式算法）')
    parser.add_argument('--only-baseline', action='store_true', help='只运行启发式算法（跳过DRL）')
    
    args = parser.parse_args()
    
    # 确定轮次
    if args.quick:
        num_episodes = 50
    elif args.full:
        num_episodes = 500
    else:
        num_episodes = args.episodes
    
    print("\n" + "="*80)
    print("Baseline对比实验配置")
    print("="*80)
    print(f"  训练轮次: {num_episodes}")
    print(f"  车辆数量: {args.num_vehicles}")
    print(f"  固定拓扑: 4 RSU + 2 UAV")
    print(f"  随机种子: {args.seed}")
    print(f"  多seed运行: {args.multi_seed} 次")
    print(f"  模式: {'快速测试' if args.quick else ('完整实验' if args.full else '标准对比')}")
    print("="*80)
    
    # 多seed运行
    if args.multi_seed > 1:
        print(f"\n将运行 {args.multi_seed} 个不同随机种子的实验...")
        all_seeds_results = []
        
        for seed_idx in range(args.multi_seed):
            current_seed = args.seed + seed_idx * 100
            print(f"\n{'='*80}")
            print(f"运行第 {seed_idx + 1}/{args.multi_seed} 个seed: {current_seed}")
            print("="*80)
            
            # 创建实验环境
            experiment = BaselineComparisonExperiment(num_vehicles=args.num_vehicles)
            
            # 运行实验
            if args.algorithm:
                experiment.run_single_algorithm(args.algorithm, num_episodes, current_seed)
            else:
                experiment.run_all_algorithms(num_episodes, current_seed)
            
            all_seeds_results.append(experiment.results)
        
        # 合并多seed结果（计算平均和方差）
        print("\n" + "="*80)
        print(f"多seed结果汇总（{args.multi_seed}个seeds）")
        print("="*80)
        # TODO: 实现多seed结果聚合
        print("[INFO] 多seed结果聚合功能开发中...")
        
        # 使用最后一个seed的experiment对象进行分析
        experiment.analyze_results()
        experiment.generate_plots()
        experiment.save_all_results()
        
    else:
        # 单seed运行
        experiment = BaselineComparisonExperiment(num_vehicles=args.num_vehicles)
        
        # 运行实验
        if args.algorithm:
            # 单独运行某个算法
            print(f"\n单独运行算法: {args.algorithm}")
            experiment.run_single_algorithm(args.algorithm, num_episodes, args.seed)
        elif args.only_drl:
            # 只运行DRL算法
            print("\n只运行DRL算法...")
            drl_algorithms = ['TD3', 'DDPG', 'SAC', 'PPO', 'DQN']
            for algo in drl_algorithms:
                result = experiment.run_drl_algorithm(algo, num_episodes, args.seed)
                experiment.results[algo] = result
        elif args.only_baseline:
            # 只运行启发式算法
            print("\n只运行启发式算法...")
            baseline_algorithms = ['Random', 'Greedy', 'RoundRobin', 'LocalFirst', 'NearestNode']
            for algo in baseline_algorithms:
                result = experiment.run_baseline_algorithm(algo, num_episodes, args.seed)
                experiment.results[algo] = result
        else:
            # 运行所有算法
            experiment.run_all_algorithms(num_episodes, args.seed)
        
        # 分析结果
        experiment.analyze_results()
        
        # 生成图表
        experiment.generate_plots()
        
        # 保存结果
        experiment.save_all_results()
    
    print("\n" + "="*80)
    print("实验全部完成!")
    print("="*80)
    print(f"  结果目录: {experiment.save_dir}")
    print(f"  分析目录: {experiment.analysis_dir}")
    print(f"  车辆配置: {args.num_vehicles}辆车 + 4 RSU + 2 UAV")
    print("="*80)


if __name__ == "__main__":
    main()

