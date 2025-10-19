#!/usr/bin/env python3
"""
GA (Genetic Algorithm) - 遗传算法
使用进化策略优化任务迁移和缓存决策
"""

import numpy as np
import time
import argparse
from pathlib import Path
import sys
from typing import List, Tuple, Optional

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from train_single_agent import SingleAgentTrainingEnvironment
from baseline_comparison.individual_runners.common import ResultsManager
from baseline_comparison.improved_baseline_algorithms import BaselineAlgorithm


class GABaseline(BaselineAlgorithm):
    """
    遗传算法 (Genetic Algorithm)
    
    策略：
    1. 维护一个种群，每个个体是一个动作向量
    2. 评估每个个体的适应度（系统奖励）
    3. 选择优秀个体进行交叉和变异
    4. 产生新一代种群，迭代优化
    """
    
    def __init__(self, name="GA", population_size=20, elite_ratio=0.2, 
                 mutation_rate=0.1, mutation_std=0.1, crossover_rate=0.8):
        super().__init__(name)
        
        # GA参数
        self.population_size = population_size
        self.elite_size = int(population_size * elite_ratio)
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.crossover_rate = crossover_rate
        
        # 种群和适应度
        self.population = None
        self.fitness_scores = None
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # 进化历史
        self.generation = 0
        self.fitness_history = []
        
        # 评估缓存
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.total_evaluations = 0
        
    def reset(self):
        """重置算法状态"""
        super().reset()
        self.population = None
        self.fitness_scores = None
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.generation = 0
        self.fitness_history = []
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.total_evaluations = 0
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        选择动作 - 使用遗传算法优化
        
        对于每个状态，运行一次快速进化来找到最优动作
        """
        # 初始化种群（如果是新的episode）
        if self.population is None:
            self.initialize_population()
        
        # 快速进化几代找到当前状态的最优动作
        num_generations = 3  # 快速进化代数
        for _ in range(num_generations):
            # 评估种群适应度
            self.evaluate_population(state)
            
            # 选择、交叉、变异产生新种群
            self.evolve_population()
            
            self.generation += 1
        
        # 返回最佳个体作为动作
        if self.best_individual is not None:
            action = self.best_individual.copy()
        else:
            # 如果没有找到最佳个体，返回随机动作
            action = np.random.uniform(-1, 1, 16)
            
        return action
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        
        # 生成多样化的初始种群
        for i in range(self.population_size):
            if i < self.population_size // 4:
                # 1/4 偏向本地处理
                individual = self._create_biased_individual('local')
            elif i < self.population_size // 2:
                # 1/4 偏向RSU处理
                individual = self._create_biased_individual('rsu')
            elif i < 3 * self.population_size // 4:
                # 1/4 偏向UAV处理
                individual = self._create_biased_individual('uav')
            else:
                # 1/4 完全随机
                individual = np.random.uniform(-1, 1, 16)
                
            self.population.append(individual)
            
        self.population = np.array(self.population)
        self.fitness_scores = np.zeros(self.population_size)
        
    def _create_biased_individual(self, bias_type: str) -> np.ndarray:
        """创建有偏向的个体"""
        individual = np.random.uniform(-0.5, 0.5, 16)  # 较小的随机值
        
        if bias_type == 'local':
            individual[0] = 0.8  # 本地处理
            individual[1:3] = -0.8  # 不卸载到RSU/UAV
        elif bias_type == 'rsu':
            individual[0] = -0.8  # 不本地处理
            individual[1] = 0.8   # 卸载到RSU
            individual[2] = -0.8  # 不卸载到UAV
            # 随机选择一个RSU
            rsu_idx = np.random.randint(0, 4)
            individual[3:7] = -0.5
            individual[3 + rsu_idx] = 0.8
        elif bias_type == 'uav':
            individual[0:2] = -0.8  # 不本地/RSU处理
            individual[2] = 0.8     # 卸载到UAV
            # 随机选择一个UAV
            uav_idx = np.random.randint(0, 2)
            individual[7:9] = -0.5
            individual[7 + uav_idx] = 0.8
            
        return individual
    
    def evaluate_population(self, state: np.ndarray):
        """评估种群中每个个体的适应度"""
        for i, individual in enumerate(self.population):
            # 使用缓存避免重复评估
            individual_key = tuple(np.round(individual, 3))  # 降低精度作为key
            
            if individual_key in self.evaluation_cache:
                fitness = self.evaluation_cache[individual_key]
                self.cache_hits += 1
            else:
                # 实际评估：模拟执行动作获得奖励
                fitness = self._evaluate_individual(individual, state)
                self.evaluation_cache[individual_key] = fitness
                
            self.fitness_scores[i] = fitness
            self.total_evaluations += 1
            
            # 更新最佳个体
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()
        
        # 记录平均适应度
        avg_fitness = np.mean(self.fitness_scores)
        self.fitness_history.append(avg_fitness)
        
    def _evaluate_individual(self, individual: np.ndarray, state: np.ndarray) -> float:
        """
        评估单个个体的适应度
        
        注意：这里使用简化的评估函数，基于动作的合理性给出估计值
        在实际应用中，应该运行完整的仿真来获得真实奖励
        """
        # 提取动作组成部分
        task_allocation = individual[0:3]
        rsu_selection = individual[3:7]
        uav_selection = individual[7:9]
        control_params = individual[9:16]
        
        # 基础适应度
        fitness = 0.0
        
        # 1. 任务分配合理性（softmax确保和为1）
        allocation_probs = self._softmax(task_allocation)
        allocation_entropy = -np.sum(allocation_probs * np.log(allocation_probs + 1e-8))
        fitness -= allocation_entropy * 0.1  # 鼓励明确的决策
        
        # 2. 根据状态评估决策质量
        local_load, rsu_loads, uav_loads = self._extract_loads_from_state(state)
        
        # 如果选择本地处理，检查本地负载
        if allocation_probs[0] > 0.5:  # 主要选择本地
            if local_load < 0.5:
                fitness += 0.5  # 本地负载低，好决策
            else:
                fitness -= 0.3  # 本地负载高，差决策
                
        # 如果选择RSU，检查RSU负载
        elif allocation_probs[1] > 0.5:  # 主要选择RSU
            rsu_probs = self._softmax(rsu_selection)
            selected_rsu = np.argmax(rsu_probs)
            if selected_rsu < len(rsu_loads):
                if rsu_loads[selected_rsu] < 0.5:
                    fitness += 0.5  # 选择了低负载RSU
                else:
                    fitness -= 0.2  # 选择了高负载RSU
                    
        # 如果选择UAV，检查UAV负载
        elif allocation_probs[2] > 0.5:  # 主要选择UAV
            uav_probs = self._softmax(uav_selection)
            selected_uav = np.argmax(uav_probs)
            if selected_uav < len(uav_loads):
                if uav_loads[selected_uav] < 0.5:
                    fitness += 0.4  # UAV负载低
                else:
                    fitness -= 0.3  # UAV负载高
        
        # 3. 控制参数合理性
        # 参数应该在合理范围内，不要太极端
        param_penalty = np.sum(np.abs(control_params)) / len(control_params)
        fitness -= param_penalty * 0.1
        
        # 4. 额外奖励：平衡负载
        if allocation_probs[1] > 0.3 and np.min(rsu_loads) < 0.3:
            fitness += 0.2  # 有效利用空闲RSU
            
        return fitness
    
    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def evolve_population(self):
        """进化种群：选择、交叉、变异"""
        # 1. 选择（精英保留 + 锦标赛选择）
        # 对适应度排序
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # 精英直接进入下一代
        new_population = []
        for i in range(self.elite_size):
            new_population.append(self.population[sorted_indices[i]].copy())
        
        # 2. 生成其余个体
        while len(new_population) < self.population_size:
            # 锦标赛选择父代
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交叉
            if np.random.rand() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # 变异
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            
            # 添加到新种群
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # 更新种群
        self.population = np.array(new_population[:self.population_size])
        
    def _tournament_selection(self, tournament_size=3):
        """锦标赛选择"""
        candidates = np.random.choice(self.population_size, tournament_size, replace=False)
        fitness_values = [self.fitness_scores[i] for i in candidates]
        winner_idx = candidates[np.argmax(fitness_values)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作 - 使用分段交叉"""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # 为不同部分使用不同的交叉策略
        # 任务分配部分（0:3）- 单点交叉
        if np.random.rand() < 0.5:
            crossover_point = np.random.randint(1, 3)
            offspring1[0:crossover_point] = parent2[0:crossover_point]
            offspring2[0:crossover_point] = parent1[0:crossover_point]
        
        # RSU选择部分（3:7）- 均匀交叉
        for i in range(3, 7):
            if np.random.rand() < 0.5:
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
        
        # UAV选择部分（7:9）- 整体交换
        if np.random.rand() < 0.5:
            offspring1[7:9], offspring2[7:9] = offspring2[7:9].copy(), offspring1[7:9].copy()
        
        # 控制参数部分（9:16）- 算术交叉
        alpha = np.random.rand()
        offspring1[9:16] = alpha * parent1[9:16] + (1 - alpha) * parent2[9:16]
        offspring2[9:16] = alpha * parent2[9:16] + (1 - alpha) * parent1[9:16]
        
        return offspring1, offspring2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                if i < 9:  # 离散决策部分
                    # 使用较大的变异
                    mutated[i] += np.random.normal(0, self.mutation_std * 2)
                else:  # 连续控制参数
                    # 使用较小的变异
                    mutated[i] += np.random.normal(0, self.mutation_std)
                
                # 确保在有效范围内
                mutated[i] = np.clip(mutated[i], -1, 1)
        
        return mutated


def parse_args():
    parser = argparse.ArgumentParser(description='GA算法 - 遗传算法优化')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='训练回合数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--num_vehicles', type=int, default=12,
                       help='车辆数量')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='每回合最大步数')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    # GA特定参数
    parser.add_argument('--population_size', type=int, default=20,
                       help='种群大小')
    parser.add_argument('--elite_ratio', type=float, default=0.2,
                       help='精英比例')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                       help='变异率')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                       help='交叉率')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print(f"GA (Genetic Algorithm) - 遗传算法优化")
    print("=" * 70)
    print(f"参数配置:")
    print(f"  - 回合数: {args.episodes}")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 车辆数: {args.num_vehicles}")
    print(f"  - 每回合步数: {args.max_steps}")
    print(f"  - 种群大小: {args.population_size}")
    print(f"  - 精英比例: {args.elite_ratio}")
    print(f"  - 变异率: {args.mutation_rate}")
    print(f"  - 交叉率: {args.crossover_rate}")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建环境
    env = SingleAgentTrainingEnvironment("TD3")
    algorithm = GABaseline(
        population_size=args.population_size,
        elite_ratio=args.elite_ratio,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )
    algorithm.update_environment(env)
    
    # 运行实验
    episode_rewards = []
    episode_delays = []
    episode_energies = []
    episode_completions = []
    
    start_time = time.time()
    
    for episode in range(1, args.episodes + 1):
        state = env.reset_environment()
        algorithm.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(args.max_steps):
            action = algorithm.select_action(state)
            actions_dict = env._build_actions_from_vector(action)
            next_state, reward, done, info = env.step(action, state, actions_dict)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # 记录指标
        metrics = info.get('system_metrics', {})
        episode_rewards.append(episode_reward / max(1, episode_steps))
        episode_delays.append(metrics.get('avg_task_delay', 0))
        episode_energies.append(metrics.get('total_energy_consumption', 0))
        episode_completions.append(metrics.get('task_completion_rate', 0))
        
        # 打印进度
        if episode % 20 == 0 or episode == args.episodes:
            cache_rate = algorithm.cache_hits / max(1, algorithm.total_evaluations)
            print(f"Episode {episode}/{args.episodes}: "
                  f"Reward={episode_rewards[-1]:.3f}, "
                  f"Delay={episode_delays[-1]:.3f}s, "
                  f"Energy={episode_energies[-1]:.1f}J, "
                  f"CacheRate={cache_rate:.1%}")
    
    execution_time = time.time() - start_time
    
    # 汇总结果
    stable_start = args.episodes // 2
    results = {
        'algorithm': 'GA',
        'algorithm_type': 'MetaHeuristic',
        'strategy': 'Genetic Algorithm',
        'num_episodes': args.episodes,
        'seed': args.seed,
        'num_vehicles': args.num_vehicles,
        'execution_time': execution_time,
        'episode_rewards': episode_rewards,
        'episode_delays': episode_delays,
        'episode_energies': episode_energies,
        'episode_completion_rates': episode_completions,
        'avg_delay': float(np.mean(episode_delays[stable_start:])),
        'std_delay': float(np.std(episode_delays[stable_start:])),
        'avg_energy': float(np.mean(episode_energies[stable_start:])),
        'std_energy': float(np.std(episode_energies[stable_start:])),
        'avg_completion_rate': float(np.mean(episode_completions[stable_start:])),
        'final_reward': float(np.mean(episode_rewards[stable_start:])),
        # GA特定指标
        'population_size': args.population_size,
        'total_evaluations': algorithm.total_evaluations,
        'cache_hits': algorithm.cache_hits,
        'cache_hit_rate': algorithm.cache_hits / max(1, algorithm.total_evaluations)
    }
    
    # 保存结果
    results_manager = ResultsManager()
    save_path = results_manager.save_results(
        algorithm='GA',
        results=results,
        algorithm_type='MetaHeuristic',
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"执行时间: {execution_time:.2f}秒")
    print(f"平均时延: {results['avg_delay']:.3f} ± {results['std_delay']:.3f}s")
    print(f"平均能耗: {results['avg_energy']:.1f} ± {results['std_energy']:.1f}J")
    print(f"平均完成率: {results['avg_completion_rate']:.1%}")
    print(f"最终奖励: {results['final_reward']:.3f}")
    print(f"总评估次数: {algorithm.total_evaluations}")
    print(f"缓存命中率: {results['cache_hit_rate']:.1%}")
    print(f"结果已保存至: {save_path}")
    print("=" * 70)
    
    # 显示详细结果
    results_manager.print_summary(results)
    
    return results


if __name__ == "__main__":
    main()











