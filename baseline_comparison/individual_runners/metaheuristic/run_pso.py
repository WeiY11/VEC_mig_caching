#!/usr/bin/env python3
"""
PSO (Particle Swarm Optimization) - 粒子群优化算法
使用群体智能优化任务迁移和缓存决策
"""

import numpy as np
import time
import argparse
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from train_single_agent import SingleAgentTrainingEnvironment
from baseline_comparison.individual_runners.common import ResultsManager
from baseline_comparison.improved_baseline_algorithms import BaselineAlgorithm


class Particle:
    """粒子类"""
    def __init__(self, dim: int, bounds: tuple = (-1, 1)):
        """
        初始化粒子
        
        Args:
            dim: 维度（动作空间大小）
            bounds: 位置边界
        """
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-0.1, 0.1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        self.fitness = float('-inf')
        

class PSOBaseline(BaselineAlgorithm):
    """
    粒子群优化算法 (Particle Swarm Optimization)
    
    策略：
    1. 初始化粒子群，每个粒子代表一个解（动作向量）
    2. 评估每个粒子的适应度
    3. 更新个体最优和全局最优
    4. 根据速度更新公式更新粒子位置
    5. 迭代优化找到最优解
    """
    
    def __init__(self, name="PSO", swarm_size=20, w=0.7, c1=1.5, c2=1.5, 
                 w_decay=0.99, v_max=0.2):
        super().__init__(name)
        
        # PSO参数
        self.swarm_size = swarm_size
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.w_decay = w_decay  # 惯性权重衰减
        self.v_max = v_max  # 最大速度
        
        # 粒子群
        self.swarm = None
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        # 优化历史
        self.iteration = 0
        self.fitness_history = []
        self.diversity_history = []
        
        # 评估缓存
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.total_evaluations = 0
        
    def reset(self):
        """重置算法状态"""
        super().reset()
        self.swarm = None
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.iteration = 0
        self.fitness_history = []
        self.diversity_history = []
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.total_evaluations = 0
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        选择动作 - 使用PSO优化
        
        对于每个状态，运行几次迭代来找到最优动作
        """
        # 初始化粒子群（如果是新的episode）
        if self.swarm is None:
            self.initialize_swarm()
        
        # 运行PSO迭代
        num_iterations = 5  # 每步的迭代次数
        for _ in range(num_iterations):
            # 评估所有粒子
            self.evaluate_swarm(state)
            
            # 更新粒子位置和速度
            self.update_swarm()
            
            # 衰减惯性权重
            self.w *= self.w_decay
            self.w = max(0.4, self.w)  # 最小惯性权重
            
            self.iteration += 1
        
        # 返回全局最优位置作为动作
        if self.global_best_position is not None:
            action = self.global_best_position.copy()
        else:
            # 如果没有找到最优位置，返回随机动作
            action = np.random.uniform(-1, 1, 16)
            
        return action
    
    def initialize_swarm(self):
        """初始化粒子群"""
        self.swarm = []
        
        for i in range(self.swarm_size):
            particle = Particle(dim=16)
            
            # 初始化策略：部分粒子有特定偏向
            if i < self.swarm_size // 4:
                # 偏向本地处理
                particle.position[0] = np.random.uniform(0.5, 1.0)
                particle.position[1:3] = np.random.uniform(-1.0, -0.5, 2)
            elif i < self.swarm_size // 2:
                # 偏向RSU处理
                particle.position[0] = np.random.uniform(-1.0, -0.5)
                particle.position[1] = np.random.uniform(0.5, 1.0)
                particle.position[2] = np.random.uniform(-1.0, -0.5)
            elif i < 3 * self.swarm_size // 4:
                # 偏向UAV处理
                particle.position[0:2] = np.random.uniform(-1.0, -0.5, 2)
                particle.position[2] = np.random.uniform(0.5, 1.0)
            # 其余粒子保持随机初始化
            
            self.swarm.append(particle)
            
    def evaluate_swarm(self, state: np.ndarray):
        """评估粒子群中每个粒子的适应度"""
        fitness_values = []
        
        for particle in self.swarm:
            # 使用缓存避免重复评估
            position_key = tuple(np.round(particle.position, 3))
            
            if position_key in self.evaluation_cache:
                fitness = self.evaluation_cache[position_key]
                self.cache_hits += 1
            else:
                # 实际评估
                fitness = self._evaluate_particle(particle.position, state)
                self.evaluation_cache[position_key] = fitness
                
            particle.fitness = fitness
            fitness_values.append(fitness)
            self.total_evaluations += 1
            
            # 更新个体最优
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # 更新全局最优
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        # 记录统计信息
        avg_fitness = np.mean(fitness_values)
        self.fitness_history.append(avg_fitness)
        
        # 计算多样性（粒子间的平均距离）
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
    def _evaluate_particle(self, position: np.ndarray, state: np.ndarray) -> float:
        """
        评估单个粒子的适应度
        
        使用启发式评估函数，考虑：
        1. 负载均衡
        2. 通信开销
        3. 能耗效率
        """
        # 提取动作组成部分
        task_allocation = position[0:3]
        rsu_selection = position[3:7]
        uav_selection = position[7:9]
        control_params = position[9:16]
        
        # 基础适应度
        fitness = 0.0
        
        # 提取系统负载信息
        local_load, rsu_loads, uav_loads = self._extract_loads_from_state(state)
        
        # 1. 负载均衡奖励
        # 使用softmax将动作转换为概率
        allocation_probs = self._softmax(task_allocation)
        
        # 根据负载情况评估决策
        if allocation_probs[0] > 0.5:  # 主要选择本地
            if local_load < 0.4:
                fitness += 0.6  # 本地负载低，好决策
            elif local_load > 0.7:
                fitness -= 0.4  # 本地负载高，差决策
            else:
                fitness += 0.2  # 中等负载，一般决策
                
        elif allocation_probs[1] > 0.5:  # 主要选择RSU
            rsu_probs = self._softmax(rsu_selection)
            # 计算加权平均负载
            weighted_load = np.sum(rsu_probs * rsu_loads)
            if weighted_load < 0.5:
                fitness += 0.5
            else:
                fitness -= 0.2 * (weighted_load - 0.5)
                
        elif allocation_probs[2] > 0.5:  # 主要选择UAV
            uav_probs = self._softmax(uav_selection)
            selected_uav = np.argmax(uav_probs)
            if selected_uav < len(uav_loads):
                if uav_loads[selected_uav] < 0.5:
                    fitness += 0.4
                else:
                    fitness -= 0.3
        
        # 2. 决策清晰度奖励（避免模糊决策）
        max_prob = np.max(allocation_probs)
        if max_prob > 0.7:
            fitness += 0.2  # 清晰的决策
        elif max_prob < 0.4:
            fitness -= 0.1  # 模糊的决策
        
        # 3. 控制参数合理性
        # 功率控制（index 9）- 应该根据任务类型调整
        if allocation_probs[0] > 0.5:  # 本地处理
            if control_params[0] < 0:  # 低功率
                fitness += 0.1
        else:  # 卸载处理
            if control_params[0] > 0:  # 高功率用于传输
                fitness += 0.1
                
        # 带宽分配（index 10）- 卸载时需要更多带宽
        if allocation_probs[1] > 0.5 or allocation_probs[2] > 0.5:
            if control_params[1] > 0.3:
                fitness += 0.1
                
        # 4. 系统负载均衡奖励
        all_loads = [local_load] + list(rsu_loads) + list(uav_loads)
        load_variance = np.var(all_loads)
        if load_variance < 0.1:
            fitness += 0.3  # 负载很均衡
        elif load_variance > 0.3:
            fitness -= 0.2  # 负载很不均衡
            
        return fitness
    
    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def update_swarm(self):
        """更新粒子群的速度和位置"""
        for particle in self.swarm:
            # 生成随机数
            r1 = np.random.rand(16)
            r2 = np.random.rand(16)
            
            # 更新速度
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social
            
            # 限制速度
            particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)
            
            # 更新位置
            particle.position = particle.position + particle.velocity
            
            # 确保位置在边界内
            particle.position = np.clip(particle.position, -1, 1)
            
            # 边界反弹策略
            for i in range(len(particle.position)):
                if abs(particle.position[i]) >= 0.99:
                    particle.velocity[i] *= -0.5  # 反弹并减速
                    
    def _calculate_diversity(self) -> float:
        """计算粒子群的多样性（平均距离）"""
        positions = np.array([p.position for p in self.swarm])
        
        # 计算质心
        centroid = np.mean(positions, axis=0)
        
        # 计算每个粒子到质心的距离
        distances = np.linalg.norm(positions - centroid, axis=1)
        
        # 返回平均距离
        return np.mean(distances)


def parse_args():
    parser = argparse.ArgumentParser(description='PSO算法 - 粒子群优化')
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
    # PSO特定参数
    parser.add_argument('--swarm_size', type=int, default=20,
                       help='粒子群大小')
    parser.add_argument('--w', type=float, default=0.7,
                       help='惯性权重')
    parser.add_argument('--c1', type=float, default=1.5,
                       help='个体学习因子')
    parser.add_argument('--c2', type=float, default=1.5,
                       help='社会学习因子')
    parser.add_argument('--w_decay', type=float, default=0.99,
                       help='惯性权重衰减率')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print(f"PSO (Particle Swarm Optimization) - 粒子群优化")
    print("=" * 70)
    print(f"参数配置:")
    print(f"  - 回合数: {args.episodes}")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 车辆数: {args.num_vehicles}")
    print(f"  - 每回合步数: {args.max_steps}")
    print(f"  - 粒子群大小: {args.swarm_size}")
    print(f"  - 惯性权重: {args.w}")
    print(f"  - 个体学习因子: {args.c1}")
    print(f"  - 社会学习因子: {args.c2}")
    print(f"  - 权重衰减率: {args.w_decay}")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建环境
    env = SingleAgentTrainingEnvironment("TD3")
    algorithm = PSOBaseline(
        swarm_size=args.swarm_size,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        w_decay=args.w_decay
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
            avg_diversity = np.mean(algorithm.diversity_history[-20:]) if algorithm.diversity_history else 0
            print(f"Episode {episode}/{args.episodes}: "
                  f"Reward={episode_rewards[-1]:.3f}, "
                  f"Delay={episode_delays[-1]:.3f}s, "
                  f"Energy={episode_energies[-1]:.1f}J, "
                  f"Diversity={avg_diversity:.3f}")
    
    execution_time = time.time() - start_time
    
    # 汇总结果
    stable_start = args.episodes // 2
    results = {
        'algorithm': 'PSO',
        'algorithm_type': 'MetaHeuristic',
        'strategy': 'Particle Swarm Optimization',
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
        # PSO特定指标
        'swarm_size': args.swarm_size,
        'total_evaluations': algorithm.total_evaluations,
        'cache_hits': algorithm.cache_hits,
        'cache_hit_rate': algorithm.cache_hits / max(1, algorithm.total_evaluations),
        'final_diversity': algorithm.diversity_history[-1] if algorithm.diversity_history else 0
    }
    
    # 保存结果
    results_manager = ResultsManager()
    save_path = results_manager.save_results(
        algorithm='PSO',
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
    print(f"最终多样性: {results['final_diversity']:.3f}")
    print(f"结果已保存至: {save_path}")
    print("=" * 70)
    
    # 显示详细结果
    results_manager.print_summary(results)
    
    return results


if __name__ == "__main__":
    main()









