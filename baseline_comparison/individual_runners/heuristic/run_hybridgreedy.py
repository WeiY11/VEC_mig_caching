#!/usr/bin/env python3
"""
HybridGreedy算法 - 混合贪婪策略
同时考虑时延和能耗，使用加权优化
"""

import numpy as np
import time
import argparse
from pathlib import Path
import sys

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from train_single_agent import SingleAgentTrainingEnvironment
from baseline_comparison.individual_runners.common import ResultsManager
from baseline_comparison.improved_baseline_algorithms import BaselineAlgorithm


class HybridGreedyBaseline(BaselineAlgorithm):
    """
    混合贪婪算法
    
    策略：
    1. 同时考虑时延和能耗（使用系统配置的权重）
    2. 计算每个决策的综合成本
    3. 选择综合成本最小的方案
    4. 动态调整权重以适应不同场景
    """
    
    def __init__(self, name="HybridGreedy"):
        super().__init__(name)
        # 使用系统配置的权重
        self.delay_weight = 2.0  # 与奖励函数一致
        self.energy_weight = 1.2  # 与奖励函数一致
        self.adaptive = True  # 是否自适应调整权重
        
    def select_action(self, state):
        """
        选择动作 - 混合优化
        
        动作向量（16维）：
        - [0:3]: 任务卸载决策（每个UAV）
        - [3:7]: 缓存决策（每个缓存位置）  
        - [7:9]: UAV迁移决策
        - [9:16]: 控制参数
        """
        action = np.zeros(16)
        
        # 动态调整权重（如果启用）
        if self.adaptive:
            current_weights = self._adjust_weights_based_on_state(state)
        else:
            current_weights = (self.delay_weight, self.energy_weight)
        
        # 任务卸载决策 - 最小化加权成本
        for uav_idx in range(3):
            # 获取当前UAV的状态
            uav_state_idx = 12 * 5 + 4 * 5 + uav_idx * 5
            if uav_state_idx + 5 <= len(state):
                cpu_util = state[uav_state_idx + 1] if uav_state_idx + 1 < len(state) else 0.5
                queue_len = state[uav_state_idx + 3] if uav_state_idx + 3 < len(state) else 5
                
                # 计算不同卸载选项的综合成本
                min_cost = float('inf')
                best_option = 0  # 默认本地处理
                
                # 选项1：本地处理（UAV）
                local_delay = self._estimate_local_delay(cpu_util, queue_len)
                local_energy = self._estimate_local_energy(cpu_util, is_uav=True)
                local_cost = current_weights[0] * local_delay + current_weights[1] * local_energy / 600
                
                if local_cost < min_cost:
                    min_cost = local_cost
                    best_option = -1.0  # 本地处理
                
                # 选项2：卸载到RSU
                for rsu_idx in range(4):
                    rsu_state_idx = 12 * 5 + rsu_idx * 5
                    if rsu_state_idx + 5 <= len(state):
                        rsu_cpu = state[rsu_state_idx + 1] if rsu_state_idx + 1 < len(state) else 0.3
                        rsu_queue = state[rsu_state_idx + 3] if rsu_state_idx + 3 < len(state) else 3
                        
                        # 计算时延和能耗
                        trans_delay = self._estimate_transmission_delay(uav_idx, rsu_idx)
                        proc_delay = self._estimate_processing_delay(rsu_cpu, rsu_queue)
                        total_delay = trans_delay + proc_delay
                        
                        trans_energy = self._estimate_transmission_energy(uav_idx, rsu_idx)
                        proc_energy = self._estimate_processing_energy(rsu_cpu, is_rsu=True)
                        total_energy = trans_energy + proc_energy
                        
                        # 计算综合成本
                        total_cost = current_weights[0] * total_delay + current_weights[1] * total_energy / 600
                        
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_option = (rsu_idx / 3.0) * 2 - 1  # 映射到[-1, 1]
                
                action[uav_idx] = np.clip(best_option, -1, 1)
        
        # 缓存决策 - 平衡时延和能耗收益
        cache_priorities = self._calculate_hybrid_cache_priorities(state)
        for i in range(4):
            action[3 + i] = cache_priorities[i] if i < len(cache_priorities) else 0.5
        
        # UAV迁移决策 - 综合考虑迁移成本和收益
        action[7:9] = self._calculate_hybrid_migration(state, current_weights)
        
        # 控制参数优化 - 平衡性能和能耗
        action[9] = 0.4   # 功率控制 - 中等功率
        action[10] = 0.5  # 带宽分配 - 均衡分配
        action[11] = 0.6  # 计算资源 - 较高资源
        action[12] = 0.4  # 缓存大小 - 适中
        action[13] = 0.5  # 迁移阈值 - 适中
        action[14] = 0.6  # 协作程度 - 较高协作
        action[15] = 0.6  # 预测窗口 - 适中窗口
        
        return action
    
    def _adjust_weights_based_on_state(self, state):
        """根据系统状态动态调整权重"""
        # 获取系统平均负载
        avg_cpu = np.mean(state[61:81:5]) if len(state) > 81 else 0.5
        avg_queue = np.mean(state[63:83:5]) if len(state) > 83 else 5.0
        
        # 高负载时更重视时延，低负载时更重视能耗
        if avg_cpu > 0.8 or avg_queue > 10:
            # 高负载：增加时延权重
            delay_w = self.delay_weight * 1.2
            energy_w = self.energy_weight * 0.8
        elif avg_cpu < 0.3 and avg_queue < 3:
            # 低负载：增加能耗权重
            delay_w = self.delay_weight * 0.8
            energy_w = self.energy_weight * 1.2
        else:
            # 正常负载：使用默认权重
            delay_w = self.delay_weight
            energy_w = self.energy_weight
            
        return (delay_w, energy_w)
    
    def _estimate_local_delay(self, cpu_util, queue_len):
        """估算本地处理时延"""
        base_delay = 0.1
        queue_delay = queue_len * 0.02
        cpu_delay = cpu_util * 0.3
        return base_delay + queue_delay + cpu_delay
    
    def _estimate_transmission_delay(self, uav_idx, rsu_idx):
        """估算传输时延"""
        base_delay = 0.05
        distance_factor = abs(uav_idx - rsu_idx) * 0.02
        return base_delay + distance_factor
    
    def _estimate_processing_delay(self, cpu_util, queue_len):
        """估算处理时延"""
        base_delay = 0.05
        queue_delay = queue_len * 0.01
        cpu_delay = cpu_util * 0.2
        return base_delay + queue_delay + cpu_delay
    
    def _estimate_local_energy(self, cpu_util, is_uav=True):
        """估算本地处理能耗"""
        if is_uav:
            base_energy = 50.0
            cpu_energy = cpu_util * 100.0
        else:
            base_energy = 30.0
            cpu_energy = cpu_util * 60.0
        return base_energy + cpu_energy
    
    def _estimate_transmission_energy(self, uav_idx, rsu_idx):
        """估算传输能耗"""
        base_energy = 20.0
        distance_factor = abs(uav_idx - rsu_idx) * 10.0
        return base_energy + distance_factor
    
    def _estimate_processing_energy(self, cpu_util, is_rsu=True):
        """估算处理能耗"""
        if is_rsu:
            base_energy = 30.0
            cpu_energy = cpu_util * 60.0
        else:
            base_energy = 50.0
            cpu_energy = cpu_util * 100.0
        return base_energy + cpu_energy
    
    def _calculate_hybrid_cache_priorities(self, state):
        """计算混合缓存优先级"""
        priorities = []
        for i in range(4):
            # 综合考虑访问频率和内容大小
            # 高频小文件和低频大文件都有价值
            priority = np.random.rand() * 0.4 + 0.5  # [0.5, 0.9]
            priorities.append(priority * 2 - 1)  # 映射到[-1, 1]
        return priorities
    
    def _calculate_hybrid_migration(self, state, weights):
        """计算混合迁移策略"""
        migration = np.zeros(2)
        
        # 获取负载信息
        avg_load = np.mean(state[60:80]) if len(state) > 80 else 0.5
        
        # 综合考虑迁移成本（能耗）和收益（时延改善）
        migration_cost = 50.0  # 迁移能耗成本
        
        # 高负载时迁移收益大
        if avg_load > 0.75:
            expected_delay_improvement = 0.2  # 预期时延改善
            migration_benefit = weights[0] * expected_delay_improvement
            
            if migration_benefit > weights[1] * migration_cost / 600:
                # 收益大于成本，执行迁移
                migration[0] = 0.4  # 向负载低的方向
                migration[1] = 0.2
        elif avg_load < 0.3:
            # 低负载时避免迁移节省能耗
            migration[0] = 0.0
            migration[1] = 0.0
        else:
            # 中等负载时小幅迁移
            migration[0] = 0.2
            migration[1] = 0.1
            
        return migration


def parse_args():
    parser = argparse.ArgumentParser(description='HybridGreedy算法 - 混合贪婪策略')
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
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print(f"HybridGreedy算法 - 混合贪婪策略")
    print("=" * 70)
    print(f"参数配置:")
    print(f"  - 回合数: {args.episodes}")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 车辆数: {args.num_vehicles}")
    print(f"  - 每回合步数: {args.max_steps}")
    print(f"  - 权重: 时延={2.0}, 能耗={1.2} (与系统奖励一致)")
    print(f"  - 自适应权重: 启用")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建环境
    env = SingleAgentTrainingEnvironment("TD3")
    algorithm = HybridGreedyBaseline()
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
            print(f"Episode {episode}/{args.episodes}: "
                  f"Reward={episode_rewards[-1]:.3f}, "
                  f"Delay={episode_delays[-1]:.3f}s, "
                  f"Energy={episode_energies[-1]:.1f}J")
    
    execution_time = time.time() - start_time
    
    # 汇总结果
    stable_start = args.episodes // 2
    results = {
        'algorithm': 'HybridGreedy',
        'algorithm_type': 'Heuristic',
        'strategy': 'Hybrid Optimization (Delay + Energy)',
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
        'weights': {'delay': 2.0, 'energy': 1.2},
        'adaptive': True
    }
    
    # 保存结果
    results_manager = ResultsManager()
    save_path = results_manager.save_results(
        algorithm='HybridGreedy',
        results=results,
        algorithm_type='Heuristic',
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"执行时间: {execution_time:.2f}秒")
    print(f"平均时延: {results['avg_delay']:.3f} ± {results['std_delay']:.3f}s")
    print(f"平均能耗: {results['avg_energy']:.1f} ± {results['std_energy']:.1f}J")
    print(f"平均完成率: {results['avg_completion_rate']:.1%}")
    print(f"最终奖励: {results['final_reward']:.3f}")
    print(f"结果已保存至: {save_path}")
    print("=" * 70)
    
    # 显示详细结果
    results_manager.print_summary(results)
    
    return results


if __name__ == "__main__":
    main()
