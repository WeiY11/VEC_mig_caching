#!/usr/bin/env python3
"""
MinEnergy算法 - 最小化能耗优先策略
优先选择预期能耗最小的节点进行任务卸载
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


class MinEnergyBaseline(BaselineAlgorithm):
    """
    最小化能耗优先算法
    
    策略：
    1. 评估每个可能的卸载决策的预期能耗
    2. 选择能耗最小的方案
    3. 考虑传输能耗、计算能耗和空闲能耗
    """
    
    def __init__(self, name="MinEnergy"):
        super().__init__(name)
        self.power_window = 10  # 能耗预测窗口
        
    def select_action(self, state):
        """
        选择动作 - 最小化能耗
        
        动作向量（16维）：
        - [0:3]: 任务卸载决策（每个UAV）
        - [3:7]: 缓存决策（每个缓存位置）  
        - [7:9]: UAV迁移决策
        - [9:16]: 控制参数
        """
        action = np.zeros(16)
        
        # 任务卸载决策 - 选择能耗最小的节点
        for uav_idx in range(3):
            # 获取当前UAV的状态信息
            uav_state_idx = 12 * 5 + 4 * 5 + uav_idx * 5
            if uav_state_idx + 5 <= len(state):
                cpu_util = state[uav_state_idx + 1] if uav_state_idx + 1 < len(state) else 0.5
                
                # 计算不同卸载选项的预期能耗
                min_energy = float('inf')
                best_option = 0  # 默认本地处理
                
                # 选项1：本地处理（UAV）- 通常能耗较高
                local_energy = self._estimate_local_energy(cpu_util, is_uav=True)
                if local_energy < min_energy:
                    min_energy = local_energy
                    best_option = -1.0  # 本地处理
                
                # 选项2：卸载到RSU - 传输能耗 + RSU处理能耗
                for rsu_idx in range(4):
                    rsu_state_idx = 12 * 5 + rsu_idx * 5
                    if rsu_state_idx + 5 <= len(state):
                        rsu_cpu = state[rsu_state_idx + 1] if rsu_state_idx + 1 < len(state) else 0.3
                        
                        # 估算传输能耗 + 处理能耗
                        trans_energy = self._estimate_transmission_energy(uav_idx, rsu_idx)
                        proc_energy = self._estimate_processing_energy(rsu_cpu, is_rsu=True)
                        total_energy = trans_energy + proc_energy
                        
                        if total_energy < min_energy:
                            min_energy = total_energy
                            best_option = (rsu_idx / 3.0) * 2 - 1  # 映射到[-1, 1]
                
                action[uav_idx] = np.clip(best_option, -1, 1)
        
        # 缓存决策 - 缓存能减少重复传输能耗的内容
        cache_priorities = self._calculate_cache_priorities_for_energy(state)
        for i in range(4):
            action[3 + i] = cache_priorities[i] if i < len(cache_priorities) else 0.5
        
        # UAV迁移决策 - 尽量减少迁移以节省能耗
        action[7:9] = self._calculate_migration_for_energy(state)
        
        # 控制参数优化 - 优先考虑能耗
        action[9] = 0.1   # 功率控制 - 低功率
        action[10] = 0.3  # 带宽分配 - 适中带宽
        action[11] = 0.4  # 计算资源 - 适中资源
        action[12] = 0.5  # 缓存大小 - 较大缓存减少重传
        action[13] = 0.2  # 迁移阈值 - 保守迁移
        action[14] = 0.5  # 协作程度 - 适中协作
        action[15] = 0.4  # 预测窗口 - 较短窗口
        
        return action
    
    def _estimate_local_energy(self, cpu_util, is_uav=True):
        """估算本地处理能耗"""
        # UAV能耗通常高于RSU
        if is_uav:
            base_energy = 50.0  # UAV基础处理能耗
            cpu_energy = cpu_util * 100.0  # CPU负载影响
        else:
            base_energy = 30.0  # RSU基础处理能耗
            cpu_energy = cpu_util * 60.0
        return base_energy + cpu_energy
    
    def _estimate_transmission_energy(self, uav_idx, rsu_idx):
        """估算传输能耗"""
        # 基于距离和传输功率
        base_energy = 20.0  # 基础传输能耗
        distance_factor = abs(uav_idx - rsu_idx) * 10.0  # 距离影响
        return base_energy + distance_factor
    
    def _estimate_processing_energy(self, cpu_util, is_rsu=True):
        """估算处理能耗"""
        if is_rsu:
            base_energy = 30.0  # RSU基础处理能耗
            cpu_energy = cpu_util * 60.0
        else:
            base_energy = 50.0  # UAV基础处理能耗
            cpu_energy = cpu_util * 100.0
        return base_energy + cpu_energy
    
    def _calculate_cache_priorities_for_energy(self, state):
        """计算缓存优先级 - 优先缓存频繁访问的大文件"""
        # 缓存可以减少重复传输的能耗
        priorities = []
        for i in range(4):
            # 大文件和高频访问内容优先级更高
            priority = np.random.rand() * 0.4 + 0.6  # [0.6, 1.0]
            priorities.append(priority * 2 - 1)  # 映射到[-1, 1]
        return priorities
    
    def _calculate_migration_for_energy(self, state):
        """计算UAV迁移决策 - 最小化迁移能耗"""
        # 保守迁移策略，除非必要否则不迁移
        migration = np.zeros(2)
        
        # 只有在极端情况下才迁移
        avg_load = np.mean(state[60:80]) if len(state) > 80 else 0.5
        if avg_load > 0.85:  # 非常高的负载
            migration[0] = 0.2  # 小幅迁移
            migration[1] = 0.1
        else:
            migration[0] = 0.0  # 不迁移
            migration[1] = 0.0
            
        return migration


def parse_args():
    parser = argparse.ArgumentParser(description='MinEnergy算法 - 最小化能耗优先')
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
    print(f"MinEnergy算法 - 最小化能耗优先策略")
    print("=" * 70)
    print(f"参数配置:")
    print(f"  - 回合数: {args.episodes}")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 车辆数: {args.num_vehicles}")
    print(f"  - 每回合步数: {args.max_steps}")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建环境
    env = SingleAgentTrainingEnvironment("TD3")
    algorithm = MinEnergyBaseline()
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
        'algorithm': 'MinEnergy',
        'algorithm_type': 'Heuristic',
        'strategy': 'Minimize Energy First',
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
        'final_reward': float(np.mean(episode_rewards[stable_start:]))
    }
    
    # 保存结果
    results_manager = ResultsManager()
    save_path = results_manager.save_results(
        algorithm='MinEnergy',
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
