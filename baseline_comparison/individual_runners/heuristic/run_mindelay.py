#!/usr/bin/env python3
"""
MinDelay算法 - 最小化时延优先策略
优先选择预期时延最小的节点进行任务卸载
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


class MinDelayBaseline(BaselineAlgorithm):
    """
    最小化时延优先算法
    
    策略：
    1. 评估每个可能的卸载决策的预期时延
    2. 选择时延最小的方案
    3. 考虑传输时延、处理时延和排队时延
    """
    
    def __init__(self, name="MinDelay"):
        super().__init__(name)
        self.time_window = 10  # 时延预测窗口
        
    def select_action(self, state):
        """
        选择动作 - 最小化时延
        
        动作向量（16维）：
        - [0:3]: 任务卸载决策（每个UAV）
        - [3:7]: 缓存决策（每个缓存位置）  
        - [7:9]: UAV迁移决策
        - [9:16]: 控制参数
        """
        action = np.zeros(16)
        
        # 任务卸载决策 - 选择时延最小的节点
        for uav_idx in range(3):
            # 获取当前UAV的任务信息
            uav_state_idx = 12 * 5 + 4 * 5 + uav_idx * 5
            if uav_state_idx + 5 <= len(state):
                cpu_util = state[uav_state_idx + 1] if uav_state_idx + 1 < len(state) else 0.5
                queue_len = state[uav_state_idx + 3] if uav_state_idx + 3 < len(state) else 5
                
                # 计算不同卸载选项的预期时延
                min_delay = float('inf')
                best_option = 0  # 默认本地处理
                
                # 选项1：本地处理（UAV）
                local_delay = self._estimate_local_delay(cpu_util, queue_len)
                if local_delay < min_delay:
                    min_delay = local_delay
                    best_option = -1.0  # 本地处理
                
                # 选项2：卸载到最近的RSU
                for rsu_idx in range(4):
                    rsu_state_idx = 12 * 5 + rsu_idx * 5
                    if rsu_state_idx + 5 <= len(state):
                        rsu_cpu = state[rsu_state_idx + 1] if rsu_state_idx + 1 < len(state) else 0.3
                        rsu_queue = state[rsu_state_idx + 3] if rsu_state_idx + 3 < len(state) else 3
                        
                        # 估算传输时延 + 处理时延
                        trans_delay = self._estimate_transmission_delay(uav_idx, rsu_idx)
                        proc_delay = self._estimate_processing_delay(rsu_cpu, rsu_queue)
                        total_delay = trans_delay + proc_delay
                        
                        if total_delay < min_delay:
                            min_delay = total_delay
                            best_option = (rsu_idx / 3.0) * 2 - 1  # 映射到[-1, 1]
                
                action[uav_idx] = np.clip(best_option, -1, 1)
        
        # 缓存决策 - 缓存最常访问且时延敏感的内容
        cache_priorities = self._calculate_cache_priorities(state)
        for i in range(4):
            action[3 + i] = cache_priorities[i] if i < len(cache_priorities) else 0.5
        
        # UAV迁移决策 - 向任务密集区域迁移以减少传输时延
        action[7:9] = self._calculate_migration_for_delay(state)
        
        # 控制参数优化
        action[9] = 0.3   # 功率控制 - 适中功率
        action[10] = 0.7  # 带宽分配 - 较高带宽减少传输时延
        action[11] = 0.8  # 计算资源 - 较高资源减少处理时延
        action[12] = 0.2  # 缓存大小 - 适中
        action[13] = 0.9  # 迁移阈值 - 积极迁移
        action[14] = 0.7  # 协作程度 - 较高协作
        action[15] = 0.8  # 预测窗口 - 较长窗口
        
        return action
    
    def _estimate_local_delay(self, cpu_util, queue_len):
        """估算本地处理时延"""
        # 考虑CPU利用率和队列长度
        base_delay = 0.1  # 基础处理时延
        queue_delay = queue_len * 0.02  # 排队时延
        cpu_delay = cpu_util * 0.3  # CPU负载影响
        return base_delay + queue_delay + cpu_delay
    
    def _estimate_transmission_delay(self, uav_idx, rsu_idx):
        """估算传输时延"""
        # 简化模型：基于距离和网络条件
        base_delay = 0.05  # 基础传输时延
        distance_factor = abs(uav_idx - rsu_idx) * 0.02
        return base_delay + distance_factor
    
    def _estimate_processing_delay(self, cpu_util, queue_len):
        """估算处理时延"""
        base_delay = 0.05  # RSU基础处理时延（比UAV快）
        queue_delay = queue_len * 0.01  # RSU排队时延
        cpu_delay = cpu_util * 0.2  # CPU负载影响
        return base_delay + queue_delay + cpu_delay
    
    def _calculate_cache_priorities(self, state):
        """计算缓存优先级 - 优先缓存时延敏感内容"""
        # 基于内容类型和访问频率
        priorities = []
        for i in range(4):
            # 时延敏感内容优先级更高
            priority = np.random.rand() * 0.3 + 0.7  # [0.7, 1.0]
            priorities.append(priority * 2 - 1)  # 映射到[-1, 1]
        return priorities
    
    def _calculate_migration_for_delay(self, state):
        """计算UAV迁移决策 - 减少传输时延"""
        # 向任务密集区域迁移
        migration = np.zeros(2)
        
        # 简化策略：向负载较高的区域迁移（任务多的地方）
        avg_load = np.mean(state[60:80]) if len(state) > 80 else 0.5
        if avg_load > 0.7:
            migration[0] = 0.5  # 向东迁移
            migration[1] = 0.3  # 向北迁移
        else:
            migration[0] = -0.3  # 向西迁移
            migration[1] = -0.2  # 向南迁移
            
        return migration


def parse_args():
    parser = argparse.ArgumentParser(description='MinDelay算法 - 最小化时延优先')
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
    print(f"MinDelay算法 - 最小化时延优先策略")
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
    algorithm = MinDelayBaseline()
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
        'algorithm': 'MinDelay',
        'algorithm_type': 'Heuristic',
        'strategy': 'Minimize Delay First',
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
        algorithm='MinDelay',
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
