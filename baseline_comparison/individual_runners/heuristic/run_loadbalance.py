#!/usr/bin/env python3
"""
LoadBalance算法 - 负载均衡策略
保持各节点负载均衡，避免单点过载
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


class LoadBalanceBaseline(BaselineAlgorithm):
    """
    负载均衡算法
    
    策略：
    1. 监控各节点的负载情况（CPU、内存、队列长度）
    2. 将任务分配到负载最低的节点
    3. 定期重新平衡负载
    """
    
    def __init__(self, name="LoadBalance"):
        super().__init__(name)
        self.load_history = {}  # 记录历史负载
        self.balance_threshold = 0.3  # 负载差异阈值
        
    def select_action(self, state):
        """
        选择动作 - 负载均衡
        
        动作向量（16维）：
        - [0:3]: 任务卸载决策（每个UAV）
        - [3:7]: 缓存决策（每个缓存位置）  
        - [7:9]: UAV迁移决策
        - [9:16]: 控制参数
        """
        action = np.zeros(16)
        
        # 计算各节点的综合负载
        node_loads = self._calculate_node_loads(state)
        
        # 任务卸载决策 - 选择负载最低的节点
        for uav_idx in range(3):
            # 获取当前UAV的负载
            uav_state_idx = 12 * 5 + 4 * 5 + uav_idx * 5
            if uav_state_idx + 5 <= len(state):
                uav_load = self._get_node_load(state, uav_state_idx, is_uav=True)
                
                # 找到负载最低的目标节点
                min_load = uav_load  # 当前UAV负载
                best_option = -1.0  # 默认本地处理
                
                # 检查各RSU的负载
                for rsu_idx in range(4):
                    rsu_state_idx = 12 * 5 + rsu_idx * 5
                    if rsu_state_idx + 5 <= len(state):
                        rsu_load = self._get_node_load(state, rsu_state_idx, is_uav=False)
                        
                        # 选择负载更低的节点
                        if rsu_load < min_load - self.balance_threshold:
                            min_load = rsu_load
                            best_option = (rsu_idx / 3.0) * 2 - 1  # 映射到[-1, 1]
                
                action[uav_idx] = np.clip(best_option, -1, 1)
        
        # 缓存决策 - 在负载较低的节点缓存热点内容
        cache_priorities = self._calculate_cache_for_balance(state, node_loads)
        for i in range(4):
            action[3 + i] = cache_priorities[i] if i < len(cache_priorities) else 0.5
        
        # UAV迁移决策 - 从高负载区域向低负载区域迁移
        action[7:9] = self._calculate_migration_for_balance(state, node_loads)
        
        # 控制参数优化 - 支持负载均衡
        action[9] = 0.5   # 功率控制 - 适中
        action[10] = 0.6  # 带宽分配 - 动态分配
        action[11] = 0.7  # 计算资源 - 较高资源支持均衡
        action[12] = 0.6  # 缓存大小 - 较大缓存
        action[13] = 0.6  # 迁移阈值 - 适中迁移
        action[14] = 0.8  # 协作程度 - 高协作支持均衡
        action[15] = 0.6  # 预测窗口 - 适中窗口
        
        return action
    
    def _calculate_node_loads(self, state):
        """计算各节点的综合负载"""
        loads = {}
        
        # 计算RSU负载
        for rsu_idx in range(4):
            state_idx = 12 * 5 + rsu_idx * 5
            if state_idx + 5 <= len(state):
                load = self._get_node_load(state, state_idx, is_uav=False)
                loads[f'rsu_{rsu_idx}'] = load
        
        # 计算UAV负载
        for uav_idx in range(2):  # 只有2个UAV
            state_idx = 12 * 5 + 4 * 5 + uav_idx * 5
            if state_idx + 5 <= len(state):
                load = self._get_node_load(state, state_idx, is_uav=True)
                loads[f'uav_{uav_idx}'] = load
                
        return loads
    
    def _get_node_load(self, state, state_idx, is_uav):
        """获取单个节点的综合负载"""
        if state_idx + 5 > len(state):
            return 0.5  # 默认负载
            
        # 提取负载指标
        cpu_util = state[state_idx + 1] if state_idx + 1 < len(state) else 0.5
        mem_util = state[state_idx + 2] if state_idx + 2 < len(state) else 0.5
        queue_len = state[state_idx + 3] if state_idx + 3 < len(state) else 5
        
        # 归一化队列长度
        max_queue = 20 if is_uav else 15
        norm_queue = min(queue_len / max_queue, 1.0)
        
        # 综合负载计算（CPU权重更高）
        load = cpu_util * 0.5 + mem_util * 0.3 + norm_queue * 0.2
        
        return load
    
    def _calculate_cache_for_balance(self, state, node_loads):
        """计算缓存策略以支持负载均衡"""
        priorities = []
        
        # 在负载较低的节点优先缓存
        for i in range(4):
            rsu_load = node_loads.get(f'rsu_{i}', 0.5)
            # 负载越低，缓存优先级越高
            priority = (1.0 - rsu_load) * 0.8
            priorities.append(priority * 2 - 1)  # 映射到[-1, 1]
            
        return priorities
    
    def _calculate_migration_for_balance(self, state, node_loads):
        """计算UAV迁移以平衡负载"""
        migration = np.zeros(2)
        
        # 获取各区域的平均负载
        east_load = np.mean([node_loads.get(f'rsu_{i}', 0.5) for i in [0, 1]])
        west_load = np.mean([node_loads.get(f'rsu_{i}', 0.5) for i in [2, 3]])
        
        # 从高负载向低负载区域迁移
        if east_load > west_load + self.balance_threshold:
            migration[0] = -0.5  # 向西迁移
        elif west_load > east_load + self.balance_threshold:
            migration[0] = 0.5   # 向东迁移
            
        # 简单的南北迁移策略
        avg_load = np.mean(list(node_loads.values()))
        if avg_load > 0.7:
            migration[1] = np.random.choice([-0.3, 0.3])  # 随机南北迁移
            
        return migration
    
    def reset(self):
        """重置算法状态"""
        super().reset()
        self.load_history.clear()


def parse_args():
    parser = argparse.ArgumentParser(description='LoadBalance算法 - 负载均衡策略')
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
    print(f"LoadBalance算法 - 负载均衡策略")
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
    algorithm = LoadBalanceBaseline()
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
        'algorithm': 'LoadBalance',
        'algorithm_type': 'Heuristic',
        'strategy': 'Load Balancing',
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
        algorithm='LoadBalance',
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
