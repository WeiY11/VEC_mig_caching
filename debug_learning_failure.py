"""
调试脚本：诊断智能体学习失败的根本原因

使用方法：
python debug_learning_failure.py

这个脚本会运行5个episode，详细打印：
1. 环境实际产生的指标
2. 奖励函数的输入和输出
3. 训练更新的状态
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from train_single_agent import SingleAgentTrainingEnvironment
from config import config

def debug_baseline_policy():
    """测试一个固定策略，看环境是否可解"""
    print("=" * 80)
    print("基准测试：固定策略（全部本地处理）")
    print("=" * 80)
    
    # 创建训练环境
    env = SingleAgentTrainingEnvironment(
        algorithm="OPTIMIZED_TD3",
        override_scenario={"num_vehicles": 4, "num_rsus": 2, "num_uavs": 1}  # 简化拓扑
    )
    
    action_dim = env.agent_env.action_dim
    
    for ep in range(5):
        # 重置环境
        node_states, system_metrics, resource_state = env.simulator.reset()
        state = env.agent_env.get_state_vector(node_states, system_metrics, resource_state)
        
        episode_reward = 0
        step_count = 0
        
        # 固定策略：全部本地处理
        action = np.zeros(action_dim, dtype=np.float32)
        action[0] = 1.0  # 本地偏好 = 1.0
        
        for step in range(100):  # 最多100步
            # 执行动作
            metrics_before = env.strategy_coordinator.get_latest_metrics()
            node_states, system_metrics, resource_state = env.simulator.step_with_actions(
                env.agent_env.decompose_action(action)
            )
            
            # 计算奖励
            cache_metrics, migration_metrics = env.strategy_coordinator.get_latest_metrics()
            reward = env.agent_env.calculate_reward(system_metrics, cache_metrics, migration_metrics)
            episode_reward += reward
            step_count += 1
            
            # 打印前3步和最后1步的详细信息
            if step < 3 or step == 99:
                print(f"\n  Episode {ep+1}, Step {step+1}:")
                print(f"    Delay: {system_metrics.get('avg_task_delay', 'N/A'):.4f}s")
                print(f"    Energy: {system_metrics.get('total_energy_consumption', 'N/A'):.2f}J")
                print(f"    Completion: {system_metrics.get('task_completion_rate', 0)*100:.1f}%")
                print(f"    Data Loss: {system_metrics.get('data_loss_ratio_bytes', 0)*100:.1f}%")
                print(f"    Cache Hit: {system_metrics.get('cache_hit_rate', 0)*100:.1f}%")
                print(f"    Step Reward: {reward:.4f}")
            
            # 检查是否结束
            if system_metrics.get('done', False):
                break
        
        avg_step_reward = episode_reward / max(step_count, 1)
        print(f"\n  Episode {ep+1} 总结:")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Avg Step Reward: {avg_step_reward:.4f}")
        print(f"    Steps: {step_count}")
    
    print("\n" + "=" * 80)
    print("如果avg_step_reward在 -0.1 到 -2.0 之间，说明环境可解")
    print("如果avg_step_reward < -5.0，说明环境配置有严重问题")
    print("=" * 80)

def debug_reward_calculation():
    """详细检查奖励计算过程"""
    print("\n" + "=" * 80)
    print("奖励计算测试：手动构造指标，验证奖励函数")
    print("=" * 80)
    
    from utils.unified_reward_calculator import calculate_unified_reward
    
    # 测试案例1：理想状态
    test_metrics_ideal = {
        'avg_task_delay': 0.3,  # 低于目标0.4
        'total_energy_consumption': 3000.0,  # 低于目标3500
        'task_completion_rate': 0.95,
        'data_loss_ratio_bytes': 0.01,
        'cache_hit_rate': 0.80,
    }
    
    # 测试案例2：中等状态
    test_metrics_medium = {
        'avg_task_delay': 0.5,  #  略高于目标
        'total_energy_consumption': 4000.0,  # 略高于目标
        'task_completion_rate': 0.90,
        'data_loss_ratio_bytes': 0.05,
        'cache_hit_rate': 0.60,
    }
    
    # 测试案例3：糟糕状态
    test_metrics_bad = {
        'avg_task_delay': 1.5,
        'total_energy_consumption': 8000.0,
        'task_completion_rate': 0.70,
        'data_loss_ratio_bytes': 0.20,
        'cache_hit_rate': 0.20,
    }
    
    for name, metrics in [("理想", test_metrics_ideal), ("中等", test_metrics_medium), ("糟糕", test_metrics_bad)]:
        reward = calculate_unified_reward(metrics, algorithm="general")
        print(f"\n  {name}状态:")
        print(f"    Delay: {metrics['avg_task_delay']:.2f}s, Energy: {metrics['total_energy_consumption']:.0f}J")
        print(f"    Completion: {metrics['task_completion_rate']*100:.0f}%, Loss: {metrics['data_loss_ratio_bytes']*100:.1f}%")
        print(f"    --> Reward: {reward:.4f}")
    
    print("\n" + "=" * 80)
    print("预期：理想 > -0.1, 中等 -0.1~-0.3, 糟糕 -0.5~-1.0")
    print("=" * 80)

if __name__ == "__main__":
    print("\n\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "智能体学习失败诊断工具" + " " * 20 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")
    
    debug_reward_calculation()
    print("\n\n")
    debug_baseline_policy()
    
    print("\n\n诊断完成。请检查上述输出。\n")
