#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline对比环境测试脚本
验证所有算法是否可以正常运行
"""

import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def test_all():
    """运行所有测试"""
    print("="*60)
    print("Baseline对比实验环境测试")
    print("="*60)
    
    success_count = 0
    total_tests = 6
    
    # 测试1: Baseline算法加载
    print("\n[测试1] Baseline算法加载...")
    try:
        from baseline_comparison.baseline_algorithms import create_baseline_algorithm
        baselines = ['Random', 'Greedy', 'RoundRobin', 'LocalFirst', 'NearestNode']
        for name in baselines:
            algo = create_baseline_algorithm(name)
            print(f"    - {name}: 成功")
        success_count += 1
    except Exception as e:
        print(f"    失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: DRL算法导入
    print("\n[测试2] DRL算法导入...")
    try:
        from single_agent.td3 import TD3Environment
        from single_agent.ddpg import DDPGEnvironment
        from single_agent.sac import SACEnvironment
        from single_agent.ppo import PPOEnvironment
        from single_agent.dqn import DQNEnvironment
        print("    成功: 所有DRL算法导入正常")
        print(f"      - TD3, DDPG, SAC, PPO, DQN")
        success_count += 1
    except Exception as e:
        print(f"    失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 训练环境创建
    print("\n[测试3] 训练环境创建...")
    try:
        from train_single_agent import SingleAgentTrainingEnvironment
        env = SingleAgentTrainingEnvironment("TD3")
        print(f"    成功: TD3训练环境创建正常")
        success_count += 1
    except Exception as e:
        print(f"    失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 系统仿真器
    print("\n[测试4] 系统仿真器...")
    try:
        from evaluation.system_simulator import CompleteSystemSimulator
        simulator = CompleteSystemSimulator()
        print(f"    成功: 仿真器创建正常")
        success_count += 1
    except Exception as e:
        print(f"    失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试5: 奖励计算
    print("\n[测试5] 奖励计算...")
    try:
        from utils.unified_reward_calculator import calculate_unified_reward
        test_metrics = {
            'avg_task_delay': 0.15,
            'total_energy_consumption': 500,
            'dropped_tasks': 2,
            'task_completion_rate': 0.95
        }
        reward = calculate_unified_reward(test_metrics)
        print(f"    成功: 奖励={reward:.3f}")
        success_count += 1
    except Exception as e:
        print(f"    失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试6: Baseline动作生成
    print("\n[测试6] Baseline动作生成...")
    try:
        from baseline_comparison.baseline_algorithms import create_baseline_algorithm
        import numpy as np
        
        algo = create_baseline_algorithm('Random')
        test_state = np.random.rand(130)
        actions = algo.select_action(test_state, {})
        
        print(f"    成功: 动作生成正常")
        print(f"      vehicle_agent: {len(actions['vehicle_agent'])}维")
        print(f"      rsu_agent: {len(actions['rsu_agent'])}维")
        print(f"      uav_agent: {len(actions['uav_agent'])}维")
        success_count += 1
    except Exception as e:
        print(f"    失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 汇总结果
    print("\n" + "="*60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    print("="*60)
    
    if success_count == total_tests:
        print("\n所有测试通过！可以开始运行Baseline对比实验")
        print("\n建议命令:")
        print("  快速测试: python run_baseline_comparison.py --episodes 50 --quick")
        print("  单独算法: python run_baseline_comparison.py --algorithm TD3 --episodes 100")
        print("  完整对比: python run_baseline_comparison.py --episodes 200")
    else:
        print(f"\n警告: {total_tests - success_count} 个测试失败")
        print("请检查环境配置")
    
    print("="*60)
    
    return success_count == total_tests


if __name__ == "__main__":
    test_all()


