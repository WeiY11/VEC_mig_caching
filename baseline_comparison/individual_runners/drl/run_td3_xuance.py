#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3算法独立运行脚本（基于xuance框架）

【使用方法】
基础训练:
  python baseline_comparison/individual_runners/drl/run_td3_xuance.py --episodes 200
  
指定随机种子:
  python baseline_comparison/individual_runners/drl/run_td3_xuance.py --episodes 200 --seed 42
  
改变车辆数:
  python baseline_comparison/individual_runners/drl/run_td3_xuance.py --episodes 200 --num-vehicles 16
  
完整参数:
  python baseline_comparison/individual_runners/drl/run_td3_xuance.py --episodes 200 --seed 42 --num-vehicles 12 --max-steps 100

【说明】
- 使用xuance框架的TD3实现
- 结果保存在 baseline_comparison/results/td3/
- 与现有系统配置保持一致
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 修复Windows编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 导入通用组件
from baseline_comparison.individual_runners.common import (
    VECGymEnv,
    create_xuance_config,
    ResultsManager
)

# 导入xuance深度集成模块
try:
    import xuance
    from baseline_comparison.individual_runners.common.xuance_integration import XuanceTrainer
    XUANCE_AVAILABLE = True
    print("✓ xuance框架已加载 (v1.3.2+)")
except ImportError as e:
    XUANCE_AVAILABLE = False
    print(f"⚠️  xuance未安装或版本过低: {e}")
    print("将使用fallback模式")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='TD3算法独立运行（xuance）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--episodes', type=int, default=200,
                        help='训练轮次')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num-vehicles', type=int, default=12,
                        help='车辆数量')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='每轮最大步数')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='结果保存目录（默认：baseline_comparison/results/td3/）')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细输出')
    
    return parser.parse_args()


def run_td3_fallback(env, args, xuance_config):
    """
    兼容模式：使用项目现有的TD3实现
    当xuance不可用时使用
    """
    print("\n使用兼容模式（项目自带TD3）...")
    
    from single_agent.td3 import TD3Environment
    
    # 创建TD3环境
    td3_env = TD3Environment()
    
    episode_rewards = []
    episode_delays = []
    episode_energies = []
    episode_completions = []
    
    start_time = time.time()
    
    for episode in range(1, args.episodes + 1):
        state = td3_env.reset_environment()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(args.max_steps):
            # 选择动作
            action = td3_env.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = td3_env.step(action, state)
            
            # 存储经验
            td3_env.remember(state, action, reward, next_state, done)
            
            # 更新
            if len(td3_env.memory) >= td3_env.batch_size:
                td3_env.update()
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # 记录指标
        metrics = info.get('system_metrics', {})
        episode_rewards.append(episode_reward)
        episode_delays.append(metrics.get('avg_task_delay', 0))
        episode_energies.append(metrics.get('total_energy_consumption', 0))
        episode_completions.append(metrics.get('task_completion_rate', 0))
        
        # 打印进度
        if episode % 20 == 0 or episode == args.episodes:
            print(f"Episode {episode}/{args.episodes}: "
                  f"Reward={episode_reward:.3f}, "
                  f"Delay={episode_delays[-1]:.3f}s, "
                  f"Energy={episode_energies[-1]:.1f}J, "
                  f"Completion={episode_completions[-1]:.2%}")
    
    execution_time = time.time() - start_time
    
    # 构建结果
    stable_start = args.episodes // 2
    results = {
        'algorithm': 'TD3',
        'algorithm_type': 'DRL',
        'implementation': 'fallback',
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
        'initial_reward': float(np.mean(episode_rewards[:10])),
        'final_reward': float(np.mean(episode_rewards[-10:])),
    }
    
    return results


def run_td3_xuance(args):
    """
    使用xuance框架运行TD3（深度集成版）
    """
    print("\n🚀 使用xuance框架深度集成训练TD3...")
    
    # 创建xuance训练器
    trainer = XuanceTrainer(
        algorithm='TD3',
        num_episodes=args.episodes,
        seed=args.seed,
        num_vehicles=args.num_vehicles,
        save_dir=args.save_dir
    )
    
    # 执行训练
    results = trainer.train()
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("TD3算法独立运行（xuance版）")
    print("="*80)
    print(f"训练轮次: {args.episodes}")
    print(f"随机种子: {args.seed}")
    print(f"车辆数量: {args.num_vehicles}")
    print(f"每轮步数: {args.max_steps}")
    print(f"xuance可用: {XUANCE_AVAILABLE}")
    print("="*80)
    
    # 创建配置（用于fallback模式）
    xuance_config = create_xuance_config(
        algorithm='TD3',
        num_episodes=args.episodes,
        seed=args.seed,
        num_vehicles=args.num_vehicles,
        max_steps=args.max_steps
    )
    
    # 运行训练
    if XUANCE_AVAILABLE:
        # 使用xuance深度集成
        results = run_td3_xuance(args)
    else:
        # 使用fallback模式
        env = VECGymEnv(xuance_config['env_config'])
        results = run_td3_fallback(env, args, xuance_config)
        env.close()
    
    # 保存结果
    manager = ResultsManager()
    save_path = manager.save_results(
        algorithm='TD3',
        results=results,
        algorithm_type='DRL',
        save_dir=args.save_dir
    )
    
    # 打印摘要
    manager.print_summary(results)
    
    print("\n" + "="*80)
    print(f"✓ TD3训练完成！结果已保存到: {save_path}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

