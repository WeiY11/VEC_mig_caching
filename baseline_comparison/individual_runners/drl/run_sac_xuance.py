#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAC算法独立运行脚本（基于xuance框架）

【使用方法】
  python baseline_comparison/individual_runners/drl/run_sac_xuance.py --episodes 200 --seed 42
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from baseline_comparison.individual_runners.common import (
    VECGymEnv, create_xuance_config, ResultsManager
)

try:
    import xuance
    XUANCE_AVAILABLE = True
except ImportError:
    XUANCE_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='SAC算法独立运行（xuance）')
    parser.add_argument('--episodes', type=int, default=200, help='训练轮次')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-vehicles', type=int, default=12, help='车辆数量')
    parser.add_argument('--max-steps', type=int, default=100, help='每轮最大步数')
    parser.add_argument('--save-dir', type=str, default=None, help='结果保存目录')
    return parser.parse_args()


def run_sac_fallback(env, args, config):
    """使用项目自带SAC实现"""
    print("\n使用兼容模式（项目自带SAC）...")
    
    from single_agent.sac import SACEnvironment
    
    sac_env = SACEnvironment()
    episode_rewards, episode_delays, episode_energies, episode_completions = [], [], [], []
    
    start_time = time.time()
    
    for episode in range(1, args.episodes + 1):
        state = sac_env.reset_environment()
        episode_reward = 0.0
        
        for step in range(args.max_steps):
            action = sac_env.select_action(state)
            next_state, reward, done, info = sac_env.step(action, state)
            sac_env.remember(state, action, reward, next_state, done)
            
            if len(sac_env.memory) >= sac_env.batch_size:
                sac_env.update()
            
            episode_reward += reward
            state = next_state
            if done:
                break
        
        metrics = info.get('system_metrics', {})
        episode_rewards.append(episode_reward)
        episode_delays.append(metrics.get('avg_task_delay', 0))
        episode_energies.append(metrics.get('total_energy_consumption', 0))
        episode_completions.append(metrics.get('task_completion_rate', 0))
        
        if episode % 20 == 0 or episode == args.episodes:
            print(f"Episode {episode}/{args.episodes}: Reward={episode_reward:.3f}, "
                  f"Delay={episode_delays[-1]:.3f}s")
    
    execution_time = time.time() - start_time
    stable_start = args.episodes // 2
    
    results = {
        'algorithm': 'SAC',
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


def main():
    args = parse_args()
    
    print("="*80)
    print("SAC算法独立运行")
    print("="*80)
    print(f"训练轮次: {args.episodes} | 随机种子: {args.seed} | 车辆数: {args.num_vehicles}")
    print("="*80)
    
    config = create_xuance_config('SAC', args.episodes, args.seed, args.num_vehicles, args.max_steps)
    env = VECGymEnv(config['env_config'])
    
    results = run_sac_fallback(env, args, config)
    env.close()
    
    manager = ResultsManager()
    save_path = manager.save_results('SAC', results, 'DRL', args.save_dir)
    manager.print_summary(results)
    
    print(f"\n✓ SAC训练完成！结果: {save_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())








