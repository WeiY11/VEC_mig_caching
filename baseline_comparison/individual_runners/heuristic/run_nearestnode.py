#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NearestNode策略独立运行脚本

【使用方法】
  python baseline_comparison/individual_runners/heuristic/run_nearestnode.py --episodes 200 --seed 42
"""

import os
import sys
import json
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

from config import config
from train_single_agent import SingleAgentTrainingEnvironment
from baseline_comparison.improved_baseline_algorithms import NearestNodeBaseline
from baseline_comparison.individual_runners.common import ResultsManager


def parse_args():
    parser = argparse.ArgumentParser(description='NearestNode策略独立运行')
    parser.add_argument('--episodes', type=int, default=200, help='运行轮次')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-vehicles', type=int, default=12, help='车辆数量')
    parser.add_argument('--max-steps', type=int, default=100, help='每轮最大步数')
    parser.add_argument('--save-dir', type=str, default=None, help='结果保存目录')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("NearestNode策略独立运行")
    print("="*80)
    print(f"运行轮次: {args.episodes} | 随机种子: {args.seed} | 车辆数: {args.num_vehicles}")
    print("="*80)
    
    np.random.seed(args.seed)
    os.environ['RANDOM_SEED'] = str(args.seed)
    
    if args.num_vehicles != 12:
        os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps({"num_vehicles": args.num_vehicles})
    
    env = SingleAgentTrainingEnvironment("TD3")
    algorithm = NearestNodeBaseline()
    algorithm.update_environment(env)
    
    episode_rewards, episode_delays, episode_energies, episode_completions = [], [], [], []
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
        
        metrics = info.get('system_metrics', {})
        episode_rewards.append(episode_reward / max(1, episode_steps))
        episode_delays.append(metrics.get('avg_task_delay', 0))
        episode_energies.append(metrics.get('total_energy_consumption', 0))
        episode_completions.append(metrics.get('task_completion_rate', 0))
        
        if episode % 20 == 0 or episode == args.episodes:
            print(f"Episode {episode}/{args.episodes}: Reward={episode_rewards[-1]:.3f}, "
                  f"Delay={episode_delays[-1]:.3f}s")
    
    execution_time = time.time() - start_time
    stable_start = args.episodes // 2
    
    results = {
        'algorithm': 'NearestNode',
        'algorithm_type': 'Heuristic',
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
    }
    
    manager = ResultsManager()
    save_path = manager.save_results('NearestNode', results, 'Heuristic', args.save_dir)
    manager.print_summary(results)
    
    print(f"\n✓ NearestNode策略运行完成！结果: {save_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())








