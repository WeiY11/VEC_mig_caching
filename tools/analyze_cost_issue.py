#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析成本计算是否正确
检查为什么高到达率反而成本更低
"""

import sys
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
from pathlib import Path

# 分析每个到达率的训练结果
result_files = [
    ('1.0', 'results/single_agent/td3/training_results_20251031_202734.json'),
    ('1.5', 'results/single_agent/td3/training_results_20251031_195709.json'),
    ('2.0', 'results/single_agent/td3/training_results_20251031_192845.json'),
    ('2.5', 'results/single_agent/td3/training_results_20251031_190006.json'),
    ('3.0', 'results/single_agent/td3/training_results_20251031_183056.json'),
]

print("="*100)
print("Detailed Cost Analysis - Checking Why High Arrival Rate Has Lower Cost")
print("="*100)

for rate, filepath in result_files:
    print(f"\n{'='*100}")
    print(f"Arrival Rate: {rate} tasks/s")
    print(f"File: {filepath}")
    print(f"{'='*100}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取最后50个episode的数据
    last_n = 50
    start_idx = -last_n
    
    episode_rewards = data['episode_rewards'][start_idx:]
    ep_metrics = data['episode_metrics']
    
    delays = ep_metrics['avg_delay'][start_idx:]
    energies = ep_metrics['total_energy'][start_idx:]
    steps = ep_metrics['episode_steps'][start_idx:]
    completion_rates = ep_metrics['task_completion_rate'][start_idx:]
    
    # 计算平均值
    avg_reward = np.mean(episode_rewards)
    avg_delay = np.mean(delays)
    avg_energy = np.mean(energies)
    avg_steps = np.mean(steps)
    avg_completion = np.mean(completion_rates)
    
    # 计算成本
    avg_cost = -avg_reward  # Cost = -Reward
    
    print(f"\nLast {last_n} episodes statistics:")
    print(f"  Avg Episode Reward:        {avg_reward:>12.2f}")
    print(f"  Avg Episode Cost:          {avg_cost:>12.2f}")
    print(f"  Avg Steps per Episode:     {avg_steps:>12.2f}")
    print(f"  Avg Delay (s):             {avg_delay:>12.4f}")
    print(f"  Avg Total Energy (J):      {avg_energy:>12.2f}")
    print(f"  Avg Completion Rate:       {avg_completion:>12.4f}")
    
    # 关键问题：检查episode_reward是什么
    print(f"\n  Analysis:")
    print(f"  - Cost per step:           {avg_cost / avg_steps:>12.2f}")
    print(f"  - Energy per step:         {avg_energy / avg_steps:>12.2f}")
    print(f"  - Delay per step:          {avg_delay:>12.4f}  (already per-step)")
    
    # 检查实际任务数量
    print(f"\n  Estimated tasks per episode:")
    # 假设simulation_time和time_slot
    time_slot = 0.2  # 从配置得知
    sim_time = avg_steps * time_slot
    estimated_tasks = float(rate) * sim_time
    print(f"    Simulation time:         {sim_time:>12.2f} s")
    print(f"    Estimated total tasks:   {estimated_tasks:>12.2f}")
    print(f"    Completed tasks:         {estimated_tasks * avg_completion:>12.2f}")
    print(f"    Cost per task:           {avg_cost / (estimated_tasks * avg_completion):>12.4f}")

print("\n" + "="*100)
print("SUMMARY: Cost Per Task Analysis")
print("="*100)
print(f"{'Rate':>6} | {'Episode Cost':>13} | {'Steps':>6} | {'Tasks':>7} | {'Cost/Task':>11}")
print("-"*100)

for rate, filepath in result_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    last_n = 50
    start_idx = -last_n
    
    episode_rewards = data['episode_rewards'][start_idx:]
    ep_metrics = data['episode_metrics']
    steps = ep_metrics['episode_steps'][start_idx:]
    completion_rates = ep_metrics['task_completion_rate'][start_idx:]
    
    avg_reward = np.mean(episode_rewards)
    avg_cost = -avg_reward
    avg_steps = np.mean(steps)
    avg_completion = np.mean(completion_rates)
    
    time_slot = 0.2
    sim_time = avg_steps * time_slot
    estimated_tasks = float(rate) * sim_time
    completed_tasks = estimated_tasks * avg_completion
    cost_per_task = avg_cost / completed_tasks if completed_tasks > 0 else 0
    
    print(f"{rate:>6} | {avg_cost:>13.2f} | {avg_steps:>6.1f} | {completed_tasks:>7.1f} | {cost_per_task:>11.4f}")

print("="*100)

