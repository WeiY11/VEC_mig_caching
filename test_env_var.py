#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试环境变量是否能传递"""
import os
import subprocess
import sys

# 测试1: 直接运行train_single_agent.py并传递环境变量
print("="*80)
print("Test: Can environment variable TASK_ARRIVAL_RATE be passed?")
print("="*80)

env = os.environ.copy()
env['TASK_ARRIVAL_RATE'] = '99.9'  # 明显的测试值

cmd = [sys.executable, 'train_single_agent.py', '--algorithm', 'TD3', '--episodes', '1', '--num-vehicles', '12']

print(f"Running: {' '.join(cmd)}")
print(f"With TASK_ARRIVAL_RATE={env['TASK_ARRIVAL_RATE']}")
print()

result = subprocess.run(
    cmd,
    env=env,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# 搜索输出中是否包含我们设置的值
if '99.9' in result.stdout:
    print("[OK] Environment variable WAS passed!")
    # 找到包含99.9的行
    for line in result.stdout.split('\n'):
        if '99.9' in line:
            print(f"  Found: {line.strip()}")
else:
    print("[ERROR] Environment variable NOT passed!")
    print("\nSearching for 'arrival' or 'rate' in output...")
    for line in result.stdout.split('\n'):
        if 'arrival' in line.lower() or 'task_arrival_rate' in line.lower():
            print(f"  {line.strip()}")

print("\n" + "="*80)

