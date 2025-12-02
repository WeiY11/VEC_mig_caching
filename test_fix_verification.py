#!/usr/bin/env python3
"""验证修复后的代码"""
from train_single_agent import SingleAgentTrainingEnvironment
import numpy as np

# 测试环境创建
print("创建 OPTIMIZED_TD3 环境...")
env = SingleAgentTrainingEnvironment(algorithm='OPTIMIZED_TD3')
print('环境创建成功')
print(f'State dim: {env.agent_env.state_dim}')
print(f'Action dim: {env.agent_env.action_dim}')

# 测试重置
state = env.reset_environment()
print(f'Reset OK, state shape: {state.shape}')

# 测试step
action = np.random.randn(env.agent_env.action_dim)
next_state, reward, done, info = env.step(action)
print(f'Step OK, reward: {reward:.4f}')
print(f'task_feedback in info: {"task_feedback" in info}')

if 'task_feedback' in info:
    fb = info['task_feedback']
    print(f'  step_generated: {fb.get("step_generated", 0)}')
    print(f'  offload_distribution: {fb.get("offload_distribution", {})}')

print("\n验证完成!")
