#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试运行脚本（无emoji，适配Windows gbk编码）
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# 导入项目模块
from config import config
from single_agent.td3 import TD3Environment
from evaluation.system_simulator import CompleteSystemSimulator
from ablation_experiments.ablation_configs import get_config_by_name


def suppress_emoji_output():
    """抑制emoji输出，避免gbk编码错误"""
    import io
    import sys
    
    class SafeWriter(io.TextIOWrapper):
        def write(self, s):
            try:
                return super().write(s)
            except UnicodeEncodeError:
                # 忽略无法编码的字符
                safe_s = s.encode('gbk', errors='ignore').decode('gbk')
                return super().write(safe_s)
    
    # 替换stdout
    sys.stdout = SafeWriter(sys.stdout.buffer, encoding='gbk', errors='replace')


def quick_test_single_config():
    """快速测试单个配置"""
    
    print("="*60)
    print("快速测试：Full-System配置（5轮）")
    print("="*60)
    
    # 抑制emoji输出
    suppress_emoji_output()
    
    # 设置随机种子
    np.random.seed(42)
    
    # 获取Full-System配置
    ablation_config = get_config_by_name('Full-System')
    ablation_config.apply_to_system()
    
    # 创建TD3环境和仿真器
    print("\n正在初始化...")
    try:
        td3_env = TD3Environment()
        simulator = CompleteSystemSimulator()
        print("初始化成功！")
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 快速训练5轮
    num_episodes = 5
    print(f"\n开始训练 {num_episodes} 轮...")
    
    for episode in range(1, num_episodes + 1):
        simulator.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(50):  # 每轮只运行50步
            # 获取状态
            node_states = simulator.get_all_node_states()
            system_metrics = simulator.get_system_metrics()
            state = td3_env.get_state_vector(node_states, system_metrics)
            
            # 选择动作
            actions = td3_env.get_actions(state, training=True)
            
            # 执行动作
            step_result = simulator.step(actions)
            
            # 获取下一状态
            next_node_states = simulator.get_all_node_states()
            next_system_metrics = simulator.get_system_metrics()
            next_state = td3_env.get_state_vector(next_node_states, next_system_metrics)
            
            # 计算奖励
            reward = td3_env.calculate_reward(
                next_system_metrics,
                step_result.get('cache_metrics'),
                step_result.get('migration_metrics')
            )
            
            # 存储经验并更新
            td3_env.train_step(state, actions['vehicle_agent'], reward, next_state, False)
            
            episode_reward += reward
            episode_steps += 1
        
        # Episode结束
        final_metrics = simulator.get_system_metrics()
        avg_reward = episode_reward / episode_steps
        
        print(f"Episode {episode}/{num_episodes}: "
              f"Reward={avg_reward:.3f}, "
              f"Delay={final_metrics.get('avg_task_delay', 0):.3f}s, "
              f"Energy={final_metrics.get('total_energy_consumption', 0):.1f}J")
    
    print("\n" + "="*60)
    print("快速测试完成！")
    print("="*60)
    print("\n实验环境工作正常！")
    print("\n下一步可以运行完整实验:")
    print("  python run_ablation_td3.py --episodes 30 --quick")
    print("="*60)


if __name__ == "__main__":
    quick_test_single_config()

