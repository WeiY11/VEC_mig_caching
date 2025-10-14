#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试状态维度不匹配问题"""
import sys
import numpy as np
from pathlib import Path

# 添加父目录到sys.path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# 导入必要的模块
from train_single_agent import SingleAgentTrainingEnvironment
from single_agent.td3 import TD3Environment

def debug_state_dimensions():
    """调试状态维度"""
    print("=" * 60)
    print("状态维度调试")
    print("=" * 60)
    
    # 1. 创建SingleAgentTrainingEnvironment
    print("\n1. 创建SingleAgentTrainingEnvironment...")
    training_env = SingleAgentTrainingEnvironment("TD3")
    
    # 2. 获取网络拓扑
    num_vehicles = len(training_env.simulator.vehicles)
    num_rsus = len(training_env.simulator.rsus) 
    num_uavs = len(training_env.simulator.uavs)
    
    print(f"\n仿真器拓扑：")
    print(f"  - 车辆数: {num_vehicles}")
    print(f"  - RSU数: {num_rsus}")
    print(f"  - UAV数: {num_uavs}")
    
    # 3. 获取agent_env的状态维度
    print(f"\nTD3Environment状态维度: {training_env.agent_env.state_dim}")
    
    # 4. 重置环境并获取初始状态
    print("\n重置环境...")
    initial_state = training_env.reset_environment()
    print(f"实际状态维度: {len(initial_state)}")
    print(f"状态形状: {initial_state.shape}")
    
    # 5. 手动计算预期维度
    expected_dim = num_vehicles * 5 + num_rsus * 5 + num_uavs * 5 + 8
    print(f"\n预期状态维度计算:")
    print(f"  - 车辆状态: {num_vehicles} × 5 = {num_vehicles * 5}")
    print(f"  - RSU状态: {num_rsus} × 5 = {num_rsus * 5}")
    print(f"  - UAV状态: {num_uavs} × 5 = {num_uavs * 5}")
    print(f"  - 全局状态: 8")
    print(f"  - 总计: {expected_dim}")
    
    # 6. 分析状态组成
    print(f"\n状态组成分析:")
    idx = 0
    
    # 车辆状态
    vehicle_start = idx
    vehicle_end = idx + num_vehicles * 5
    print(f"  - 车辆状态 [{vehicle_start}:{vehicle_end}]: shape=({num_vehicles}, 5)")
    idx = vehicle_end
    
    # RSU状态
    rsu_start = idx
    rsu_end = idx + num_rsus * 5
    print(f"  - RSU状态 [{rsu_start}:{rsu_end}]: shape=({num_rsus}, 5)")
    idx = rsu_end
    
    # UAV状态
    uav_start = idx
    uav_end = idx + num_uavs * 5
    print(f"  - UAV状态 [{uav_start}:{uav_end}]: shape=({num_uavs}, 5)")
    idx = uav_end
    
    # 全局状态
    global_start = idx
    global_end = idx + 8
    print(f"  - 全局状态 [{global_start}:{global_end}]: shape=(8,)")
    
    # 7. 检查是否匹配
    print(f"\n状态维度匹配检查:")
    print(f"  - TD3Environment.state_dim = {training_env.agent_env.state_dim}")
    print(f"  - 实际状态长度 = {len(initial_state)}")
    print(f"  - 预期状态长度 = {expected_dim}")
    
    if len(initial_state) == training_env.agent_env.state_dim == expected_dim:
        print("  ✅ 状态维度完全匹配!")
    else:
        print("  ❌ 状态维度不匹配!")
        
        # 进一步分析
        if len(initial_state) != training_env.agent_env.state_dim:
            print(f"     - 实际状态与TD3Environment不一致!")
        if len(initial_state) != expected_dim:
            print(f"     - 实际状态与预期计算不一致!")
        if training_env.agent_env.state_dim != expected_dim:
            print(f"     - TD3Environment与预期计算不一致!")
    
    return training_env, initial_state

if __name__ == "__main__":
    training_env, state = debug_state_dimensions()

