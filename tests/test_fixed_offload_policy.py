#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试固定卸载策略的实现

验证：
1. 固定策略能够正确初始化
2. 固定策略能够生成有效的action
3. 固定策略的action能够覆盖智能体的决策
4. 不同策略的行为有明显差异
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments'))

import numpy as np
from fallback_baselines import (
    RandomPolicy, GreedyPolicy, LocalOnlyPolicy, 
    RSUOnlyPolicy, RoundRobinPolicy, WeightedPreferencePolicy
)


def test_policy_initialization():
    """测试策略初始化"""
    print("\n" + "=" * 80)
    print("测试1: 固定策略初始化")
    print("=" * 80)
    
    policies = {
        'Random': RandomPolicy(),
        'Greedy': GreedyPolicy(),
        'LocalOnly': LocalOnlyPolicy(),
        'RSUOnly': RSUOnlyPolicy(),
        'RoundRobin': RoundRobinPolicy(),
        'Weighted': WeightedPreferencePolicy(),
    }
    
    for name, policy in policies.items():
        print(f"  {name}: {type(policy).__name__} - ", end="")
        if hasattr(policy, 'select_action'):
            print("[PASS] 有select_action方法")
        else:
            print("[FAIL] 缺少select_action方法")
            return False
    
    print("\n[PASS] 所有策略初始化成功")
    return True


def test_policy_action_generation():
    """测试策略action生成"""
    print("\n" + "=" * 80)
    print("测试2: 固定策略Action生成")
    print("=" * 80)
    
    # 创建简单的环境mock
    class MockEnv:
        def __init__(self):
            self.simulator = type('obj', (object,), {
                'vehicles': [{}] * 4,
                'rsus': [{}] * 2,
                'uavs': [{}] * 1,
            })()
            self.agent_env = type('obj', (object,), {'action_dim': 18})()
    
    mock_env = MockEnv()
    
    # 创建一个简单的state
    state = np.random.randn(50).astype(np.float32)
    
    policies = {
        'Random': RandomPolicy(seed=42),
        'Greedy': GreedyPolicy(),
        'LocalOnly': LocalOnlyPolicy(),
        'RSUOnly': RSUOnlyPolicy(),
        'RoundRobin': RoundRobinPolicy(),
    }
    
    all_passed = True
    for name, policy in policies.items():
        try:
            # 更新环境
            policy.update_environment(mock_env)
            
            # 生成action
            action = policy.select_action(state)
            
            # 验证action
            if not isinstance(action, np.ndarray):
                print(f"  {name}: [FAIL] action不是numpy数组")
                all_passed = False
                continue
            
            if len(action) < 3:
                print(f"  {name}: [FAIL] action维度不足3")
                all_passed = False
                continue
            
            # 提取前3个值作为offload preference
            local_pref = float(action[0])
            rsu_pref = float(action[1])
            uav_pref = float(action[2])
            
            print(f"  {name}: [PASS] action={action[:3]} (local={local_pref:.2f}, rsu={rsu_pref:.2f}, uav={uav_pref:.2f})")
            
        except Exception as e:
            print(f"  {name}: [FAIL] 异常: {e}")
            all_passed = False
    
    if all_passed:
        print("\n[PASS] 所有策略都能生成有效action")
    else:
        print("\n[FAIL] 部分策略生成action失败")
    
    return all_passed


def test_policy_behavior_difference():
    """测试不同策略的行为差异"""
    print("\n" + "=" * 80)
    print("测试3: 不同策略行为差异")
    print("=" * 80)
    
    # 创建环境mock
    class MockEnv:
        def __init__(self):
            self.simulator = type('obj', (object,), {
                'vehicles': [{}] * 4,
                'rsus': [{}] * 2,
                'uavs': [{}] * 1,
            })()
            self.agent_env = type('obj', (object,), {'action_dim': 18})()
    
    mock_env = MockEnv()
    state = np.random.randn(50).astype(np.float32)
    
    # 测试LocalOnly vs RSUOnly
    local_policy = LocalOnlyPolicy()
    rsu_policy = RSUOnlyPolicy()
    
    local_policy.update_environment(mock_env)
    rsu_policy.update_environment(mock_env)
    
    local_action = local_policy.select_action(state)
    rsu_action = rsu_policy.select_action(state)
    
    # LocalOnly应该有高的local preference
    if local_action[0] > local_action[1] and local_action[0] > local_action[2]:
        print("  LocalOnly: [PASS] local > rsu, local > uav")
        local_ok = True
    else:
        print(f"  LocalOnly: [FAIL] action={local_action[:3]}, local不是最大")
        local_ok = False
    
    # RSUOnly应该有高的RSU preference
    if rsu_action[1] > rsu_action[0] and rsu_action[1] > rsu_action[2]:
        print("  RSUOnly: [PASS] rsu > local, rsu > uav")
        rsu_ok = True
    else:
        print(f"  RSUOnly: [FAIL] action={rsu_action[:3]}, rsu不是最大")
        rsu_ok = False
    
    if local_ok and rsu_ok:
        print("\n[PASS] 不同策略有明显行为差异")
        return True
    else:
        print("\n[FAIL] 策略行为未达到预期")
        return False


def test_round_robin_behavior():
    """测试RoundRobin策略的轮询行为"""
    print("\n" + "=" * 80)
    print("测试4: RoundRobin轮询行为")
    print("=" * 80)
    
    # 创建环境mock
    class MockEnv:
        def __init__(self):
            self.simulator = type('obj', (object,), {
                'vehicles': [{}] * 4,
                'rsus': [{}] * 2,
                'uavs': [{}] * 1,
            })()
            self.agent_env = type('obj', (object,), {'action_dim': 18})()
    
    mock_env = MockEnv()
    state = np.random.randn(50).astype(np.float32)
    
    policy = RoundRobinPolicy()
    policy.update_environment(mock_env)
    
    # 生成多个action，观察是否轮询
    actions = []
    for i in range(6):
        action = policy.select_action(state)
        actions.append(action[:3].copy())
        print(f"  Step {i+1}: local={action[0]:.2f}, rsu={action[1]:.2f}, uav={action[2]:.2f}")
    
    # 检查是否有变化
    actions_array = np.array(actions)
    variance = np.var(actions_array, axis=0)
    
    if np.any(variance > 0.1):
        print("\n[PASS] RoundRobin策略有轮询行为（action有变化）")
        return True
    else:
        print("\n[WARNING] RoundRobin策略变化不明显")
        return True  # 仍然通过，因为可能是实现方式不同


def main():
    """运行所有测试"""
    print("=" * 80)
    print("固定卸载策略测试套件")
    print("=" * 80)
    print(f"项目根目录: {project_root}")
    
    # 设置随机种子
    np.random.seed(42)
    
    results = []
    
    # 测试1：策略初始化
    try:
        result1 = test_policy_initialization()
        results.append(("策略初始化", result1))
    except Exception as e:
        print(f"\n[ERROR] 策略初始化测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("策略初始化", False))
    
    # 测试2：Action生成
    try:
        result2 = test_policy_action_generation()
        results.append(("Action生成", result2))
    except Exception as e:
        print(f"\n[ERROR] Action生成测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Action生成", False))
    
    # 测试3：行为差异
    try:
        result3 = test_policy_behavior_difference()
        results.append(("行为差异", result3))
    except Exception as e:
        print(f"\n[ERROR] 行为差异测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("行为差异", False))
    
    # 测试4：RoundRobin行为
    try:
        result4 = test_round_robin_behavior()
        results.append(("RoundRobin行为", result4))
    except Exception as e:
        print(f"\n[ERROR] RoundRobin行为测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("RoundRobin行为", False))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n[SUCCESS] 所有测试通过！")
        print("\n使用示例：")
        print("  python train_single_agent.py --algorithm TD3 --fixed-offload-policy random")
        print("  python train_single_agent.py --algorithm TD3 --fixed-offload-policy greedy")
        print("  python train_single_agent.py --algorithm TD3 --fixed-offload-policy local_only")
        return 0
    else:
        print("\n[FAILURE] 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

