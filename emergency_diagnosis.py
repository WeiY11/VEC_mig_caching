"""
紧急诊断脚本 - 验证智能体动作是否真的影响系统

测试逻辑:
1. 固定动作A（全本地） vs 固定动作B（全远程）
2. 如果结果一样 → 动作没起作用
3. 如果结果不同 → 动作有效，但奖励函数有问题
"""

import numpy as np
import subprocess
import sys

def test_fixed_action(action_type, episodes=10):
    """测试固定动作的效果"""
    
    if action_type == "local":
        print("\n" + "="*70)
        print("🧪 测试 A: 强制本地处理")
        print("="*70)
        cmd = [
            sys.executable, "train_single_agent.py",
            "--algorithm", "OPTIMIZED_TD3",
            "--episodes", str(episodes),
            "--num-vehicles", "12",
            "--enforce-offload-mode", "local_only"
        ]
    elif action_type == "remote":
        print("\n" + "="*70)
        print("🧪 测试 B: 强制远程卸载")
        print("="*70)
        cmd = [
            sys.executable, "train_single_agent.py",
            "--algorithm", "OPTIMIZED_TD3",
            "--episodes", str(episodes),
            "--num-vehicles", "12",
            "--enforce-offload-mode", "remote_only"
        ]
    else:
        print("\n" + "="*70)
        print("🧪 测试 C: 智能体自由决策")
        print("="*70)
        cmd = [
            sys.executable, "train_single_agent.py",
            "--algorithm", "OPTIMIZED_TD3",
            "--episodes", str(episodes),
            "--num-vehicles", "12"
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 从输出中提取平均奖励
    for line in result.stdout.split('\n'):
        if 'Average Reward' in line or '平均奖励' in line:
            print(f"   结果: {line}")
    
    return result.returncode == 0

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║             紧急诊断：动作是否真的影响系统？                   ║
╚════════════════════════════════════════════════════════════════╝

理论预测:
  • 全本地处理: 时延高(~5s), 能耗高, 奖励应该 -100
  • 全远程卸载: 时延低(~2s), 能耗中, 奖励应该 -30
  
如果两者奖励都是-90:
  → 证明动作根本没影响系统！！！
  → 智能体在做"无用功"
    """)
    
    input("按Enter开始诊断 (约5分钟)...")
    
    # 测试1: 本地
    test_fixed_action("local", episodes=10)
    
    # 测试2: 远程
    test_fixed_action("remote", episodes=10)
    
    # 测试3: 智能体
    test_fixed_action("agent", episodes=10)
    
    print("\n" + "="*70)
    print("📊 诊断结果分析")
    print("="*70)
    print("""
请对比上面三个测试的奖励值:

情况1: 三者奖励都差不多 (-90左右)
  → 问题确诊: 动作根本没影响系统
  → 需要检查: 仿真器是否真的使用了动作
  
情况2: 本地(-100) vs 远程(-30) vs 智能体(-90)
  → 问题确诊: 动作有效，但智能体没学会
  → 需要: 增加探索、调整奖励函数
  
情况3: 三者都是0附近
  → 奖励函数彻底坏了
    """)

if __name__ == "__main__":
    main()
