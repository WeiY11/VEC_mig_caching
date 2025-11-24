#!/usr/bin/env python3
"""
🚀 Enhanced TD3 优化对比实验
配置优化后的奖励函数，公平对比Enhanced TD3与标准TD3

优化要点：
1. ✅ 增加缓存命中奖励 (weight_cache_bonus=2.0)
2. ✅ 降低能耗惩罚 (weight_energy=0.4, 从0.7降低43%)
3. ✅ 降低丢包惩罚 (penalty_dropped=50, 估计降低50%)
4. ✅ 增加迁移成功奖励 (migration_bonus=0.5*effectiveness)
"""

import os
import sys
import subprocess

def main():
    # 设置优化后的奖励权重
    env_vars = {
        # 核心权重
        'RL_WEIGHT_DELAY': '2.0',           # 保持延迟权重
        'RL_WEIGHT_ENERGY': '0.4',          # 降低能耗惩罚 (从0.7→0.4)
        'RL_PENALTY_DROPPED': '50',         # 降低丢包惩罚

        # 缓存优化奖励
        'RL_WEIGHT_CACHE_BONUS': '2.0',     # 缓存命中奖励！24%→+0.48

        # 迁移优化
        'RL_WEIGHT_MIGRATION': '0.1',       # 降低迁移成本惩罚

        # 目标值
        'RL_LATENCY_TARGET': '0.4',
        'RL_ENERGY_TARGET': '1200',
    }

    # 应用环境变量
    for key, value in env_vars.items():
        os.environ[key] = value

    print("🚀 Enhanced TD3 优化对比实验")
    print("=" * 60)
    print("优化配置：")
    print("  ✅ 缓存命中奖励: +2.0 (使24%命中率→+0.48奖励)")
    print("  ✅ 能耗惩罚: 0.4 (降低43%，适应复杂网络)")
    print("  ✅ 丢包惩罚: 50 (降低50%，减少0.6%差异影响)")
    print("  ✅ 迁移成功奖励: 0.5*effectiveness (新增!)")
    print("=" * 60)
    print()

    # 运行对比实验
    cmd = [
        sys.executable,
        'compare_enhanced_td3.py',
        '--algorithms', 'TD3', 'ENHANCED_TD3', 'CAM_TD3', 'ENHANCED_CAM_TD3',
        '--episodes', '1500',
        '--num-vehicles', '12',
        '--seed', '42'
    ]

    print(f"运行命令: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 实验完成！查看 results/td3_comparison/ 目录")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
