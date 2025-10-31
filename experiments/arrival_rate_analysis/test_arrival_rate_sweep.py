#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证任务到达率扫描功能
执行超快速测试（每个到达率只训练5轮）
"""

import subprocess
import sys

print("="*80)
print("🧪 快速测试：TD3任务到达率扫描功能")
print("="*80)
print("测试配置:")
print("  - 到达率: [1.5, 2.5, 3.5] tasks/s (3个测试点)")
print("  - 轮次: 5 (超快速)")
print("  - 车辆数: 12")
print("  - 预计时间: 2-3分钟")
print("="*80)

# 运行测试
cmd = [
    sys.executable,
    "experiments/run_td3_arrival_rate_sweep.py",
    "--rates", "1.5", "2.5", "3.5",
    "--episodes", "5",
    "--num-vehicles", "12",
    "--output-dir", "results/test_arrival_rate"
]

print(f"\n执行命令: {' '.join(cmd)}\n")

try:
    result = subprocess.run(cmd, check=True)
    print("\n" + "="*80)
    print("✅ 测试成功!")
    print("="*80)
    print("📁 查看结果: results/test_arrival_rate/")
    print("="*80)
except subprocess.CalledProcessError as e:
    print("\n" + "="*80)
    print(f"❌ 测试失败: {e}")
    print("="*80)
    sys.exit(1)

