#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM-TD3优化训练脚本
应用优化后的配置参数解决奖励无法收敛的问题

使用方法：
python train_cam_td3_optimized.py
"""
import sys
import os

# 在导入其他模块之前加载优化配置
print("="*80)
print("🚀 CAM-TD3 优化训练")
print("="*80)

# 加载优化配置
from config.cam_td3_optimized_config import *

# 现在导入并运行训练
sys.argv = [
    "train_single_agent.py",
    "--algorithm", "CAM_TD3",
    "--episodes", "1000",
    "--num-vehicles", "12"
]

print("\n开始训练...")
print("="*80)

# 导入并执行主训练脚本
import train_single_agent

if __name__ == '__main__':
    # 主脚本已经通过sys.argv运行
    pass
