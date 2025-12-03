#!/bin/bash
# ============================================================================
# VEC边缘计算系统 - 训练命令速查表
# ============================================================================

# ============================================================================
# 方式一：原有框架 (train_single_agent.py) - 功能完整版
# ============================================================================

# -------------------- OPTIMIZED_TD3 (推荐) --------------------
# 基础训练
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200

# 带超参数覆盖
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 500 \
    --actor-lr 0.0001 --batch-size 256 --gamma 0.99 --num-vehicles 16

# 静默模式（不弹出交互）
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --silent-mode

# 使用YAML配置文件
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --config config/defaults.yaml

# 查看当前配置
python train_single_agent.py --print-config

# -------------------- TD3 --------------------
python train_single_agent.py --algorithm TD3 --episodes 200
python train_single_agent.py --algorithm TD3 --episodes 500 --actor-lr 0.0001 --batch-size 128

# -------------------- SAC --------------------
python train_single_agent.py --algorithm SAC --episodes 200
python train_single_agent.py --algorithm SAC --episodes 500 --num-vehicles 16

# -------------------- PPO --------------------
python train_single_agent.py --algorithm PPO --episodes 200

# -------------------- DDPG --------------------
python train_single_agent.py --algorithm DDPG --episodes 200

# -------------------- DQN --------------------
python train_single_agent.py --algorithm DQN --episodes 200

# -------------------- TD3-LE (Latency-Energy) --------------------
python train_single_agent.py --algorithm TD3-LE --episodes 200

# -------------------- CAM_TD3 (Cache-Aware Migration) --------------------
python train_single_agent.py --algorithm CAM_TD3 --episodes 200

# -------------------- 算法对比 --------------------
python train_single_agent.py --compare --episodes 200


# ============================================================================
# 方式二：xuance框架 (xuance/train.py) - 简洁统一版
# ============================================================================

# -------------------- DRL算法 --------------------
python xuance/train.py --method optimized_td3 --episodes 200
python xuance/train.py --method td3 --episodes 200
python xuance/train.py --method sac --episodes 200
python xuance/train.py --method ppo --episodes 200
python xuance/train.py --method ddpg --episodes 200
python xuance/train.py --method dqn --episodes 200
python xuance/train.py --method cam_td3 --episodes 200
python xuance/train.py --method td3_le --episodes 200

# 带参数
python xuance/train.py --method optimized_td3 --episodes 500 --seed 42

# -------------------- 对比方案 (Benchmarks) --------------------
python xuance/train.py --method local --episodes 200      # 本地计算基准
python xuance/train.py --method heuristic --episodes 200  # 启发式基准
python xuance/train.py --method sa --episodes 200         # 模拟退火

# 查看帮助
python xuance/train.py --help


# ============================================================================
# 高级选项
# ============================================================================

# 指定随机种子（可复现实验）
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --seed 42

# 修改车辆数量
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --num-vehicles 20

# 禁用实时可视化
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --no-realtime-vis

# 禁用中央资源分配架构
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --no-central-resource

# 从已有模型继续训练
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 \
    --resume-from results/single_agent/optimized_td3/model.pth

# 启用通信模型优化（3GPP标准）
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 200 --comm-enhancements


# ============================================================================
# 超参数覆盖参数说明
# ============================================================================
# --actor-lr          Actor学习率 (默认: 9e-5)
# --critic-lr         Critic学习率 (默认: 9e-5)
# --batch-size        批次大小 (默认: 384)
# --buffer-size       经验回放缓冲区大小 (默认: 100000)
# --gamma             折扣因子 (默认: 0.99)
# --tau               软更新系数 (默认: 0.005)
# --exploration-noise 初始探索噪声 (默认: 0.18)
# --hidden-dim        隐藏层维度 (默认: 256)
# --arrival-rate      任务到达率 (默认: 3.5 tasks/s)


# ============================================================================
# 配置优先级（从高到低）
# ============================================================================
# 1. 命令行参数 (--actor-lr 0.0001)
# 2. 环境变量 (TD3_ACTOR_LR=0.0001)
# 3. YAML配置 (config/defaults.yaml)
# 4. Python默认值


# ============================================================================
# 结果输出位置
# ============================================================================
# 训练结果: results/single_agent/<算法名>/
# 模型文件: results/single_agent/<算法名>/model.pth
# HTML报告: results/single_agent/<算法名>/training_report_*.html
# JSON数据: results/single_agent/<算法名>/training_results_*.json
