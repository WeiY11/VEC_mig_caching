"""
TD3 v2.0 配置备份 - 用于对比实验
保存旧版配置参数，方便验证v3.0优化效果
"""
from dataclasses import dataclass

@dataclass
class TD3ConfigV2:
    """TD3算法配置 - v2.0 (优化前)"""
    # 网络结构
    hidden_dim: int = 512
    actor_lr: float = 1e-4
    critic_lr: float = 8e-5
    graph_embed_dim: int = 128
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 100000
    tau: float = 0.005
    gamma: float = 0.99
    
    # TD3特有参数
    policy_delay: int = 2
    target_noise: float = 0.05
    noise_clip: float = 0.2
    
    # 探索参数
    exploration_noise: float = 0.15
    noise_decay: float = 0.999
    min_noise: float = 0.05
    
    # 梯度裁剪
    gradient_clip_norm: float = 0.7
    use_gradient_clip: bool = True
    
    # PER参数
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 400000
    
    # 后期稳定策略
    late_stage_start_updates: int = 90000
    late_stage_tau: float = 0.003
    late_stage_policy_delay: int = 3
    late_stage_noise_floor: float = 0.03
    td_error_clip: float = 4.0
    
    # 训练频率
    update_freq: int = 1
    warmup_steps: int = 4000


# 对比表
COMPARISON_TABLE = """
TD3 配置对比: v2.0 → v3.0

关键参数           v2.0        v3.0        变化      预期效果
═══════════════════════════════════════════════════════════════
actor_lr           1e-4        8e-5        -20%      策略更新更平滑
critic_lr          8e-5        6e-5        -25%      Q值估计更稳定
batch_size         256         384         +50%      梯度方差降低
tau                0.005       0.003       -40%      目标网络更稳定
exploration_noise  0.15        0.12        -20%      初始探索适中
noise_decay        0.999       0.9995      +         加快衰减
min_noise          0.05        0.02        -60%      后期极致稳定⭐
target_noise       0.05        0.03        -40%      目标策略噪声降低
noise_clip         0.2         0.15        -25%      噪声裁剪收紧
late_stage_start   90000       60000       -33%      提前触发稳定期
td_error_clip      4.0         3.0         -25%      减少outliers影响

预期改进：
✅ 收敛后Reward Variance: 0.22 → 0.12 (降低45%)
✅ 800轮探索噪声: 0.045 → 0.020 (降低55%)
✅ 奖励曲线波动幅度缩小30%+
✅ 保持任务完成率 ≥97%
"""

if __name__ == "__main__":
    print(COMPARISON_TABLE)

