"""
增强型TD3智能体配置

扩展标准TD3Config，增加5项高级优化的配置参数：
1. 分布式Critic
2. 熵正则化
3. 模型化队列预测
4. 队列感知回放
5. GAT路由器

作者：VEC_mig_caching Team
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EnhancedTD3Config:
    """增强型TD3配置 - 集成5项高级优化"""
    
    # ========== 基础TD3参数 ==========
    # 🔧 v10优化：大幅增加网络容量 + 更激进学习率
    hidden_dim: int = 1024          # 🔧 v10: 768 → 1024
    actor_lr: float = 3e-3          # 🔧 v10: 1e-3 → 3e-3 (更激进)
    critic_lr: float = 5e-3         # 🔧 v10: 2e-3 → 5e-3 (更激进)
    graph_embed_dim: int = 384      # 🔧 v10: 256 → 384
    
    # 训练参数
    batch_size: int = 1024  # 🔧 v22: 512 → 1024 (更大batch提高GPU利用率)
    buffer_size: int = 500000  # 🔧 v22: 300000 → 500000 (支持更长训练)
    tau: float = 0.005          # 🔧 v10: 0.01 → 0.005 (更慢更新更稳定)
    gamma: float = 0.99         # 🔧 v10: 0.98 → 0.99 (更长视野)
    
    # TD3特有
    policy_delay: int = 2
    target_noise: float = 0.05
    noise_clip: float = 0.2
    
    # 探索参数 - 🔧 v10优化
    exploration_noise: float = 0.35  # 🔧 v10: 0.30 → 0.35 (更高初始噪声)
    noise_decay: float = 0.9998      # 🔧 v10: 0.9995 → 0.9998 (慢衰减)
    min_noise: float = 0.10          # 🔧 v10: 0.08 → 0.10 (更高最小噪声)
    
    # 梯度裁剪
    gradient_clip_norm: float = 0.5
    use_gradient_clip: bool = True
    use_reward_normalization: bool = True
    reward_norm_clip: float = 6.0
    reward_norm_beta: float = 0.996
    
    # 保守增强
    cql_alpha: float = 0.12
    cql_num_samples: int = 4
    uncertainty_weight: float = 0.05
    
    # ========== Feature 1: 分布式Critic ==========
    use_distributional_critic: bool = False  # 是否启用分布式Critic
    n_quantiles: int = 51  # 分位数个数
    quantile_embedding_dim: int = 64  # 分位数嵌入维度
    quantile_kappa: float = 1.0  # Huber损失阈值
    cvar_alpha: float = 0.1  # CVaR的alpha参数（关注最差10%）
    tail_penalty_weight: float = 0.5  # 尾部惩罚权重
    tail_percentiles: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])  # 尾部百分位
    tail_percentile_weights: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])  # 尾部权重
    
    # ========== Feature 2: 熵正则化 ==========
    use_entropy_reg: bool = False  # 是否启用熵正则化
    auto_tune_alpha: bool = True  # 自动调节温度参数
    initial_alpha: float = 0.2  # 初始温度
    alpha_lr: float = 3e-4  # 温度参数学习率
    target_entropy_ratio: float = 0.6  # 目标熵 = -action_dim * ratio
    
    # 分组温度（卸载 vs 缓存）
    use_grouped_temperature: bool = False  # 是否使用分组温度
    offload_temp: float = 1.5  # 卸载决策温度（高温，鼓励探索）
    cache_temp: float = 0.5  # 缓存决策温度（低温，稳定策略）
    
    # ========== Feature 3: 模型化队列预测 ==========
    use_model_based_rollout: bool = False  # 是否启用模型化rollout
    model_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])  # 动态模型隐藏层
    model_lr: float = 3e-4  # 模型学习率
    rollout_horizon: int = 5  # rollout步数
    model_train_freq: int = 200  # 模型训练频率（每N步训练一次）
    model_train_iterations: int = 10  # 每次训练的迭代次数
    rollout_batch_size: int = 256  # rollout批次大小
    num_rollouts_per_state: int = 1  # 每个状态的rollout次数
    imagined_reward_weight: float = 0.3  # 想象奖励权重
    overflow_penalty: float = -10.0  # 队列溢出惩罚
    min_model_buffer_size: int = 1000  # 开始训练模型的最小buffer大小
    
    # ========== Feature 4: 队列感知回放 ==========
    use_queue_aware_replay: bool = False  # 是否启用队列感知回放
    queue_priority_weight: float = 0.3  # 队列因素在优先级中的权重
    queue_metrics_ema_decay: float = 0.9  # 队列指标EMA平滑系数
    queue_occ_coef: float = 0.5  # 队列占用权重
    packet_loss_coef: float = 0.3  # 丢包率权重
    migration_cong_coef: float = 0.2  # 迁移拥塞权重
    
    # ========== Feature 5: GAT路由器 ==========
    use_gat_router: bool = False  # 是否启用GAT路由器
    num_attention_heads: int = 4  # 注意力头数
    gat_hidden_dim: int = 128  # GAT隐藏维度
    edge_feature_dim: int = 8  # 边特征维度
    gat_dropout: float = 0.1  # GAT Dropout率
    
    # ========== 优先级回放参数（通用） ==========
    alpha: float = 0.6  # PER的alpha参数
    beta_start: float = 0.4  # PER的beta初始值
    beta_increment: float = 5e-6  # beta增长率
    
    # ========== 其他 ==========
    warmup_steps: int = 200  # 🔧 v9: 500 → 200 (更快开始学习)
    update_freq: int = 1     # 🔧 v9: 2 → 1 (每步都更新)
    gradient_steps: int = 8  # 🔧 v22: 4 → 8 每步多次梯度更新 (大幅提高GPU利用率)
    device: str = 'cuda'  # 设备
    
    # ========== 🚀 性能优化参数 (v22新增) ==========
    use_amp: bool = True  # 混合精度训练(AMP) - 减少显存占用，加速计算
    use_async_transfer: bool = True  # 异步数据传输
    num_workers: int = 2  # 数据预取进程数
    pin_memory: bool = True  # 锁页内存加速CPU-GPU传输
    prefetch_factor: int = 2  # 预取批次数
    
    # 后期稳定策略
    late_stage_start_updates: int = 50000
    late_stage_tau: float = 0.002
    late_stage_policy_delay: int = 3
    late_stage_noise_floor: float = 0.02
    
    # TD误差裁剪
    td_error_clip: float = 4.0
    
    def __post_init__(self):
        """后处理配置"""
        import os
        
        # 从环境变量读取设备配置
        if 'DEVICE' in os.environ:
            self.device = os.environ['DEVICE']
        
        # 确保队列感知回放的权重系数之和为1
        total_queue_coef = self.queue_occ_coef + self.packet_loss_coef + self.migration_cong_coef
        if total_queue_coef > 0 and abs(total_queue_coef - 1.0) > 0.01:
            # 归一化
            self.queue_occ_coef /= total_queue_coef
            self.packet_loss_coef /= total_queue_coef
            self.migration_cong_coef /= total_queue_coef
        
        # 确保尾部百分位和权重长度一致
        if len(self.tail_percentiles) != len(self.tail_percentile_weights):
            print(f"[Warning] tail_percentiles和tail_percentile_weights长度不一致，使用默认值")
            self.tail_percentiles = [0.90, 0.95, 0.99]
            self.tail_percentile_weights = [0.3, 0.5, 0.8]


def create_baseline_config() -> EnhancedTD3Config:
    """创建基线配置（所有优化禁用）"""
    return EnhancedTD3Config()


def create_full_enhanced_config() -> EnhancedTD3Config:
    """创建完全增强配置（所有优化启用）"""
    return EnhancedTD3Config(
        use_distributional_critic=True,
        use_entropy_reg=True,
        use_model_based_rollout=True,
        use_queue_aware_replay=True,
        use_gat_router=True,
    )


def create_queue_focused_config() -> EnhancedTD3Config:
    """创建队列优化焦点配置（针对高拥塞场景）"""
    return EnhancedTD3Config(
        use_distributional_critic=True,  # 尾部时延抑制
        use_queue_aware_replay=True,  # 队列感知采样
        use_model_based_rollout=True,  # 预测队列溢出
        n_quantiles=51,
        tail_penalty_weight=0.7,  # 更高的尾部惩罚
        queue_priority_weight=0.4,  # 更高的队列权重
        rollout_horizon=5,
    )


def create_exploration_focused_config() -> EnhancedTD3Config:
    """创建探索优化焦点配置（针对不确定性高的环境）"""
    return EnhancedTD3Config(
        use_entropy_reg=True,  # 熵正则化维持探索
        auto_tune_alpha=True,
        use_grouped_temperature=True,
        offload_temp=2.0,  # 更高的卸载探索温度
        cache_temp=0.3,  # 更低的缓存温度
        exploration_noise=0.2,  # 更高的初始噪声
    )


def create_dynamic_topology_config() -> EnhancedTD3Config:
    """创建动态拓扑配置（针对移动性高的场景）"""
    return EnhancedTD3Config(
        use_gat_router=True,  # GAT适应动态拓扑
        num_attention_heads=6,  # 更多注意力头
        gat_hidden_dim=256,  # 更大的表示能力
    )


def create_optimized_gat_config() -> EnhancedTD3Config:
    """✨ 创建优化的GAT配置（最新优化）
    
    优化点：
    1. 启用Queue-aware Replay + GAT
    2. 增加注意力头数
    3. 优化学习率和batch size
    4. 增强探索策略
    """
    return EnhancedTD3Config(
        # 核心优化
        use_queue_aware_replay=True,
        use_gat_router=True,
        
        # GAT优化
        num_attention_heads=6,  # 增加到6个头
        gat_hidden_dim=192,  # 适度增大隐藏层
        gat_dropout=0.15,  # 轻微增加dropout
        
        # 训练优化
        batch_size=640,  # 增大batch size
        actor_lr=1.5e-4,  # 调低学习率
        critic_lr=2.5e-4,
        
        # 探索优化
        exploration_noise=0.20,
        noise_decay=0.9985,  # 更温和的衰减
        min_noise=0.08,  # 较高的最小噪声
        
        # 队列感知优化
        queue_priority_weight=0.5,  # 更高的队列权重
    )
