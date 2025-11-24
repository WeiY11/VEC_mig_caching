"""
分布式Critic网络 - QR-DQN风格队列约束Critic

基于Quantile Regression DQN (QR-DQN)的分布式价值网络，用于建模队列长度和时延的风险分布。
通过输出多个分位数的Q值，能够有效惩罚尾部时延，抑制队列溢出。

主要特点：
1. 输出51个分位数的Q值分布（覆盖2%-100%）
2. 使用分位数Huber损失进行训练
3. 支持CVaR（条件风险值）计算，关注最差情况
4. 对P90/P95/P99等高分位数施加额外惩罚
5. 双Q网络架构，减少过估计

作者：VEC_mig_caching Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class QuantileNetwork(nn.Module):
    """
    分位数神经网络
    
    输出N个分位数的Q值，每个分位数对应累积分布函数(CDF)的一个点。
    例如：tau=[0.02, 0.04, ..., 1.0] 对应P2, P4, ..., P100百分位数。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_quantiles: int = 51,
        quantile_embedding_dim: int = 64,
    ):
        super(QuantileNetwork, self).__init__()
        self.n_quantiles = n_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        
        # 状态-动作特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 分位数嵌入网络 - 将tau值嵌入到高维空间
        self.quantile_embedding = nn.Sequential(
            nn.Linear(1, quantile_embedding_dim),
            nn.ReLU(),
        )
        
        # 融合网络 - 将状态-动作特征与分位数嵌入融合
        self.merge_network = nn.Sequential(
            nn.Linear(hidden_dim + quantile_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 输出单个Q值
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        taus: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            taus: [n_quantiles] or None，分位数点，默认使用均匀分布
            
        Returns:
            quantile_values: [batch_size, n_quantiles] 每个分位数的Q值
        """
        batch_size = state.size(0)
        
        # 默认使用均匀分布的分位数
        if taus is None:
            taus = torch.linspace(
                1 / (2 * self.n_quantiles),
                1 - 1 / (2 * self.n_quantiles),
                self.n_quantiles,
                device=state.device,
            )
        
        # 提取状态-动作特征 [batch_size, hidden_dim]
        sa = torch.cat([state, action], dim=1)
        sa_features = self.feature_extractor(sa)
        
        # 嵌入分位数 [n_quantiles, quantile_embedding_dim]
        tau_embedding = self.quantile_embedding(taus.unsqueeze(-1))
        
        # 扩展维度以进行批量计算
        # sa_features: [batch_size, 1, hidden_dim]
        # tau_embedding: [1, n_quantiles, quantile_embedding_dim]
        sa_features = sa_features.unsqueeze(1).expand(-1, self.n_quantiles, -1)
        tau_embedding = tau_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 融合特征 [batch_size, n_quantiles, hidden_dim + quantile_embedding_dim]
        merged = torch.cat([sa_features, tau_embedding], dim=-1)
        
        # 输出每个分位数的Q值 [batch_size, n_quantiles]
        quantile_values = self.merge_network(merged).squeeze(-1)
        
        return quantile_values


class QuantileHuberLoss(nn.Module):
    """
    分位数Huber损失
    
    结合了分位数回归损失和Huber损失的优点：
    - 分位数回归：非对称损失，适合建模分布
    - Huber损失：对异常值鲁棒
    """
    
    def __init__(self, kappa: float = 1.0):
        super(QuantileHuberLoss, self).__init__()
        self.kappa = kappa
    
    def forward(
        self,
        quantile_values: torch.Tensor,
        target_values: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算分位数Huber损失
        
        Args:
            quantile_values: [batch_size, n_quantiles] 预测的分位数Q值
            target_values: [batch_size, n_quantiles] 目标分位数Q值
            taus: [n_quantiles] 分位数点
            
        Returns:
            loss: 标量损失值
        """
        # TD误差 [batch_size, n_quantiles]
        td_errors = target_values - quantile_values
        
        # Huber损失部分
        abs_errors = torch.abs(td_errors)
        huber_loss = torch.where(
            abs_errors <= self.kappa,
            0.5 * td_errors ** 2,
            self.kappa * (abs_errors - 0.5 * self.kappa)
        )
        
        # 分位数权重 [batch_size, n_quantiles]
        # tau * (td_error < 0) + (1 - tau) * (td_error >= 0)
        quantile_weight = torch.abs(taus.unsqueeze(0) - (td_errors < 0).float())
        
        # 加权Huber损失
        loss = (quantile_weight * huber_loss).mean()
        
        return loss


class DistributionalCritic(nn.Module):
    """
    分布式Critic - Twin Q网络，每个输出分位数分布
    
    用于TD3算法的增强Critic，输出Q值的完整分布而非单点估计。
    通过建模分布，能够：
    1. 识别高风险状态-动作对（高方差）
    2. 惩罚尾部时延（高分位数）
    3. 计算CVaR进行保守策略评估
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_quantiles: int = 51,
        quantile_embedding_dim: int = 64,
        kappa: float = 1.0,
    ):
        super(DistributionalCritic, self).__init__()
        self.n_quantiles = n_quantiles
        
        # 双Q网络
        self.q1_network = QuantileNetwork(
            state_dim, action_dim, hidden_dim, n_quantiles, quantile_embedding_dim
        )
        self.q2_network = QuantileNetwork(
            state_dim, action_dim, hidden_dim, n_quantiles, quantile_embedding_dim
        )
        
        # 损失函数
        self.loss_fn = QuantileHuberLoss(kappa=kappa)
        
        # 缓存分位数点（tau值）
        self.register_buffer(
            'taus',
            torch.linspace(
                1 / (2 * n_quantiles),
                1 - 1 / (2 * n_quantiles),
                n_quantiles,
            )
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 返回两个Q网络的分位数分布
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            q1_quantiles: [batch_size, n_quantiles]
            q2_quantiles: [batch_size, n_quantiles]
        """
        q1_quantiles = self.q1_network(state, action, self.taus)
        q2_quantiles = self.q2_network(state, action, self.taus)
        return q1_quantiles, q2_quantiles
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        获取Q1的期望值（策略更新使用）
        
        Returns:
            q1_mean: [batch_size, 1] 分位数Q值的均值
        """
        q1_quantiles = self.q1_network(state, action, self.taus)
        return q1_quantiles.mean(dim=1, keepdim=True)
    
    def compute_cvar(
        self,
        quantile_values: torch.Tensor,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """
        计算CVaR（条件风险值）- 最差alpha比例情况的平均Q值
        
        Args:
            quantile_values: [batch_size, n_quantiles]
            alpha: CVaR的alpha参数（例如0.1表示最差10%）
            
        Returns:
            cvar: [batch_size, 1] CVaR值
        """
        # 计算对应alpha的分位数索引
        n_tail = max(1, int(self.n_quantiles * alpha))
        
        # 取最小的n_tail个分位数（最差情况）
        tail_quantiles, _ = torch.topk(
            quantile_values, n_tail, dim=1, largest=False, sorted=True
        )
        
        # 返回平均值
        cvar = tail_quantiles.mean(dim=1, keepdim=True)
        return cvar
    
    def compute_tail_penalty(
        self,
        quantile_values: torch.Tensor,
        tail_percentiles: list = [0.90, 0.95, 0.99],
        weights: list = [0.3, 0.5, 0.8],
    ) -> torch.Tensor:
        """
        计算尾部惩罚项 - 对高分位数（P90, P95, P99）施加额外惩罚
        
        适用于抑制队列长时延和溢出风险。
        
        Args:
            quantile_values: [batch_size, n_quantiles]
            tail_percentiles: 关注的尾部百分位数列表
            weights: 对应的权重列表
            
        Returns:
            penalty: [batch_size, 1] 尾部惩罚值（负值，越负越差）
        """
        penalty = 0.0
        
        for percentile, weight in zip(tail_percentiles, weights):
            # 找到对应百分位数的索引
            idx = int(percentile * self.n_quantiles)
            idx = min(idx, self.n_quantiles - 1)
            
            # 提取该分位数的Q值
            tail_q = quantile_values[:, idx:idx+1]
            
            # 累加加权惩罚
            penalty = penalty + weight * tail_q
        
        # 归一化
        penalty = penalty / len(tail_percentiles)
        
        return penalty
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_quantiles: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算分布式Critic损失
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            target_quantiles: [batch_size, n_quantiles] 目标分位数Q值
            weights: [batch_size, 1] 重要性采样权重（可选）
            
        Returns:
            loss: 总损失
            td_errors: [batch_size] TD误差（用于更新优先级）
        """
        # 获取当前分位数Q值
        q1_quantiles, q2_quantiles = self.forward(state, action)
        
        # 计算分位数Huber损失
        loss_q1 = self.loss_fn(q1_quantiles, target_quantiles, self.taus)
        loss_q2 = self.loss_fn(q2_quantiles, target_quantiles, self.taus)
        
        # 如果有重要性采样权重，则应用
        if weights is not None:
            # 注意：分位数损失已经是均值，此处权重需要重新计算
            # 简化处理：直接用权重缩放总损失
            loss = weights.mean() * (loss_q1 + loss_q2)
        else:
            loss = loss_q1 + loss_q2
        
        # 计算TD误差（用于优先级更新）- 使用Q值均值
        q1_mean = q1_quantiles.mean(dim=1)
        target_mean = target_quantiles.mean(dim=1)
        td_errors = torch.abs(q1_mean - target_mean)
        
        return loss, td_errors


def create_distributional_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 512,
    n_quantiles: int = 51,
) -> DistributionalCritic:
    """
    便捷函数：创建分布式Critic
    
    默认配置：
    - 51个分位数（覆盖2%-100%，步长2%）
    - Huber阈值kappa=1.0
    - 隐藏层维度512
    """
    return DistributionalCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_quantiles=n_quantiles,
        quantile_embedding_dim=64,
        kappa=1.0,
    )
