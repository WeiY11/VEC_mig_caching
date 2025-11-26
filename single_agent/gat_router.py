"""
GAT路由器 - 图注意力网络用于协同缓存

基于Graph Attention Networks (GAT)的路由特征提取器，显式建模：
1. 车辆-RSU的注意力关系（卸载决策）
2. RSU-RSU的协同缓存关系（内容共享）

主要特点：
- 多头注意力机制
- 边特征融合（距离、信号强度、带宽）
- 动态拓扑支持（邻接矩阵掩码）
- 输出协同缓存概率矩阵

作者：VEC_mig_caching Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class GATLayer(nn.Module):
    """
    单层图注意力
    
    实现多头注意力机制，融合边特征
    
    ✨ 优化点：
    1. 添加残差连接
    2. 添加LayerNorm稳定训练
    3. 改进初始化策略
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        edge_feature_dim: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
        use_residual: bool = True,
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度（每个头）
            num_heads: 注意力头数
            edge_feature_dim: 边特征维度
            dropout: Dropout率
            concat: 是否拼接多头输出（否则平均）
            use_residual: 是否使用残差连接
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.edge_feature_dim = edge_feature_dim
        self.concat = concat
        self.use_residual = use_residual
        
        # 每个头的变换矩阵
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # 注意力权重参数 [num_heads, 2*out_features + edge_feature_dim]
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features + edge_feature_dim, 1))
        
        # 边特征嵌入
        self.edge_embedding = nn.Linear(edge_feature_dim, edge_feature_dim)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # ✨ 添加LayerNorm提升稳定性
        if concat:
            self.layer_norm = nn.LayerNorm(out_features * num_heads)
            # 残差投影（如果维度不匹配）
            if use_residual and in_features != out_features * num_heads:
                self.residual_proj = nn.Linear(in_features, out_features * num_heads)
            else:
                self.residual_proj = None
        else:
            self.layer_norm = nn.LayerNorm(out_features)
            if use_residual and in_features != out_features:
                self.residual_proj = nn.Linear(in_features, out_features)
            else:
                self.residual_proj = None
        
        self._init_weights()
    
    def _init_weights(self):
        """✨ 改进的初始化策略"""
        # Xavier初始化变换矩阵
        nn.init.xavier_uniform_(self.W, gain=1.414)  # 使用更大的gain
        # Xavier初始化注意力参数
        nn.init.xavier_uniform_(self.a, gain=1.414)
    
    def forward(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        adjacency_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h_source: [batch_size, num_source, in_features] 源节点特征
            h_target: [batch_size, num_target, in_features] 目标节点特征
            edge_features: [batch_size, num_source, num_target, edge_feature_dim] 边特征
            adjacency_mask: [batch_size, num_source, num_target] 邻接掩码（True表示存在边）
            
        Returns:
            h_out: [batch_size, num_target, out_features*num_heads] （concat=True）
                   或 [batch_size, num_target, out_features] （concat=False）
        """
        batch_size = h_source.size(0)
        num_source = h_source.size(1)
        num_target = h_target.size(1)
        
        # 线性变换 [batch_size, num_heads, num_nodes, out_features]
        Wh_source = torch.einsum('bni,hio->bhno', h_source, self.W)
        Wh_target = torch.einsum('bni,hio->bhno', h_target, self.W)
        
        # 计算注意力分数
        # 扩展维度进行广播
        Wh_source_expanded = Wh_source.unsqueeze(3)  # [batch, heads, num_source, 1, out]
        Wh_target_expanded = Wh_target.unsqueeze(2)  # [batch, heads, 1, num_target, out]
        
        # 拼接源和目标特征 [batch, heads, num_source, num_target, 2*out]
        concat_features = torch.cat([
            Wh_source_expanded.expand(-1, -1, -1, num_target, -1),
            Wh_target_expanded.expand(-1, -1, num_source, -1, -1),
        ], dim=-1)
        
        # 融合边特征
        if edge_features is not None:
            edge_emb = self.edge_embedding(edge_features)  # [batch, num_source, num_target, edge_dim]
            edge_emb = edge_emb.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)  # [batch, heads, src, tgt, edge_dim]
            concat_features = torch.cat([concat_features, edge_emb], dim=-1)
        else:
            # 零边特征
            zero_edge = torch.zeros(
                batch_size, self.num_heads, num_source, num_target, self.edge_feature_dim,
                device=h_source.device
            )
            concat_features = torch.cat([concat_features, zero_edge], dim=-1)
        
        # 计算注意力logits [batch, heads, num_source, num_target]
        e = torch.einsum('bhsti,hio->bhst', concat_features, self.a).squeeze(-1)
        e = self.leakyrelu(e)
        
        # 应用邻接掩码
        if adjacency_mask is not None:
            # Ensure boolean
            if adjacency_mask.dtype != torch.bool:
                adjacency_mask = adjacency_mask > 0.5
                
            # adjacency_mask: [batch, num_source, num_target]
            adjacency_mask = adjacency_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            e = e.masked_fill(~adjacency_mask, float('-inf'))
        
        # Softmax归一化（对源节点维度）
        attention = F.softmax(e, dim=2)  # [batch, heads, num_source, num_target]
        attention = self.dropout(attention)
        
        # 加权聚合源节点特征
        # attention: [batch, heads, num_source, num_target]
        # Wh_source: [batch, heads, num_source, out]
        # 结果: [batch, heads, num_target, out]
        h_prime = torch.einsum('bhst,bhso->bhto', attention, Wh_source)
        
        # 多头输出处理
        if self.concat:
            # 拼接所有头 [batch, num_target, heads*out]
            h_out = h_prime.permute(0, 2, 1, 3).reshape(batch_size, num_target, -1)
        else:
            # 平均所有头 [batch, num_target, out]
            h_out = h_prime.mean(dim=1)
        
        # ✨ 添加残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                # 维度不匹配，需要投影
                residual = self.residual_proj(h_target)
            else:
                # 维度匹配，直接相加
                residual = h_target
            h_out = h_out + residual
        
        # ✨ LayerNorm稳定训练
        h_out = self.layer_norm(h_out)
        
        return h_out


class VehicleRSUAttention(nn.Module):
    """
    车辆-RSU注意力模块
    
    建模车辆到RSU的卸载决策，考虑距离、信号质量、RSU负载等因素
    """
    
    def __init__(
        self,
        vehicle_feature_dim: int = 5,
        rsu_feature_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        edge_feature_dim: int = 8,
    ):
        super(VehicleRSUAttention, self).__init__()
        
        # 节点特征投影
        self.vehicle_proj = nn.Linear(vehicle_feature_dim, hidden_dim)
        self.rsu_proj = nn.Linear(rsu_feature_dim, hidden_dim)
        
        # GAT层
        self.gat_layer = GATLayer(
            in_features=hidden_dim,
            out_features=hidden_dim // num_heads,
            num_heads=num_heads,
            edge_feature_dim=edge_feature_dim,
            concat=True,
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        vehicle_features: torch.Tensor,
        rsu_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        adjacency_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vehicle_features: [batch_size, num_vehicles, vehicle_feature_dim]
            rsu_features: [batch_size, num_rsus, rsu_feature_dim]
            edge_features: [batch_size, num_vehicles, num_rsus, edge_feature_dim]
            adjacency_mask: [batch_size, num_vehicles, num_rsus]
            
        Returns:
            vehicle_representations: [batch_size, num_vehicles, hidden_dim]
        """
        # 投影节点特征
        h_vehicles = F.relu(self.vehicle_proj(vehicle_features))
        h_rsus = F.relu(self.rsu_proj(rsu_features))
        
        # GAT注意力（RSU -> Vehicles）
        # 注意：edge_features是[batch, V, R, dim]，GAT期望[batch, src, tgt, dim]
        # 这里src=RSU, tgt=Vehicle，所以需要转置
        if edge_features is not None:
            edge_features_t = edge_features.permute(0, 2, 1, 3)
        else:
            edge_features_t = None
            
        if adjacency_mask is not None:
            adjacency_mask_t = adjacency_mask.permute(0, 2, 1)
        else:
            adjacency_mask_t = None
            
        h_out = self.gat_layer(h_rsus, h_vehicles, edge_features_t, adjacency_mask_t)
        
        # 输出投影
        vehicle_representations = self.output_proj(h_out)
        
        return vehicle_representations


class RSURSUCollaborativeAttention(nn.Module):
    """
    RSU-RSU协同缓存注意力模块
    
    建模RSU之间的内容协作关系，输出协同缓存决策概率
    """
    
    def __init__(
        self,
        rsu_feature_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        edge_feature_dim: int = 8,
    ):
        super(RSURSUCollaborativeAttention, self).__init__()
        
        # RSU特征投影
        self.rsu_proj = nn.Linear(rsu_feature_dim, hidden_dim)
        
        # 物理邻接注意力
        self.physical_gat = GATLayer(
            in_features=hidden_dim,
            out_features=hidden_dim // num_heads,
            num_heads=num_heads,
            edge_feature_dim=edge_feature_dim,
            concat=True,
        )
        
        # 缓存内容相似度注意力（self-attention）
        self.content_gat = GATLayer(
            in_features=hidden_dim,
            out_features=hidden_dim // num_heads,
            num_heads=num_heads,
            edge_feature_dim=edge_feature_dim,
            concat=True,
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 协同缓存概率预测头
        self.collab_cache_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        rsu_features: torch.Tensor,
        physical_adjacency: Optional[torch.Tensor] = None,
        cache_similarity: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rsu_features: [batch_size, num_rsus, rsu_feature_dim]
            physical_adjacency: [batch_size, num_rsus, num_rsus] 物理邻接掩码
            cache_similarity: [batch_size, num_rsus, num_rsus] 缓存相似度矩阵（可选）
            edge_features: [batch_size, num_rsus, num_rsus, edge_feature_dim] 边特征
            
        Returns:
            rsu_representations: [batch_size, num_rsus, hidden_dim]
            collab_cache_probs: [batch_size, num_rsus, 1] 协同缓存概率
        """
        # 投影RSU特征
        h_rsus = F.relu(self.rsu_proj(rsu_features))
        
        # 物理邻接注意力（使用边特征）
        h_physical = self.physical_gat(h_rsus, h_rsus, edge_features, physical_adjacency)
        
        # 缓存内容相似度注意力（使用cache_similarity作为额外掩码）
        # 注：不使用边特征，因为这是基于内容的注意力
        h_content = self.content_gat(h_rsus, h_rsus, None, cache_similarity)
        
        # 融合两路注意力
        h_fused = self.fusion(torch.cat([h_physical, h_content], dim=-1))
        
        # 预测协同缓存概率
        collab_cache_probs = self.collab_cache_head(h_fused)
        
        return h_fused, collab_cache_probs


class GATRouterActor(nn.Module):
    """
    基于GAT的Actor网络
    
    替换GraphFeatureExtractor，显式建模车辆-RSU和RSU-RSU的注意力关系
    """
    
    def __init__(
        self,
        num_vehicles: int,
        num_rsus: int,
        num_uavs: int,
        vehicle_feature_dim: int = 5,
        rsu_feature_dim: int = 6,  # 🔧 修复: RSU现在6维（+cpu_frequency）
        uav_feature_dim: int = 6,  # 🔧 修复: UAV现在6维（+cpu_frequency）
        global_feature_dim: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
        edge_feature_dim: int = 8,
        central_state_dim: int = 0,
    ):
        super(GATRouterActor, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.vehicle_feature_dim = vehicle_feature_dim
        self.rsu_feature_dim = rsu_feature_dim  # 🔧 新增: 保存维度
        self.uav_feature_dim = uav_feature_dim  # 🔧 新增: 保存维度
        # 全局特征维度 = 基础全局特征 + 中央状态特征
        self.actual_global_dim = global_feature_dim + central_state_dim
        
        # 车辆-RSU注意力
        self.vehicle_rsu_attention = VehicleRSUAttention(
            vehicle_feature_dim, rsu_feature_dim, hidden_dim, num_heads, edge_feature_dim
        )
        
        # RSU-RSU协同缓存注意力
        self.rsu_rsu_attention = RSURSUCollaborativeAttention(
            rsu_feature_dim, hidden_dim, num_heads, edge_feature_dim
        )
        
        # UAV特征编码（简化处理，使用MLP）
        self.uav_encoder = nn.Sequential(
            nn.Linear(uav_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 全局特征编码（包含中央状态）
        self.global_encoder = nn.Sequential(
            nn.Linear(self.actual_global_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # vehicle + rsu + uav + global
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # 记录最后一次的协同缓存概率（供外部访问）
        self.last_collab_cache_probs = None
    
    def forward(
        self,
        state: torch.Tensor,
        adjacency_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim] 完整状态向量
            adjacency_info: 邻接信息字典（可选），包含：
                - 'vehicle_rsu_mask': [batch, num_vehicles, num_rsus]
                - 'rsu_rsu_mask': [batch, num_rsus, num_rsus]
                - 'edge_features': ...
                
        Returns:
            global_representation: [batch_size, hidden_dim] 全局表示
        """
        batch_size = state.size(0)
        
        # 解析状态向量
        idx = 0
        vehicle_features = state[:, idx:idx + self.num_vehicles * self.vehicle_feature_dim].view(batch_size, self.num_vehicles, self.vehicle_feature_dim)
        idx += self.num_vehicles * self.vehicle_feature_dim
        
        # 🔧 修复: 使用实际维度（RSU=6, UAV=6）
        rsu_features = state[:, idx:idx + self.num_rsus * self.rsu_feature_dim].view(batch_size, self.num_rsus, self.rsu_feature_dim)
        idx += self.num_rsus * self.rsu_feature_dim
        
        uav_features = state[:, idx:idx + self.num_uavs * self.uav_feature_dim].view(batch_size, self.num_uavs, self.uav_feature_dim)
        idx += self.num_uavs * self.uav_feature_dim
        
        # 假设剩余为全局特征
        global_features = state[:, idx:]
        
        # ✨ 如果没有提供邻接信息，则动态构建
        if adjacency_info is None:
            adjacency_info = self._build_adjacency_info(state)
        
        # 车辆-RSU注意力
        vehicle_repr = self.vehicle_rsu_attention(
            vehicle_features,
            rsu_features,
            edge_features=adjacency_info.get('vehicle_rsu_edge_features'),
            adjacency_mask=adjacency_info.get('vehicle_rsu_mask'),
        )
        vehicle_repr_pooled = vehicle_repr.mean(dim=1)  # [batch, hidden_dim]
        
        # RSU-RSU协同缓存注意力
        rsu_repr, collab_cache_probs = self.rsu_rsu_attention(
            rsu_features,
            physical_adjacency=adjacency_info.get('rsu_rsu_mask'),
            cache_similarity=adjacency_info.get('cache_similarity'),
            edge_features=adjacency_info.get('rsu_rsu_edge_features'),
        )
        rsu_repr_pooled = rsu_repr.mean(dim=1)  # [batch, hidden_dim]
        self.last_collab_cache_probs = collab_cache_probs  # 缓存协同缓存概率
        
        # UAV编码
        uav_repr = self.uav_encoder(uav_features).mean(dim=1)  # [batch, hidden_dim]
        
        # 全局特征编码
        global_repr = self.global_encoder(global_features)  # [batch, hidden_dim]
        
        # 融合所有表示
        fused_repr = self.final_fusion(torch.cat([
            vehicle_repr_pooled, rsu_repr_pooled, uav_repr, global_repr
        ], dim=-1))
        
        return fused_repr
    
    def get_collab_cache_probs(self) -> Optional[torch.Tensor]:
        """获取最后一次forward的协同缓存概率"""
        return self.last_collab_cache_probs
    
    def _build_adjacency_info(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从状态向量构建动态邻接信息
        
        Args:
            state: [batch_size, state_dim] 状态向量
            
        Returns:
            adjacency_info: 包含邻接掩码和边特征的字典
        """
        batch_size = state.size(0)
        device = state.device
        
        # 解析位置和负载信息
        idx = 0
        vehicle_features = state[:, idx:idx + self.num_vehicles * self.vehicle_feature_dim].view(batch_size, self.num_vehicles, self.vehicle_feature_dim)
        idx += self.num_vehicles * self.vehicle_feature_dim
        
        # 🔧 修复: 使用实际维度（RSU=6, UAV=6）
        rsu_features = state[:, idx:idx + self.num_rsus * self.rsu_feature_dim].view(batch_size, self.num_rsus, self.rsu_feature_dim)
        idx += self.num_rsus * self.rsu_feature_dim
        
        uav_features = state[:, idx:idx + self.num_uavs * self.uav_feature_dim].view(batch_size, self.num_uavs, self.uav_feature_dim)
        
        # 提取位置信息 (前2维是位置)
        vehicle_pos = vehicle_features[:, :, :2]  # [batch, num_vehicles, 2]
        rsu_pos = rsu_features[:, :, :2]  # [batch, num_rsus, 2]
        
        # 计算车辆-RSU距离矩阵
        # vehicle_pos: [batch, num_vehicles, 2] -> [batch, num_vehicles, 1, 2]
        # rsu_pos: [batch, num_rsus, 2] -> [batch, 1, num_rsus, 2]
        v_expanded = vehicle_pos.unsqueeze(2)  # [batch, num_vehicles, 1, 2]
        r_expanded = rsu_pos.unsqueeze(1)  # [batch, 1, num_rsus, 2]
        
        # 欧氏距离
        vehicle_rsu_dist = torch.sqrt(torch.sum((v_expanded - r_expanded) ** 2, dim=-1) + 1e-8)  # [batch, num_vehicles, num_rsus]
        
        # 计算RSU-RSU距离矩阵
        r_expanded_i = rsu_pos.unsqueeze(2)  # [batch, num_rsus, 1, 2]
        r_expanded_j = rsu_pos.unsqueeze(1)  # [batch, 1, num_rsus, 2]
        rsu_rsu_dist = torch.sqrt(torch.sum((r_expanded_i - r_expanded_j) ** 2, dim=-1) + 1e-8)  # [batch, num_rsus, num_rsus]
        
        # 构建邻接掩码（基于距离阈值）
        # ✨ 优化：根据信号强度动态调整覆盖范围
        vehicle_rsu_coverage = 500.0  # RSU基础覆盖范围500m
        rsu_rsu_collaboration_range = 1500.0  # RSU协作范围1500m
        
        # ✨ 动态覆盖：基于信号质量的软掩码
        # 计算信号强度权重
        signal_strength = 1.0 / (1.0 + vehicle_rsu_dist / 100.0)
        # 软掩码：信号强度 > 0.3
        vehicle_rsu_mask = signal_strength > 0.3  # [batch, num_vehicles, num_rsus]
        rsu_rsu_mask = rsu_rsu_dist <= rsu_rsu_collaboration_range  # [batch, num_rsus, num_rsus]
        
        # 构建车辆-RSU边特征 [batch, num_vehicles, num_rsus, edge_dim=8]
        # 特征包括：距离(归一化), 信号强度, RSU负载, 缓存利用率, 队列长度等
        vehicle_rsu_edge_features = torch.zeros(batch_size, self.num_vehicles, self.num_rsus, 8, device=device)
        
        # 距离归一化 (0-1)
        vehicle_rsu_edge_features[:, :, :, 0] = torch.clamp(vehicle_rsu_dist / 1000.0, 0.0, 1.0)
        
        # 信号强度估计（基于距离，简化的路径损耗模型）
        # SINR = -32.4 - 20*log10(d_km) - 20*log10(f_GHz) + tx_power + antenna_gain
        # 简化为: signal_strength = 1 / (1 + distance/100)
        vehicle_rsu_edge_features[:, :, :, 1] = 1.0 / (1.0 + vehicle_rsu_dist / 100.0)
        
        # 带宽估计（基于负载，假设均分）
        # rsu_features[:, :, 3] 是队列长度/负载
        rsu_load = rsu_features[:, :, 3].unsqueeze(1).expand(-1, self.num_vehicles, -1)  # [batch, num_vehicles, num_rsus]
        vehicle_rsu_edge_features[:, :, :, 2] = torch.clamp(1.0 - rsu_load, 0.1, 1.0)  # 可用带宽
        
        # RSU缓存利用率 (rsu_features[:, :, 2])
        rsu_cache = rsu_features[:, :, 2].unsqueeze(1).expand(-1, self.num_vehicles, -1)
        vehicle_rsu_edge_features[:, :, :, 3] = rsu_cache
        
        # RSU能耗状态 (rsu_features[:, :, 4])
        rsu_energy = rsu_features[:, :, 4].unsqueeze(1).expand(-1, self.num_vehicles, -1)
        vehicle_rsu_edge_features[:, :, :, 4] = rsu_energy
        
        # 传输延迟估计（基于距离和带宽）
        # delay = distance / speed_of_light + data_size / bandwidth
        propagation_delay = vehicle_rsu_dist / 300.0  # 归一化到ms级别
        vehicle_rsu_edge_features[:, :, :, 5] = torch.clamp(propagation_delay / 10.0, 0.0, 1.0)
        
        # 链路质量（综合指标）
        link_quality = vehicle_rsu_edge_features[:, :, :, 1] * vehicle_rsu_edge_features[:, :, :, 2]  # 信号*带宽
        vehicle_rsu_edge_features[:, :, :, 6] = link_quality
        
        # 是否在覆盖范围内（二值特征）
        vehicle_rsu_edge_features[:, :, :, 7] = vehicle_rsu_mask.float()
        
        # 构建RSU-RSU边特征 [batch, num_rsus, num_rsus, edge_dim=8]
        rsu_rsu_edge_features = torch.zeros(batch_size, self.num_rsus, self.num_rsus, 8, device=device)
        
        # 距离归一化
        rsu_rsu_edge_features[:, :, :, 0] = torch.clamp(rsu_rsu_dist / 2000.0, 0.0, 1.0)
        
        # 回传带宽（假设有线回传，带宽固定）
        rsu_rsu_edge_features[:, :, :, 1] = 0.9  # 高带宽有线回传
        
        # 负载相似度（用于缓存协作）
        rsu_load_i = rsu_features[:, :, 3].unsqueeze(2)  # [batch, num_rsus, 1]
        rsu_load_j = rsu_features[:, :, 3].unsqueeze(1)  # [batch, 1, num_rsus]
        load_diff = torch.abs(rsu_load_i - rsu_load_j)
        rsu_rsu_edge_features[:, :, :, 2] = 1.0 - torch.clamp(load_diff, 0.0, 1.0)  # 负载相似度
        
        # 缓存相似度（用于协同缓存）
        rsu_cache_i = rsu_features[:, :, 2].unsqueeze(2)
        rsu_cache_j = rsu_features[:, :, 2].unsqueeze(1)
        cache_similarity = 1.0 - torch.abs(rsu_cache_i - rsu_cache_j)
        rsu_rsu_edge_features[:, :, :, 3] = cache_similarity
        
        # 回传延迟（基于距离）
        backhaul_delay = rsu_rsu_dist / 200000.0  # 光纤速度约2e5 km/s
        rsu_rsu_edge_features[:, :, :, 4] = torch.clamp(backhaul_delay / 5.0, 0.0, 1.0)
        
        # 是否物理相邻
        rsu_rsu_edge_features[:, :, :, 5] = rsu_rsu_mask.float()
        
        # 协作潜力（综合负载差和缓存相似度）
        collaboration_potential = (rsu_rsu_edge_features[:, :, :, 2] + rsu_rsu_edge_features[:, :, :, 3]) / 2.0
        rsu_rsu_edge_features[:, :, :, 6] = collaboration_potential
        
        # 预留字段
        rsu_rsu_edge_features[:, :, :, 7] = 0.0
        
        return {
            'vehicle_rsu_mask': vehicle_rsu_mask,
            'vehicle_rsu_edge_features': vehicle_rsu_edge_features,
            'rsu_rsu_mask': rsu_rsu_mask,
            'rsu_rsu_edge_features': rsu_rsu_edge_features,
            'cache_similarity': cache_similarity,  # 用于内容相似度注意力
        }
