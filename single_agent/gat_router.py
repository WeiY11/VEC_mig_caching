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
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        edge_feature_dim: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度（每个头）
            num_heads: 注意力头数
            edge_feature_dim: 边特征维度
            dropout: Dropout率
            concat: 是否拼接多头输出（否则平均）
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.edge_feature_dim = edge_feature_dim
        self.concat = concat
        
        # 每个头的变换矩阵
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # 注意力权重参数 [num_heads, 2*out_features + edge_feature_dim]
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features + edge_feature_dim, 1))
        
        # 边特征嵌入
        self.edge_embedding = nn.Linear(edge_feature_dim, edge_feature_dim)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
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
        h_out = self.gat_layer(h_rsus, h_vehicles, edge_features, adjacency_mask)
        
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
            edge_features: [batch_size, num_rsus, num_rsus, edge_feature_dim]
            
        Returns:
            rsu_representations: [batch_size, num_rsus, hidden_dim]
            collab_cache_probs: [batch_size, num_rsus, 1] 协同缓存概率
        """
        # 投影RSU特征
        h_rsus = F.relu(self.rsu_proj(rsu_features))
        
        # 物理邻接注意力
        h_physical = self.physical_gat(h_rsus, h_rsus, edge_features, physical_adjacency)
        
        # 缓存内容相似度注意力（全连接，使用cache_similarity作为额外掩码）
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
        rsu_feature_dim: int = 5,
        uav_feature_dim: int = 5,
        global_feature_dim: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
        edge_feature_dim: int = 8,
    ):
        super(GATRouterActor, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        
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
        
        # 全局特征编码
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
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
        vehicle_features = state[:, idx:idx + self.num_vehicles * 5].view(batch_size, self.num_vehicles, 5)
        idx += self.num_vehicles * 5
        
        rsu_features = state[:, idx:idx + self.num_rsus * 5].view(batch_size, self.num_rsus, 5)
        idx += self.num_rsus * 5
        
        uav_features = state[:, idx:idx + self.num_uavs * 5].view(batch_size, self.num_uavs, 5)
        idx += self.num_uavs * 5
        
        # 假设剩余为全局特征
        global_features = state[:, idx:]
        
        # 车辆-RSU注意力
        vehicle_repr = self.vehicle_rsu_attention(
            vehicle_features,
            rsu_features,
            edge_features=adjacency_info.get('vehicle_rsu_edge_features') if adjacency_info else None,
            adjacency_mask=adjacency_info.get('vehicle_rsu_mask') if adjacency_info else None,
        )
        vehicle_repr_pooled = vehicle_repr.mean(dim=1)  # [batch, hidden_dim]
        
        # RSU-RSU协同缓存注意力
        rsu_repr, collab_cache_probs = self.rsu_rsu_attention(
            rsu_features,
            physical_adjacency=adjacency_info.get('rsu_rsu_mask') if adjacency_info else None,
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
