#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
有线回传网络模型
RSU间通过有线网络进行通信和任务迁移
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WiredBackhaulConfig:
    """有线回传网络配置"""
    # 网络参数
    fiber_capacity_gbps: float = 10.0        # 光纤容量 10Gbps
    ethernet_capacity_mbps: float = 1000.0   # 以太网容量 1Gbps
    
    # 延迟参数 (基于真实网络测量)
    fiber_propagation_delay_per_km: float = 5.0e-6  # 光纤传播延迟 5μs/km
    switch_processing_delay: float = 0.001           # 交换机处理延迟 1ms
    router_processing_delay: float = 0.002           # 路由器处理延迟 2ms
    
    # 能耗参数
    switch_power_w: float = 50.0             # 交换机功耗 50W
    router_power_w: float = 100.0            # 路由器功耗 100W
    fiber_repeater_power_w: float = 30.0     # 光纤中继器功耗 30W
    
    # 拓扑参数
    max_hops: int = 3                        # 最大跳数
    redundancy_factor: float = 1.2           # 冗余系数


class WiredBackhaulModel:
    """🔌 有线回传网络模型"""
    
    def __init__(self, config: WiredBackhaulConfig = None):
        self.config = config or WiredBackhaulConfig()
        
        # 🌐 构建RSU回传网络拓扑
        self.backhaul_topology = self._build_backhaul_topology()
        
        # 📊 网络状态监控
        self.network_stats = {
            'total_transmissions': 0,
            'total_data_transferred': 0.0,  # MB
            'total_energy_consumed': 0.0,   # J
            'avg_delay': 0.0,
            'link_utilization': {},         # 每条链路的利用率
            'congestion_events': 0
        }
        
        print("🔌 有线回传网络模型初始化完成")
    
    def _build_backhaul_topology(self) -> Dict[str, Dict]:
        """
        🌐 构建RSU回传网络拓扑
        
        典型城市RSU回传拓扑：
        - 中央RSU(RSU_2) 作为汇聚点，直连核心路由器
        - 其他RSU通过光纤环网或星型网络连接到中央RSU
        """
        topology = {
            # 中央RSU作为网络汇聚点
            'RSU_2': {
                'type': 'central_hub',
                'connected_to': ['core_router'],
                'fiber_links': {
                    'RSU_0': {'distance_km': 0.8, 'capacity_gbps': 10.0, 'hops': 1},
                    'RSU_1': {'distance_km': 1.2, 'capacity_gbps': 10.0, 'hops': 1},
                    'RSU_3': {'distance_km': 0.9, 'capacity_gbps': 10.0, 'hops': 1},
                    'RSU_4': {'distance_km': 1.1, 'capacity_gbps': 10.0, 'hops': 1},
                    'RSU_5': {'distance_km': 0.5, 'capacity_gbps': 10.0, 'hops': 1},
                }
            },
            
            # 接入RSU - 星型拓扑连接到中央RSU
            'RSU_0': {
                'type': 'access_node',
                'connected_to': ['RSU_2'],
                'fiber_links': {
                    'RSU_2': {'distance_km': 0.8, 'capacity_gbps': 10.0, 'hops': 1}
                }
            },
            'RSU_1': {
                'type': 'access_node', 
                'connected_to': ['RSU_2'],
                'fiber_links': {
                    'RSU_2': {'distance_km': 1.2, 'capacity_gbps': 10.0, 'hops': 1}
                }
            },
            'RSU_3': {
                'type': 'access_node',
                'connected_to': ['RSU_2'],
                'fiber_links': {
                    'RSU_2': {'distance_km': 0.9, 'capacity_gbps': 10.0, 'hops': 1}
                }
            },
            'RSU_4': {
                'type': 'access_node',
                'connected_to': ['RSU_2'],
                'fiber_links': {
                    'RSU_2': {'distance_km': 1.1, 'capacity_gbps': 10.0, 'hops': 1}
                }
            },
            'RSU_5': {
                'type': 'access_node',
                'connected_to': ['RSU_2'],
                'fiber_links': {
                    'RSU_2': {'distance_km': 0.5, 'capacity_gbps': 10.0, 'hops': 1}
                }
            }
        }
        
        return topology
    
    def calculate_wired_transmission_delay(self, data_size_mb: float, 
                                         source_rsu: str, target_rsu: str) -> Tuple[float, Dict]:
        """
        🔌 计算RSU间有线传输延迟
        
        Args:
            data_size_mb: 数据大小 (MB)
            source_rsu: 源RSU ID
            target_rsu: 目标RSU ID
            
        Returns:
            Tuple[传输延迟(秒), 详细信息]
        """
        # 🔍 查找最佳路径
        path_info = self._find_optimal_path(source_rsu, target_rsu)
        
        if not path_info:
            return float('inf'), {'error': 'No path found'}
        
        # 📡 计算各部分延迟
        
        # 1. 传播延迟 (光纤中光速传播)
        total_distance_km = path_info['total_distance']
        propagation_delay = total_distance_km * self.config.fiber_propagation_delay_per_km
        
        # 2. 网络设备处理延迟
        num_hops = path_info['hops']
        device_processing_delay = (
            num_hops * self.config.switch_processing_delay +
            (num_hops - 1) * self.config.router_processing_delay
        )
        
        # 3. 传输延迟 (基于可用带宽)
        available_bandwidth_mbps = path_info['min_capacity'] * 1000  # Gbps转Mbps
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_delay = data_size_bits / (available_bandwidth_mbps * 1e6)  # 传输时间
        
        # 4. 队列延迟 (基于网络拥塞)
        congestion_factor = self._calculate_congestion_factor(path_info['path'])
        queuing_delay = transmission_delay * congestion_factor
        
        # 🔗 总延迟
        total_delay = propagation_delay + device_processing_delay + transmission_delay + queuing_delay
        
        # 📊 详细信息
        details = {
            'path': path_info['path'],
            'total_distance_km': total_distance_km,
            'hops': num_hops,
            'available_bandwidth_mbps': available_bandwidth_mbps,
            'propagation_delay': propagation_delay,
            'device_processing_delay': device_processing_delay,
            'transmission_delay': transmission_delay,
            'queuing_delay': queuing_delay,
            'total_delay': total_delay,
            'congestion_factor': congestion_factor
        }
        
        # 更新统计
        self.network_stats['total_transmissions'] += 1
        self.network_stats['total_data_transferred'] += data_size_mb
        
        return total_delay, details
    
    def calculate_wired_transmission_energy(self, data_size_mb: float, 
                                          source_rsu: str, target_rsu: str,
                                          transmission_time: float) -> Tuple[float, Dict]:
        """
        ⚡ 计算有线传输能耗
        
        Args:
            data_size_mb: 数据大小
            source_rsu: 源RSU
            target_rsu: 目标RSU
            transmission_time: 传输时间
            
        Returns:
            Tuple[能耗(焦耳), 详细信息]
        """
        path_info = self._find_optimal_path(source_rsu, target_rsu)
        
        if not path_info:
            return 0.0, {'error': 'No path found'}
        
        # 🔌 网络设备能耗计算
        
        # 1. 交换机能耗 (固定功耗 * 传输时间)
        switch_energy = self.config.switch_power_w * transmission_time * path_info['hops']
        
        # 2. 路由器能耗 (仅中央节点有路由器)
        router_energy = self.config.router_power_w * transmission_time
        
        # 3. 光纤设备能耗 (中继器等)
        fiber_distance = path_info['total_distance']
        # 每10km需要一个光纤中继器
        num_repeaters = max(1, int(fiber_distance / 10))
        repeater_energy = self.config.fiber_repeater_power_w * transmission_time * num_repeaters
        
        # 🔋 总能耗
        total_energy = switch_energy + router_energy + repeater_energy
        
        # 📊 详细分解
        energy_breakdown = {
            'switch_energy': switch_energy,
            'router_energy': router_energy,
            'repeater_energy': repeater_energy,
            'total_energy': total_energy,
            'num_switches': path_info['hops'],
            'num_repeaters': num_repeaters,
            'transmission_time': transmission_time
        }
        
        # 更新统计
        self.network_stats['total_energy_consumed'] += total_energy
        
        return total_energy, energy_breakdown
    
    def _find_optimal_path(self, source: str, target: str) -> Optional[Dict]:
        """
        🛣️ 查找最优路径
        
        Args:
            source: 源RSU ID
            target: 目标RSU ID
            
        Returns:
            路径信息字典或None
        """
        if source == target:
            return None
        
        if source not in self.backhaul_topology or target not in self.backhaul_topology:
            return None
        
        # 🌟 简化路径算法：由于采用星型拓扑，大部分路径都经过中央RSU
        
        # 1️⃣ 直连情况
        source_links = self.backhaul_topology[source].get('fiber_links', {})
        if target in source_links:
            link_info = source_links[target]
            return {
                'path': [source, target],
                'total_distance': link_info['distance_km'],
                'min_capacity': link_info['capacity_gbps'],
                'hops': link_info['hops']
            }
        
        # 2️⃣ 通过中央RSU中转
        central_rsu = 'RSU_2'
        if source != central_rsu and target != central_rsu:
            # source -> central -> target
            if (central_rsu in source_links and 
                target in self.backhaul_topology[central_rsu].get('fiber_links', {})):
                
                source_to_central = source_links[central_rsu]
                central_to_target = self.backhaul_topology[central_rsu]['fiber_links'][target]
                
                return {
                    'path': [source, central_rsu, target],
                    'total_distance': source_to_central['distance_km'] + central_to_target['distance_km'],
                    'min_capacity': min(source_to_central['capacity_gbps'], central_to_target['capacity_gbps']),
                    'hops': source_to_central['hops'] + central_to_target['hops']
                }
        
        # 3️⃣ 从中央RSU到其他RSU
        elif source == central_rsu:
            central_links = self.backhaul_topology[central_rsu].get('fiber_links', {})
            if target in central_links:
                link_info = central_links[target]
                return {
                    'path': [source, target],
                    'total_distance': link_info['distance_km'],
                    'min_capacity': link_info['capacity_gbps'],
                    'hops': link_info['hops']
                }
        
        # 4️⃣ 其他RSU到中央RSU
        elif target == central_rsu:
            if central_rsu in source_links:
                link_info = source_links[central_rsu]
                return {
                    'path': [source, target],
                    'total_distance': link_info['distance_km'],
                    'min_capacity': link_info['capacity_gbps'], 
                    'hops': link_info['hops']
                }
        
        return None
    
    def _calculate_congestion_factor(self, path: List[str]) -> float:
        """
        📊 计算网络拥塞因子
        
        Args:
            path: 网络路径
            
        Returns:
            拥塞因子 (1.0 = 无拥塞, >1.0 = 有拥塞)
        """
        # 🔍 基于网络使用历史计算拥塞
        base_congestion = 1.0
        
        # 考虑路径长度：跳数越多，拥塞可能性越大
        hop_penalty = len(path) * 0.1
        
        # 考虑时间变化：模拟网络流量波动
        import time
        time_factor = 1.0 + 0.1 * np.sin(time.time() * 0.1)  # 周期性波动
        
        # 考虑随机网络抖动
        random_jitter = np.random.uniform(0.95, 1.05)
        
        congestion_factor = base_congestion + hop_penalty
        congestion_factor *= time_factor * random_jitter
        
        # 限制在合理范围
        return np.clip(congestion_factor, 1.0, 2.0)
    
    def estimate_migration_cost(self, data_size_mb: float, 
                              source_rsu: str, target_rsu: str) -> Dict:
        """
        💰 估算任务迁移成本
        
        Args:
            data_size_mb: 迁移数据大小
            source_rsu: 源RSU
            target_rsu: 目标RSU
            
        Returns:
            迁移成本信息
        """
        # 计算传输延迟和能耗
        delay, delay_details = self.calculate_wired_transmission_delay(
            data_size_mb, source_rsu, target_rsu
        )
        energy, energy_details = self.calculate_wired_transmission_energy(
            data_size_mb, source_rsu, target_rsu, delay
        )
        
        # 🎯 综合成本评估
        # 延迟成本：每毫秒延迟的成本
        delay_cost = delay * 1000 * 0.1  # 0.1 cost units per ms
        
        # 能耗成本：每焦耳能耗的成本
        energy_cost = energy * 0.001     # 0.001 cost units per joule
        
        # 网络使用成本：基于数据量
        bandwidth_cost = data_size_mb * 0.01  # 0.01 cost units per MB
        
        total_cost = delay_cost + energy_cost + bandwidth_cost
        
        return {
            'total_cost': total_cost,
            'delay_cost': delay_cost,
            'energy_cost': energy_cost,
            'bandwidth_cost': bandwidth_cost,
            'transmission_delay': delay,
            'energy_consumption': energy,
            'path_info': delay_details,
            'energy_breakdown': energy_details
        }
    
    def get_backhaul_status(self) -> Dict:
        """📊 获取回传网络状态"""
        # 计算链路利用率
        total_capacity = sum(
            sum(link['capacity_gbps'] for link in node_info.get('fiber_links', {}).values())
            for node_info in self.backhaul_topology.values()
        ) / 2  # 避免双重计算
        
        utilization = self.network_stats['total_data_transferred'] / (total_capacity * 1000)  # GB
        
        status = {
            'network_topology': 'star_with_central_hub',
            'central_hub': 'RSU_2',
            'total_links': len([
                link for node_info in self.backhaul_topology.values()
                for link in node_info.get('fiber_links', {}).keys()
            ]) // 2,
            'total_capacity_gbps': total_capacity,
            'network_utilization': min(1.0, utilization),
            'statistics': self.network_stats.copy(),
            'congestion_level': 'low' if utilization < 0.3 else 'medium' if utilization < 0.7 else 'high'
        }
        
        return status


# ==================== 全局接口 ====================

# 全局有线回传模型实例
_global_backhaul_model = None

def get_backhaul_model() -> WiredBackhaulModel:
    """获取全局有线回传模型实例"""
    global _global_backhaul_model
    if _global_backhaul_model is None:
        _global_backhaul_model = WiredBackhaulModel()
    return _global_backhaul_model

def calculate_rsu_to_rsu_delay(data_size_mb: float, source_rsu: str, target_rsu: str) -> float:
    """🔌 简化接口：计算RSU间有线传输延迟"""
    model = get_backhaul_model()
    delay, _ = model.calculate_wired_transmission_delay(data_size_mb, source_rsu, target_rsu)
    return delay

def calculate_rsu_to_rsu_energy(data_size_mb: float, source_rsu: str, target_rsu: str, 
                               transmission_time: float) -> float:
    """⚡ 简化接口：计算RSU间有线传输能耗"""
    model = get_backhaul_model()
    energy, _ = model.calculate_wired_transmission_energy(
        data_size_mb, source_rsu, target_rsu, transmission_time
    )
    return energy


if __name__ == "__main__":
    # 🧪 测试有线回传网络模型
    print("🧪 测试有线回传网络模型")
    print("=" * 50)
    
    model = WiredBackhaulModel()
    
    # 测试不同RSU间的传输
    test_cases = [
        ("RSU_0", "RSU_1", 5.0),   # 通过中央RSU中转
        ("RSU_2", "RSU_3", 3.0),   # 中央RSU直连
        ("RSU_4", "RSU_5", 2.0),   # 通过中央RSU中转
    ]
    
    for source, target, data_size in test_cases:
        delay, delay_info = model.calculate_wired_transmission_delay(data_size, source, target)
        energy, energy_info = model.calculate_wired_transmission_energy(data_size, source, target, delay)
        cost_info = model.estimate_migration_cost(data_size, source, target)
        
        print(f"\n🔌 {source} → {target} ({data_size}MB):")
        print(f"   📡 路径: {' → '.join(delay_info.get('path', []))}")
        print(f"   ⏱️ 延迟: {delay*1000:.2f}ms")
        print(f"   ⚡ 能耗: {energy:.3f}J")
        print(f"   💰 总成本: {cost_info['total_cost']:.3f}")
    
    # 网络状态
    status = model.get_backhaul_status()
    print(f"\n📊 回传网络状态:")
    print(f"   🌐 拓扑类型: {status['network_topology']}")
    print(f"   🏢 中央节点: {status['central_hub']}")
    print(f"   🔗 总链路数: {status['total_links']}")
    print(f"   📈 网络利用率: {status['network_utilization']:.1%}")
    print(f"   🚥 拥塞水平: {status['congestion_level']}")
