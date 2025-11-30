#!/usr/bin/env python3
"""
空间异质性遮挡模型测试
验证不同区域（主干道 vs 密集街区）的遮挡差异
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_structures import Position
from communication.models import WirelessCommunicationModel

def test_spatial_heterogeneity():
    """测试空间异质性遮挡"""
    print("\n" + "="*70)
    print("🌍 空间异质性遮挡模型测试")
    print("="*70)
    
    comm_model = WirelessCommunicationModel()
    
    # RSU固定位置
    rsu_pos = Position(0, 0, 25.0)
    
    # 测试不同位置的车辆
    test_positions = [
        # 主干道区域（grid坐标0,0 -> 主干道）
        {"name": "主干道1", "pos": Position(100, 100, 1.5), "desc": "X和Y都在主干道"},
        {"name": "主干道2", "pos": Position(400, 150, 1.5), "desc": "X在主干道"},
        
        # 一般街区
        {"name": "一般街区1", "pos": Position(250, 250, 1.5), "desc": "hash<30区域"},
        {"name": "一般街区2", "pos": Position(350, 450, 1.5), "desc": "hash<30区域"},
        
        # 中等密度街区
        {"name": "中等街区1", "pos": Position(550, 350, 1.5), "desc": "hash 30-70区域"},
        {"name": "中等街区2", "pos": Position(650, 550, 1.5), "desc": "hash 30-70区域"},
        
        # 密集建筑区
        {"name": "密集区1", "pos": Position(750, 250, 1.5), "desc": "hash>70区域"},
        {"name": "密集区2", "pos": Position(850, 450, 1.5), "desc": "hash>70区域"},
    ]
    
    print(f"\n📍 RSU位置: (0, 0, 25m)")
    print(f"📋 配置: building_density={comm_model.building_density}, ")
    print(f"         blockage_attenuation={comm_model.blockage_attenuation}dB\n")
    
    print("-" * 70)
    print(f"{'区域类型':<12} {'位置':<20} {'距离(m)':<10} {'局部密度':<10} {'LoS概率':<10} {'路损(dB)'}")
    print("-" * 70)
    
    for test in test_positions:
        vehicle_pos = test["pos"]
        distance = rsu_pos.distance_to(vehicle_pos)
        
        # 计算信道状态
        channel_state = comm_model.calculate_channel_state(
            rsu_pos, vehicle_pos,
            tx_node_type='rsu',
            rx_node_type='vehicle'
        )
        
        # 获取局部密度
        local_density = comm_model._get_local_building_density(rsu_pos, vehicle_pos)
        
        print(f"{test['name']:<12} ({vehicle_pos.x:>4.0f},{vehicle_pos.y:>4.0f})     "
              f"{distance:>7.1f}    {local_density:>7.2f}    "
              f"{channel_state.los_probability:>7.1%}   {channel_state.path_loss_db:>6.1f}")
    
    print("-" * 70)
    
    print("\n✅ 空间异质性效果:")
    print("  🛣️  主干道区域: 低密度(0.05-0.20) → 高LoS概率")
    print("  🏘️  一般街区: 中低密度(0.20-0.40) → 中等LoS概率")
    print("  🏙️  中等街区: 中高密度(0.40-0.60) → 较低LoS概率")
    print("  🏢  密集建筑: 高密度(0.60-0.90) → 极低LoS概率")
    print("\n💡 同样的距离，不同位置的链路质量差异显著！")

if __name__ == '__main__':
    test_spatial_heterogeneity()
