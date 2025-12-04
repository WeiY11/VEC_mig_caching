#!/usr/bin/env python3
"""
网络配置
🔧 2024-12-04 修复：与 system_config.py 保持一致
"""

from typing import Dict, Any

class NetworkConfig:
    """网络配置类"""
    
    def __init__(self):
        # 车辆配置
        self.vehicle_config = {
            'num_vehicles': 12,
            'velocity_range': (8, 15),   # 🔧 修复: m/s (与仿真器一致)
            'transmission_power': 0.1,   # W
            'computation_capacity': 100, # MIPS
            'battery_capacity': 1000     # J
        }
        
        # RSU配置
        # 🔧 2024-12-04 修复：与 system_config.py 保持一致
        self.rsu_config = {
            'num_rsus': 4,               # 🔧 修复: 与system_config一致
            'coverage_radius': 300,      # 🔧 修复: 200 → 300m
            'transmission_power': 1.0,   # W
            'computation_capacity': 1000, # MIPS
            'cache_capacity': 200,       # 🔧 修复: 100 → 200 MB
            'bandwidth': 40              # 🔧 修复: 20 → 40 MHz (与表格2一致)
        }
        
        # UAV配置
        self.uav_config = {
            'num_uavs': 2,
            'altitude': 100,             # m
            'velocity_range': (20, 50),  # m/s
            'transmission_power': 0.5,   # W
            'computation_capacity': 500, # MIPS
            'cache_capacity': 150,       # 🔧 修复: 50 → 150 MB (与CacheConfig一致)
            'battery_capacity': 5000     # J
        }
        
        # 3GPP标准通信配置
        self.communication_config = {
            'carrier_frequency': 3.5e9,  # 3.5 GHz - 3GPP NR n78频段
            'bandwidth': 40e6,           # 🔧 修复: 20e6 → 40e6 (与表格2一致)
            'thermal_noise_density': -100.0,  # 🔧 修复: -174 → -100 dBm (与表格2一致)
            'los_threshold': 50.0,       # m - 3GPP TS 38.901
            'los_decay_factor': 100.0,   # m - 3GPP标准
            'shadowing_std_los': 4.0,    # dB - 3GPP标准（LoS）
            'shadowing_std_nlos': 7.82,  # dB - 3GPP标准（NLoS）
            'antenna_gain_rsu': 15.0,    # dBi
            'antenna_gain_uav': 5.0,     # dBi
            'antenna_gain_vehicle': 3.0, # dBi
            'max_tx_power_rsu': 40.0,    # 🔧 修复: 46 → 40 dBm (10W，与表格2一致)
            'max_tx_power_uav': 23.0,    # 🔧 修复: 30 → 23 dBm (0.2W)
            'max_tx_power_vehicle': 30.0 # 🔧 修复: 23 → 30 dBm (1W，与表格2一致)
        }
        
        # 任务配置
        self.task_config = {
            'arrival_rate': 3.5,         # 🔧 修复: 0.8 → 3.5 tasks/second (高负载)
            'data_size_mean': 7.5,       # 🔧 修复: 1.0 → 7.5 MB (5-10MB范围中点)
            'computation_mean': 100,     # MIPS
            'deadline_mean': 3.5,        # 🔧 修复: 1.0 → 3.5 seconds (1-6s范围中点)
            'num_content_types': 100
        }
    
    def get_network_config(self) -> Dict[str, Any]:
        """获取完整网络配置"""
        return {
            'vehicle': self.vehicle_config,
            'rsu': self.rsu_config,
            'uav': self.uav_config,
            'communication': self.communication_config,
            'task': self.task_config
        }
    
    def update_config(self, component: str, **kwargs):
        """更新指定组件的配置"""
        config_map = {
            'vehicle': self.vehicle_config,
            'rsu': self.rsu_config,
            'uav': self.uav_config,
            'communication': self.communication_config,
            'task': self.task_config
        }
        
        if component in config_map:
            config_map[component].update(kwargs)