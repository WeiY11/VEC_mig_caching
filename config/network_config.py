#!/usr/bin/env python3
"""
网络配置
"""

from typing import Dict, Any

class NetworkConfig:
    """网络配置类"""
    
    def __init__(self):
        # 车辆配置
        self.vehicle_config = {
            'num_vehicles': 12,
            'velocity_range': (10, 30),  # m/s
            'transmission_power': 0.1,   # W
            'computation_capacity': 100, # MIPS
            'battery_capacity': 1000     # J
        }
        
        # RSU配置
        self.rsu_config = {
            'num_rsus': 6,
            'coverage_radius': 200,      # m
            'transmission_power': 1.0,   # W
            'computation_capacity': 1000, # MIPS
            'cache_capacity': 100,       # MB
            'bandwidth': 20              # MHz
        }
        
        # UAV配置
        self.uav_config = {
            'num_uavs': 2,
            'altitude': 100,             # m
            'velocity_range': (20, 50),  # m/s
            'transmission_power': 0.5,   # W
            'computation_capacity': 500, # MIPS
            'cache_capacity': 50,        # MB
            'battery_capacity': 5000     # J
        }
        
        # 通信配置
        self.communication_config = {
            'frequency': 2.4,            # GHz
            'bandwidth': 20,             # MHz
            'noise_power': -100,         # dBm
            'path_loss_exponent': 2.5,
            'shadowing_std': 8           # dB
        }
        
        # 任务配置
        self.task_config = {
            'arrival_rate': 0.8,         # tasks/second
            'data_size_mean': 1.0,       # MB
            'computation_mean': 100,     # MIPS
            'deadline_mean': 1.0,        # seconds
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