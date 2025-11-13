#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEC系统外部配置文件
允许运行时调整关键参数，提高系统灵活性
"""

import json
import os
from typing import Dict, Any

# 默认配置参数
DEFAULT_CONFIG = {
    "time_settings": {
        "time_slot_duration": 0.1,  # seconds - 100 ms slot length
        "simulation_time": 1000     # seconds
    },
    
    "task_generation": {
        "arrival_rate": 3.0,        # tasks/second - 🔧 优化: 3.0 tasks/s/vehicle (高负载但不极端)
        "data_size_range": [0.5e6/8, 15e6/8],  # 🔧 恢复: 0.5-15 Mbits = 0.0625-1.875 MB
        "compute_density": 100,     # cycles/bit - 🔧 优化：适度提高（视频处理级别）
        "deadline_range": [0.3, 0.9],  # seconds - 3-9 slots @100 ms
        "output_ratio": 0.05         # 输出大小比例
    },
    
    "network_topology": {
        "num_vehicles": 30,         # 增加车辆密度
        "num_rsus": 8,             # 增加RSU数量
        "num_uavs": 3,             # 增加UAV数量
        "area_width": 3000,        # meters - 缩小区域提高密度
        "area_height": 3000,       # meters
        "rsu_coverage_radius": 400  # meters
    },
    
    "compute_resources": {
        "vehicle_cpu_freq_range": [1.5e9, 3.5e9],  # 1.5-3.5 GHz
        "rsu_cpu_freq_range": [3e9, 6e9],          # 3-6 GHz  
        "uav_cpu_freq_range": [1.5e9, 2.5e9],     # 1.5-2.5 GHz
        "parallel_efficiency": 0.85                 # 提高并行效率
    },
    
    "communication": {
        "total_bandwidth": 40e6,    # 40 MHz - 增加带宽
        "vehicle_tx_power": 25,     # dBm - 略增加发射功率
        "rsu_tx_power": 33,        # dBm
        "uav_tx_power": 23         # dBm
    },
    
    "migration_parameters": {
        "migration_threshold": 0.75,        # 降低迁移阈值
        "rsu_overload_threshold": 0.85,     # RSU过载阈值
        "uav_overload_threshold": 0.8,      # UAV过载阈值
        "cooldown_period": 8.0,             # seconds - 缩短冷却期
        "max_migration_distance": 800       # meters - 减少最大迁移距离
    },
    
    "cache_settings": {
        "vehicle_cache_capacity": 2e9,      # 2 GB - 增加缓存容量
        "rsu_cache_capacity": 20e9,        # 20 GB
        "uav_cache_capacity": 4e9,         # 4 GB
        "cache_hit_threshold": 0.85,       # 提高缓存命中阈值
        "prediction_window": 15            # 增加预测窗口
    },
    
    "performance_optimization": {
        "enable_adaptive_scheduling": True,
        "enable_load_balancing": True,
        "enable_energy_optimization": True,
        "batch_size_optimization": True,
        "parallel_environments": 8
    }
}

class ExternalConfigManager:
    """外部配置管理器"""
    
    def __init__(self, config_file: str = "vec_system_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # 合并默认配置和加载的配置
                return self._merge_configs(DEFAULT_CONFIG, loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ 配置文件加载失败: {e}, 使用默认配置")
                return DEFAULT_CONFIG.copy()
        else:
            # 创建默认配置文件
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self, config: Dict[str, Any] | None = None):
        """保存配置到文件"""
        config_to_save = config if config is not None else self.config
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, ensure_ascii=False)
            print(f"✅ 配置已保存到 {self.config_file}")
        except IOError as e:
            print(f"❌ 配置保存失败: {e}")
    
    def get(self, *keys):
        """获取配置值 (支持嵌套访问)"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def set(self, *keys, value):
        """设置配置值 (支持嵌套设置)"""
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config()
    
    def update_from_system_config(self, system_config):
        """从系统配置对象更新外部配置"""
        # 时间设置
        self.set("time_settings", "time_slot_duration", value=system_config.network.time_slot_duration)
        self.set("time_settings", "simulation_time", value=system_config.simulation_time)
        
        # 任务生成参数
        self.set("task_generation", "arrival_rate", value=system_config.task.arrival_rate)
        self.set("task_generation", "data_size_range", value=list(system_config.task.data_size_range))
        self.set("task_generation", "compute_density", value=system_config.task.task_compute_density)
        
        # 网络拓扑
        self.set("network_topology", "num_vehicles", value=system_config.network.num_vehicles)
        self.set("network_topology", "num_rsus", value=system_config.network.num_rsus)
        self.set("network_topology", "num_uavs", value=system_config.network.num_uavs)
        
        print("✅ 外部配置已从系统配置更新")
    
    def apply_to_system_config(self, system_config):
        """将外部配置应用到系统配置对象"""
        # 时间设置
        if self.get("time_settings", "time_slot_duration"):
            system_config.network.time_slot_duration = self.get("time_settings", "time_slot_duration")
            system_config.time_slot = self.get("time_settings", "time_slot_duration")
        
        # 任务生成参数
        if self.get("task_generation", "arrival_rate"):
            system_config.task.arrival_rate = self.get("task_generation", "arrival_rate")
        
        data_range_raw = self.get("task_generation", "data_size_range")
        if data_range_raw and isinstance(data_range_raw, list):
            system_config.task.data_size_range = tuple(data_range_raw)
            system_config.task.task_data_size_range = system_config.task.data_size_range
        
        if self.get("task_generation", "compute_density"):
            system_config.task.task_compute_density = self.get("task_generation", "compute_density")
        
        # 网络拓扑
        if self.get("network_topology", "num_vehicles"):
            system_config.network.num_vehicles = self.get("network_topology", "num_vehicles")
        
        if self.get("network_topology", "num_rsus"):
            system_config.network.num_rsus = self.get("network_topology", "num_rsus")
        
        if self.get("network_topology", "num_uavs"):
            system_config.network.num_uavs = self.get("network_topology", "num_uavs")
        
        # 计算资源
        if self.get("compute_resources", "parallel_efficiency"):
            system_config.compute.parallel_efficiency = self.get("compute_resources", "parallel_efficiency")
        
        # 通信参数
        if self.get("communication", "total_bandwidth"):
            system_config.communication.total_bandwidth = self.get("communication", "total_bandwidth")
        
        print("✅ 系统配置已从外部配置更新")
    
    def validate_config(self) -> bool:
        """验证配置的合理性"""
        print("🔍 验证外部配置合理性...")
        
        valid = True
        
        # 验证时隙长度
        time_slot = self.get("time_settings", "time_slot_duration")
        if not time_slot or not isinstance(time_slot, (int, float)) or time_slot < 0.1 or time_slot > 1.0:
            print(f"❌ 时隙长度不合理: {time_slot}s (应为0.1-1.0s)")
            valid = False
        
        # 验证任务到达率
        arrival_rate = self.get("task_generation", "arrival_rate")
        if not arrival_rate or not isinstance(arrival_rate, (int, float)) or arrival_rate < 0.1 or arrival_rate > 10.0:
            print(f"❌ 任务到达率不合理: {arrival_rate} tasks/s (应为0.1-10.0)")
            valid = False
        
        # 验证数据大小范围
        data_range = self.get("task_generation", "data_size_range")
        if (not data_range or not isinstance(data_range, list) or len(data_range) != 2 or
            data_range[0] >= data_range[1] or data_range[0] < 1e4):
            print(f"❌ 数据大小范围不合理: {data_range}")
            valid = False
        
        # 验证网络规模
        vehicles = self.get("network_topology", "num_vehicles")
        rsus = self.get("network_topology", "num_rsus")
        if (not vehicles or not isinstance(vehicles, int) or vehicles < 1 or 
            not rsus or not isinstance(rsus, int) or rsus < 1):
            print(f"❌ 网络规模不合理: {vehicles}车辆, {rsus}RSU")
            valid = False
        
        if valid:
            print("✅ 外部配置验证通过")
        
        return valid
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 当前配置摘要:")
        print(f"   时隙长度: {self.get('time_settings', 'time_slot_duration')}s")
        print(f"   任务到达率: {self.get('task_generation', 'arrival_rate')} tasks/s")
        
        data_range = self.get('task_generation', 'data_size_range')
        if data_range and isinstance(data_range, list) and len(data_range) >= 2:
            print(f"   数据大小: {data_range[0]/1e6:.1f}-{data_range[1]/1e6:.1f}MB")
        
        print(f"   网络规模: {self.get('network_topology', 'num_vehicles')}车辆 + {self.get('network_topology', 'num_rsus')}RSU + {self.get('network_topology', 'num_uavs')}UAV")
        
        bandwidth = self.get('communication', 'total_bandwidth')
        if bandwidth and isinstance(bandwidth, (int, float)):
            print(f"   总带宽: {bandwidth/1e6:.0f}MHz")

# 全局配置管理器实例
external_config = ExternalConfigManager()

def apply_external_config_to_system():
    """将外部配置应用到系统配置"""
    from config.system_config import config
    external_config.apply_to_system_config(config)
    return config
