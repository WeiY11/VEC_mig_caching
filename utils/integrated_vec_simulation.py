#!/usr/bin/env python3
"""
整合VEC仿真系统
结合realistic缓存、用户行为模式和地理移动性
"""

import numpy as np
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
import math

# 导入之前实现的组件
from realistic_vec_cache import RealisticVECCacheSimulator, VECContent, ContentType
from user_behavior_patterns import TimeBasedBehaviorModel, UserProfile, UserType

@dataclass
class VehicleState:
    """车辆状态"""
    vehicle_id: str
    current_position: Tuple[float, float]
    speed: float  # km/h
    direction: float  # 度
    route_progress: float  # 路线进度 0-1
    current_route: List[Tuple[float, float]]
    user_profile: UserProfile
    last_request_time: float
    request_frequency: float  # 每分钟请求次数

@dataclass
class RSUState:
    """RSU状态"""
    rsu_id: str
    position: Tuple[float, float]
    coverage_radius: float  # 覆盖半径(米)
    cache_simulator: RealisticVECCacheSimulator
    connected_vehicles: Set[str]
    load_factor: float
    last_update_time: float

class IntegratedVECSimulation:
    """整合VEC仿真系统"""
    
    def __init__(self, num_vehicles: int = 50, num_rsus: int = 8):
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        
        # 初始化组件
        self.behavior_model = TimeBasedBehaviorModel()
        self.vehicles: Dict[str, VehicleState] = {}
        self.rsus: Dict[str, RSUState] = {}
        
        # 仿真参数
        self.simulation_start_time = time.time()
        self.current_simulation_time = datetime.now()
        self.time_acceleration = 60  # 1秒仿真时间 = 60秒现实时间
        
        # 统计数据
        self.global_stats = {
            'total_requests': 0,
            'total_cache_hits': 0,
            'total_handovers': 0,
            'content_type_requests': defaultdict(int),
            'hourly_patterns': defaultdict(lambda: defaultdict(int)),
            'user_type_behaviors': defaultdict(lambda: defaultdict(int)),
            'geographic_hotspots': defaultdict(int)
        }
        
        # 初始化仿真环境
        self._setup_simulation()
    
    def _setup_simulation(self):
        """设置仿真环境"""
        print("🚀 初始化整合VEC仿真系统...")
        
        # 生成用户画像
        user_profiles = self.behavior_model.generate_realistic_user_profiles(self.num_vehicles)
        
        # 创建车辆
        for i, profile in enumerate(user_profiles):
            vehicle_id = f"vehicle_{i:03d}"
            
            # 初始位置（在用户家附近）
            start_lat = profile.home_location[0] + np.random.uniform(-0.01, 0.01)
            start_lon = profile.home_location[1] + np.random.uniform(-0.01, 0.01)
            
            # 选择初始路线
            initial_route = self._select_initial_route(profile)
            
            vehicle = VehicleState(
                vehicle_id=vehicle_id,
                current_position=(start_lat, start_lon),
                speed=np.random.uniform(20, 80),  # 20-80 km/h
                direction=np.random.uniform(0, 360),
                route_progress=0.0,
                current_route=initial_route,
                user_profile=profile,
                last_request_time=0.0,
                request_frequency=self._calculate_request_frequency(profile.user_type)
            )
            
            self.vehicles[vehicle_id] = vehicle
        
        # 创建RSU
        for i in range(self.num_rsus):
            rsu_id = f"rsu_{i:03d}"
            
            # RSU位置（分布在城市区域）
            rsu_lat = 39.9042 + np.random.uniform(-0.1, 0.1)
            rsu_lon = 116.4074 + np.random.uniform(-0.1, 0.1)
            
            # 创建缓存仿真器
            cache_sim = RealisticVECCacheSimulator(rsu_id, "rsu", (rsu_lat, rsu_lon))
            
            rsu = RSUState(
                rsu_id=rsu_id,
                position=(rsu_lat, rsu_lon),
                coverage_radius=2000.0,  # 2km覆盖半径
                cache_simulator=cache_sim,
                connected_vehicles=set(),
                load_factor=0.0,
                last_update_time=time.time()
            )
            
            self.rsus[rsu_id] = rsu
        
        print(f"✅ 创建了 {len(self.vehicles)} 个车辆和 {len(self.rsus)} 个RSU")
    
    def _select_initial_route(self, profile: UserProfile) -> List[Tuple[float, float]]:
        """选择初始路线"""
        if profile.frequent_routes:
            return random.choice(profile.frequent_routes)
        else:
            # 默认家-工作路线
            return [profile.home_location, profile.work_location]
    
    def _calculate_request_frequency(self, user_type: UserType) -> float:
        """计算请求频率（每分钟）"""
        frequency_map = {
            UserType.DELIVERY: 3.0,      # 配送司机请求频繁
            UserType.TAXI_DRIVER: 2.5,   # 出租车司机
            UserType.BUSINESS: 2.0,      # 商务人士
            UserType.COMMUTER: 1.5,      # 通勤族
            UserType.TOURIST: 1.2,       # 游客
            UserType.STUDENT: 1.0,       # 学生
            UserType.LEISURE: 0.8        # 休闲用户
        }
        return frequency_map.get(user_type, 1.0)
    
    def step_simulation(self, time_delta_seconds: float = 10.0):
        """推进仿真一步"""
        # 更新仿真时间
        real_time_delta = timedelta(seconds=time_delta_seconds * self.time_acceleration)
        self.current_simulation_time += real_time_delta
        current_time = time.time()
        
        # 更新车辆状态
        for vehicle in self.vehicles.values():
            self._update_vehicle_state(vehicle, time_delta_seconds)
        
        # 更新RSU连接
        self._update_rsu_connections()
        
        # 处理内容请求
        self._process_content_requests()
        
        # 更新缓存状态
        self._update_cache_states()
        
        # 收集统计数据
        self._collect_statistics()
    
    def _update_vehicle_state(self, vehicle: VehicleState, time_delta: float):
        """更新车辆状态"""
        if not vehicle.current_route or len(vehicle.current_route) < 2:
            return
        
        # 计算移动距离
        distance_km = vehicle.speed * (time_delta / 3600)  # 转换为公里
        
        # 更新路线进度
        route_length = self._calculate_route_length(vehicle.current_route)
        if route_length > 0:
            progress_delta = distance_km / route_length
            vehicle.route_progress = min(1.0, vehicle.route_progress + progress_delta)
        
        # 更新位置
        new_position = self._interpolate_position_on_route(
            vehicle.current_route, vehicle.route_progress
        )
        vehicle.current_position = new_position
        
        # 检查是否需要新路线
        if vehicle.route_progress >= 1.0:
            self._assign_new_route(vehicle)
    
    def _calculate_route_length(self, route: List[Tuple[float, float]]) -> float:
        """计算路线长度（公里）"""
        total_length = 0.0
        for i in range(len(route) - 1):
            lat1, lon1 = route[i]
            lat2, lon2 = route[i + 1]
            # 简化距离计算
            distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # 约111km/度
            total_length += distance
        return total_length
    
    def _interpolate_position_on_route(self, route: List[Tuple[float, float]], 
                                     progress: float) -> Tuple[float, float]:
        """在路线上插值位置"""
        if len(route) < 2:
            return route[0] if route else (0, 0)
        
        if progress <= 0:
            return route[0]
        if progress >= 1:
            return route[-1]
        
        # 计算在哪个路段
        total_segments = len(route) - 1
        segment_progress = progress * total_segments
        segment_index = int(segment_progress)
        local_progress = segment_progress - segment_index
        
        if segment_index >= total_segments:
            return route[-1]
        
        # 在当前路段内插值
        start_pos = route[segment_index]
        end_pos = route[segment_index + 1]
        
        lat = start_pos[0] + local_progress * (end_pos[0] - start_pos[0])
        lon = start_pos[1] + local_progress * (end_pos[1] - start_pos[1])
        
        return (lat, lon)
    
    def _assign_new_route(self, vehicle: VehicleState):
        """为车辆分配新路线"""
        profile = vehicle.user_profile
        current_hour = self.current_simulation_time.hour
        
        # 根据时间和用户类型选择路线
        if profile.frequent_routes:
            # 在早晚高峰优先选择通勤路线
            if 7 <= current_hour <= 9:
                # 早高峰：家到工作
                route = [profile.home_location, profile.work_location]
            elif 17 <= current_hour <= 19:
                # 晚高峰：工作到家
                route = [profile.work_location, profile.home_location]
            else:
                # 其他时间随机选择
                route = random.choice(profile.frequent_routes)
        else:
            # 默认往返路线
            route = [profile.home_location, profile.work_location]
        
        vehicle.current_route = route
        vehicle.route_progress = 0.0
        
        # 根据路线调整速度
        if profile.user_type in [UserType.DELIVERY, UserType.TAXI_DRIVER]:
            vehicle.speed = np.random.uniform(30, 60)  # 职业司机稍快
        else:
            vehicle.speed = np.random.uniform(20, 80)  # 普通用户
    
    def _update_rsu_connections(self):
        """更新RSU连接状态"""
        # 清空所有连接
        for rsu in self.rsus.values():
            rsu.connected_vehicles.clear()
        
        # 重新计算连接
        for vehicle in self.vehicles.values():
            closest_rsu = self._find_closest_rsu(vehicle.current_position)
            if closest_rsu:
                distance = self._calculate_distance(
                    vehicle.current_position, closest_rsu.position
                )
                if distance <= closest_rsu.coverage_radius:
                    closest_rsu.connected_vehicles.add(vehicle.vehicle_id)
        
        # 更新负载因子
        for rsu in self.rsus.values():
            max_capacity = 20  # 假设最大连接20个车辆
            rsu.load_factor = len(rsu.connected_vehicles) / max_capacity
    
    def _find_closest_rsu(self, position: Tuple[float, float]) -> Optional[RSUState]:
        """寻找最近的RSU"""
        closest_rsu = None
        min_distance = float('inf')
        
        for rsu in self.rsus.values():
            distance = self._calculate_distance(position, rsu.position)
            if distance < min_distance:
                min_distance = distance
                closest_rsu = rsu
        
        return closest_rsu
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """计算距离（米）"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111000  # 转换为米
    
    def _process_content_requests(self):
        """处理内容请求"""
        current_time = time.time()
        
        for vehicle in self.vehicles.values():
            # 检查是否应该发起请求
            time_since_last = current_time - vehicle.last_request_time
            avg_interval = 60.0 / vehicle.request_frequency  # 平均间隔时间
            
            # 🔧 修复：降低请求间隔，增加请求概率
            if time_since_last >= avg_interval * 0.5:  # 更频繁的请求
                # 增加请求概率
                if np.random.random() < 0.3:  # 30%概率发起请求
                    self._generate_and_process_request(vehicle, current_time)
                    vehicle.last_request_time = current_time
    
    def _generate_and_process_request(self, vehicle: VehicleState, current_time: float):
        """生成并处理内容请求"""
        # 计算内容需求概率
        demand_probs = self.behavior_model.calculate_content_demand_probability(
            vehicle.user_profile,
            vehicle.current_position,
            self.current_simulation_time
        )
        
        # 选择内容类型
        content_types = list(demand_probs.keys())
        probabilities = list(demand_probs.values())
        
        if not content_types:
            return
        
        selected_type = np.random.choice(content_types, p=probabilities)
        
        # 找到连接的RSU
        connected_rsu = None
        for rsu in self.rsus.values():
            if vehicle.vehicle_id in rsu.connected_vehicles:
                connected_rsu = rsu
                break
        
        if not connected_rsu:
            return  # 没有RSU连接，无法请求
        
        # 生成具体内容请求
        content_type_enum = ContentType(selected_type) if selected_type in [ct.value for ct in ContentType] else ContentType.TRAFFIC_INFO
        content = connected_rsu.cache_simulator.generate_realistic_content_request(
            vehicle.current_position,
            vehicle.user_profile if hasattr(vehicle.user_profile, 'preferences') else None
        )
        
        # 处理请求
        hit, action, metrics = connected_rsu.cache_simulator.request_content(
            content, vehicle.current_position
        )
        
        # 更新统计
        self.global_stats['total_requests'] += 1
        if hit:
            self.global_stats['total_cache_hits'] += 1
        
        self.global_stats['content_type_requests'][selected_type] += 1
        self.global_stats['user_type_behaviors'][vehicle.user_profile.user_type.value][selected_type] += 1
        
        # 记录地理热点
        lat_zone = int(vehicle.current_position[0] * 100) / 100
        lon_zone = int(vehicle.current_position[1] * 100) / 100
        self.global_stats['geographic_hotspots'][(lat_zone, lon_zone)] += 1
    
    def _update_cache_states(self):
        """更新缓存状态"""
        for rsu in self.rsus.values():
            rsu.cache_simulator.periodic_cleanup()
    
    def _collect_statistics(self):
        """收集统计数据"""
        hour = self.current_simulation_time.hour
        
        # 记录每小时统计
        for content_type, count in self.global_stats['content_type_requests'].items():
            if count > 0:  # 只记录有请求的类型
                self.global_stats['hourly_patterns'][hour][content_type] = count
    
    def run_simulation(self, duration_hours: float = 24.0, step_seconds: float = 10.0):
        """运行仿真"""
        print(f"🎬 开始运行VEC仿真 - 持续 {duration_hours} 小时")
        
        total_steps = int(duration_hours * 3600 / step_seconds)
        
        for step in range(total_steps):
            self.step_simulation(step_seconds)
            
            # 定期输出进度
            if step % 360 == 0:  # 每小时输出一次
                current_hour = step * step_seconds / 3600
                hit_rate = (self.global_stats['total_cache_hits'] / 
                           max(1, self.global_stats['total_requests']))
                print(f"⏰ 仿真进度: {current_hour:.1f}h - "
                      f"请求总数: {self.global_stats['total_requests']}, "
                      f"缓存命中率: {hit_rate:.2%}")
        
        print("✅ 仿真完成")
        return self._generate_final_report()
    
    def _generate_final_report(self) -> Dict:
        """生成最终报告"""
        total_requests = self.global_stats['total_requests']
        hit_rate = self.global_stats['total_cache_hits'] / max(1, total_requests)
        
        # 分析内容类型分布
        content_distribution = dict(self.global_stats['content_type_requests'])
        
        # 分析用户行为
        user_behaviors = {}
        for user_type, behaviors in self.global_stats['user_type_behaviors'].items():
            user_behaviors[user_type] = dict(behaviors)
        
        # 分析地理热点
        geographic_hotspots = dict(self.global_stats['geographic_hotspots'])
        top_hotspots = sorted(geographic_hotspots.items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        # RSU性能统计
        rsu_stats = {}
        for rsu_id, rsu in self.rsus.items():
            stats = rsu.cache_simulator.get_comprehensive_statistics()
            rsu_stats[rsu_id] = {
                'cache_hit_rate': stats['cache_performance']['hit_rate'],
                'load_factor': rsu.load_factor,
                'connected_vehicles': len(rsu.connected_vehicles),
                'cached_items': stats['resource_utilization']['cached_items']
            }
        
        return {
            'simulation_summary': {
                'total_requests': total_requests,
                'total_cache_hits': self.global_stats['total_cache_hits'],
                'overall_hit_rate': hit_rate,
                'total_vehicles': len(self.vehicles),
                'total_rsus': len(self.rsus)
            },
            'content_analysis': {
                'content_type_distribution': content_distribution,
                'hourly_patterns': dict(self.global_stats['hourly_patterns'])
            },
            'user_behavior_analysis': user_behaviors,
            'geographic_analysis': {
                'top_hotspots': top_hotspots[:5],
                'total_zones': len(geographic_hotspots)
            },
            'rsu_performance': rsu_stats
        }


def test_integrated_simulation():
    """测试整合仿真系统"""
    print("🧪 测试整合VEC仿真系统...")
    
    # 创建小规模仿真
    simulation = IntegratedVECSimulation(num_vehicles=10, num_rsus=3)
    
    # 运行短时间仿真
    report = simulation.run_simulation(duration_hours=2.0, step_seconds=30.0)
    
    print("\n📊 仿真报告:")
    print("="*50)
    
    # 仿真总结
    summary = report['simulation_summary']
    print(f"📈 总请求数: {summary['total_requests']}")
    print(f"🎯 总命中数: {summary['total_cache_hits']}")
    print(f"📊 整体命中率: {summary['overall_hit_rate']:.2%}")
    
    # 内容分析
    print(f"\n📱 内容类型分布:")
    content_dist = report['content_analysis']['content_type_distribution']
    for content_type, count in sorted(content_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = count / max(1, summary['total_requests']) * 100
        print(f"  {content_type}: {count} ({percentage:.1f}%)")
    
    # 用户行为分析
    print(f"\n👥 用户行为分析:")
    user_behaviors = report['user_behavior_analysis']
    for user_type, behaviors in user_behaviors.items():
        if behaviors:
            top_content = max(behaviors.items(), key=lambda x: x[1])
            print(f"  {user_type}: 最常请求 {top_content[0]} ({top_content[1]}次)")
    
    # RSU性能
    print(f"\n🏢 RSU性能:")
    rsu_perf = report['rsu_performance']
    for rsu_id, stats in rsu_perf.items():
        print(f"  {rsu_id}: 命中率{stats['cache_hit_rate']:.1%}, "
              f"负载{stats['load_factor']:.1%}, "
              f"连接{stats['connected_vehicles']}辆车")
    
    print("\n✅ 整合仿真测试完成")


if __name__ == "__main__":
    test_integrated_simulation()
