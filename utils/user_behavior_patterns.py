#!/usr/bin/env python3
"""
VEC用户行为模式建模
基于时间、地点、用户类型的需求变化模拟
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

class UserType(Enum):
    """用户类型枚举"""
    COMMUTER = "commuter"           # 通勤族
    BUSINESS = "business"           # 商务人士  
    LEISURE = "leisure"             # 休闲用户
    DELIVERY = "delivery"           # 配送司机
    TAXI_DRIVER = "taxi_driver"     # 出租车司机
    TOURIST = "tourist"             # 游客
    STUDENT = "student"             # 学生

class TimePattern(Enum):
    """时间模式枚举"""
    MORNING_RUSH = "morning_rush"   # 早高峰 7-9
    NOON_BREAK = "noon_break"       # 午休 12-14
    EVENING_RUSH = "evening_rush"   # 晚高峰 17-19
    NIGHT_TIME = "night_time"       # 夜间 22-6
    WEEKEND = "weekend"             # 周末
    HOLIDAY = "holiday"             # 节假日
    NORMAL_TIME = "normal_time"     # 正常时间

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    user_type: UserType
    home_location: Tuple[float, float]      # 家庭位置
    work_location: Tuple[float, float]      # 工作位置
    frequent_routes: List[List[Tuple[float, float]]]  # 常用路线
    content_preferences: Dict[str, float]   # 内容偏好权重
    active_hours: Tuple[int, int]          # 活跃时间段
    weekend_behavior_diff: float          # 周末行为差异系数

class TimeBasedBehaviorModel:
    """基于时间的用户行为模型"""
    
    def __init__(self):
        # 时间段内容需求基础分布
        self.time_content_patterns = {
            TimePattern.MORNING_RUSH: {
                'traffic_info': 0.40,      # 40% - 高峰期最关心交通
                'navigation': 0.25,        # 25% - 导航需求高
                'parking_info': 0.15,      # 15% - 停车需求
                'safety_alert': 0.10,      # 10% - 安全警报
                'weather_info': 0.05,      # 5% - 天气关注
                'map_data': 0.03,          # 3% - 地图更新
                'entertainment': 0.01,     # 1% - 娱乐内容极少
                'sensor_data': 0.01        # 1% - 传感器数据
            },
            
            TimePattern.NOON_BREAK: {
                'entertainment': 0.30,     # 30% - 午休娱乐需求高
                'parking_info': 0.20,      # 20% - 找餐厅停车
                'traffic_info': 0.15,      # 15% - 中等交通关注
                'navigation': 0.12,        # 12% - 觅食导航
                'weather_info': 0.10,      # 10% - 天气查看
                'map_data': 0.08,          # 8% - 地图浏览
                'safety_alert': 0.03,      # 3% - 安全警报
                'sensor_data': 0.02        # 2% - 传感器数据
            },
            
            TimePattern.EVENING_RUSH: {
                'traffic_info': 0.35,      # 35% - 晚高峰交通重要
                'navigation': 0.22,        # 22% - 回家导航
                'parking_info': 0.18,      # 18% - 回家停车
                'safety_alert': 0.12,      # 12% - 夜间安全关注增加
                'entertainment': 0.08,     # 8% - 开始娱乐需求
                'weather_info': 0.03,      # 3% - 天气查看
                'map_data': 0.01,          # 1% - 地图数据
                'sensor_data': 0.01        # 1% - 传感器数据
            },
            
            TimePattern.NIGHT_TIME: {
                'entertainment': 0.35,     # 35% - 夜间娱乐主导
                'safety_alert': 0.25,      # 25% - 夜间安全重要
                'navigation': 0.15,        # 15% - 夜间出行导航
                'traffic_info': 0.10,      # 10% - 夜间交通查看
                'parking_info': 0.08,      # 8% - 夜间停车
                'weather_info': 0.04,      # 4% - 天气查看
                'map_data': 0.02,          # 2% - 地图浏览
                'sensor_data': 0.01        # 1% - 传感器数据
            },
            
            TimePattern.WEEKEND: {
                'entertainment': 0.30,     # 30% - 周末娱乐需求高
                'navigation': 0.20,        # 20% - 周末出游导航
                'parking_info': 0.18,      # 18% - 商圈停车需求高
                'traffic_info': 0.12,      # 12% - 周末交通较少关注
                'weather_info': 0.10,      # 10% - 出游天气重要
                'map_data': 0.06,          # 6% - 探索新地点
                'safety_alert': 0.03,      # 3% - 安全警报
                'sensor_data': 0.01        # 1% - 传感器数据
            },
            
            TimePattern.HOLIDAY: {
                'navigation': 0.25,        # 25% - 节假日出行导航高
                'entertainment': 0.25,     # 25% - 节假日娱乐
                'traffic_info': 0.20,      # 20% - 节假日交通拥堵
                'parking_info': 0.15,      # 15% - 景点停车难
                'weather_info': 0.08,      # 8% - 出游天气
                'map_data': 0.05,          # 5% - 探索新地方
                'safety_alert': 0.01,      # 1% - 安全警报
                'sensor_data': 0.01        # 1% - 传感器数据
            }
        }
        
        # 用户类型行为偏好修正
        self.user_type_modifiers = {
            UserType.COMMUTER: {
                'traffic_info': 1.5,       # 通勤族特别关注交通
                'navigation': 1.3,         # 导航需求高
                'parking_info': 1.2,       # 停车重要
                'entertainment': 0.6       # 娱乐需求低
            },
            
            UserType.BUSINESS: {
                'parking_info': 1.8,       # 商务人士停车需求很高
                'navigation': 1.4,         # 商务导航重要
                'traffic_info': 1.3,       # 时间就是金钱
                'entertainment': 0.4       # 工作时间娱乐少
            },
            
            UserType.LEISURE: {
                'entertainment': 2.0,      # 休闲用户娱乐需求高
                'weather_info': 1.5,       # 关注天气
                'map_data': 1.3,           # 喜欢探索
                'traffic_info': 0.7        # 不太关心交通效率
            },
            
            UserType.DELIVERY: {
                'navigation': 2.0,         # 配送司机导航需求极高
                'traffic_info': 1.8,       # 实时交通重要
                'parking_info': 1.5,       # 临时停车需求
                'entertainment': 0.2       # 工作中无娱乐
            },
            
            UserType.TAXI_DRIVER: {
                'traffic_info': 2.0,       # 出租车司机最关心交通
                'navigation': 1.8,         # 导航是工具
                'parking_info': 1.3,       # 等客停车
                'entertainment': 0.3       # 很少娱乐需求
            },
            
            UserType.TOURIST: {
                'navigation': 1.8,         # 游客导航需求高
                'map_data': 1.6,           # 探索新地方
                'entertainment': 1.4,      # 旅游娱乐
                'weather_info': 1.3,       # 旅游天气重要
                'parking_info': 1.2        # 景点停车
            },
            
            UserType.STUDENT: {
                'entertainment': 1.6,      # 学生娱乐需求较高
                'navigation': 1.2,         # 校园导航
                'traffic_info': 0.8,       # 对交通效率要求不高
                'parking_info': 0.6        # 多数不开车
            }
        }
        
        # 位置类型影响（城市中心、郊区、高速等）
        self.location_modifiers = {
            'city_center': {
                'parking_info': 1.8,       # 市中心停车难
                'traffic_info': 1.5,       # 拥堵严重
                'safety_alert': 1.2        # 人员密集安全重要
            },
            'suburb': {
                'navigation': 1.3,         # 郊区路线复杂
                'weather_info': 1.2,       # 郊区天气影响大
                'entertainment': 0.8       # 娱乐资源少
            },
            'highway': {
                'traffic_info': 1.6,       # 高速路况重要
                'safety_alert': 1.4,       # 高速安全重要
                'navigation': 1.2,         # 路线规划重要
                'entertainment': 0.4       # 高速上不娱乐
            }
        }
    
    def get_current_time_pattern(self, current_time: Optional[datetime] = None) -> TimePattern:
        """获取当前时间模式"""
        if current_time is None:
            current_time = datetime.now()
        
        # 检查是否为节假日（简化实现）
        if self._is_holiday(current_time):
            return TimePattern.HOLIDAY
        
        # 检查是否为周末
        if current_time.weekday() >= 5:  # 周六、周日
            return TimePattern.WEEKEND
        
        # 工作日时间段判断
        hour = current_time.hour
        
        if 7 <= hour <= 9:
            return TimePattern.MORNING_RUSH
        elif 12 <= hour <= 14:
            return TimePattern.NOON_BREAK
        elif 17 <= hour <= 19:
            return TimePattern.EVENING_RUSH
        elif hour >= 22 or hour <= 6:
            return TimePattern.NIGHT_TIME
        else:
            return TimePattern.NORMAL_TIME
    
    def _is_holiday(self, date_time: datetime) -> bool:
        """简化的节假日判断"""
        # 简单实现：可以扩展为真实的节假日数据
        month = date_time.month
        day = date_time.day
        
        # 示例节假日
        holidays = [
            (1, 1),   # 元旦
            (2, 14),  # 情人节
            (5, 1),   # 劳动节
            (10, 1),  # 国庆节
            (12, 25), # 圣诞节
        ]
        
        return (month, day) in holidays
    
    def calculate_content_demand_probability(self, 
                                           user_profile: UserProfile,
                                           current_location: Tuple[float, float],
                                           current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        计算用户在特定时间和地点对各类内容的需求概率
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 获取时间模式
        time_pattern = self.get_current_time_pattern(current_time)
        
        # 获取基础时间分布
        if time_pattern in self.time_content_patterns:
            base_distribution = self.time_content_patterns[time_pattern].copy()
        else:
            # 使用正常时间分布
            base_distribution = {
                'traffic_info': 0.25,
                'navigation': 0.20,
                'parking_info': 0.15,
                'entertainment': 0.15,
                'weather_info': 0.10,
                'safety_alert': 0.08,
                'map_data': 0.05,
                'sensor_data': 0.02
            }
        
        # 应用用户类型修正
        user_modifiers = self.user_type_modifiers.get(user_profile.user_type, {})
        for content_type, modifier in user_modifiers.items():
            if content_type in base_distribution:
                base_distribution[content_type] *= modifier
        
        # 应用用户个人偏好
        for content_type, preference in user_profile.content_preferences.items():
            if content_type in base_distribution:
                base_distribution[content_type] *= preference
        
        # 应用地理位置修正
        location_type = self._classify_location(current_location)
        location_modifiers = self.location_modifiers.get(location_type, {})
        for content_type, modifier in location_modifiers.items():
            if content_type in base_distribution:
                base_distribution[content_type] *= modifier
        
        # 应用时间特殊修正
        base_distribution = self._apply_time_specific_modifiers(
            base_distribution, current_time, user_profile
        )
        
        # 归一化概率分布
        total = sum(base_distribution.values())
        if total > 0:
            base_distribution = {k: v/total for k, v in base_distribution.items()}
        
        return base_distribution
    
    def _classify_location(self, location: Tuple[float, float]) -> str:
        """简化的位置分类"""
        lat, lon = location
        
        # 简化实现：可以集成真实的地理数据
        # 这里使用模拟分类
        city_center_lat, city_center_lon = 39.9042, 116.4074  # 北京市中心
        
        distance_to_center = math.sqrt((lat - city_center_lat)**2 + (lon - city_center_lon)**2)
        
        if distance_to_center < 0.05:  # 约5km内
            return 'city_center'
        elif distance_to_center < 0.2:  # 约20km内
            return 'suburb'
        else:
            return 'highway'
    
    def _apply_time_specific_modifiers(self, 
                                     distribution: Dict[str, float],
                                     current_time: datetime,
                                     user_profile: UserProfile) -> Dict[str, float]:
        """应用特定时间修正"""
        hour = current_time.hour
        minute = current_time.minute
        
        # 早高峰强化效应
        if 7 <= hour <= 9:
            peak_intensity = 1.0 + 0.5 * math.sin(math.pi * (hour - 7) / 2)
            distribution['traffic_info'] *= peak_intensity
            distribution['navigation'] *= peak_intensity
        
        # 午餐时间效应
        elif 11 <= hour <= 13:
            lunch_intensity = 1.0 + 0.3 * math.sin(math.pi * (hour - 11) / 2)
            distribution['parking_info'] *= lunch_intensity
            distribution['entertainment'] *= lunch_intensity
        
        # 晚高峰效应
        elif 17 <= hour <= 19:
            evening_intensity = 1.0 + 0.4 * math.sin(math.pi * (hour - 17) / 2)
            distribution['traffic_info'] *= evening_intensity
            distribution['parking_info'] *= evening_intensity
        
        # 夜间安全效应
        elif hour >= 22 or hour <= 6:
            night_intensity = 1.0 + 0.6 * (1 - abs(hour - 24) / 12 if hour >= 12 else 1 - hour / 6)
            distribution['safety_alert'] *= night_intensity
            distribution['entertainment'] *= (1.5 if 20 <= hour <= 23 else 1.0)
        
        # 天气相关时间效应
        if hour in [6, 7, 18, 19]:  # 出门/回家时间
            distribution['weather_info'] *= 1.3
        
        return distribution
    
    def generate_realistic_user_profiles(self, num_users: int = 100) -> List[UserProfile]:
        """生成现实的用户画像"""
        profiles = []
        
        # 用户类型分布（基于现实比例）
        user_type_distribution = [
            (UserType.COMMUTER, 0.35),      # 35% 通勤族
            (UserType.LEISURE, 0.25),       # 25% 休闲用户
            (UserType.BUSINESS, 0.15),      # 15% 商务人士
            (UserType.STUDENT, 0.10),       # 10% 学生
            (UserType.DELIVERY, 0.08),      # 8% 配送司机
            (UserType.TAXI_DRIVER, 0.05),   # 5% 出租车司机
            (UserType.TOURIST, 0.02),       # 2% 游客
        ]
        
        for i in range(num_users):
            # 选择用户类型
            user_type = np.random.choice(
                [ut for ut, _ in user_type_distribution],
                p=[prob for _, prob in user_type_distribution]
            )
            
            # 生成基础位置（北京市范围）
            home_lat = 39.9042 + np.random.uniform(-0.2, 0.2)
            home_lon = 116.4074 + np.random.uniform(-0.2, 0.2)
            
            work_lat = 39.9042 + np.random.uniform(-0.15, 0.15)
            work_lon = 116.4074 + np.random.uniform(-0.15, 0.15)
            
            # 生成内容偏好
            preferences = self._generate_user_preferences(user_type)
            
            # 生成活跃时间
            active_hours = self._generate_active_hours(user_type)
            
            # 生成常用路线
            routes = self._generate_frequent_routes(
                (home_lat, home_lon), 
                (work_lat, work_lon), 
                user_type
            )
            
            profile = UserProfile(
                user_id=f"user_{i:04d}",
                user_type=user_type,
                home_location=(home_lat, home_lon),
                work_location=(work_lat, work_lon),
                frequent_routes=routes,
                content_preferences=preferences,
                active_hours=active_hours,
                weekend_behavior_diff=np.random.uniform(0.7, 1.3)
            )
            
            profiles.append(profile)
        
        return profiles
    
    def _generate_user_preferences(self, user_type: UserType) -> Dict[str, float]:
        """生成用户偏好"""
        base_preferences = {
            'traffic_info': 1.0,
            'navigation': 1.0,
            'parking_info': 1.0,
            'entertainment': 1.0,
            'weather_info': 1.0,
            'safety_alert': 1.0,
            'map_data': 1.0,
            'sensor_data': 1.0
        }
        
        # 根据用户类型调整
        type_adjustments = self.user_type_modifiers.get(user_type, {})
        for content_type, adjustment in type_adjustments.items():
            if content_type in base_preferences:
                base_preferences[content_type] = adjustment
        
        # 添加个性化随机变化
        for content_type in base_preferences:
            base_preferences[content_type] *= np.random.uniform(0.8, 1.2)
        
        return base_preferences
    
    def _generate_active_hours(self, user_type: UserType) -> Tuple[int, int]:
        """生成活跃时间段"""
        if user_type == UserType.COMMUTER:
            return (6, 22)  # 通勤族早出晚归
        elif user_type == UserType.BUSINESS:
            return (7, 23)  # 商务人士工作时间长
        elif user_type == UserType.STUDENT:
            return (8, 24)  # 学生晚睡
        elif user_type in [UserType.DELIVERY, UserType.TAXI_DRIVER]:
            return (0, 24)  # 职业司机全天
        else:
            return (9, 22)  # 休闲用户正常时间
    
    def _generate_frequent_routes(self, 
                                home: Tuple[float, float], 
                                work: Tuple[float, float],
                                user_type: UserType) -> List[List[Tuple[float, float]]]:
        """生成常用路线"""
        routes = []
        
        # 基本通勤路线
        if user_type != UserType.TOURIST:
            routes.append([home, work])  # 家-工作
            routes.append([work, home])  # 工作-家
        
        # 根据用户类型添加特殊路线
        if user_type == UserType.LEISURE:
            # 添加休闲场所
            for _ in range(3):
                leisure_spot = (
                    home[0] + np.random.uniform(-0.05, 0.05),
                    home[1] + np.random.uniform(-0.05, 0.05)
                )
                routes.append([home, leisure_spot, home])
        
        elif user_type == UserType.DELIVERY:
            # 添加配送路线
            for _ in range(5):
                delivery_points = []
                current = work
                for _ in range(np.random.randint(3, 8)):
                    next_point = (
                        current[0] + np.random.uniform(-0.02, 0.02),
                        current[1] + np.random.uniform(-0.02, 0.02)
                    )
                    delivery_points.append(next_point)
                    current = next_point
                routes.append(delivery_points)
        
        return routes


def test_user_behavior_patterns():
    """测试用户行为模式"""
    print("🧪 测试基于时间的用户行为模式...")
    
    behavior_model = TimeBasedBehaviorModel()
    
    # 生成用户画像
    users = behavior_model.generate_realistic_user_profiles(5)
    
    print(f"\n👥 生成了 {len(users)} 个用户画像:")
    for user in users:
        print(f"- {user.user_id}: {user.user_type.value}")
    
    # 测试不同时间的需求模式
    test_times = [
        datetime(2024, 1, 15, 8, 0),   # 早高峰
        datetime(2024, 1, 15, 13, 0),  # 午休
        datetime(2024, 1, 15, 18, 0),  # 晚高峰
        datetime(2024, 1, 15, 23, 0),  # 夜间
        datetime(2024, 1, 13, 15, 0),  # 周末
    ]
    
    test_user = users[0]  # 选择第一个用户测试
    
    print(f"\n📊 用户 {test_user.user_id} ({test_user.user_type.value}) 的需求模式:")
    
    for test_time in test_times:
        time_pattern = behavior_model.get_current_time_pattern(test_time)
        demand_prob = behavior_model.calculate_content_demand_probability(
            test_user, test_user.home_location, test_time
        )
        
        print(f"\n⏰ {test_time.strftime('%Y-%m-%d %H:%M')} ({time_pattern.value}):")
        
        # 排序并显示top 3需求
        sorted_demands = sorted(demand_prob.items(), key=lambda x: x[1], reverse=True)
        for content_type, probability in sorted_demands[:3]:
            print(f"   {content_type}: {probability:.1%}")
    
    print("\n✅ 用户行为模式测试完成")


if __name__ == "__main__":
    test_user_behavior_patterns()
