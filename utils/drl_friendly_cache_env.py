#!/usr/bin/env python3
"""
DRL友好的缓存环境
支持可控现实度的渐进式训练
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

class RealismLevel(Enum):
    """现实度等级"""
    MINIMAL = "minimal"          # 最小化：固定模式，便于学习
    BASIC = "basic"              # 基础：简单时间模式
    MODERATE = "moderate"        # 中等：用户类型差异
    REALISTIC = "realistic"      # 现实：完整行为模式
    CHAOTIC = "chaotic"          # 混沌：高随机性测试

@dataclass
class CacheEnvironmentConfig:
    """缓存环境配置"""
    realism_level: RealismLevel = RealismLevel.MINIMAL
    num_content_types: int = 4           # 内容类型数量
    cache_capacity: int = 10             # 缓存容量(项目数)
    episode_length: int = 100            # Episode长度
    request_frequency: float = 0.1       # 请求频率
    reward_shaping: bool = True          # 奖励塑形
    state_simplification: bool = True    # 状态简化
    user_behavior_noise: float = 0.1     # 用户行为噪声
    temporal_patterns: bool = False      # 时间模式
    user_diversity: bool = False         # 用户多样性

class DRLFriendlyCacheEnvironment:
    """DRL友好的缓存环境"""
    
    def __init__(self, config: CacheEnvironmentConfig):
        self.config = config
        self.current_step = 0
        self.episode_step = 0
        
        # 状态维度设计
        self._setup_state_space()
        
        # 动作空间设计  
        self._setup_action_space()
        
        # 内容请求模式
        self._setup_request_patterns()
        
        # 缓存状态
        self.cache_items = []  # 简化为列表
        self.cache_ages = []   # 缓存年龄
        self.cache_access_counts = []  # 访问次数
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'episode_reward': 0.0,
            'hit_rate_history': []
        }
        
        print(f"🤖 DRL缓存环境初始化 - 现实度: {config.realism_level.value}")
    
    def _setup_state_space(self):
        """设置状态空间"""
        # 基础状态维度
        base_dims = [
            self.config.cache_capacity,      # 缓存占用率 
            self.config.num_content_types,   # 当前请求类型
            1,                               # 时间步
        ]
        
        # 根据现实度增加维度
        if self.config.realism_level.value in ['moderate', 'realistic']:
            base_dims.extend([
                1,  # 用户类型
                1,  # 时间模式
            ])
        
        if self.config.realism_level.value in ['realistic', 'chaotic']:
            base_dims.extend([
                1,  # 位置相关性
                1,  # 内容新鲜度
            ])
        
        self.state_dim = sum(base_dims)
        print(f"  状态维度: {self.state_dim}")
    
    def _setup_action_space(self):
        """设置动作空间"""
        # 简化的动作设计
        self.actions = {
            0: "不缓存",
            1: "缓存(低优先级)",  
            2: "缓存(高优先级)",
            3: "替换最旧项目",
            4: "替换最少使用项目"
        }
        
        self.action_dim = len(self.actions)
        print(f"  动作维度: {self.action_dim}")
    
    def _setup_request_patterns(self):
        """设置请求模式"""
        if self.config.realism_level == RealismLevel.MINIMAL:
            # 固定循环模式，便于学习
            self.request_pattern = [0, 1, 2, 3] * 25  # 简单循环
            
        elif self.config.realism_level == RealismLevel.BASIC:
            # 基础时间模式
            morning_pattern = [0, 0, 1, 2]  # 早高峰：交通导航
            noon_pattern = [3, 3, 2, 1]     # 午休：娱乐停车
            self.request_pattern = morning_pattern * 12 + noon_pattern * 13
            
        elif self.config.realism_level == RealismLevel.MODERATE:
            # 中等复杂度：3种用户类型
            commuter_pattern = [0, 0, 1, 1, 2]      # 通勤族
            leisure_pattern = [3, 3, 2, 1, 0]       # 休闲用户  
            business_pattern = [1, 2, 0, 1, 2]      # 商务人士
            self.request_pattern = (commuter_pattern * 7 + 
                                  leisure_pattern * 6 + 
                                  business_pattern * 7)
        else:
            # 现实/混沌模式：使用概率分布
            self.request_pattern = None  # 动态生成
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.episode_step = 0
        
        # 清空缓存
        self.cache_items = []
        self.cache_ages = []
        self.cache_access_counts = []
        
        # 重置统计
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'episode_reward': 0.0,
            'hit_rate_history': []
        }
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """环境步进"""
        self.episode_step += 1
        
        # 生成内容请求
        content_type = self._generate_content_request()
        
        # 检查缓存命中
        cache_hit = self._check_cache_hit(content_type)
        
        # 执行缓存动作
        self._execute_cache_action(action, content_type, cache_hit)
        
        # 计算奖励
        reward = self._calculate_reward(cache_hit, action, content_type)
        
        # 更新缓存状态
        self._update_cache_state()
        
        # 更新统计
        self._update_statistics(cache_hit, reward)
        
        # 检查episode结束
        done = self.episode_step >= self.config.episode_length
        
        # 获取新状态
        next_state = self._get_state()
        
        # 信息字典
        info = {
            'cache_hit': cache_hit,
            'content_type': content_type,
            'hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_requests']),
            'cache_utilization': len(self.cache_items) / self.config.cache_capacity
        }
        
        return next_state, reward, done, info
    
    def _generate_content_request(self) -> int:
        """生成内容请求"""
        if self.request_pattern is not None:
            # 使用预定义模式
            pattern_idx = self.episode_step % len(self.request_pattern)
            content_type = self.request_pattern[pattern_idx]
        else:
            # 使用概率分布（现实模式）
            content_type = self._generate_realistic_request()
        
        # 添加噪声
        if np.random.random() < self.config.user_behavior_noise:
            content_type = np.random.randint(0, self.config.num_content_types)
        
        return content_type
    
    def _generate_realistic_request(self) -> int:
        """生成现实请求（复杂模式）"""
        # 简化的时间模式
        hour_of_day = (self.episode_step // 4) % 24
        
        if 7 <= hour_of_day <= 9:  # 早高峰
            probs = [0.4, 0.3, 0.2, 0.1]  # 交通，导航，停车，娱乐
        elif 12 <= hour_of_day <= 14:  # 午休
            probs = [0.2, 0.2, 0.2, 0.4]  # 均匀分布
        elif 17 <= hour_of_day <= 19:  # 晚高峰
            probs = [0.5, 0.25, 0.15, 0.1]  # 交通优先
        else:  # 其他时间
            probs = [0.25, 0.25, 0.25, 0.25]  # 均匀分布
        
        return np.random.choice(self.config.num_content_types, p=probs)
    
    def _check_cache_hit(self, content_type: int) -> bool:
        """检查缓存命中"""
        if content_type in self.cache_items:
            # 更新访问计数和年龄
            idx = self.cache_items.index(content_type)
            self.cache_access_counts[idx] += 1
            self.cache_ages[idx] = 0  # 重置年龄
            return True
        return False
    
    def _execute_cache_action(self, action: int, content_type: int, cache_hit: bool):
        """执行缓存动作"""
        if cache_hit:
            return  # 已命中，无需操作
        
        if action == 0:  # 不缓存
            return
        
        elif action in [1, 2]:  # 缓存（低/高优先级）
            if len(self.cache_items) < self.config.cache_capacity:
                # 有空间，直接添加
                self.cache_items.append(content_type)
                self.cache_ages.append(0)
                self.cache_access_counts.append(1)
            elif action == 2:  # 高优先级，强制替换
                self._replace_cache_item(content_type, method='random')
        
        elif action == 3:  # 替换最旧项目
            if len(self.cache_items) >= self.config.cache_capacity:
                self._replace_cache_item(content_type, method='oldest')
        
        elif action == 4:  # 替换最少使用项目
            if len(self.cache_items) >= self.config.cache_capacity:
                self._replace_cache_item(content_type, method='lfu')
    
    def _replace_cache_item(self, new_content: int, method: str):
        """替换缓存项目"""
        if not self.cache_items:
            return
        
        if method == 'random':
            idx = np.random.randint(len(self.cache_items))
        elif method == 'oldest':
            idx = np.argmax(self.cache_ages)
        elif method == 'lfu':
            idx = np.argmin(self.cache_access_counts)
        else:
            idx = 0
        
        # 替换
        self.cache_items[idx] = new_content
        self.cache_ages[idx] = 0
        self.cache_access_counts[idx] = 1
    
    def _calculate_reward(self, cache_hit: bool, action: int, content_type: int) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 基础奖励：命中获得正奖励
        if cache_hit:
            reward += 1.0
        else:
            reward -= 0.1  # 小的负奖励
        
        # 奖励塑形
        if self.config.reward_shaping:
            # 缓存利用率奖励
            utilization = len(self.cache_items) / self.config.cache_capacity
            if 0.6 <= utilization <= 0.9:  # 鼓励合理利用率
                reward += 0.1
            
            # 动作合理性奖励
            if not cache_hit and action in [1, 2]:  # 未命中时缓存
                reward += 0.05
            elif cache_hit and action == 0:  # 命中时不重复缓存
                reward += 0.05
        
        return reward
    
    def _update_cache_state(self):
        """更新缓存状态"""
        # 增加所有项目年龄
        self.cache_ages = [age + 1 for age in self.cache_ages]
    
    def _update_statistics(self, cache_hit: bool, reward: float):
        """更新统计信息"""
        self.stats['total_requests'] += 1
        if cache_hit:
            self.stats['cache_hits'] += 1
        
        self.stats['episode_reward'] += reward
        
        # 记录命中率历史
        hit_rate = self.stats['cache_hits'] / self.stats['total_requests']
        self.stats['hit_rate_history'].append(hit_rate)
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 基础状态
        cache_utilization = len(self.cache_items) / self.config.cache_capacity
        state.append(cache_utilization)
        
        # 缓存内容分布
        content_counts = [0] * self.config.num_content_types
        for item in self.cache_items:
            content_counts[item] += 1
        
        # 归一化内容计数
        if self.config.cache_capacity > 0:
            content_counts = [c / self.config.cache_capacity for c in content_counts]
        
        state.extend(content_counts)
        
        # 时间步（归一化）
        time_step = self.episode_step / self.config.episode_length
        state.append(time_step)
        
        # 根据现实度添加其他状态
        if self.config.realism_level.value in ['moderate', 'realistic']:
            # 添加时间模式状态
            hour = (self.episode_step // 4) % 24
            time_pattern = self._get_time_pattern(hour)
            state.append(time_pattern)
        
        # 填充到固定维度
        while len(state) < self.state_dim:
            state.append(0.0)
        
        return np.array(state[:self.state_dim], dtype=np.float32)
    
    def _get_time_pattern(self, hour: int) -> float:
        """获取时间模式特征"""
        if 7 <= hour <= 9:
            return 0.8  # 早高峰
        elif 12 <= hour <= 14:
            return 0.6  # 午休
        elif 17 <= hour <= 19:
            return 0.9  # 晚高峰
        else:
            return 0.3  # 其他时间
    
    def get_statistics(self) -> Dict:
        """获取环境统计"""
        return {
            'hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_requests']),
            'total_requests': self.stats['total_requests'],
            'cache_utilization': len(self.cache_items) / self.config.cache_capacity,
            'episode_reward': self.stats['episode_reward'],
            'cache_items': len(self.cache_items)
        }


class ProgressiveTrainingManager:
    """渐进式训练管理器"""
    
    def __init__(self):
        self.training_stages = [
            {
                'name': 'Stage 1: 基础学习',
                'config': CacheEnvironmentConfig(
                    realism_level=RealismLevel.MINIMAL,
                    num_content_types=3,
                    cache_capacity=5,
                    episode_length=50,
                    reward_shaping=True,
                    state_simplification=True
                ),
                'episodes': 1000,
                'success_criteria': {'hit_rate': 0.6}
            },
            {
                'name': 'Stage 2: 时间模式',
                'config': CacheEnvironmentConfig(
                    realism_level=RealismLevel.BASIC,
                    num_content_types=4,
                    cache_capacity=8,
                    episode_length=100,
                    temporal_patterns=True,
                    reward_shaping=True
                ),
                'episodes': 2000,
                'success_criteria': {'hit_rate': 0.5}
            },
            {
                'name': 'Stage 3: 用户多样性',
                'config': CacheEnvironmentConfig(
                    realism_level=RealismLevel.MODERATE,
                    num_content_types=6,
                    cache_capacity=10,
                    episode_length=150,
                    user_diversity=True,
                    user_behavior_noise=0.1
                ),
                'episodes': 3000,
                'success_criteria': {'hit_rate': 0.45}
            },
            {
                'name': 'Stage 4: 现实场景',
                'config': CacheEnvironmentConfig(
                    realism_level=RealismLevel.REALISTIC,
                    num_content_types=8,
                    cache_capacity=15,
                    episode_length=200,
                    user_behavior_noise=0.2,
                    reward_shaping=False  # 移除奖励塑形
                ),
                'episodes': 5000,
                'success_criteria': {'hit_rate': 0.4}
            }
        ]
    
    def get_next_stage(self, current_performance: Dict) -> Optional[Dict]:
        """获取下一个训练阶段"""
        for stage in self.training_stages:
            # 检查是否满足条件
            meets_criteria = True
            for metric, threshold in stage['success_criteria'].items():
                if current_performance.get(metric, 0) < threshold:
                    meets_criteria = False
                    break
            
            if not meets_criteria:
                return stage
        
        return None  # 所有阶段完成


def test_drl_environment():
    """测试DRL环境"""
    print("🧪 测试DRL友好的缓存环境...")
    
    # 测试不同现实度级别
    realism_levels = [RealismLevel.MINIMAL, RealismLevel.BASIC, RealismLevel.MODERATE]
    
    for level in realism_levels:
        print(f"\n🎯 测试现实度: {level.value}")
        
        config = CacheEnvironmentConfig(realism_level=level)
        env = DRLFriendlyCacheEnvironment(config)
        
        # 运行几个episode
        total_reward = 0
        for episode in range(3):
            state = env.reset()
            episode_reward = 0
            
            for step in range(20):
                # 随机动作（实际中用DRL agent）
                action = np.random.randint(env.action_dim)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_reward += episode_reward
            stats = env.get_statistics()
            print(f"  Episode {episode+1}: 奖励={episode_reward:.2f}, 命中率={stats['hit_rate']:.2%}")
        
        avg_reward = total_reward / 3
        print(f"  平均奖励: {avg_reward:.2f}")
    
    print("\n✅ DRL环境测试完成")


if __name__ == "__main__":
    test_drl_environment()
