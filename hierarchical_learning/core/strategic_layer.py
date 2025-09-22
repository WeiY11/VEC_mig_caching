"""
战略层（Strategic Layer）实现
使用SAC算法进行高层决策：计算卸载 vs 内容缓存

主要功能：
1. 分析整个区域的车辆密度、网络负载等宏观信息
2. 决定当前时刻系统应该优先采取的总体策略
3. 为战术层提供高层指导
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from single_agent.sac import SACAgent, SACConfig
from hierarchical_learning.core.base_layer import BaseLayer


class StrategicLayer(BaseLayer):
    """战略层 - 使用SAC算法进行高层战略决策"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 战略层状态维度：区域级别的宏观信息
        # 包括：车辆密度、网络负载、RSU状态、UAV状态、历史性能指标等
        self.strategic_state_dim = config.get('strategic_state_dim', 20)
        
        # 战略层动作维度：高层策略选择
        # 动作空间：[计算卸载权重, 内容缓存权重, 资源分配策略, 优先级设置]
        self.strategic_action_dim = config.get('strategic_action_dim', 4)
        
        # SAC配置 - 全面优化
        sac_config = SACConfig(
            hidden_dim=config.get('strategic_hidden_dim', 512),   # 增大网络容量
            actor_lr=config.get('strategic_actor_lr', 8e-5),      # 优化学习率
            critic_lr=config.get('strategic_critic_lr', 1e-4),    # 稍高的critic学习率
            alpha_lr=config.get('strategic_alpha_lr', 1e-4),      # 温度参数学习率
            initial_temperature=config.get('strategic_temperature', 0.05),  # 更低的初始温度
            tau=config.get('strategic_tau', 0.002),               # 适中的软更新率
            gamma=config.get('strategic_gamma', 0.998),           # 更高的折扣因子（长期规划）
            batch_size=config.get('strategic_batch_size', 64),    # 降低批次大小，加速触发更新
            buffer_size=config.get('strategic_buffer_size', 100000)  # 增大缓冲区
        )
        
        # 初始化SAC智能体
        self.sac_agent = SACAgent(
            state_dim=self.strategic_state_dim,
            action_dim=self.strategic_action_dim,
            config=sac_config
        )
        
        # 战略决策历史
        self.decision_history = []
        self.performance_history = []
        
        # 学习率调度
        self.initial_lr = {
            'actor': sac_config.actor_lr,
            'critic': sac_config.critic_lr,
            'alpha': sac_config.alpha_lr
        }
        self.lr_decay_rate = config.get('strategic_lr_decay', 0.9995)
        self.min_lr = config.get('strategic_min_lr', 1e-6)
        self.training_steps = 0
        
        # 算法增强功能
        self.use_gradient_clipping = config.get('strategic_gradient_clip', True)
        self.gradient_clip_value = config.get('strategic_clip_value', 1.0)
        self.use_adaptive_lr = config.get('strategic_adaptive_lr', True)
        self.lr_patience = config.get('strategic_lr_patience', 100)
        self.performance_window = []
        self.best_performance = float('-inf')
        self.patience_counter = 0
        
        # 高级探索策略
        self.exploration_schedule = config.get('strategic_exploration_schedule', 'linear')
        self.initial_exploration = config.get('strategic_initial_exploration', 0.9)
        self.final_exploration = config.get('strategic_final_exploration', 0.1)
        self.exploration_steps = config.get('strategic_exploration_steps', 50000)
        
    def process_state(self, raw_state: Dict) -> np.ndarray:
        """
        处理原始环境状态，提取战略层需要的宏观信息
        
        Args:
            raw_state: 包含车辆、RSU、UAV等详细状态的字典
            
        Returns:
            strategic_state: 战略层状态向量
        """
        strategic_features = []
        
        # 1. 车辆密度和分布特征
        if 'vehicles' in raw_state:
            vehicles = raw_state['vehicles']
            vehicle_count = len(vehicles)
            
            # 车辆密度
            strategic_features.append(vehicle_count / 100.0)  # 归一化
            
            # 车辆计算需求统计
            total_compute_demand = sum([v.get('compute_demand', 0) for v in vehicles])
            avg_compute_demand = total_compute_demand / max(vehicle_count, 1)
            strategic_features.append(avg_compute_demand / 1000.0)  # 归一化
            
            # 车辆移动性统计
            avg_velocity = np.mean([v.get('velocity', 0) for v in vehicles])
            strategic_features.append(avg_velocity / 30.0)  # 归一化到0-1
            
        else:
            strategic_features.extend([0.0, 0.0, 0.0])
        
        # 2. RSU网络负载和状态
        if 'rsus' in raw_state:
            rsus = raw_state['rsus']
            rsu_count = len(rsus)
            
            # RSU平均负载
            total_rsu_load = sum([rsu.get('cpu_usage', 0) for rsu in rsus])
            avg_rsu_load = total_rsu_load / max(rsu_count, 1)
            strategic_features.append(avg_rsu_load)
            
            # RSU可用计算资源
            total_available_compute = sum([rsu.get('available_compute', 0) for rsu in rsus])
            strategic_features.append(total_available_compute / 10000.0)  # 归一化
            
            # RSU网络拥塞程度
            total_network_load = sum([rsu.get('network_load', 0) for rsu in rsus])
            avg_network_load = total_network_load / max(rsu_count, 1)
            strategic_features.append(avg_network_load)
            
        else:
            strategic_features.extend([0.0, 0.0, 0.0])
        
        # 3. UAV状态（固定位置，但状态可变）
        if 'uavs' in raw_state:
            uavs = raw_state['uavs']
            uav_count = len(uavs)
            
            # UAV平均负载
            total_uav_load = sum([uav.get('cpu_usage', 0) for uav in uavs])
            avg_uav_load = total_uav_load / max(uav_count, 1)
            strategic_features.append(avg_uav_load)
            
            # UAV可用计算资源
            total_uav_compute = sum([uav.get('available_compute', 0) for uav in uavs])
            strategic_features.append(total_uav_compute / 5000.0)  # 归一化
            
            # UAV覆盖效率
            coverage_efficiency = sum([uav.get('coverage_efficiency', 0) for uav in uavs]) / max(uav_count, 1)
            strategic_features.append(coverage_efficiency)
            
        else:
            strategic_features.extend([0.0, 0.0, 0.0])
        
        # 4. 系统整体性能指标
        if 'system_metrics' in raw_state:
            metrics = raw_state['system_metrics']
            
            # 整体延迟
            strategic_features.append(metrics.get('avg_latency', 0) / 100.0)  # 归一化
            
            # 能耗效率
            strategic_features.append(metrics.get('energy_efficiency', 0))
            
            # 成功率
            strategic_features.append(metrics.get('success_rate', 0))
            
            # 网络利用率
            strategic_features.append(metrics.get('network_utilization', 0))
            
        else:
            strategic_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 5. 历史性能趋势（简化版）
        if len(self.performance_history) > 0:
            recent_performance = np.mean(self.performance_history[-5:])  # 最近5步的平均性能
            strategic_features.append(recent_performance)
        else:
            strategic_features.append(0.0)
        
        # 6. 时间特征（可选）
        time_of_day = raw_state.get('time_of_day', 0) / 24.0  # 归一化到0-1
        strategic_features.append(time_of_day)
        
        # 确保特征向量长度正确
        while len(strategic_features) < self.strategic_state_dim:
            strategic_features.append(0.0)
        
        strategic_features = strategic_features[:self.strategic_state_dim]
        
        return np.array(strategic_features, dtype=np.float32)
    
    def get_action(self, processed_state: np.ndarray) -> np.ndarray:
        """
        根据处理后的战略状态生成高层决策
        
        Args:
            processed_state: 战略层状态向量
            
        Returns:
            strategic_action: 战略层动作向量
            [计算卸载权重, 内容缓存权重, 资源分配策略, 优先级设置]
        """
        action = self.sac_agent.select_action(processed_state, training=True)
        
        # 对动作进行后处理，确保符合战略决策的语义
        strategic_action = self._post_process_action(action)
        
        # 记录决策历史
        self.decision_history.append({
            'state': processed_state.copy(),
            'action': strategic_action.copy(),
            'timestamp': len(self.decision_history)
        })
        
        return strategic_action
    
    def _post_process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        对原始动作进行后处理，确保符合战略决策的语义
        
        Args:
            raw_action: SAC输出的原始动作
            
        Returns:
            processed_action: 处理后的战略动作
        """
        processed_action = raw_action.copy()
        
        # 1. 计算卸载权重和内容缓存权重（使用softmax确保和为1）
        offloading_weight = torch.softmax(torch.tensor([raw_action[0], raw_action[1]]), dim=0)
        processed_action[0] = offloading_weight[0].item()  # 计算卸载权重
        processed_action[1] = offloading_weight[1].item()  # 内容缓存权重
        
        # 2. 资源分配策略（映射到0-1范围）
        processed_action[2] = torch.sigmoid(torch.tensor(raw_action[2])).item()
        
        # 3. 优先级设置（映射到0-1范围）
        processed_action[3] = torch.sigmoid(torch.tensor(raw_action[3])).item()
        
        return processed_action
    
    def train(self, replay_buffer=None) -> Dict[str, float]:
        """
        训练战略层SAC模型
        
        Args:
            replay_buffer: 经验回放缓冲区（可选，SAC有自己的缓冲区）
            
        Returns:
            training_stats: 训练统计信息
        """
        if len(self.sac_agent.replay_buffer) < self.sac_agent.config.batch_size:
            return {}
        
        # 使用SAC的更新方法
        training_stats = self.sac_agent.update()
        
        # 应用梯度裁剪
        if self.use_gradient_clipping and training_stats:
            self._apply_gradient_clipping()
        
        # 更新学习率调度
        self.training_steps += 1
        self._update_learning_rate()
        
        # 自适应学习率调整
        if self.use_adaptive_lr and training_stats:
            self._adaptive_lr_update(training_stats.get('critic_loss', 0.0))
        
        # 添加增强统计信息
        if training_stats:
            training_stats['current_lr'] = self._get_current_lr()
            training_stats['exploration_rate'] = self._get_exploration_rate()
            training_stats['training_steps'] = self.training_steps
        
        return training_stats
    
    def _update_learning_rate(self):
        """更新学习率（衰减）"""
        if hasattr(self.sac_agent, 'actor_optimizer'):
            for param_group in self.sac_agent.actor_optimizer.param_groups:
                new_lr = max(self.min_lr, param_group['lr'] * self.lr_decay_rate)
                param_group['lr'] = new_lr
        
        if hasattr(self.sac_agent, 'critic_optimizer'):
            for param_group in self.sac_agent.critic_optimizer.param_groups:
                new_lr = max(self.min_lr, param_group['lr'] * self.lr_decay_rate)
                param_group['lr'] = new_lr
                
        if hasattr(self.sac_agent, 'alpha_optimizer'):
            for param_group in self.sac_agent.alpha_optimizer.param_groups:
                new_lr = max(self.min_lr, param_group['lr'] * self.lr_decay_rate)
                param_group['lr'] = new_lr
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        if hasattr(self.sac_agent, 'actor_optimizer'):
            return self.sac_agent.actor_optimizer.param_groups[0]['lr']
        return self.initial_lr['actor']
    
    def _apply_gradient_clipping(self):
        """应用梯度裁剪"""
        if hasattr(self.sac_agent, 'actor'):
            torch.nn.utils.clip_grad_norm_(self.sac_agent.actor.parameters(), self.gradient_clip_value)
        if hasattr(self.sac_agent, 'critic'):
            torch.nn.utils.clip_grad_norm_(self.sac_agent.critic.parameters(), self.gradient_clip_value)
    
    def _adaptive_lr_update(self, current_loss: float):
        """自适应学习率更新"""
        self.performance_window.append(current_loss)
        
        # 保持窗口大小
        if len(self.performance_window) > self.lr_patience:
            self.performance_window.pop(0)
        
        # 检查性能改进
        if len(self.performance_window) >= self.lr_patience:
            avg_performance = np.mean(self.performance_window)
            
            if avg_performance < self.best_performance:
                self.best_performance = avg_performance
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                # 如果性能没有改进，降低学习率
                if self.patience_counter >= self.lr_patience:
                    self._reduce_learning_rate(0.5)
                    self.patience_counter = 0
                    print(f"🔄 战略层自适应降低学习率，当前损失: {avg_performance:.4f}")
    
    def _reduce_learning_rate(self, factor: float):
        """降低学习率"""
        if hasattr(self.sac_agent, 'actor_optimizer'):
            for param_group in self.sac_agent.actor_optimizer.param_groups:
                param_group['lr'] = max(self.min_lr, param_group['lr'] * factor)
        
        if hasattr(self.sac_agent, 'critic_optimizer'):
            for param_group in self.sac_agent.critic_optimizer.param_groups:
                param_group['lr'] = max(self.min_lr, param_group['lr'] * factor)
    
    def _get_exploration_rate(self) -> float:
        """获取当前探索率"""
        if self.exploration_schedule == 'linear':
            progress = min(1.0, self.training_steps / self.exploration_steps)
            return self.initial_exploration - (self.initial_exploration - self.final_exploration) * progress
        elif self.exploration_schedule == 'exponential':
            decay_factor = np.exp(-self.training_steps / (self.exploration_steps / 3))
            return self.final_exploration + (self.initial_exploration - self.final_exploration) * decay_factor
        else:
            return self.final_exploration
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """存储经验到SAC的回放缓冲区"""
        self.sac_agent.store_experience(state, action, reward, next_state, done)
        
        # 更新性能历史
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:  # 保持历史长度
            self.performance_history.pop(0)
    
    def save_model(self, path: str):
        """保存战略层模型"""
        self.sac_agent.save_model(path)
    
    def load_model(self, path: str):
        """加载战略层模型"""
        self.sac_agent.load_model(path)
    
    def get_strategic_guidance(self) -> Dict[str, float]:
        """
        获取当前的战略指导信息，供战术层使用
        
        Returns:
            guidance: 战略指导字典
        """
        if len(self.decision_history) == 0:
            return {
                'offloading_priority': 0.5,
                'caching_priority': 0.5,
                'resource_allocation_strategy': 0.5,
                'system_priority': 0.5
            }
        
        latest_decision = self.decision_history[-1]['action']
        
        return {
            'offloading_priority': latest_decision[0],
            'caching_priority': latest_decision[1],
            'resource_allocation_strategy': latest_decision[2],
            'system_priority': latest_decision[3]
        }
    
    def get_layer_stats(self) -> Dict[str, float]:
        """获取战略层统计信息"""
        if len(self.performance_history) == 0:
            return {}
        
        return {
            'avg_performance': np.mean(self.performance_history),
            'recent_performance': np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else np.mean(self.performance_history),
            'decision_count': len(self.decision_history),
            'performance_trend': np.mean(self.performance_history[-5:]) - np.mean(self.performance_history[-10:-5]) if len(self.performance_history) >= 10 else 0.0
        }