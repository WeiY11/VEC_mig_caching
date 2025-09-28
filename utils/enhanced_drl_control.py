#!/usr/bin/env python3
"""
增强DRL控制机制
让DRL能够做更具体和fine-grained的决策
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SpecificDecision:
    """具体决策"""
    task_id: str
    target_node: str           # 具体目标节点
    cache_action: str          # 具体缓存动作
    migration_trigger: bool    # 是否触发迁移
    priority_boost: float      # 优先级提升
    confidence: float          # 决策置信度

class EnhancedDRLController:
    """
    增强DRL控制器
    让DRL做更具体的决策，而不只是概率调整
    """
    
    def __init__(self):
        # DRL扩展动作空间
        self.enhanced_action_space = {
            # 原有11维：卸载偏好
            'offload_preferences': 11,
            
            # 新增14维：具体控制
            'specific_node_selection': 6,    # 直接指定RSU选择权重
            'cache_decisions': 4,            # 具体缓存决策
            'migration_triggers': 3,         # 具体迁移触发
            'priority_adjustments': 1        # 任务优先级调整
        }
        
        self.total_action_dim = sum(self.enhanced_action_space.values())  # 25维
        
        # 决策历史
        self.decision_history = []
        self.performance_feedback = {}
        
        print(f"🤖 增强DRL控制器初始化 - 动作维度: {self.total_action_dim}")
    
    def parse_enhanced_actions(self, action: np.ndarray) -> Dict:
        """
        解析增强动作向量为具体决策
        """
        if len(action) < self.total_action_dim:
            action = np.pad(action, (0, self.total_action_dim - len(action)), mode='constant')
        
        parsed_actions = {}
        start_idx = 0
        
        # 1. 卸载偏好(11维)
        end_idx = start_idx + self.enhanced_action_space['offload_preferences']
        parsed_actions['offload_preferences'] = action[start_idx:end_idx]
        start_idx = end_idx
        
        # 2. 具体节点选择(6维) - 直接指定每个RSU的选择权重
        end_idx = start_idx + self.enhanced_action_space['specific_node_selection']
        rsu_weights = action[start_idx:end_idx]
        # Softmax归一化
        rsu_weights_exp = np.exp(rsu_weights - np.max(rsu_weights))
        parsed_actions['rsu_selection_weights'] = rsu_weights_exp / np.sum(rsu_weights_exp)
        start_idx = end_idx
        
        # 3. 具体缓存决策(4维)
        end_idx = start_idx + self.enhanced_action_space['cache_decisions']
        cache_actions = action[start_idx:end_idx]
        parsed_actions['cache_decisions'] = {
            'high_priority_cache_threshold': np.tanh(cache_actions[0]) * 0.5 + 0.5,  # [0,1]
            'low_priority_cache_threshold': np.tanh(cache_actions[1]) * 0.3 + 0.3,   # [0,0.6]
            'cache_replacement_aggressiveness': np.tanh(cache_actions[2]) * 0.5 + 0.5,  # [0,1]
            'prefetch_probability': np.sigmoid(cache_actions[3])  # [0,1]
        }
        start_idx = end_idx
        
        # 4. 具体迁移触发(3维)
        end_idx = start_idx + self.enhanced_action_space['migration_triggers']
        migration_actions = action[start_idx:end_idx]
        parsed_actions['migration_decisions'] = {
            'rsu_migration_sensitivity': np.sigmoid(migration_actions[0]),  # [0,1]
            'uav_migration_sensitivity': np.sigmoid(migration_actions[1]),  # [0,1] 
            'global_balancing_weight': np.sigmoid(migration_actions[2])     # [0,1]
        }
        start_idx = end_idx
        
        # 5. 优先级调整(1维)
        priority_adjustment = action[start_idx]
        parsed_actions['priority_boost'] = np.tanh(priority_adjustment)  # [-1,1]
        
        return parsed_actions
    
    def make_specific_task_decision(self, task: Dict, system_state: Dict, 
                                  parsed_actions: Dict) -> SpecificDecision:
        """
        基于DRL输出做具体的任务决策
        """
        task_id = task.get('id', 'unknown')
        
        # 1. 具体节点选择（不再是概率，而是确定性选择）
        rsu_weights = parsed_actions['rsu_selection_weights']
        offload_prefs = parsed_actions['offload_preferences']
        
        # 基于系统状态和DRL偏好做具体选择
        candidate_nodes = self._get_available_nodes(system_state)
        best_node = self._select_best_node(candidate_nodes, rsu_weights, offload_prefs, task)
        
        # 2. 具体缓存决策
        cache_decision = self._decide_cache_action(task, parsed_actions['cache_decisions'])
        
        # 3. 具体迁移触发
        migration_trigger = self._should_trigger_migration(
            task, system_state, parsed_actions['migration_decisions']
        )
        
        # 4. 优先级调整
        priority_boost = parsed_actions['priority_boost']
        
        # 5. 计算决策置信度
        confidence = self._calculate_decision_confidence(task, system_state, parsed_actions)
        
        return SpecificDecision(
            task_id=task_id,
            target_node=best_node,
            cache_action=cache_decision,
            migration_trigger=migration_trigger,
            priority_boost=priority_boost,
            confidence=confidence
        )
    
    def _get_available_nodes(self, system_state: Dict) -> List[str]:
        """获取可用节点列表"""
        nodes = ['local']
        
        # 添加RSU节点
        rsus = system_state.get('rsus', [])
        for i, rsu in enumerate(rsus):
            queue_len = len(rsu.get('computation_queue', []))
            if queue_len < 25:  # 只考虑非极度过载的RSU
                nodes.append(f'rsu_{i}')
        
        # 添加UAV节点
        uavs = system_state.get('uavs', [])
        for i, uav in enumerate(uavs):
            queue_len = len(uav.get('computation_queue', []))
            battery = uav.get('battery_level', 1.0)
            if queue_len < 15 and battery > 0.3:  # UAV容量和电量检查
                nodes.append(f'uav_{i}')
        
        return nodes
    
    def _select_best_node(self, candidates: List[str], rsu_weights: np.ndarray, 
                         offload_prefs: np.ndarray, task: Dict) -> str:
        """
        基于DRL权重选择具体最佳节点
        """
        if not candidates:
            return 'local'
        
        # 基于任务特征和DRL偏好计算每个节点的适合度
        node_scores = {}
        
        for node in candidates:
            if node == 'local':
                # 本地处理评分
                score = float(offload_prefs[0])
            elif node.startswith('rsu_'):
                # RSU评分：DRL权重 + 负载状态
                rsu_idx = int(node.split('_')[1])
                if rsu_idx < len(rsu_weights):
                    drl_preference = float(rsu_weights[rsu_idx])
                    score = drl_preference + float(offload_prefs[1])
                else:
                    score = float(offload_prefs[1])
            elif node.startswith('uav_'):
                # UAV评分
                score = float(offload_prefs[2])
            else:
                score = 0.0
            
            # 基于任务类型调整评分
            task_type = task.get('task_type', 3)
            if task_type <= 2 and node == 'local':
                score += 0.5  # 紧急任务偏好本地
            elif task_type >= 3 and node != 'local':
                score += 0.3  # 容忍任务偏好卸载
            
            node_scores[node] = score
        
        # 选择评分最高的节点
        best_node = max(node_scores.items(), key=lambda x: x[1])[0]
        return best_node
    
    def _decide_cache_action(self, task: Dict, cache_decisions: Dict) -> str:
        """决定具体缓存动作"""
        task_priority = task.get('task_type', 3)
        
        if task_priority <= 2:
            # 高优先级任务
            threshold = cache_decisions['high_priority_cache_threshold']
            if np.random.random() < threshold:
                return 'cache_high_priority'
            else:
                return 'no_cache'
        else:
            # 低优先级任务
            threshold = cache_decisions['low_priority_cache_threshold']
            if np.random.random() < threshold:
                return 'cache_low_priority'
            else:
                return 'no_cache'
    
    def _should_trigger_migration(self, task: Dict, system_state: Dict, 
                                migration_decisions: Dict) -> bool:
        """决定是否触发迁移"""
        # 基于DRL学习的敏感度参数
        rsu_sensitivity = migration_decisions['rsu_migration_sensitivity']
        uav_sensitivity = migration_decisions['uav_migration_sensitivity']
        
        # 简化的迁移触发逻辑
        task_urgency = 1.0 / max(1.0, task.get('deadline', 10.0))
        
        if task_urgency > rsu_sensitivity:
            return True
        
        return False
    
    def _calculate_decision_confidence(self, task: Dict, system_state: Dict, 
                                     parsed_actions: Dict) -> float:
        """计算决策置信度"""
        # 基于系统状态的一致性计算置信度
        base_confidence = 0.8
        
        # 基于历史决策效果调整
        if len(self.decision_history) > 10:
            recent_decisions = self.decision_history[-10:]
            success_rate = sum(1 for d in recent_decisions if d.get('success', False)) / len(recent_decisions)
            base_confidence = 0.3 * success_rate + 0.7 * base_confidence
        
        return min(1.0, max(0.1, base_confidence))
    
    def record_decision_outcome(self, decision: SpecificDecision, success: bool, 
                              performance_metrics: Dict):
        """记录决策结果，用于学习改进"""
        outcome = {
            'decision': decision,
            'success': success,
            'metrics': performance_metrics,
            'timestamp': time.time()
        }
        
        self.decision_history.append(outcome)
        
        # 保持历史长度
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        # 更新性能反馈
        self.performance_feedback[decision.target_node] = self.performance_feedback.get(
            decision.target_node, []
        ) + [success]
        
        # 保持反馈历史长度
        if len(self.performance_feedback[decision.target_node]) > 20:
            self.performance_feedback[decision.target_node].pop(0)

# 全局增强DRL控制器
_enhanced_drl_controller = EnhancedDRLController()

def parse_enhanced_drl_actions(action: np.ndarray) -> Dict:
    """解析增强DRL动作"""
    return _enhanced_drl_controller.parse_enhanced_actions(action)

def make_specific_decision(task: Dict, system_state: Dict, action: np.ndarray) -> SpecificDecision:
    """基于DRL动作做具体决策"""
    parsed_actions = _enhanced_drl_controller.parse_enhanced_actions(action)
    return _enhanced_drl_controller.make_specific_task_decision(task, system_state, parsed_actions)

import time
