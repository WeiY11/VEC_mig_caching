#!/usr/bin/env python3
"""
智能迁移策略
基于全局负载均衡和系统性能优化的迁移决策
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class IntelligentMigrationStrategy:
    """
    智能迁移策略
    注重策略质量而非成功率，敢于尝试有益的迁移
    """
    
    def __init__(self):
        self.strategy_params = {
            'global_balance_weight': 0.6,      # 全局均衡权重
            'local_optimization_weight': 0.4,  # 局部优化权重
            'exploration_probability': 0.15,   # 探索性迁移概率
            'risk_tolerance': 0.3,             # 风险容忍度
            'min_benefit_threshold': 0.1       # 最小收益阈值
        }
        
        # 系统状态跟踪
        self.system_load_history = []
        self.migration_outcomes = []
        self.global_performance_trend = []
        
        print("🎯 智能迁移策略初始化 - 注重策略质量而非成功率")
    
    def evaluate_migration_necessity(self, node_states: Dict) -> Dict:
        """
        评估迁移必要性 - 从全局系统优化角度
        """
        # 计算系统负载分布
        load_distribution = self._calculate_load_distribution(node_states)
        
        # 评估负载不均衡程度
        imbalance_score = self._calculate_imbalance_score(load_distribution)
        
        # 识别迁移机会
        migration_opportunities = self._identify_migration_opportunities(
            node_states, load_distribution, imbalance_score
        )
        
        return {
            'load_distribution': load_distribution,
            'imbalance_score': imbalance_score,
            'migration_opportunities': migration_opportunities,
            'global_recommendation': self._generate_global_recommendation(
                imbalance_score, migration_opportunities
            )
        }
    
    def _calculate_load_distribution(self, node_states: Dict) -> Dict:
        """计算真实的负载分布"""
        distribution = {
            'rsu_loads': {},
            'uav_loads': {},
            'system_average': 0.0,
            'load_variance': 0.0
        }
        
        # RSU负载计算
        rsu_loads = []
        for node_id, state in node_states.items():
            if node_id.startswith('rsu_'):
                # 🔧 基于队列长度计算真实负载
                queue_len = state.get('queue_length', 0)
                # 使用realistic的负载模型：队列/20为满负载
                real_load = queue_len / 20.0
                distribution['rsu_loads'][node_id] = real_load
                rsu_loads.append(real_load)
        
        # UAV负载计算
        uav_loads = []
        for node_id, state in node_states.items():
            if node_id.startswith('uav_'):
                queue_len = state.get('queue_length', 0)
                battery = state.get('battery_level', 1.0)
                # UAV负载还要考虑电池状态
                real_load = (queue_len / 10.0) * battery  # 电池低时有效负载下降
                distribution['uav_loads'][node_id] = real_load
                uav_loads.append(real_load)
        
        # 系统统计
        all_loads = rsu_loads + uav_loads
        if all_loads:
            distribution['system_average'] = np.mean(all_loads)
            distribution['load_variance'] = np.var(all_loads)
        
        return distribution
    
    def _calculate_imbalance_score(self, distribution: Dict) -> float:
        """
        计算负载不均衡分数
        0.0 = 完全均衡, 1.0 = 极度不均衡
        """
        all_loads = list(distribution['rsu_loads'].values()) + list(distribution['uav_loads'].values())
        
        if len(all_loads) < 2:
            return 0.0
        
        # 基于标准差和极值差计算不均衡程度
        mean_load = np.mean(all_loads)
        std_load = np.std(all_loads)
        max_load = np.max(all_loads)
        min_load = np.min(all_loads)
        
        # 综合不均衡分数
        variance_component = std_load / (mean_load + 0.1)  # 相对标准差
        range_component = (max_load - min_load) / (max_load + 0.1)  # 相对极值差
        
        imbalance_score = 0.6 * variance_component + 0.4 * range_component
        return min(1.0, imbalance_score)
    
    def _identify_migration_opportunities(self, node_states: Dict, 
                                        distribution: Dict, 
                                        imbalance_score: float) -> List[Dict]:
        """
        识别迁移机会 - 基于全局优化而非局部规则
        """
        opportunities = []
        
        if imbalance_score < 0.2:  # 系统已经比较均衡
            return opportunities
        
        # 找出过载节点和空闲节点
        system_avg = distribution['system_average']
        overloaded_nodes = []
        underloaded_nodes = []
        
        # RSU分析
        for node_id, load in distribution['rsu_loads'].items():
            if load > system_avg + 0.3:  # 比平均高30%算过载
                queue_len = node_states[node_id].get('queue_length', 0)
                overloaded_nodes.append({
                    'node_id': node_id,
                    'node_type': 'rsu',
                    'load': load,
                    'queue_length': queue_len,
                    'excess_load': load - system_avg
                })
            elif load < system_avg - 0.2:  # 比平均低20%算空闲
                queue_len = node_states[node_id].get('queue_length', 0)
                underloaded_nodes.append({
                    'node_id': node_id,
                    'node_type': 'rsu',
                    'load': load,
                    'queue_length': queue_len,
                    'available_capacity': system_avg - load
                })
        
        # UAV分析
        for node_id, load in distribution['uav_loads'].items():
            battery = node_states[node_id].get('battery_level', 1.0)
            if load > system_avg + 0.2 and battery > 0.3:  # UAV过载且有电
                queue_len = node_states[node_id].get('queue_length', 0)
                overloaded_nodes.append({
                    'node_id': node_id,
                    'node_type': 'uav',
                    'load': load,
                    'queue_length': queue_len,
                    'battery': battery,
                    'excess_load': load - system_avg
                })
        
        # 生成迁移机会
        for overloaded in overloaded_nodes:
            for underloaded in underloaded_nodes:
                # 计算迁移收益
                potential_benefit = self._calculate_migration_benefit(
                    overloaded, underloaded, system_avg
                )
                
                if potential_benefit > self.strategy_params['min_benefit_threshold']:
                    opportunity = {
                        'source': overloaded,
                        'target': underloaded,
                        'potential_benefit': potential_benefit,
                        'estimated_success_rate': self._estimate_success_rate(
                            overloaded, underloaded
                        ),
                        'risk_level': self._calculate_risk_level(overloaded, underloaded)
                    }
                    opportunities.append(opportunity)
        
        # 按潜在收益排序
        opportunities.sort(key=lambda x: x['potential_benefit'], reverse=True)
        return opportunities
    
    def _calculate_migration_benefit(self, source: Dict, target: Dict, 
                                   system_avg: float) -> float:
        """
        计算迁移的潜在收益
        """
        # 负载均衡收益
        source_excess = source['excess_load']
        target_capacity = target.get('available_capacity', 0)
        balance_benefit = min(source_excess, target_capacity) * 0.5
        
        # 系统整体优化收益
        current_variance = (source['load'] - system_avg)**2 + (target['load'] - system_avg)**2
        
        # 假设迁移部分任务后的新负载
        migration_ratio = 0.3  # 迁移30%的超额负载
        new_source_load = source['load'] - source_excess * migration_ratio
        new_target_load = target['load'] + source_excess * migration_ratio * 0.8  # 80%效率
        
        new_variance = (new_source_load - system_avg)**2 + (new_target_load - system_avg)**2
        variance_reduction = current_variance - new_variance
        
        # 综合收益
        total_benefit = 0.6 * balance_benefit + 0.4 * variance_reduction
        return max(0.0, total_benefit)
    
    def _estimate_success_rate(self, source: Dict, target: Dict) -> float:
        """
        估算迁移成功率 - 基于实际系统状态
        """
        base_success_rate = 0.7  # 基础成功率70%
        
        # 基于源节点负载调整
        if source['load'] > 1.5:  # 严重过载时迁移成功率下降
            base_success_rate *= 0.8
        
        # 基于目标节点状态调整
        if target['load'] < 0.3:  # 目标很空闲时成功率提高
            base_success_rate *= 1.2
        elif target['load'] > 0.8:  # 目标已较忙时成功率下降
            base_success_rate *= 0.7
        
        # 基于节点类型调整
        if source['node_type'] == 'uav' and target['node_type'] == 'rsu':
            base_success_rate *= 0.9  # UAV到RSU稍难
        elif source['node_type'] == 'rsu' and target['node_type'] == 'rsu':
            base_success_rate *= 1.1  # RSU间迁移稍易
        
        return min(0.95, max(0.4, base_success_rate))
    
    def _calculate_risk_level(self, source: Dict, target: Dict) -> float:
        """
        计算迁移风险等级
        0.0 = 低风险, 1.0 = 高风险
        """
        risk_factors = []
        
        # 源节点风险
        if source['load'] > 1.5:
            risk_factors.append(0.3)  # 严重过载时迁移有风险
        
        # 目标节点风险
        if target['load'] > 0.7:
            risk_factors.append(0.4)  # 目标较忙时有风险
        
        # 跨节点类型风险
        if source['node_type'] != target['node_type']:
            risk_factors.append(0.2)
        
        return min(1.0, sum(risk_factors))
    
    def _generate_global_recommendation(self, imbalance_score: float, 
                                      opportunities: List[Dict]) -> Dict:
        """
        生成全局迁移建议
        """
        if imbalance_score < 0.2:
            return {
                'action': 'maintain',
                'reason': '系统负载均衡良好',
                'priority': 'low'
            }
        
        if not opportunities:
            return {
                'action': 'wait',
                'reason': '暂无有益迁移机会',
                'priority': 'low'
            }
        
        # 选择最佳迁移机会
        best_opportunity = opportunities[0]
        
        if best_opportunity['potential_benefit'] > 0.3:
            priority = 'high'
            action = 'migrate_aggressive'
        elif best_opportunity['potential_benefit'] > 0.15:
            priority = 'medium'
            action = 'migrate_balanced'
        else:
            priority = 'low'
            action = 'migrate_conservative'
        
        return {
            'action': action,
            'reason': f"全局负载不均衡({imbalance_score:.2f})",
            'priority': priority,
            'best_opportunity': best_opportunity
        }
    
    def should_attempt_risky_migration(self, opportunity: Dict) -> bool:
        """
        决定是否尝试有风险的迁移
        好的策略应该敢于承担calculated risk
        """
        benefit = opportunity['potential_benefit']
        risk = opportunity['risk_level']
        estimated_success = opportunity['estimated_success_rate']
        
        # 期望收益 = 收益 × 成功率 - 风险成本
        expected_value = benefit * estimated_success - risk * 0.1
        
        # 🎯 关键：即使成功率不高，但期望收益为正就尝试
        if expected_value > 0.05:
            return True
        
        # 🎯 探索性迁移：低概率尝试未知策略
        if np.random.random() < self.strategy_params['exploration_probability']:
            return True
        
        return False
    
    def get_strategy_quality_metrics(self) -> Dict:
        """
        获取策略质量指标 - 比成功率更重要的指标
        """
        return {
            'system_balance_improvement': self._calculate_balance_improvement(),
            'exploration_ratio': len([o for o in self.migration_outcomes if o.get('exploratory', False)]) / max(1, len(self.migration_outcomes)),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(),
            'global_optimization_score': self._calculate_global_optimization_score(),
            'strategy_diversity': self._calculate_strategy_diversity()
        }
    
    def _calculate_balance_improvement(self) -> float:
        """计算负载均衡改善程度"""
        if len(self.system_load_history) < 10:
            return 0.0
        
        recent_variance = np.var(self.system_load_history[-5:])
        early_variance = np.var(self.system_load_history[:5])
        
        if early_variance == 0:
            return 0.0
        
        improvement = (early_variance - recent_variance) / early_variance
        return max(-1.0, min(1.0, improvement))
    
    def _calculate_risk_reward_ratio(self) -> float:
        """计算风险收益比"""
        if not self.migration_outcomes:
            return 0.0
        
        total_risk = sum(o.get('risk_taken', 0) for o in self.migration_outcomes)
        total_reward = sum(o.get('actual_benefit', 0) for o in self.migration_outcomes)
        
        if total_risk == 0:
            return float('inf') if total_reward > 0 else 0.0
        
        return total_reward / total_risk
    
    def _calculate_global_optimization_score(self) -> float:
        """计算全局优化分数"""
        if len(self.global_performance_trend) < 5:
            return 0.5
        
        # 基于系统性能趋势
        recent_performance = np.mean(self.global_performance_trend[-5:])
        early_performance = np.mean(self.global_performance_trend[:5])
        
        improvement = (recent_performance - early_performance) / abs(early_performance + 0.1)
        return max(0.0, min(1.0, 0.5 + improvement))
    
    def _calculate_strategy_diversity(self) -> float:
        """计算策略多样性"""
        if len(self.migration_outcomes) < 10:
            return 0.5
        
        # 统计不同类型的迁移
        migration_types = {}
        for outcome in self.migration_outcomes[-20:]:
            mig_type = f"{outcome.get('source_type', 'unknown')}_{outcome.get('target_type', 'unknown')}"
            migration_types[mig_type] = migration_types.get(mig_type, 0) + 1
        
        # 计算多样性指数
        total_migrations = sum(migration_types.values())
        diversity_score = 0.0
        
        for count in migration_types.values():
            probability = count / total_migrations
            diversity_score -= probability * np.log(probability + 1e-10)
        
        # 归一化到[0,1]
        max_diversity = np.log(len(migration_types) + 1)
        return diversity_score / max_diversity if max_diversity > 0 else 0.0

def create_quality_focused_migration_strategy() -> IntelligentMigrationStrategy:
    """创建注重质量的迁移策略"""
    return IntelligentMigrationStrategy()

# 全局智能迁移策略
_global_migration_strategy = IntelligentMigrationStrategy()

def evaluate_migration_from_quality_perspective(node_states: Dict) -> Dict:
    """从质量角度评估迁移策略"""
    return _global_migration_strategy.evaluate_migration_necessity(node_states)

def get_migration_strategy_metrics() -> Dict:
    """获取迁移策略质量指标"""
    return _global_migration_strategy.get_strategy_quality_metrics()
