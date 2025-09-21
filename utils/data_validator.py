#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验数据验证模块
确保训练和测试数据的合理性，检测异常值和错误数据
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """验证级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    message: str
    suggestion: str = ""


class SystemMetricsValidator:
    """系统指标验证器"""
    
    def __init__(self):
        # 定义各项指标的合理范围
        self.metric_ranges = {
            # 延迟相关 (秒)
            'avg_task_delay': (0.001, 10.0),
            'max_task_delay': (0.001, 30.0),
            'transmission_delay': (0.0001, 5.0),
            'processing_delay': (0.001, 20.0),
            'waiting_delay': (0.0, 15.0),
            
            # 能耗相关 (焦耳)
            'total_energy_consumption': (0.0, 5000.0),
            'avg_energy_per_task': (0.1, 500.0),
            'vehicle_energy': (0.0, 1000.0),
            'rsu_energy': (0.0, 2000.0),
            'uav_energy': (0.0, 800.0),
            
            # 率类指标 (0-1)
            'task_completion_rate': (0.0, 1.0),
            'cache_hit_rate': (0.0, 1.0),
            'data_loss_rate': (0.0, 1.0),
            'migration_success_rate': (0.0, 1.0),
            'delay_violation_rate': (0.0, 1.0),
            
            # 利用率指标 (0-1)
            'cpu_utilization': (0.0, 1.0),
            'bandwidth_utilization': (0.0, 1.0),
            'cache_utilization': (0.0, 1.0),
            'queue_utilization': (0.0, 1.0),
            
            # 电池相关 (0-1)
            'avg_uav_battery': (0.0, 1.0),
            'min_uav_battery': (0.0, 1.0),
            
            # 负载相关
            'load_factor': (0.0, 0.99),
            'queue_length': (0, 1000),
            'system_load_ratio': (0.0, 2.0),
        }
        
        # 定义指标间的逻辑关系
        self.logical_constraints = [
            ('task_completion_rate', 'data_loss_rate', 'completion_loss_consistency'),
            ('cache_hit_rate', 'avg_task_delay', 'cache_delay_correlation'),
            ('cpu_utilization', 'total_energy_consumption', 'utilization_energy_correlation'),
        ]
        
        # 历史数据用于趋势分析
        self.history_buffer = []
        self.max_history_size = 100
        
        # 异常检测参数
        self.outlier_threshold = 3.0  # 标准差倍数
        
    def validate_single_metric(self, metric_name: str, value: Any) -> List[ValidationResult]:
        """
        验证单个指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            
        Returns:
            验证结果列表
        """
        results = []
        
        # 基本类型检查
        if not isinstance(value, (int, float)):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                metric_name=metric_name,
                value=0.0,
                expected_range=(0.0, 0.0),
                message=f"指标 {metric_name} 的值类型错误: {type(value)}",
                suggestion="检查数据生成逻辑，确保返回数值类型"
            ))
            return results
        
        # 数值有效性检查
        if not np.isfinite(value):
            level = ValidationLevel.CRITICAL if np.isnan(value) else ValidationLevel.ERROR
            results.append(ValidationResult(
                level=level,
                metric_name=metric_name,
                value=float(value),
                expected_range=(0.0, 0.0),
                message=f"指标 {metric_name} 包含无效值: {value}",
                suggestion="检查计算逻辑中的除零错误或数值溢出"
            ))
            return results
        
        # 范围检查
        if metric_name in self.metric_ranges:
            min_val, max_val = self.metric_ranges[metric_name]
            if not (min_val <= value <= max_val):
                level = ValidationLevel.ERROR if value < 0 else ValidationLevel.WARNING
                results.append(ValidationResult(
                    level=level,
                    metric_name=metric_name,
                    value=float(value),
                    expected_range=(min_val, max_val),
                    message=f"指标 {metric_name} 超出合理范围: {value} (期望: {min_val}-{max_val})",
                    suggestion=self._get_range_suggestion(metric_name, value, min_val, max_val)
                ))
        
        return results
    
    def validate_system_metrics(self, metrics: Dict) -> List[ValidationResult]:
        """
        验证完整的系统指标
        
        Args:
            metrics: 系统指标字典
            
        Returns:
            验证结果列表
        """
        results = []
        
        # 验证每个指标
        for metric_name, value in metrics.items():
            results.extend(self.validate_single_metric(metric_name, value))
        
        # 验证指标间的逻辑关系
        results.extend(self._validate_logical_constraints(metrics))
        
        # 异常值检测
        results.extend(self._detect_outliers(metrics))
        
        # 更新历史记录
        self._update_history(metrics)
        
        return results
    
    def _validate_logical_constraints(self, metrics: Dict) -> List[ValidationResult]:
        """验证指标间的逻辑约束"""
        results = []
        
        # 完成率与丢失率的一致性检查
        completion_rate = metrics.get('task_completion_rate', 0.0)
        loss_rate = metrics.get('data_loss_rate', 0.0)
        if completion_rate + loss_rate > 1.1:  # 允许小误差
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                metric_name='completion_loss_consistency',
                value=completion_rate + loss_rate,
                expected_range=(0.0, 1.0),
                message=f"完成率({completion_rate:.3f})与丢失率({loss_rate:.3f})之和超过1",
                suggestion="检查任务统计逻辑，确保分类互斥且完整"
            ))
        
        # 缓存命中率与延迟的相关性检查
        cache_hit_rate = metrics.get('cache_hit_rate', 0.0)
        avg_delay = metrics.get('avg_task_delay', 0.0)
        if cache_hit_rate > 0.8 and avg_delay > 2.0:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                metric_name='cache_delay_correlation',
                value=avg_delay,
                expected_range=(0.0, 1.0),
                message=f"高缓存命中率({cache_hit_rate:.3f})但延迟较高({avg_delay:.3f}s)",
                suggestion="检查缓存策略实现或延迟计算逻辑"
            ))
        
        return results
    
    def _detect_outliers(self, metrics: Dict) -> List[ValidationResult]:
        """检测异常值"""
        results = []
        
        if len(self.history_buffer) < 10:
            return results  # 历史数据不足，跳过异常检测
        
        for metric_name, current_value in metrics.items():
            if not isinstance(current_value, (int, float)) or not np.isfinite(current_value):
                continue
            
            # 获取历史数据
            historical_values = [h.get(metric_name, current_value) for h in self.history_buffer[-20:]]
            historical_values = [v for v in historical_values if isinstance(v, (int, float)) and np.isfinite(v)]
            
            if len(historical_values) < 5:
                continue
            
            # 计算统计量
            mean_val = np.mean(historical_values)
            std_val = np.std(historical_values)
            
            if std_val > 1e-6:  # 避免除零
                z_score = abs(current_value - mean_val) / std_val
                if z_score > self.outlier_threshold:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        metric_name=metric_name,
                        value=float(current_value),
                        expected_range=(mean_val - 2*std_val, mean_val + 2*std_val),
                        message=f"指标 {metric_name} 可能是异常值: {current_value:.4f} (Z-score: {z_score:.2f})",
                        suggestion="检查该时步的特殊情况或数据记录错误"
                    ))
        
        return results
    
    def _update_history(self, metrics: Dict):
        """更新历史记录"""
        self.history_buffer.append(metrics.copy())
        if len(self.history_buffer) > self.max_history_size:
            self.history_buffer.pop(0)
    
    def _get_range_suggestion(self, metric_name: str, value: float, 
                            min_val: float, max_val: float) -> str:
        """获取范围错误的建议"""
        if value < min_val:
            if 'rate' in metric_name or 'ratio' in metric_name:
                return "检查分母是否为零或计算逻辑错误"
            elif 'delay' in metric_name:
                return "检查时间计算是否正确，可能存在负值"
            elif 'energy' in metric_name:
                return "检查能耗计算逻辑，负能耗不合理"
            else:
                return "检查计算逻辑，确保结果非负"
        else:
            if 'rate' in metric_name or 'ratio' in metric_name:
                return "检查分子是否超过分母，或计算公式错误"
            elif 'delay' in metric_name:
                return "检查是否存在任务堆积或处理能力不足"
            elif 'energy' in metric_name:
                return "检查能耗模型参数，可能存在数值过大"
            else:
                return "检查数值计算是否合理，可能存在累积错误"
    
    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """生成验证报告"""
        if not results:
            return "✅ 所有指标验证通过，数据质量良好"
        
        # 按级别分组
        by_level = {}
        for result in results:
            level = result.level.value
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(result)
        
        report = "📊 数据验证报告\n"
        report += "=" * 50 + "\n"
        
        for level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO']:
            if level in by_level:
                icon = {'CRITICAL': '🔴', 'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️'}[level]
                report += f"\n{icon} {level} ({len(by_level[level])} 项):\n"
                
                for result in by_level[level]:
                    report += f"  • {result.metric_name}: {result.message}\n"
                    if result.suggestion:
                        report += f"    建议: {result.suggestion}\n"
        
        return report


class ExperimentDataValidator:
    """实验数据验证器"""
    
    def __init__(self):
        self.metrics_validator = SystemMetricsValidator()
        self.training_validator = TrainingDataValidator()
        
    def validate_experiment_results(self, results_file: str) -> Dict:
        """
        验证实验结果文件
        
        Args:
            results_file: 结果文件路径
            
        Returns:
            验证结果字典
        """
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f"无法读取结果文件: {e}",
                'validation_results': []
            }
        
        validation_results = []
        
        # 验证训练结果
        if 'episode_metrics' in data:
            for episode, metrics in enumerate(data['episode_metrics']):
                if isinstance(metrics, dict):
                    results = self.metrics_validator.validate_system_metrics(metrics)
                    validation_results.extend(results)
        
        # 验证最终性能
        if 'final_performance' in data:
            results = self.metrics_validator.validate_system_metrics(data['final_performance'])
            validation_results.extend(results)
        
        # 生成报告
        report = self.metrics_validator.generate_validation_report(validation_results)
        
        return {
            'status': 'PASS' if not any(r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] 
                                      for r in validation_results) else 'FAIL',
            'message': '数据验证完成',
            'validation_results': validation_results,
            'report': report,
            'total_issues': len(validation_results),
            'critical_issues': len([r for r in validation_results if r.level == ValidationLevel.CRITICAL]),
            'error_issues': len([r for r in validation_results if r.level == ValidationLevel.ERROR])
        }


class TrainingDataValidator:
    """训练数据验证器"""
    
    def __init__(self):
        self.reward_range = (-20.0, 20.0)
        self.convergence_threshold = 0.1
        
    def validate_training_convergence(self, rewards: List[float]) -> ValidationResult:
        """验证训练收敛性"""
        if len(rewards) < 10:
            return ValidationResult(
                level=ValidationLevel.WARNING,
                metric_name='training_convergence',
                value=0.0,
                expected_range=(0.0, 1.0),
                message="训练数据不足，无法判断收敛性",
                suggestion="增加训练轮次或检查训练过程"
            )
        
        # 计算最后20%数据的方差
        recent_rewards = rewards[-max(10, len(rewards)//5):]
        variance = np.var(recent_rewards)
        
        if variance > self.convergence_threshold:
            return ValidationResult(
                level=ValidationLevel.WARNING,
                metric_name='training_convergence',
                value=variance,
                expected_range=(0.0, self.convergence_threshold),
                message=f"训练可能未收敛，最近奖励方差: {variance:.4f}",
                suggestion="考虑增加训练轮次或调整学习率"
            )
        
        return ValidationResult(
            level=ValidationLevel.INFO,
            metric_name='training_convergence',
            value=variance,
            expected_range=(0.0, self.convergence_threshold),
            message=f"训练收敛良好，奖励方差: {variance:.4f}",
            suggestion=""
        )


# 全局验证器实例
experiment_validator = ExperimentDataValidator()


def validate_system_metrics(metrics: Dict) -> List[ValidationResult]:
    """
    验证系统指标的便捷接口
    
    Args:
        metrics: 系统指标字典
        
    Returns:
        验证结果列表
    """
    validator = SystemMetricsValidator()
    return validator.validate_system_metrics(metrics)


def quick_validate(metrics: Dict) -> str:
    """
    快速验证并返回简要报告
    
    Args:
        metrics: 系统指标字典
        
    Returns:
        简要验证报告
    """
    results = validate_system_metrics(metrics)
    if not results:
        return "✅ 数据验证通过"
    
    critical_count = len([r for r in results if r.level == ValidationLevel.CRITICAL])
    error_count = len([r for r in results if r.level == ValidationLevel.ERROR])
    warning_count = len([r for r in results if r.level == ValidationLevel.WARNING])
    
    if critical_count > 0:
        return f"🔴 严重问题: {critical_count} 项"
    elif error_count > 0:
        return f"❌ 错误: {error_count} 项"
    elif warning_count > 0:
        return f"⚠️ 警告: {warning_count} 项"
    else:
        return "ℹ️ 信息提示"