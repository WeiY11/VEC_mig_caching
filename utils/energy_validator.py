#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Energy Consumption Validator Module
用于验证能耗计算的合理性和准确性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class EnergyValidator:
    """能耗验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 定义合理的能耗范围 (单位: J) - 根据实际场景调整
        self.energy_ranges = {
            'vehicle_compute': (0.0, 100.0),     # 车辆计算能耗
            'vehicle_transmit': (0.0, 50.0),     # 车辆传输能耗
            'edge_compute': (0.0, 5000.0),       # 边缘服务器计算能耗（调整为更高阈值）
            'uav_compute': (0.0, 200.0),         # UAV计算能耗
            'uav_hover': (0.0, 2500.0),          # UAV悬停能耗（100步*25W*0.2s*2UAV=1000J，留余量）
            'uav_move': (0.0, 100.0),            # UAV移动能耗
            'downlink': (0.0, 500.0),            # 下行传输能耗
            'total_system': (0.0, 10000.0),      # 系统总能耗（调整为更高阈值）
        }
        
        # 能耗异常阈值
        self.anomaly_thresholds = {
            'zero_energy_rate': 0.05,      # 零能耗比例不应超过5%
            'max_energy_rate': 0.01,       # 最大能耗比例不应超过1%
            'variance_threshold': 10000,    # 能耗方差不应过大
        }
    
    def validate_energy_consumption(self, energy_data: Dict) -> Dict:
        """
        验证能耗数据的合理性
        
        Args:
            energy_data: 包含各种能耗数据的字典
            
        Returns:
            验证结果字典
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # 1. 检查能耗范围
            self._check_energy_ranges(energy_data, validation_results)
            
            # 2. 检查能耗分布
            self._check_energy_distribution(energy_data, validation_results)
            
            # 3. 检查能耗一致性
            self._check_energy_consistency(energy_data, validation_results)
            
            # 4. 计算统计信息
            self._calculate_energy_statistics(energy_data, validation_results)
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"能耗验证过程中发生错误: {str(e)}")
            self.logger.error(f"Energy validation error: {e}")
        
        return validation_results
    
    def _check_energy_ranges(self, energy_data: Dict, results: Dict) -> None:
        """检查能耗值是否在合理范围内"""
        for energy_type, values in energy_data.items():
            if energy_type not in self.energy_ranges:
                results['warnings'].append(f"未知的能耗类型: {energy_type}")
                continue
            
            values_array = np.array(values) if not isinstance(values, np.ndarray) else values
            min_val, max_val = self.energy_ranges[energy_type]
            
            # 检查是否有超出范围的值
            out_of_range = (values_array < min_val) | (values_array > max_val)
            if np.any(out_of_range):
                out_count = np.sum(out_of_range)
                total_count = len(values_array)
                percentage = (out_count / total_count) * 100
                
                if percentage > 5.0:  # 超过5%认为是错误
                    results['is_valid'] = False
                    results['errors'].append(
                        f"{energy_type}能耗有{out_count}个值({percentage:.1f}%)超出合理范围"
                        f"[{min_val}, {max_val}]"
                    )
                else:
                    results['warnings'].append(
                        f"{energy_type}能耗有{out_count}个值({percentage:.1f}%)超出合理范围"
                    )
    
    def _check_energy_distribution(self, energy_data: Dict, results: Dict) -> None:
        """检查能耗分布的合理性"""
        for energy_type, values in energy_data.items():
            values_array = np.array(values) if not isinstance(values, np.ndarray) else values
            
            if len(values_array) == 0:
                continue
            
            # 检查零能耗比例
            zero_count = np.sum(values_array == 0)
            zero_rate = zero_count / len(values_array)
            
            if zero_rate > self.anomaly_thresholds['zero_energy_rate']:
                results['warnings'].append(
                    f"{energy_type}能耗中有{zero_rate*100:.1f}%为零，可能存在问题"
                )
            
            # 检查最大值占比
            if len(values_array) > 1:
                max_val = np.max(values_array)
                max_count = np.sum(values_array == max_val)
                max_rate = max_count / len(values_array)
                
                if max_rate > self.anomaly_thresholds['max_energy_rate'] and max_count > 1:
                    results['warnings'].append(
                        f"{energy_type}能耗中有{max_rate*100:.1f}%为最大值，分布可能异常"
                    )
            
            # 检查方差
            if len(values_array) > 1:
                variance = np.var(values_array)
                if variance > self.anomaly_thresholds['variance_threshold']:
                    results['warnings'].append(
                        f"{energy_type}能耗方差过大({variance:.2f})，数据波动异常"
                    )
    
    def _check_energy_consistency(self, energy_data: Dict, results: Dict) -> None:
        """检查能耗数据的一致性"""
        # 检查总能耗是否等于各部分之和
        component_types = ['vehicle_compute', 'vehicle_transmit', 'edge_compute', 
                          'uav_compute', 'uav_hover', 'uav_move']
        
        if 'total_system' in energy_data:
            total_energy = np.array(energy_data['total_system'])
            
            # 计算各组件能耗之和
            component_sum = np.zeros_like(total_energy)
            available_components = []
            
            for comp_type in component_types:
                if comp_type in energy_data:
                    comp_values = np.array(energy_data[comp_type])
                    if len(comp_values) == len(total_energy):
                        component_sum += comp_values
                        available_components.append(comp_type)
            
            if available_components:
                # 检查总和一致性
                diff = np.abs(total_energy - component_sum)
                relative_error = diff / (total_energy + 1e-8)  # 避免除零
                
                large_error_mask = relative_error > 0.1  # 相对误差超过10%
                if np.any(large_error_mask):
                    error_count = np.sum(large_error_mask)
                    error_rate = error_count / len(total_energy)
                    
                    if error_rate > 0.05:  # 超过5%认为是错误
                        results['is_valid'] = False
                        results['errors'].append(
                            f"总能耗与组件能耗之和不一致，{error_count}个样本"
                            f"({error_rate*100:.1f}%)相对误差超过10%"
                        )
                    else:
                        results['warnings'].append(
                            f"总能耗与组件能耗之和存在轻微不一致，{error_count}个样本相对误差超过10%"
                        )
    
    def _calculate_energy_statistics(self, energy_data: Dict, results: Dict) -> None:
        """计算能耗统计信息"""
        stats = {}
        
        for energy_type, values in energy_data.items():
            values_array = np.array(values) if not isinstance(values, np.ndarray) else values
            
            if len(values_array) == 0:
                continue
            
            stats[energy_type] = {
                'count': len(values_array),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'zero_count': int(np.sum(values_array == 0)),
                'zero_rate': float(np.sum(values_array == 0) / len(values_array))
            }
        
        results['statistics'] = stats


def validate_energy_consumption(energy_data: Dict) -> Dict:
    """
    验证能耗数据的便捷函数
    
    Args:
        energy_data: 包含各种能耗数据的字典
        
    Returns:
        验证结果字典
    """
    validator = EnergyValidator()
    return validator.validate_energy_consumption(energy_data)


def check_energy_efficiency(current_energy: float, baseline_energy: float, 
                           threshold: float = 0.1) -> Tuple[bool, str]:
    """
    检查能耗效率
    
    Args:
        current_energy: 当前能耗
        baseline_energy: 基线能耗
        threshold: 效率阈值
        
    Returns:
        (是否高效, 说明信息)
    """
    if baseline_energy <= 0:
        return False, "基线能耗无效"
    
    efficiency_ratio = (baseline_energy - current_energy) / baseline_energy
    
    if efficiency_ratio >= threshold:
        return True, f"能耗效率提升{efficiency_ratio*100:.1f}%"
    elif efficiency_ratio >= 0:
        return True, f"能耗效率轻微提升{efficiency_ratio*100:.1f}%"
    else:
        return False, f"能耗增加{-efficiency_ratio*100:.1f}%"


def validate_single_task_energy(task_energy: Dict, task_size: float, 
                               task_cycles: float) -> Dict:
    """
    验证单个任务的能耗合理性
    
    Args:
        task_energy: 任务能耗字典
        task_size: 任务数据大小(MB)
        task_cycles: 任务计算周期数
        
    Returns:
        验证结果
    """
    results = {
        'is_valid': True,
        'warnings': [],
        'energy_efficiency': {}
    }
    
    # 计算能耗效率指标
    if 'compute' in task_energy and task_cycles > 0:
        compute_efficiency = task_energy['compute'] / task_cycles
        results['energy_efficiency']['compute_per_cycle'] = compute_efficiency
        
        if compute_efficiency > 10.0:  # 每周期超过10J认为异常
            results['warnings'].append(f"计算能耗效率异常: {compute_efficiency:.2f}J/cycle")
    
    if 'transmit' in task_energy and task_size > 0:
        transmit_efficiency = task_energy['transmit'] / task_size
        results['energy_efficiency']['transmit_per_mb'] = transmit_efficiency
        
        if transmit_efficiency > 5.0:  # 每MB超过5J认为异常
            results['warnings'].append(f"传输能耗效率异常: {transmit_efficiency:.2f}J/MB")
    
    return results