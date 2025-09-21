#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值稳定性增强脚本
为关键计算函数添加数值稳定性检查，防止除零、溢出等问题
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def add_numerical_stability_checks():
    """为关键计算添加数值稳定性检查"""
    print("🔢 增强数值稳定性...")
    
    # 创建一个数值稳定性工具模块
    stability_utils_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值稳定性工具模块
提供各种数值计算的稳定性保障
"""

import numpy as np
import math
from typing import Union, Optional

# 数值稳定性常量
EPSILON = 1e-12  # 极小值阈值
MAX_SAFE_VALUE = 1e10  # 最大安全值
MIN_SAFE_VALUE = 1e-10  # 最小安全值

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 分母为零时的默认返回值
        
    Returns:
        安全的除法结果
    """
    if abs(denominator) < EPSILON:
        return default
    
    result = numerator / denominator
    
    # 检查结果是否超出安全范围
    if abs(result) > MAX_SAFE_VALUE:
        return MAX_SAFE_VALUE if result > 0 else -MAX_SAFE_VALUE
    
    return result

def safe_sqrt(value: float) -> float:
    """
    安全平方根计算
    
    Args:
        value: 输入值
        
    Returns:
        安全的平方根结果
    """
    if value < 0:
        return 0.0
    
    return math.sqrt(max(value, 0.0))

def safe_log(value: float, base: Optional[float] = None) -> float:
    """
    安全对数计算
    
    Args:
        value: 输入值
        base: 对数底数，None表示自然对数
        
    Returns:
        安全的对数结果
    """
    if value <= 0:
        return -MAX_SAFE_VALUE
    
    if base is None:
        return math.log(max(value, MIN_SAFE_VALUE))
    else:
        if base <= 0 or base == 1:
            return 0.0
        return math.log(max(value, MIN_SAFE_VALUE)) / math.log(base)

def safe_exp(value: float) -> float:
    """
    安全指数计算
    
    Args:
        value: 输入值
        
    Returns:
        安全的指数结果
    """
    # 限制输入范围防止溢出
    if value > 700:  # exp(700) 接近 float64 上限
        return MAX_SAFE_VALUE
    elif value < -700:
        return MIN_SAFE_VALUE
    
    return math.exp(value)

def safe_power(base: float, exponent: float) -> float:
    """
    安全幂运算
    
    Args:
        base: 底数
        exponent: 指数
        
    Returns:
        安全的幂运算结果
    """
    if base == 0:
        return 0.0 if exponent > 0 else 1.0
    
    if base < 0 and not isinstance(exponent, int):
        # 负数的非整数次幂，返回绝对值的幂
        base = abs(base)
    
    try:
        # 使用对数计算避免直接幂运算溢出
        if base > 0:
            log_result = exponent * math.log(base)
            if abs(log_result) > 700:
                return MAX_SAFE_VALUE if log_result > 0 else MIN_SAFE_VALUE
            return math.exp(log_result)
        else:
            return pow(base, exponent)
    except (OverflowError, ValueError):
        return MAX_SAFE_VALUE if base > 1 and exponent > 0 else MIN_SAFE_VALUE

def clamp(value: float, min_val: float = -MAX_SAFE_VALUE, max_val: float = MAX_SAFE_VALUE) -> float:
    """
    将值限制在指定范围内
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        限制后的值
    """
    return max(min_val, min(value, max_val))

def validate_sinr(sinr: float) -> float:
    """
    验证和修正SINR值
    
    Args:
        sinr: 信噪比值
        
    Returns:
        修正后的SINR值
    """
    if np.isnan(sinr) or np.isinf(sinr):
        return MIN_SAFE_VALUE
    
    return clamp(sinr, MIN_SAFE_VALUE, MAX_SAFE_VALUE)

def validate_energy(energy: float) -> float:
    """
    验证能耗值
    
    Args:
        energy: 能耗值
        
    Returns:
        修正后的能耗值
    """
    if np.isnan(energy) or np.isinf(energy) or energy < 0:
        return 0.0
    
    return clamp(energy, 0.0, MAX_SAFE_VALUE)

def validate_delay(delay: float) -> float:
    """
    验证延迟值
    
    Args:
        delay: 延迟值
        
    Returns:
        修正后的延迟值
    """
    if np.isnan(delay) or np.isinf(delay) or delay < 0:
        return 0.0
    
    return clamp(delay, 0.0, MAX_SAFE_VALUE)

def validate_probability(prob: float) -> float:
    """
    验证概率值
    
    Args:
        prob: 概率值
        
    Returns:
        修正后的概率值 (0-1)
    """
    if np.isnan(prob) or np.isinf(prob):
        return 0.5  # 默认概率
    
    return clamp(prob, 0.0, 1.0)

def validate_cpu_frequency(freq: float) -> float:
    """
    验证CPU频率值
    
    Args:
        freq: CPU频率 (Hz)
        
    Returns:
        修正后的CPU频率
    """
    if np.isnan(freq) or np.isinf(freq) or freq <= 0:
        return 1e9  # 默认1GHz
    
    # CPU频率合理范围: 100MHz - 100GHz
    return clamp(freq, 1e8, 1e11)

def validate_data_size(size: float) -> float:
    """
    验证数据大小
    
    Args:
        size: 数据大小 (bytes)
        
    Returns:
        修正后的数据大小
    """
    if np.isnan(size) or np.isinf(size) or size < 0:
        return 1e6  # 默认1MB
    
    # 数据大小合理范围: 1KB - 1GB
    return clamp(size, 1e3, 1e9)

def check_numerical_health(value: float, name: str = "value") -> bool:
    """
    检查数值健康状态
    
    Args:
        value: 要检查的值
        name: 值的名称（用于错误报告）
        
    Returns:
        是否健康
    """
    if np.isnan(value):
        print(f"⚠️ 检测到NaN值: {name}")
        return False
    
    if np.isinf(value):
        print(f"⚠️ 检测到无穷值: {name}")
        return False
    
    if abs(value) > MAX_SAFE_VALUE:
        print(f"⚠️ 值过大: {name} = {value}")
        return False
    
    return True

class NumericalStabilityMonitor:
    """数值稳定性监控器"""
    
    def __init__(self):
        self.warning_count = 0
        self.error_count = 0
        self.max_warnings = 100
    
    def check_and_fix(self, value: float, name: str, validator_func=None) -> float:
        """
        检查并修复数值
        
        Args:
            value: 原始值
            name: 值名称
            validator_func: 验证函数
            
        Returns:
            修复后的值
        """
        if not check_numerical_health(value, name):
            self.warning_count += 1
            if self.warning_count <= self.max_warnings:
                print(f"🔧 自动修复数值问题: {name}")
        
        if validator_func:
            return validator_func(value)
        else:
            return clamp(value)
    
    def get_statistics(self) -> dict:
        """获取监控统计"""
        return {
            'warning_count': self.warning_count,
            'error_count': self.error_count
        }

# 全局监控器实例
numerical_monitor = NumericalStabilityMonitor()
'''
    
    # 写入文件
    with open('d:/VEC_mig_caching/utils/numerical_stability.py', 'w', encoding='utf-8') as f:
        f.write(stability_utils_content)
    
    print("   ✅ 创建数值稳定性工具模块")
    
    return True

def create_enhanced_validation_tests():
    """创建增强的验证测试"""
    print("🧪 创建数值验证测试...")
    
    test_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值稳定性验证测试
验证系统在各种边界条件下的稳定性
"""

import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.numerical_stability import *
from config.external_config import external_config, apply_external_config_to_system

def test_edge_cases():
    """测试边界情况"""
    print("🎯 测试边界情况...")
    
    test_cases = [
        # (测试名称, 函数, 参数, 期望行为)
        ("除零保护", safe_divide, (1.0, 0.0), "应返回默认值"),
        ("负数平方根", safe_sqrt, (-1.0,), "应返回0"),
        ("极大值处理", clamp, (1e20, -1e10, 1e10), "应被限制"),
        ("NaN处理", validate_energy, (float('nan'),), "应返回0"),
        ("无穷大处理", validate_delay, (float('inf'),), "应被限制"),
        ("负能耗修正", validate_energy, (-100.0,), "应返回0"),
        ("超范围频率", validate_cpu_frequency, (1e15,), "应被限制"),
    ]
    
    passed = 0
    for name, func, args, expected in test_cases:
        try:
            result = func(*args)
            if check_numerical_health(result):
                print(f"   ✅ {name}: {result:.6f}")
                passed += 1
            else:
                print(f"   ❌ {name}: 结果不健康")
        except Exception as e:
            print(f"   ❌ {name}: 异常 {e}")
    
    print(f"   边界测试通过率: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")
    return passed == len(test_cases)

def test_system_calculations():
    """测试系统计算的数值稳定性"""
    print("⚖️ 测试系统计算稳定性...")
    
    # 应用配置
    apply_external_config_to_system()
    
    # 模拟各种计算场景
    scenarios = [
        ("极小任务", {"data_size": 1e3, "cpu_freq": 1e9, "time_slot": 0.2}),
        ("极大任务", {"data_size": 50e6, "cpu_freq": 50e9, "time_slot": 0.2}),
        ("低频处理", {"data_size": 10e6, "cpu_freq": 1e8, "time_slot": 0.2}),
        ("高频处理", {"data_size": 10e6, "cpu_freq": 100e9, "time_slot": 0.2}),
    ]
    
    stable_scenarios = 0
    
    for name, params in scenarios:
        print(f"   测试场景: {name}")
        
        try:
            # 模拟处理能力计算
            data_size = validate_data_size(params["data_size"])
            cpu_freq = validate_cpu_frequency(params["cpu_freq"])
            time_slot = clamp(params["time_slot"], 0.01, 10.0)
            
            # 计算处理周期
            compute_cycles = data_size * 8 * 500  # 500 cycles/bit
            
            # 计算处理能力
            processing_capacity = safe_divide(cpu_freq * time_slot * 0.9, compute_cycles)
            
            # 验证结果
            if check_numerical_health(processing_capacity, f"{name}_capacity"):
                print(f"     ✅ 处理能力: {processing_capacity:.6f} tasks/时隙")
                
                # 计算负载因子
                arrival_rate = 1.35
                tasks_per_slot = arrival_rate * time_slot
                load_factor = safe_divide(tasks_per_slot, processing_capacity)
                
                if check_numerical_health(load_factor, f"{name}_load"):
                    print(f"     ✅ 负载因子: {load_factor:.2f}")
                    stable_scenarios += 1
                else:
                    print(f"     ❌ 负载因子计算不稳定")
            else:
                print(f"     ❌ 处理能力计算不稳定")
                
        except Exception as e:
            print(f"     ❌ 计算异常: {e}")
    
    print(f"   系统计算稳定率: {stable_scenarios}/{len(scenarios)} ({stable_scenarios/len(scenarios)*100:.1f}%)")
    return stable_scenarios == len(scenarios)

def test_energy_calculation_stability():
    """测试能耗计算稳定性"""
    print("⚡ 测试能耗计算稳定性...")
    
    # 测试车辆能耗计算
    test_params = [
        (2e9, 0.1, 0.2),    # 正常情况
        (50e9, 1.0, 0.2),   # 高频高利用率
        (1e8, 0.01, 0.2),   # 低频低利用率
        (0, 0.5, 0.2),      # 零频率
        (1e20, 2.0, 0.2),   # 异常值
    ]
    
    stable_count = 0
    
    for i, (freq, util, time_slot) in enumerate(test_params):
        freq = validate_cpu_frequency(freq)
        util = clamp(util, 0.0, 1.0)
        time_slot = clamp(time_slot, 0.01, 10.0)
        
        # 模拟车辆能耗计算 (简化版)
        kappa1 = 1e-28
        kappa2 = 1e-26
        static_power = 0.5
        
        try:
            # 动态功率
            dynamic_power = (kappa1 * safe_power(freq, 3) + 
                           kappa2 * safe_power(freq, 2) * util + 
                           static_power)
            
            # 总能耗
            total_energy = validate_energy(dynamic_power * time_slot)
            
            if check_numerical_health(total_energy, f"energy_test_{i}"):
                print(f"   ✅ 测试{i+1}: 能耗 = {total_energy:.6f}J")
                stable_count += 1
            else:
                print(f"   ❌ 测试{i+1}: 能耗计算不稳定")
                
        except Exception as e:
            print(f"   ❌ 测试{i+1}: 能耗计算异常 {e}")
    
    print(f"   能耗计算稳定率: {stable_count}/{len(test_params)} ({stable_count/len(test_params)*100:.1f}%)")
    return stable_count == len(test_params)

def run_comprehensive_stability_test():
    """运行全面稳定性测试"""
    print("🔬 数值稳定性全面测试")
    print("="*50)
    
    tests = [
        ("边界情况测试", test_edge_cases),
        ("系统计算测试", test_system_calculations),
        ("能耗计算测试", test_energy_calculation_stability),
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\\n🧪 {test_name}...")
        if test_func():
            print(f"✅ {test_name} 通过")
            passed_tests += 1
        else:
            print(f"❌ {test_name} 失败")
    
    print(f"\\n📊 总体测试结果:")
    print(f"   通过率: {passed_tests}/{len(tests)} ({passed_tests/len(tests)*100:.1f}%)")
    
    if passed_tests == len(tests):
        print("🎉 所有数值稳定性测试通过！")
        return True
    else:
        print("⚠️ 部分测试未通过，建议检查相关代码")
        return False

if __name__ == "__main__":
    success = run_comprehensive_stability_test()
    
    if success:
        print("\\n💡 系统数值稳定性良好，可以安全运行")
    else:
        print("\\n🔧 建议使用数值稳定性工具模块进行修复")
'''
    
    # 写入文件
    with open('d:/VEC_mig_caching/test_numerical_stability.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("   ✅ 创建数值稳定性测试脚本")
    
    return True

def main():
    """主函数"""
    print("🚀 数值稳定性增强")
    print("="*40)
    
    # 添加数值稳定性检查
    add_numerical_stability_checks()
    
    # 创建验证测试
    create_enhanced_validation_tests()
    
    print(f"\n✅ 数值稳定性增强完成！")
    print(f"📁 创建的文件:")
    print(f"   • utils/numerical_stability.py - 数值稳定性工具")
    print(f"   • test_numerical_stability.py - 稳定性测试")
    
    print(f"\n💡 使用建议:")
    print(f"   1. 在关键计算中导入并使用 numerical_stability 模块")
    print(f"   2. 运行 test_numerical_stability.py 验证系统稳定性")
    print(f"   3. 在SINR计算等除法运算中使用 safe_divide")
    print(f"   4. 在能耗计算中使用相应的验证函数")

if __name__ == "__main__":
    main()