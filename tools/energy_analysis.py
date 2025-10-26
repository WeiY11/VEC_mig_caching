#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
能耗计算全面分析脚本
手动计算 vs 代码实现对比，找出问题
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from config import config as sys_config
except Exception:
    sys_config = None


def manual_calculation_typical_task():
    """手动计算典型任务的能耗"""
    print("=== 手动计算典型任务能耗 ===")
    
    # 典型任务参数
    data_size_mb = 1.0  # 1MB
    data_size_bytes = data_size_mb * 1e6
    data_size_bits = data_size_bytes * 8
    compute_density = 120  # cycles/bit
    distance = 200.0  # meters
    
    print(f"任务：{data_size_mb}MB, 计算密度{compute_density} cycles/bit, 距离{distance}m")
    
    # 从配置获取参数
    if sys_config:
        # RSU参数（实际配置值）
        rsu_freq = sys_config.compute.rsu_default_freq  # 6e9
        rsu_kappa = sys_config.compute.rsu_kappa  # 2.8e-31
        rsu_static = sys_config.compute.rsu_static_power  # 25.0W
        
        # 车辆发射功率
        vehicle_tx_dbm = sys_config.communication.vehicle_tx_power  # 23dBm
        vehicle_tx_w = 10**((vehicle_tx_dbm - 30) / 10)  # 转换为瓦特
        
        # 通信参数
        bandwidth = sys_config.network.bandwidth  # 20MHz
        carrier_freq = sys_config.communication.carrier_frequency  # 2.4GHz
    else:
        # 默认值
        rsu_freq = 6e9
        rsu_kappa = 2.8e-31
        rsu_static = 25.0
        vehicle_tx_w = 0.2  # 23dBm ≈ 0.2W
        bandwidth = 20e6
        carrier_freq = 2.4e9
    
    print(f"参数：RSU频率{rsu_freq/1e9:.1f}GHz, kappa={rsu_kappa:.2e}, 静态功耗{rsu_static}W")
    print(f"车辆发射功率{vehicle_tx_w:.3f}W, 带宽{bandwidth/1e6:.0f}MHz")
    
    # 1. 手动计算传输能耗
    print("\n--- 传输能耗计算 ---")
    
    # 路径损耗（Free Space）
    path_loss_db = 32.45 + 20 * np.log10(distance/1000) + 20 * np.log10(carrier_freq/1e9)
    print(f"路径损耗: {path_loss_db:.1f}dB")
    
    # 接收信号功率
    rx_signal_dbm = vehicle_tx_dbm - path_loss_db
    print(f"接收信号功率: {rx_signal_dbm:.1f}dBm")
    
    # 噪声功率
    thermal_noise = -174  # dBm/Hz
    noise_figure = 9.0  # dB
    noise_power_dbm = thermal_noise + 10 * np.log10(bandwidth) + noise_figure
    print(f"噪声功率: {noise_power_dbm:.1f}dBm")
    
    # 简化SINR（不考虑干扰）
    sinr_db = rx_signal_dbm - noise_power_dbm
    print(f"SINR: {sinr_db:.1f}dB")
    
    # Shannon容量
    if sinr_db > -10:
        sinr_linear = 10**(sinr_db/10)
        capacity_bps = bandwidth * np.log2(1 + sinr_linear)
        transmission_time = data_size_bits / capacity_bps
    else:
        transmission_time = float('inf')
    
    transmission_energy = vehicle_tx_w * transmission_time
    print(f"传输时间: {transmission_time*1000:.2f}ms")
    print(f"传输能耗: {transmission_energy:.6f}J")
    
    # 2. 手动计算RSU计算能耗
    print("\n--- RSU计算能耗计算 ---")
    
    total_cycles = data_size_bits * compute_density
    computation_time = total_cycles / rsu_freq
    print(f"计算周期: {total_cycles:.0f}")
    print(f"计算时间: {computation_time:.3f}s")
    
    # 动态功率模型（论文式）
    dynamic_power = rsu_kappa * (rsu_freq ** 3)
    computation_energy = (dynamic_power + rsu_static) * computation_time
    print(f"动态功率: {dynamic_power:.1f}W")
    print(f"总功率: {dynamic_power + rsu_static:.1f}W")
    print(f"计算能耗: {computation_energy:.3f}J")
    
    # 总能耗
    total_energy_manual = transmission_energy + computation_energy
    print(f"\n手动计算总能耗: {total_energy_manual:.3f}J")
    
    return {
        'transmission_energy': transmission_energy,
        'computation_energy': computation_energy,
        'total_energy': total_energy_manual,
        'transmission_time': transmission_time,
        'computation_time': computation_time,
        'parameters': {
            'rsu_freq': rsu_freq,
            'rsu_kappa': rsu_kappa,
            'rsu_static': rsu_static,
            'vehicle_tx_w': vehicle_tx_w
        }
    }


def check_code_implementation():
    """检查代码实现中的问题"""
    print("\n=== 代码实现问题检查 ===")
    
    problems = []
    
    # 1. 检查参数一致性
    if sys_config:
        config_rsu_freq = sys_config.compute.rsu_default_freq
        config_rsu_kappa = sys_config.compute.rsu_kappa
        config_rsu_static = sys_config.compute.rsu_static_power
        
        print(f"配置文件RSU参数: freq={config_rsu_freq/1e9:.1f}GHz, kappa={config_rsu_kappa:.2e}, static={config_rsu_static}W")
        
        # 检查默认值
        default_freq = 50e9  # 代码中的默认值
        default_kappa = 1e-27
        default_static = 2.0
        
        print(f"代码默认RSU参数: freq={default_freq/1e9:.1f}GHz, kappa={default_kappa:.2e}, static={default_static}W")
        
        if abs(config_rsu_freq - default_freq) / config_rsu_freq > 0.1:
            problems.append(f"❌ RSU频率不一致: 配置{config_rsu_freq/1e9:.1f}GHz vs 代码默认{default_freq/1e9:.1f}GHz")
        
        if abs(config_rsu_kappa - default_kappa) / config_rsu_kappa > 0.1:
            problems.append(f"❌ RSU kappa不一致: 配置{config_rsu_kappa:.2e} vs 代码默认{default_kappa:.2e}")
            
        if abs(config_rsu_static - default_static) / config_rsu_static > 0.1:
            problems.append(f"❌ RSU静态功耗不一致: 配置{config_rsu_static}W vs 代码默认{default_static}W")
    
    # 2. 检查公式一致性
    print(f"\n--- 公式一致性检查 ---")
    
    # UAV公式不一致
    problems.append("❌ UAV能耗公式不一致:")
    problems.append("   - communication/models.py: kappa3 * f^2 * t （论文式28）")
    problems.append("   - evaluation/test_complete_system.py: kappa * f^3 * t")
    problems.append("   - decision/offloading_manager.py: kappa3 * f^2 * t")
    
    # RSU公式检查
    problems.append("❌ RSU能耗公式差异:")
    problems.append("   - communication/models.py: kappa2 * f^3 * t")
    problems.append("   - evaluation/test_complete_system.py: (kappa * f^3 + static) * t")
    
    # 3. 检查缺失组件
    problems.append("⚠️ 缺失的能耗组件:")
    problems.append("   - UAV悬停能耗未在主仿真器中计算")
    problems.append("   - 下行传输能耗（结果下载）未计算")
    problems.append("   - 空闲时间能耗处理不一致")
    
    # 4. 检查数值合理性
    problems.append("⚠️ 数值合理性问题:")
    problems.append("   - 某些默认参数导致频率过高（50GHz vs 6GHz）")
    problems.append("   - kappa系数差异巨大（1e-27 vs 2.8e-31）")
    
    for problem in problems:
        print(problem)
    
    return problems


def code_vs_manual_comparison():
    """代码计算 vs 手动计算对比"""
    print("\n=== 代码计算 vs 手动计算对比 ===")
    
    # 手动计算结果
    manual_result = manual_calculation_typical_task()
    
    # 模拟代码计算（使用代码中的默认值）
    print(f"\n--- 代码计算（使用默认参数）---")
    
    # 代码中的默认参数
    code_rsu_freq = 50e9  # 代码默认
    code_rsu_kappa = 1e-27  # 代码默认
    code_rsu_static = 2.0  # 代码默认
    
    data_size_bits = 8e6
    compute_density = 120
    
    # 使用代码的计算逻辑
    total_cycles = data_size_bits * compute_density
    computation_time_code = total_cycles / code_rsu_freq
    dynamic_power_code = code_rsu_kappa * (code_rsu_freq ** 3)
    computation_energy_code = (dynamic_power_code + code_rsu_static) * computation_time_code
    
    print(f"代码RSU参数: freq={code_rsu_freq/1e9:.1f}GHz, kappa={code_rsu_kappa:.2e}, static={code_rsu_static}W")
    print(f"计算时间: {computation_time_code:.6f}s")
    print(f"动态功率: {dynamic_power_code:.1f}W")
    print(f"计算能耗: {computation_energy_code:.6f}J")
    
    # 对比
    print(f"\n--- 对比结果 ---")
    manual_comp = manual_result['computation_energy']
    ratio = computation_energy_code / manual_comp if manual_comp > 0 else float('inf')
    print(f"手动计算能耗: {manual_comp:.3f}J")
    print(f"代码计算能耗: {computation_energy_code:.6f}J")
    print(f"比值（代码/手动）: {ratio:.2f}")
    
    if ratio > 10 or ratio < 0.1:
        print("❌ 能耗计算差异巨大！存在严重问题！")
    elif ratio > 2 or ratio < 0.5:
        print("⚠️ 能耗计算差异较大，需要检查参数")
    else:
        print("✅ 能耗计算基本一致")


def main():
    manual_calculation_typical_task()
    problems = check_code_implementation()
    code_vs_manual_comparison()
    
    print(f"\n=== 总结 ===")
    print(f"发现 {len([p for p in problems if '❌' in p])} 个严重问题")
    print(f"发现 {len([p for p in problems if '⚠️' in p])} 个警告")
    
    print("\n建议修复优先级：")
    print("1. 统一参数：修复代码中的默认值，使用配置文件参数")
    print("2. 统一公式：确保所有模块使用相同的能耗公式")
    print("3. 补充组件：添加UAV悬停能耗和下行传输能耗")


if __name__ == "__main__":
    main()
