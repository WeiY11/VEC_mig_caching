#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证延迟暴涨问题的脚本

比较新旧版本的性能差异
"""

print("=" * 80)
print("🔍 延迟暴涨问题验证")
print("=" * 80)

# ========== 读取当前配置 ==========
from config.system_config import config

print("\n📊 当前系统配置:")
print(f"  RSU CPU频率: {config.compute.rsu_cpu_freq / 1e9:.2f} GHz")
print(f"  UAV CPU频率: {config.compute.uav_cpu_freq / 1e9:.2f} GHz")
print(f"  带宽: {config.network.bandwidth / 1e6:.2f} MHz")

# ========== 参考值（新代码中的硬编码值） ==========
print("\n📐 system_simulator.py 中的参考值:")
reference_rsu_freq = 15e9
reference_uav_freq = 12e9
reference_bandwidth = 20e6

print(f"  RSU CPU频率参考: {reference_rsu_freq / 1e9:.2f} GHz")
print(f"  UAV CPU频率参考: {reference_uav_freq / 1e9:.2f} GHz")
print(f"  带宽参考: {reference_bandwidth / 1e6:.2f} MHz")

# ========== 计算性能比例 ==========
print("\n🔧 性能缩放比例:")
rsu_freq_ratio = config.compute.rsu_cpu_freq / reference_rsu_freq
uav_freq_ratio = config.compute.uav_cpu_freq / reference_uav_freq
bandwidth_ratio = config.network.bandwidth / reference_bandwidth

print(f"  RSU freq_ratio: {rsu_freq_ratio:.3f} ({rsu_freq_ratio * 100:.1f}%)")
print(f"  UAV freq_ratio: {uav_freq_ratio:.3f} ({uav_freq_ratio * 100:.1f}%)")
print(f"  bandwidth_ratio: {bandwidth_ratio:.3f} ({bandwidth_ratio * 100:.1f}%)")

# ========== 影响分析 ==========
print("\n🚨 性能影响分析:")

if rsu_freq_ratio < 1.0:
    print(f"  ❌ RSU性能下降 {(1 - rsu_freq_ratio) * 100:.1f}%")
    print(f"     - work_capacity 减小至 {rsu_freq_ratio * 100:.1f}%")
    print(f"     - base_divisor 减小至 {rsu_freq_ratio * 100:.1f}%")
    print(f"     → 队列延迟增加 + 计算延迟增加")
else:
    print(f"  ✅ RSU性能正常")

if uav_freq_ratio < 1.0:
    print(f"  ❌ UAV性能下降 {(1 - uav_freq_ratio) * 100:.1f}%！")
    print(f"     - work_capacity 减小至 {uav_freq_ratio * 100:.1f}%")
    print(f"     - base_divisor 减小至 {uav_freq_ratio * 100:.1f}%")
    print(f"     → 队列延迟增加 + 计算延迟增加")
    if uav_freq_ratio < 0.2:
        print(f"     ⚠️  警告：UAV性能仅剩 {uav_freq_ratio * 100:.1f}%，严重瓶颈！")
else:
    print(f"  ✅ UAV性能正常")

if bandwidth_ratio < 1.0:
    print(f"  ❌ 带宽下降 {(1 - bandwidth_ratio) * 100:.1f}%")
    print(f"     - base_rate 减小至 {bandwidth_ratio * 100:.1f}%")
    print(f"     → 传输延迟增加")
else:
    print(f"  ✅ 带宽正常")

# ========== 综合评估 ==========
print("\n📈 延迟影响综合评估:")
print("  总延迟 = 传输延迟 + 队列等待延迟 + 计算延迟")
print("")

if rsu_freq_ratio < 1.0 or uav_freq_ratio < 1.0:
    # 计算简化的延迟倍数
    # 假设任务均匀分布到RSU和UAV
    avg_compute_ratio = (rsu_freq_ratio + uav_freq_ratio) / 2
    estimated_delay_increase = 1.0 / avg_compute_ratio
    
    print(f"  估算延迟增长倍数: {estimated_delay_increase:.2f}x")
    print(f"  如果旧版本延迟 = 0.4s")
    print(f"  → 新版本延迟 ≈ {0.4 * estimated_delay_increase:.2f}s")
    print("")
    
    if abs(0.4 * estimated_delay_increase - 1.0) < 0.2:
        print("  ✅ 这与观察到的延迟暴涨（0.4s → 1.0s）**高度吻合**！")
    
    print("")
    print("🎯 结论：延迟暴涨的根本原因是:")
    print("  1. 新代码引入了 CPU 频率和带宽的动态缩放")
    print("  2. 实际配置的频率低于参考值：")
    print(f"     - RSU: {config.compute.rsu_cpu_freq / 1e9:.1f} GHz < {reference_rsu_freq / 1e9:.1f} GHz")
    print(f"     - UAV: {config.compute.uav_cpu_freq / 1e9:.1f} GHz < {reference_uav_freq / 1e9:.1f} GHz ⚠️")
    print("  3. 导致计算能力、队列处理能力大幅下降")
    print("  4. 综合效果：延迟暴涨 2.5x")

else:
    print("  ✅ 配置参数均 ≥ 参考值，不应出现延迟暴涨")
    print("  如果仍有延迟问题，需要进一步调查其他因素")

# ========== 解决方案 ==========
print("\n" + "=" * 80)
print("✅ 推荐解决方案")
print("=" * 80)

print("\n方案1: 调整配置参数（推荐）")
print("  修改 config/system_config.py:")
print("  ```python")
print("  # 在 ComputeConfig.__init__ 中：")
print(f"  self.rsu_default_freq = 15e9  # 当前: {config.compute.rsu_cpu_freq / 1e9:.1f} GHz")
print(f"  self.uav_default_freq = 12e9  # 当前: {config.compute.uav_cpu_freq / 1e9:.1f} GHz")
print("  ```")

print("\n方案2: 回退到旧版本")
print("  git reset --hard 6d5bd8f")

print("\n方案3: 修复缩放逻辑（需要重新校准）")
print("  修改 evaluation/system_simulator.py")
print("  调整 reference_rsu_freq 和 reference_uav_freq 与实际配置一致")

print("\n" + "=" * 80)

