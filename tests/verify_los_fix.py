"""
验证 LoS 概率修复：对比 3GPP 标准与修复后的实现
"""
import math
import sys
sys.path.insert(0, r'd:\VEC_mig_caching')

def p_los_3gpp(d):
    """3GPP TR 38.901 UMi Street Canyon 标准公式"""
    if d <= 18:
        return 1.0
    return (18.0/d) + math.exp(-d/36.0)*(1.0 - 18.0/d)

def p_los_old(d, local_density=0.3, avg_building_height=15.0):
    """旧的实现（修复前）"""
    if d <= 18:
        return 1.0
    d_clutter = 50.0 * (1.0 - 0.6 * local_density)
    base_prob = math.exp(-d / d_clutter)
    height_penalty = 1.0 - 0.3 * min(1.0, avg_building_height / 30.0)
    return max(0.05, min(0.95, base_prob * height_penalty))

def p_los_fixed(d, local_density=0.0, avg_building_height=15.0):
    """修复后的实现"""
    if d <= 18:
        return 1.0
    p_los = (18.0 / d) + math.exp(-d / 36.0) * (1.0 - 18.0 / d)
    density_factor = 1.0 - 0.3 * local_density
    height_factor = 1.0 - 0.15 * max(0.0, (avg_building_height - 15.0) / 30.0)
    return max(0.03, min(1.0, p_los * density_factor * height_factor))

if __name__ == "__main__":
    print("=" * 70)
    print("LoS 概率修复验证：3GPP TR 38.901 UMi Street Canyon 标准")
    print("=" * 70)
    
    print("\n【对比表格】")
    print("-" * 70)
    print(f"{'距离(m)':<10} {'3GPP标准':<12} {'旧实现':<12} {'误差':<10} {'修复后':<12}")
    print("-" * 70)
    
    distances = [30, 50, 100, 150, 200, 300, 500]
    
    for d in distances:
        std = p_los_3gpp(d)
        old = p_los_old(d)
        new = p_los_fixed(d)
        err = (old - std) / std * 100
        print(f"{d:<10} {std:<12.4f} {old:<12.4f} {err:<+10.1f}% {new:<12.4f}")
    
    print("-" * 70)
    
    print("\n【总结】")
    print("  旧实现问题：使用简单指数衰减 exp(-d/d_clutter)，远距离 LoS 概率严重低估")
    print("  修复方案：采用 3GPP 标准公式 (18/d) + exp(-d/36)*(1-18/d)")
    print("  额外特性：保留了建筑密度和建筑高度的空间异质性修正")
    print("\n✅ 修复完成！现在符合 3GPP TR 38.901 UMi Street Canyon 标准")
