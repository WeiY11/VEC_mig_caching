#!/usr/bin/env python3
# 匹配训练结果和配置名称

import os

# 14个训练结果文件（按时间排序）
result_files = [
    ("130212", "training_results_20251102_130212.json"),
    ("133219", "training_results_20251102_133219.json"),
    ("140220", "training_results_20251102_140220.json"),
    ("143208", "training_results_20251102_143208.json"),
    ("150158", "training_results_20251102_150158.json"),
    ("153226", "training_results_20251102_153226.json"),
    ("160246", "training_results_20251102_160246.json"),
    ("163726", "training_results_20251102_163726.json"),
    ("170758", "training_results_20251102_170758.json"),
    ("174023", "training_results_20251102_174023.json"),
    ("182026", "training_results_20251102_182026.json"),
    ("190111", "training_results_20251102_190111.json"),
    ("193909", "training_results_20251102_193909.json"),
    ("201444", "training_results_20251102_201444.json"),
]

# 配置目录（从list_dir结果）
config_dirs = [
    "aggressive_20251102_153233",
    "balanced_20251102_114422",
    "balanced_20251102_122216",
    "balanced_20251102_140227",
    "balanced_v2_20251102_163734",
    "cache_aggressive_20251102_170805",
    "cache_enhanced_20251102_143215",
    "comprehensive_20251102_193917",
    "conservative_20251102_160254",
    "current_20251102_123208",
    "delay_priority_20251102_130219",
    "energy_priority_20251102_133226",
    "energy_saver_20251102_190120",
    "high_reliability_20251102_150205",
    "min_cost_20251102_174031",
    "strict_latency_20251102_182034",
]

# 提取时间戳
def extract_time(s):
    import re
    match = re.search(r'(\d{6})', s)
    return match.group(1) if match else None

# 匹配
print("\n" + "="*80)
print("训练结果 <-> 配置名称 匹配")
print("="*80)

for i, (time, filename) in enumerate(result_files, 1):
    # 找到最接近的配置目录
    time_int = int(time)
    best_match = None
    min_diff = float('inf')
    
    for config_dir in config_dirs:
        config_time = extract_time(config_dir.split('_', 1)[1])
        if config_time:
            config_time_int = int(config_time)
            diff = abs(config_time_int - time_int)
            if diff < min_diff:
                min_diff = diff
                best_match = config_dir.rsplit('_', 2)[0]
    
    marker = "🏆" if i == 9 else f"{i:2d}"
    print(f"{marker}. config_{i:2d} ({time}) -> {best_match:20s} (时间差: {min_diff}秒)")

print("\n" + "="*80)
print("结论: config_9 (最优) = aggressive 配置")
print("="*80)

