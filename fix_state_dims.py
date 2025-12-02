#!/usr/bin/env python3
"""修复 head_train_single_agent.py 中的状态维度不一致问题"""

import re

# 读取文件
with open('head_train_single_agent.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到需要修改的行
modified_lines = []
skip_until_next_for = False
in_rsu_block = False
in_uav_block = False
skip_cpu_freq_line = False

i = 0
while i < len(lines):
    line = lines[i]
    
    # 检测 RSU 状态块的开始（带有 cpu_freq_norm）
    if 'cpu_freq_norm = normalize_scalar(rsu.get' in line:
        # 跳过这一行
        i += 1
        continue
    
    # 检测 UAV 状态块的开始（带有 cpu_freq_norm）
    if 'cpu_freq_norm = normalize_scalar(uav.get' in line:
        # 跳过这一行
        i += 1
        continue
    
    # 跳过 cpu_freq_norm 相关的数组元素行
    if 'cpu_freq_norm,' in line:
        i += 1
        continue
    
    # 修改 UAV 状态中的高度为队列利用率（在某些位置）
    # 检查是否是 position[2] (高度) 行，并且后面跟着 cache_utilization
    if "normalize_scalar(uav['position'][2]" in line:
        next_line = lines[i+1] if i+1 < len(lines) else ''
        # 如果下一行是 cache_utilization，说明是旧格式 [x, y, z, cache, energy, cpu]
        # 需要改为 [x, y, queue, cache, energy]
        if 'cache_utilization' in next_line or '_calculate_correct_cache_utilization' in next_line:
            # 替换为队列利用率
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + "normalize_scalar(len(uav.get('computation_queue', [])), 'uav_queue_capacity', 20.0),  # 队列利用率\n"
            modified_lines.append(new_line)
            i += 1
            continue
    
    modified_lines.append(line)
    i += 1

# 写回文件
with open('head_train_single_agent.py', 'w', encoding='utf-8') as f:
    f.writelines(modified_lines)

print(f"处理完成，原文件 {len(lines)} 行，新文件 {len(modified_lines)} 行")

# 验证修改
with open('head_train_single_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

print(f"剩余 cpu_freq_norm 出现次数: {content.count('cpu_freq_norm')}")
