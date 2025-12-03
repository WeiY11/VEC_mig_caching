# -*- coding: utf-8 -*-
import re

file_path = r'd:\VEC_mig_caching\evaluation\system_simulator.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到并删除包含乱码print的行
new_lines = []
for i, line in enumerate(lines):
    # 跳过包含乱码迁移打印的行
    if '馃幆' in line and '瑙' in line and 'urgency' in line:
        print(f"Removing line {i+1}: {line.strip()[:50]}...")
        continue
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done. Removed {len(lines) - len(new_lines)} lines")
