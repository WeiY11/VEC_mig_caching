#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json

files = [
    ('1.0', 'results/single_agent/td3/training_results_20251031_202734.json'),
    ('1.5', 'results/single_agent/td3/training_results_20251031_195709.json'),
    ('2.0', 'results/single_agent/td3/training_results_20251031_192845.json'),
    ('2.5', 'results/single_agent/td3/training_results_20251031_190006.json'),
    ('3.0', 'results/single_agent/td3/training_results_20251031_183056.json'),
]

print("="*80)
print("Checking Actual Task Arrival Rate Used in Training")
print("="*80)

for expected_rate, filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    actual_rate = data['override_scenario'].get('task_arrival_rate', 'NOT SET')
    
    print(f"Expected: {expected_rate} tasks/s  -->  Actual: {actual_rate} tasks/s")
    
    if abs(float(actual_rate) - float(expected_rate)) > 0.01:
        print(f"  [ERROR] Mismatch!")

print("="*80)

