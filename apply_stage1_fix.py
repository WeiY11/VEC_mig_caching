#!/usr/bin/env python3
"""
OPTIMIZED_TD3 收敛性优化补丁 - 阶段1 (降低探索噪声)

基于训练分析结果,本补丁修改探索策略参数以改善收敛性。

使用方法:
1. 备份原文件: python apply_stage1_fix.py --backup
2. 应用补丁: python apply_stage1_fix.py --apply
3. 恢复备份: python apply_stage1_fix.py --restore

预期效果 (300 episodes):
- 后100轮标准差 < 0.40
- 异常值 < 3%
- 变异系数降至 0.20-0.25
"""

import argparse
import shutil
from pathlib import Path
import re

# 目标文件
TARGET_FILE = Path("d:/VEC_mig_caching/single_agent/optimized_td3_wrapper.py")
BACKUP_FILE = TARGET_FILE.with_suffix('.py.stage1.backup')

# 阶段1修改: 降低探索噪声,加快衰减
STAGE1_CHANGES = {
    'exploration_noise': ('0.15', '0.08', 'L53'),
    'noise_decay': ('0.998', '0.995', 'L54'),
    'min_noise': ('0.02', '0.01', 'L55'),
    'target_noise': ('0.02', '0.015', 'L56'),
    'noise_clip': ('0.05', '0.03', 'L57'),
}

def backup_file():
    """备份原始文件"""
    if not TARGET_FILE.exists():
        print(f"❌ 目标文件不存在: {TARGET_FILE}")
        return False
    
    shutil.copy2(TARGET_FILE, BACKUP_FILE)
    print(f"✅ 已备份至: {BACKUP_FILE}")
    return True

def restore_file():
    """从备份恢复"""
    if not BACKUP_FILE.exists():
        print(f"❌ 备份文件不存在: {BACKUP_FILE}")
        return False
    
    shutil.copy2(BACKUP_FILE, TARGET_FILE)
    print(f"✅ 已从备份恢复: {TARGET_FILE}")
    return True

def apply_stage1_fix():
    """应用阶段1补丁"""
    if not TARGET_FILE.exists():
        print(f"❌ 目标文件不存在: {TARGET_FILE}")
        return False
    
    # 读取原文件
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 应用修改
    modified = content
    changes_applied = []
    
    for param, (old_val, new_val, line_hint) in STAGE1_CHANGES.items():
        # 构造正则表达式匹配
        pattern = rf'(\s+{param}={old_val})(,?\s*#.*)?(\n)'
        
        def replace_with_comment(match):
            indent = match.group(1).split('=')[0]
            comment = match.group(2) if match.group(2) else ''
            
            # 保留原注释或添加新注释
            if '→' not in comment:
                if comment:
                    comment = f"  # {old_val} → {new_val} (阶段1优化){comment.replace('#', '').strip()}"
                else:
                    comment = f"  # {old_val} → {new_val} (阶段1优化)"
            
            return f"{indent}={new_val}{comment}\n"
        
        new_content = re.sub(pattern, replace_with_comment, modified)
        
        if new_content != modified:
            modified = new_content
            changes_applied.append(f"  ✓ {param}: {old_val} → {new_val} ({line_hint})")
        else:
            print(f"  ⚠️ 未找到匹配项: {param}={old_val}")
    
    if not changes_applied:
        print("❌ 没有应用任何修改,请检查文件内容")
        return False
    
    # 写回文件
    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        f.write(modified)
    
    print(f"✅ 已应用阶段1优化补丁:")
    for change in changes_applied:
        print(change)
    
    print(f"\n📝 修改后的文件: {TARGET_FILE}")
    print("\n🚀 下一步:")
    print("   python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 300 --num-vehicles 12 --seed 42")
    
    return True

def verify_changes():
    """验证修改是否成功"""
    if not TARGET_FILE.exists():
        print(f"❌ 文件不存在: {TARGET_FILE}")
        return False
    
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    success = True
    print("\n📋 验证补丁应用状态:")
    
    for param, (old_val, new_val, line_hint) in STAGE1_CHANGES.items():
        if f"{param}={new_val}" in content:
            print(f"  ✅ {param}={new_val}")
        else:
            print(f"  ❌ {param}={new_val} (未找到)")
            success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description='OPTIMIZED_TD3 阶段1优化补丁 - 降低探索噪声',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1. 备份原文件
  python apply_stage1_fix.py --backup
  
  # 2. 应用补丁
  python apply_stage1_fix.py --apply
  
  # 3. 验证修改
  python apply_stage1_fix.py --verify
  
  # 4. 如需恢复
  python apply_stage1_fix.py --restore
        """
    )
    
    parser.add_argument('--backup', action='store_true', help='备份原始文件')
    parser.add_argument('--apply', action='store_true', help='应用阶段1补丁')
    parser.add_argument('--restore', action='store_true', help='从备份恢复')
    parser.add_argument('--verify', action='store_true', help='验证补丁应用状态')
    
    args = parser.parse_args()
    
    if args.backup:
        backup_file()
    elif args.restore:
        restore_file()
    elif args.verify:
        verify_changes()
    elif args.apply:
        print("=" * 60)
        print("🔧 OPTIMIZED_TD3 阶段1优化补丁")
        print("=" * 60)
        print("\n📌 修改内容:")
        print("   - 探索噪声: 0.15 → 0.08 (降低47%)")
        print("   - 噪声衰减: 0.998 → 0.995 (加快3倍)")
        print("   - 最小噪声: 0.02 → 0.01")
        print("   - 目标噪声: 0.02 → 0.015")
        print("   - 噪声裁剪: 0.05 → 0.03")
        print("\n" + "=" * 60 + "\n")
        
        if backup_file():
            if apply_stage1_fix():
                verify_changes()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
