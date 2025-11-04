#!/usr/bin/env python3
"""
实验优化工具：自动化配置精简和实验合并
======================================

【功能】
1. 自动将所有实验的配置数量从5个减少到3个
2. 备份原始文件
3. 生成优化报告

【使用方式】
```bash
# 预览模式（不实际修改）
python experiments/td3_strategy_suite/optimize_experiments.py --dry-run

# 执行优化（会备份原文件）
python experiments/td3_strategy_suite/optimize_experiments.py

# 恢复原始配置
python experiments/td3_strategy_suite/optimize_experiments.py --restore
```
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class ExperimentOptimizer:
    """实验配置优化器"""
    
    def __init__(self, experiments_dir: Path, dry_run: bool = False):
        self.experiments_dir = experiments_dir
        self.dry_run = dry_run
        self.backup_dir = experiments_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.changes = []
        
    def backup_file(self, file_path: Path) -> Path:
        """备份文件"""
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            return backup_path
        return file_path
    
    def optimize_config_list(self, content: str, var_name: str) -> Tuple[str, bool]:
        """
        优化配置列表：5个 → 3个
        
        策略：保留 [第1个, 中间, 最后1个]
        """
        # 匹配配置列表定义
        patterns = [
            # 数组形式
            (rf'{var_name}\s*=\s*\[(.*?)\]', lambda m: self._optimize_array(m, var_name)),
            # range形式
            (rf'{var_name}\s*=\s*list\(range\((.*?)\)\)', lambda m: self._optimize_range(m, var_name)),
        ]
        
        modified = False
        for pattern, optimizer in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                new_content = optimizer(match)
                if new_content != match.group(0):
                    content = content.replace(match.group(0), new_content)
                    modified = True
                break
        
        return content, modified
    
    def _optimize_array(self, match: re.Match, var_name: str) -> str:
        """优化数组形式的配置"""
        array_content = match.group(1)
        
        # 分割元素
        elements = [e.strip() for e in re.split(r',(?![^[\]]*\])', array_content) if e.strip()]
        
        if len(elements) <= 3:
            return match.group(0)  # 已经是3个或更少，不需要优化
        
        # 保留第1个、中间、最后1个
        middle_idx = len(elements) // 2
        optimized_elements = [elements[0], elements[middle_idx], elements[-1]]
        
        # 重新格式化
        optimized_str = f"{var_name} = [{', '.join(optimized_elements)}]"
        
        return optimized_str
    
    def _optimize_range(self, match: re.Match, var_name: str) -> str:
        """优化range形式的配置"""
        # 暂不处理range，需要手动优化
        return match.group(0)
    
    def optimize_file(self, file_path: Path) -> Dict:
        """优化单个文件"""
        print(f"\n处理: {file_path.name}")
        
        # 读取文件
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # 识别配置变量名
        config_vars = self._find_config_vars(content)
        
        if not config_vars:
            print(f"  [INFO] 未找到配置变量")
            return {"file": file_path.name, "modified": False, "configs": []}
        
        # 优化每个配置
        modifications = []
        for var_name in config_vars:
            content, modified = self.optimize_config_list(content, var_name)
            if modified:
                modifications.append(var_name)
                print(f"  [OK] 优化配置: {var_name}")
        
        # 保存文件
        if modifications and not self.dry_run:
            self.backup_file(file_path)
            file_path.write_text(content, encoding='utf-8')
            print(f"  [OK] 已保存（原文件已备份）")
        elif modifications:
            print(f"  [DRY-RUN] 将优化: {', '.join(modifications)}")
        
        return {
            "file": file_path.name,
            "modified": len(modifications) > 0,
            "configs": modifications,
            "original_lines": len(original_content.splitlines()),
            "new_lines": len(content.splitlines()),
        }
    
    def _find_config_vars(self, content: str) -> List[str]:
        """查找配置变量名"""
        # 常见的配置变量名模式
        patterns = [
            r'DEFAULT_(\w+_(?:COUNTS|LEVELS|VALUES|RATES|SPEEDS|CAPACITIES|SIZES))',
            r'(\w+_LEVELS)\s*=',
            r'(\w+_VALUES)\s*=',
            r'(\w+_CONFIGS)\s*=',
        ]
        
        var_names = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            var_names.extend(matches)
        
        return list(set(var_names))  # 去重
    
    def optimize_all(self) -> List[Dict]:
        """优化所有实验文件"""
        # 查找所有对比实验文件
        comparison_files = list(self.experiments_dir.glob("run_*_comparison.py"))
        
        print(f"\n找到 {len(comparison_files)} 个对比实验文件")
        
        results = []
        for file_path in sorted(comparison_files):
            result = self.optimize_file(file_path)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict]) -> None:
        """生成优化报告"""
        print("\n" + "=" * 70)
        print("优化报告")
        print("=" * 70)
        
        modified_count = sum(1 for r in results if r["modified"])
        total_configs = sum(len(r["configs"]) for r in results)
        
        print(f"\n总计:")
        print(f"  - 处理文件: {len(results)} 个")
        print(f"  - 修改文件: {modified_count} 个")
        print(f"  - 优化配置: {total_configs} 个")
        
        if self.dry_run:
            print(f"\n[INFO] 预览模式，未实际修改文件")
        else:
            print(f"\n[OK] 优化完成")
            print(f"备份目录: {self.backup_dir}")
        
        print("\n详细列表:")
        print("-" * 70)
        print(f"{'文件':<45} {'状态':<10} {'优化配置数':<10}")
        print("-" * 70)
        
        for result in results:
            status = "[OK] 已修改" if result["modified"] else "   跳过"
            config_count = len(result["configs"])
            print(f"{result['file']:<45} {status:<12} {config_count:<10}")
        
        print("-" * 70)
        
        # 预估节省时间
        if modified_count > 0:
            print(f"\n预估效果:")
            print(f"  - 配置数减少: 约 40%%")
            print(f"  - 训练时间节省: 约 40%%")
            print(f"  - 如果每个配置训练30秒，每个实验约节省: 2-3分钟")
            print(f"  - 总计 {modified_count} 个实验，约节省: {modified_count * 2.5:.1f} 分钟")
    
    def restore_from_backup(self, backup_dir: Path) -> None:
        """从备份恢复文件"""
        if not backup_dir.exists():
            print(f"[ERROR] 备份目录不存在: {backup_dir}")
            return
        
        backup_files = list(backup_dir.glob("*.py"))
        print(f"\n找到 {len(backup_files)} 个备份文件")
        
        for backup_file in backup_files:
            target_file = self.experiments_dir / backup_file.name
            shutil.copy2(backup_file, target_file)
            print(f"  [OK] 恢复: {backup_file.name}")
        
        print(f"\n[OK] 恢复完成")


def main():
    parser = argparse.ArgumentParser(
        description="实验配置优化工具：自动将5配置精简为3配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览模式（不实际修改）
  python optimize_experiments.py --dry-run
  
  # 执行优化（会自动备份原文件）
  python optimize_experiments.py
  
  # 恢复到优化前的状态
  python optimize_experiments.py --restore
  
  # 从指定备份恢复
  python optimize_experiments.py --restore --backup-dir backups/20231102_153045
        """
    )
    
    parser.add_argument("--dry-run", action="store_true",
                       help="预览模式，不实际修改文件")
    parser.add_argument("--restore", action="store_true",
                       help="从备份恢复文件")
    parser.add_argument("--backup-dir", type=str,
                       help="指定备份目录（用于恢复）")
    
    args = parser.parse_args()
    
    # 确定实验目录
    script_dir = Path(__file__).resolve().parent
    experiments_dir = script_dir
    
    # 创建优化器
    optimizer = ExperimentOptimizer(experiments_dir, dry_run=args.dry_run)
    
    # 执行操作
    if args.restore:
        # 恢复模式
        if args.backup_dir:
            backup_dir = experiments_dir / "backups" / args.backup_dir
        else:
            # 使用最新的备份
            backups = list((experiments_dir / "backups").glob("*"))
            if not backups:
                print("[ERROR] 没有找到备份目录")
                return
            backup_dir = max(backups, key=lambda p: p.stat().st_mtime)
            print(f"使用最新备份: {backup_dir.name}")
        
        optimizer.restore_from_backup(backup_dir)
    else:
        # 优化模式
        print("=" * 70)
        print("实验配置优化工具")
        print("=" * 70)
        print(f"\n目标: 将配置数量从 5个 减少到 3个")
        print(f"策略: 保留 [最小值, 中值, 最大值]")
        print(f"效果: 节省约 40% 训练时间")
        
        if args.dry_run:
            print(f"\n[INFO] 当前为预览模式，不会实际修改文件")
        else:
            print(f"\n[OK] 将自动备份原文件到: backups/")
        
        # 执行优化
        results = optimizer.optimize_all()
        
        # 生成报告
        optimizer.generate_report(results)
        
        if not args.dry_run and any(r["modified"] for r in results):
            print(f"\n[INFO] 提示:")
            print(f"   - 如需恢复，运行: python optimize_experiments.py --restore")
            print(f"   - 备份目录: {optimizer.backup_dir}")


if __name__ == "__main__":
    main()

