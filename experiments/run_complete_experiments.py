#!/usr/bin/env python3
"""
完整实验套件自动运行脚本
按顺序执行所有必需和推荐的实验
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def run_command(cmd, description, estimated_time):
    """运行命令并监控进度"""
    print("\n" + "=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)
    print(f"⏱️  预计耗时: {estimated_time}")
    print(f"📝 命令: {' '.join(cmd)}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=parent_dir,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ 完成！实际耗时: {elapsed/60:.1f}分钟")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断")
        return False


def main():
    print("=" * 80)
    print("🎯 VEC系统完整实验套件")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print("\n请选择实验方案:")
    print("1. 方案A - 快速完成 (约6-8小时)")
    print("   - Baseline对比")
    print("   - 消融实验分析")
    print("   - 参数敏感性（部分）")
    print()
    print("2. 方案B - 标准完成 (约15-20小时) ⭐推荐")
    print("   - 所有方案A内容")
    print("   - 完整参数敏感性")
    print("   - 多种子实验")
    print("   - 收敛性分析")
    print()
    print("3. 自定义 - 选择特定实验")
    print()
    
    choice = input("请选择方案 [1/2/3]: ").strip()
    
    experiments = []
    
    if choice == '1':
        # 方案A：快速完成
        experiments = [
            {
                'cmd': ['python', 'baseline_comparison/run_baseline_comparison.py', 
                       '--episodes', '200'],
                'desc': '[1/3] Baseline对比实验',
                'time': '4-5小时'
            },
            {
                'cmd': ['python', 'ablation_experiments/analyze_results.py'],
                'desc': '[2/3] 消融实验深度分析',
                'time': '5-10分钟'
            },
            {
                'cmd': ['python', 'experiments/run_parameter_sensitivity.py',
                       '--analysis', 'vehicle', '--episodes', '150'],
                'desc': '[3/3] 车辆数敏感性分析（快速）',
                'time': '1-2小时'
            },
        ]
        
    elif choice == '2':
        # 方案B：标准完成
        experiments = [
            {
                'cmd': ['python', 'baseline_comparison/run_baseline_comparison.py',
                       '--episodes', '200'],
                'desc': '[1/6] Baseline对比实验',
                'time': '4-5小时'
            },
            {
                'cmd': ['python', 'ablation_experiments/analyze_results.py'],
                'desc': '[2/6] 消融实验深度分析',
                'time': '5-10分钟'
            },
            {
                'cmd': ['python', 'experiments/run_parameter_sensitivity.py',
                       '--analysis', 'all', '--episodes', '200'],
                'desc': '[3/6] 完整参数敏感性分析',
                'time': '8-10小时'
            },
            {
                'cmd': ['python', 'experiments/run_td3_seed_sweep.py',
                       '--seeds', '42', '2025', '3407', '12345', '99999',
                       '--episodes', '200'],
                'desc': '[4/6] 多种子鲁棒性验证',
                'time': '2-3小时'
            },
            {
                'cmd': ['python', 'visualization/analyze_convergence.py'],
                'desc': '[5/6] 收敛性分析',
                'time': '10-15分钟'
            },
            {
                'cmd': ['python', 'visualization/generate_paper_figures.py'],
                'desc': '[6/6] 生成论文图表',
                'time': '5-10分钟'
            },
        ]
        
    elif choice == '3':
        # 自定义实验
        print("\n可选实验:")
        print("1. Baseline对比")
        print("2. 参数敏感性分析")
        print("3. 多种子实验")
        print("4. 收敛性分析")
        print("5. 消融实验分析")
        
        selections = input("\n请输入实验编号（用空格分隔，如: 1 2 3）: ").strip().split()
        
        experiment_map = {
            '1': {
                'cmd': ['python', 'baseline_comparison/run_baseline_comparison.py',
                       '--episodes', '200'],
                'desc': 'Baseline对比实验',
                'time': '4-5小时'
            },
            '2': {
                'cmd': ['python', 'experiments/run_parameter_sensitivity.py',
                       '--analysis', 'all', '--episodes', '200'],
                'desc': '参数敏感性分析',
                'time': '8-10小时'
            },
            '3': {
                'cmd': ['python', 'experiments/run_td3_seed_sweep.py',
                       '--seeds', '42', '2025', '3407', '--episodes', '200'],
                'desc': '多种子实验',
                'time': '2-3小时'
            },
            '4': {
                'cmd': ['python', 'visualization/analyze_convergence.py'],
                'desc': '收敛性分析',
                'time': '10-15分钟'
            },
            '5': {
                'cmd': ['python', 'ablation_experiments/analyze_results.py'],
                'desc': '消融实验分析',
                'time': '5-10分钟'
            },
        }
        
        for i, sel in enumerate(selections):
            if sel in experiment_map:
                exp = experiment_map[sel]
                exp['desc'] = f"[{i+1}/{len(selections)}] {exp['desc']}"
                experiments.append(exp)
    
    else:
        print("❌ 无效选择")
        return
    
    # 确认开始
    print("\n" + "=" * 80)
    print("📋 实验计划:")
    print("=" * 80)
    total_time = 0
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['desc']} ({exp['time']})")
    print("=" * 80)
    
    confirm = input("\n确认开始实验？[y/N]: ").strip().lower()
    if confirm != 'y':
        print("❌ 取消实验")
        return
    
    # 开始执行实验
    start_time = datetime.now()
    successful = 0
    failed = 0
    
    for exp in experiments:
        success = run_command(exp['cmd'], exp['desc'], exp['time'])
        if success:
            successful += 1
        else:
            failed += 1
            
            # 询问是否继续
            if failed > 0:
                cont = input("\n⚠️  实验失败。是否继续执行剩余实验？[y/N]: ").strip().lower()
                if cont != 'y':
                    print("❌ 停止实验")
                    break
    
    # 总结
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("📊 实验完成总结")
    print("=" * 80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed/3600:.1f}小时 ({elapsed/60:.0f}分钟)")
    print(f"成功: {successful}/{len(experiments)}")
    print(f"失败: {failed}/{len(experiments)}")
    print("=" * 80)
    
    if successful == len(experiments):
        print("🎉 所有实验成功完成！")
        print("\n下一步:")
        print("1. 查看 results/ 目录中的实验结果")
        print("2. 运行 python visualization/generate_paper_figures.py 生成论文图表")
        print("3. 开始撰写论文实验部分")
    else:
        print("⚠️  部分实验未完成，请检查错误信息")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断实验套件")
        sys.exit(1)
