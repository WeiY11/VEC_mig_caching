"""
按任务类别时延性能可视化工具

生成多维度的任务类别时延统计图表，包括：
1. 平均时延对比柱状图
2. deadline违约率饼图
3. 最大时延与deadline阈值对比图
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, Any
import os

# 设置中文字体和样式
try:
    # 尝试使用微软雅黑
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
except:
    # 如果失败,使用默认字体
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def visualize_task_type_delay_stats(stats: Dict[str, Any], output_dir: str = 'test_results'):
    """
    生成任务类别时延统计的可视化图表
    
    Args:
        stats: 仿真器统计数据字典
        output_dir: 输出目录
    """
    task_type_stats = stats.get('task_type_delay_stats', {})
    
    if not task_type_stats:
        print("⚠️ 未找到任务类别时延统计数据")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 任务类型名称映射
    type_names = {
        1: "类型1\n(极度敏感)",
        2: "类型2\n(敏感)",
        3: "类型3\n(中度容忍)",
        4: "类型4\n(容忍)"
    }
    
    # 提取数据
    task_types = []
    avg_delays = []
    max_delays = []
    deadlines = []
    violation_counts = []
    task_counts = []
    violation_rates = []
    
    for task_type in sorted(task_type_stats.keys()):
        type_data = task_type_stats[task_type]
        count = type_data.get('count', 0)
        
        if count > 0:
            task_types.append(task_type)
            total_delay = type_data.get('total_delay', 0.0)
            avg_delays.append(total_delay / count)
            max_delays.append(type_data.get('max_delay', 0.0))
            deadlines.append(type_data.get('deadline', 0.0))
            violations = type_data.get('deadline_violations', 0)
            violation_counts.append(violations)
            task_counts.append(count)
            violation_rates.append(violations / count if count > 0 else 0.0)
    
    if not task_types:
        print("⚠️ 没有完成的任务数据")
        return
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    
    # 1. 平均时延对比柱状图
    ax1 = plt.subplot(2, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(range(len(task_types)), avg_delays, 
                   color=colors[:len(task_types)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, delay) in enumerate(zip(bars, avg_delays)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{delay:.4f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('任务类型', fontsize=12, fontweight='bold')
    ax1.set_ylabel('平均时延 (秒)', fontsize=12, fontweight='bold')
    ax1.set_title('各任务类型平均时延对比', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(task_types)))
    ax1.set_xticklabels([type_names.get(t, f'Type{t}') for t in task_types])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F8F9FA')
    
    # 2. deadline违约率饼图
    ax2 = plt.subplot(2, 2, 2)
    
    # 计算总体违约率
    total_tasks = sum(task_counts)
    total_violations = sum(violation_counts)
    compliant_tasks = total_tasks - total_violations
    
    if total_tasks > 0:
        pie_data = [compliant_tasks, total_violations]
        pie_labels = [f'符合Deadline\n({compliant_tasks}个任务)', 
                     f'违约\n({total_violations}个任务)']
        pie_colors = ['#96CEB4', '#FF6B6B']
        explode = (0, 0.1)
        
        wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                           autopct='%1.1f%%', startangle=90, explode=explode,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'},
                                           pctdistance=0.85)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
    
    ax2.set_title(f'总体Deadline违约率\n(总计{total_tasks}个任务)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 3. 最大时延与deadline阈值对比图
    ax3 = plt.subplot(2, 2, 3)
    
    x = np.arange(len(task_types))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, max_delays, width, label='最大时延',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, deadlines, width, label='Deadline阈值',
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('任务类型', fontsize=12, fontweight='bold')
    ax3.set_ylabel('时延 (秒)', fontsize=12, fontweight='bold')
    ax3.set_title('最大时延与Deadline阈值对比', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels([type_names.get(t, f'Type{t}') for t in task_types])
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.set_facecolor('#F8F9FA')
    
    # 4. 各任务类型详细违约率对比
    ax4 = plt.subplot(2, 2, 4)
    
    bars = ax4.barh(range(len(task_types)), 
                    [rate * 100 for rate in violation_rates],
                    color=colors[:len(task_types)], alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    # 添加数值标签和任务数量
    for i, (bar, rate, count, vio_count) in enumerate(zip(bars, violation_rates, task_counts, violation_counts)):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f' {rate*100:.1f}% ({vio_count}/{count})',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_yticks(range(len(task_types)))
    ax4.set_yticklabels([type_names.get(t, f'Type{t}') for t in task_types])
    ax4.set_xlabel('违约率 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('各任务类型Deadline违约率对比', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax4.set_facecolor('#F8F9FA')
    ax4.set_xlim(0, max([rate * 100 for rate in violation_rates] + [10]) * 1.2)
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    output_path = os.path.join(output_dir, 'task_type_delay_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 任务类别时延分析图表已保存: {output_path}")
    
    plt.close()
    
    # 生成详细统计报告
    _generate_summary_chart(task_type_stats, task_types, type_names, output_dir)


def _generate_summary_chart(task_type_stats: Dict, task_types: list, 
                           type_names: Dict, output_dir: str):
    """生成汇总统计图表"""
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.axis('off')
    
    # 标题
    title_text = "任务类别时延性能统计汇总"
    ax.text(0.5, 0.95, title_text, ha='center', va='top',
           fontsize=18, fontweight='bold', transform=ax.transAxes)
    
    # 表格数据
    table_data = []
    headers = ['任务类型', '任务数量', '平均时延(s)', '最大时延(s)', 
              'Deadline(s)', '违约数量', '违约率']
    
    for task_type in task_types:
        type_data = task_type_stats[task_type]
        count = type_data.get('count', 0)
        
        if count > 0:
            avg_delay = type_data.get('total_delay', 0.0) / count
            max_delay = type_data.get('max_delay', 0.0)
            deadline = type_data.get('deadline', 0.0)
            violations = type_data.get('deadline_violations', 0)
            vio_rate = violations / count if count > 0 else 0.0
            
            row = [
                type_names.get(task_type, f'Type{task_type}'),
                str(count),
                f'{avg_delay:.4f}',
                f'{max_delay:.4f}',
                f'{deadline:.2f}',
                str(violations),
                f'{vio_rate:.1%}'
            ]
            table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0.05, 0.1, 0.9, 0.75])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    colors = ['#FFE5E5', '#E5F7F6', '#E5F2F7', '#F0F7E5']
    for i, row in enumerate(table_data):
        color = colors[i % len(colors)]
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor(color)
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
    
    # 保存表格
    output_path = os.path.join(output_dir, 'task_type_delay_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 任务类别时延汇总表已保存: {output_path}")
    
    plt.close()


def visualize_from_simulator(simulator, output_dir: str = 'test_results'):
    """
    从仿真器对象直接生成可视化图表
    
    Args:
        simulator: CompleteSystemSimulator实例
        output_dir: 输出目录
    """
    visualize_task_type_delay_stats(simulator.stats, output_dir)


if __name__ == "__main__":
    # 示例：从仿真器生成图表
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from evaluation.system_simulator import CompleteSystemSimulator
    
    print("="*80)
    print("生成任务类别时延性能可视化图表")
    print("="*80)
    
    # 创建并运行仿真
    simulator = CompleteSystemSimulator()
    
    print("\n运行仿真...")
    for step in range(100):
        simulator.run_simulation_step(step)
        if (step + 1) % 20 == 0:
            print(f"  完成第 {step + 1} 步")
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    visualize_from_simulator(simulator)
    
    print("\n" + "="*80)
    print("✅ 图表生成完成!")
    print("="*80)
