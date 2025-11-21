#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEC边缘计算系统 - 场景与算法架构综合图生成器

功能：结合实际场景拓扑与算法架构，生成学术论文级可视化图片
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def create_scenario_architecture_diagram():
    """创建场景与算法架构综合图"""
    
    # 创建大画布：左侧场景，右侧架构
    fig = plt.figure(figsize=(24, 12))
    
    # 设置浅色背景
    fig.patch.set_facecolor('#F8F9FA')
    
    # ==================== 左侧：VEC场景拓扑 ====================
    ax_scene = plt.subplot(121)
    ax_scene.set_xlim(0, 12)
    ax_scene.set_ylim(0, 12)
    ax_scene.axis('off')
    ax_scene.set_aspect('equal')
    
    # 标题
    ax_scene.text(6, 11.5, 'VEC边缘计算场景拓扑', 
                  fontsize=20, ha='center', weight='bold', color='#2C3E50')
    
    # 定义节点位置（4个路口场景）
    # RSU位置（4个路口）
    rsu_positions = [
        (3, 9),   # 左上路口
        (9, 9),   # 右上路口
        (3, 3),   # 左下路口
        (9, 3),   # 右下路口
    ]
    
    # UAV位置（2个）
    uav_positions = [
        (2, 10.5),  # 左上空域
        (10, 2.5),  # 右下空域
    ]
    
    # 车辆位置（12辆，分布在道路上）
    vehicle_positions = [
        (1.5, 9), (2.5, 9), (3.5, 9),  # 左上路口周边
        (7.5, 9), (8.5, 9), (9.5, 9),  # 右上路口周边
        (1.5, 3), (2.5, 3), (3.5, 3),  # 左下路口周边
        (7.5, 3), (8.5, 3), (9.5, 3),  # 右下路口周边
    ]
    
    # 绘制道路（灰色矩形）
    roads = [
        Rectangle((0.5, 8.5), 11, 1, color='#95A5A6', alpha=0.3, zorder=1),  # 上横路
        Rectangle((0.5, 2.5), 11, 1, color='#95A5A6', alpha=0.3, zorder=1),  # 下横路
        Rectangle((2.5, 0.5), 1, 11, color='#95A5A6', alpha=0.3, zorder=1),  # 左竖路
        Rectangle((8.5, 0.5), 1, 11, color='#95A5A6', alpha=0.3, zorder=1),  # 右竖路
    ]
    for road in roads:
        ax_scene.add_patch(road)
    
    # 绘制RSU覆盖范围（浅蓝色圆）
    for pos in rsu_positions:
        coverage = Circle(pos, 2.5, color='#3498DB', alpha=0.15, zorder=2)
        ax_scene.add_patch(coverage)
    
    # 绘制UAV覆盖范围（浅绿色圆）
    for pos in uav_positions:
        coverage = Circle(pos, 2.0, color='#2ECC71', alpha=0.15, zorder=2)
        ax_scene.add_patch(coverage)
    
    # 绘制RSU节点（塔形状）
    for i, pos in enumerate(rsu_positions):
        # 基座
        base = Rectangle((pos[0]-0.15, pos[1]-0.3), 0.3, 0.2, 
                         color='#E74C3C', zorder=5)
        ax_scene.add_patch(base)
        # 塔身
        tower = Polygon([
            (pos[0]-0.1, pos[1]-0.1),
            (pos[0]+0.1, pos[1]-0.1),
            (pos[0]+0.05, pos[1]+0.3),
            (pos[0]-0.05, pos[1]+0.3),
        ], color='#C0392B', zorder=5)
        ax_scene.add_patch(tower)
        # 天线
        ax_scene.plot([pos[0]-0.08, pos[0]+0.08], [pos[1]+0.3, pos[1]+0.3], 
                      'k-', linewidth=2, zorder=5)
        ax_scene.plot([pos[0]-0.06, pos[0]-0.06], [pos[1]+0.3, pos[1]+0.4], 
                      'k-', linewidth=1.5, zorder=5)
        ax_scene.plot([pos[0], pos[0]], [pos[1]+0.3, pos[1]+0.45], 
                      'k-', linewidth=1.5, zorder=5)
        ax_scene.plot([pos[0]+0.06, pos[0]+0.06], [pos[1]+0.3, pos[1]+0.4], 
                      'k-', linewidth=1.5, zorder=5)
        # 标签
        ax_scene.text(pos[0], pos[1]-0.6, f'RSU{i+1}', 
                      fontsize=9, ha='center', weight='bold', color='#C0392B')
        # MEC服务器图标
        server = Rectangle((pos[0]+0.3, pos[1]-0.2), 0.3, 0.25, 
                           color='#34495E', alpha=0.7, zorder=4)
        ax_scene.add_patch(server)
        ax_scene.plot([pos[0]+0.33, pos[0]+0.57], [pos[1]-0.1, pos[1]-0.1], 
                      'w-', linewidth=1, zorder=4)
        ax_scene.plot([pos[0]+0.33, pos[0]+0.57], [pos[1], pos[1]], 
                      'w-', linewidth=1, zorder=4)
    
    # 绘制UAV节点（无人机形状）
    for i, pos in enumerate(uav_positions):
        # 机身
        body = Rectangle((pos[0]-0.15, pos[1]-0.08), 0.3, 0.16, 
                         color='#2ECC71', zorder=5)
        ax_scene.add_patch(body)
        # 旋翼
        for dx, dy in [(-0.2, 0.15), (0.2, 0.15), (-0.2, -0.15), (0.2, -0.15)]:
            prop = Circle((pos[0]+dx, pos[1]+dy), 0.08, 
                          color='#27AE60', alpha=0.6, zorder=4)
            ax_scene.add_patch(prop)
        # 标签
        ax_scene.text(pos[0], pos[1]-0.5, f'UAV{i+1}', 
                      fontsize=9, ha='center', weight='bold', color='#27AE60')
    
    # 绘制车辆（红色汽车）
    for i, pos in enumerate(vehicle_positions):
        # 车身
        car = Rectangle((pos[0]-0.12, pos[1]-0.08), 0.24, 0.16, 
                        color='#E74C3C', zorder=5)
        ax_scene.add_patch(car)
        # 车窗
        window = Rectangle((pos[0]-0.08, pos[1]-0.04), 0.16, 0.08, 
                           color='#ECF0F1', zorder=6)
        ax_scene.add_patch(window)
        # 车轮
        for dx in [-0.08, 0.08]:
            wheel = Circle((pos[0]+dx, pos[1]-0.1), 0.03, 
                          color='#2C3E50', zorder=6)
            ax_scene.add_patch(wheel)
    
    # 绘制通信链路（虚线）
    # V2R链路（蓝色虚线）
    for veh_pos in vehicle_positions[:3]:
        ax_scene.plot([veh_pos[0], rsu_positions[0][0]], 
                      [veh_pos[1], rsu_positions[0][1]], 
                      'b--', alpha=0.3, linewidth=1, zorder=3)
    
    # V2U链路（绿色虚线）
    ax_scene.plot([vehicle_positions[0][0], uav_positions[0][0]], 
                  [vehicle_positions[0][1], uav_positions[0][1]], 
                  'g--', alpha=0.3, linewidth=1, zorder=3)
    
    # U2R链路（红色虚线）
    ax_scene.plot([uav_positions[1][0], rsu_positions[3][0]], 
                  [uav_positions[1][1], rsu_positions[3][1]], 
                  'r--', alpha=0.3, linewidth=1.5, zorder=3)
    
    # 绘制缓存内容流动（黄色箭头）
    arrow1 = FancyArrowPatch((1.5, 9), (3, 9),
                             arrowstyle='->', mutation_scale=15, 
                             color='#F39C12', linewidth=2, zorder=4)
    ax_scene.add_patch(arrow1)
    
    # 添加图例
    legend_y = 1.0
    ax_scene.text(0.8, legend_y+0.5, '图例:', fontsize=10, weight='bold')
    ax_scene.plot([0.8, 1.2], [legend_y+0.2, legend_y+0.2], 'b--', linewidth=1.5)
    ax_scene.text(1.4, legend_y+0.2, 'V2R通信', fontsize=8, va='center')
    ax_scene.plot([0.8, 1.2], [legend_y, legend_y], 'g--', linewidth=1.5)
    ax_scene.text(1.4, legend_y, 'V2U通信', fontsize=8, va='center')
    ax_scene.plot([0.8, 1.2], [legend_y-0.2, legend_y-0.2], 'r--', linewidth=1.5)
    ax_scene.text(1.4, legend_y-0.2, 'U2R中继', fontsize=8, va='center')
    ax_scene.arrow(0.8, legend_y-0.4, 0.3, 0, head_width=0.08, 
                   head_length=0.1, fc='#F39C12', ec='#F39C12')
    ax_scene.text(1.4, legend_y-0.4, '缓存流动', fontsize=8, va='center')
    
    # ==================== 右侧：算法架构 ====================
    ax_arch = plt.subplot(122)
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 12)
    ax_arch.axis('off')
    
    # 标题
    ax_arch.text(5, 11.5, 'CAMTD3多智能体强化学习架构', 
                 fontsize=20, ha='center', weight='bold', color='#2C3E50')
    
    # ========== 层级1：环境层 ==========
    env_box = FancyBboxPatch((0.5, 9.5), 9, 1.5, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#34495E', facecolor='#ECF0F1', 
                             linewidth=2, zorder=10)
    ax_arch.add_patch(env_box)
    ax_arch.text(5, 10.6, 'VEC环境层', fontsize=14, ha='center', weight='bold')
    ax_arch.text(5, 10.2, '状态空间：车辆位置、RSU负载、UAV电量、队列长度、缓存状态', 
                 fontsize=9, ha='center', style='italic')
    
    # ========== 层级2：智能体层 ==========
    # 中央智能体
    central_box = FancyBboxPatch((3, 7.5), 4, 1.2, 
                                 boxstyle="round,pad=0.08", 
                                 edgecolor='#E74C3C', facecolor='#FADBD8', 
                                 linewidth=2, zorder=10)
    ax_arch.add_patch(central_box)
    ax_arch.text(5, 8.4, '中央资源分配智能体', fontsize=12, ha='center', weight='bold', color='#C0392B')
    ax_arch.text(5, 8.0, 'TD3 Actor-Critic网络', fontsize=9, ha='center')
    ax_arch.text(5, 7.7, '动作：CPU频率分配 + 带宽分配', fontsize=8, ha='center', style='italic')
    
    # 决策层智能体
    decision_agents = [
        (1.5, 5.5, '缓存决策', '#3498DB'),
        (5, 5.5, '卸载决策', '#2ECC71'),
        (8.5, 5.5, '迁移决策', '#F39C12'),
    ]
    
    for x, y, name, color in decision_agents:
        box = FancyBboxPatch((x-0.9, y), 1.8, 1, 
                            boxstyle="round,pad=0.06", 
                            edgecolor=color, facecolor=color, 
                            alpha=0.3, linewidth=2, zorder=10)
        ax_arch.add_patch(box)
        ax_arch.text(x, y+0.7, f'{name}智能体', fontsize=11, ha='center', weight='bold')
        ax_arch.text(x, y+0.3, 'TD3网络', fontsize=8, ha='center')
    
    # ========== 层级3：核心算法模块 ==========
    modules_y = 3.5
    
    # TD3核心模块
    td3_modules = [
        (2, modules_y, 'Actor网络\n策略输出', '#9B59B6'),
        (5, modules_y, 'Critic网络\nQ值估计', '#E67E22'),
        (8, modules_y, '目标网络\n软更新', '#16A085'),
    ]
    
    for x, y, text, color in td3_modules:
        box = FancyBboxPatch((x-0.7, y), 1.4, 0.8, 
                            boxstyle="round,pad=0.05", 
                            edgecolor=color, facecolor=color, 
                            alpha=0.25, linewidth=1.5, zorder=10)
        ax_arch.add_patch(box)
        ax_arch.text(x, y+0.4, text, fontsize=9, ha='center', weight='bold')
    
    # ========== 层级4：优化目标 ==========
    objective_y = 1.5
    objective_box = FancyBboxPatch((1, objective_y), 8, 1.2, 
                                   boxstyle="round,pad=0.08", 
                                   edgecolor='#C0392B', facecolor='#FADBD8', 
                                   linewidth=2, zorder=10)
    ax_arch.add_patch(objective_box)
    ax_arch.text(5, objective_y+0.8, '优化目标函数', fontsize=12, ha='center', weight='bold')
    ax_arch.text(5, objective_y+0.4, 'R = -(ω_T × 时延 + ω_E × 能耗) - 0.1 × 丢弃任务数', 
                 fontsize=10, ha='center', style='italic', family='monospace')
    
    # ========== 数据流箭头 ==========
    # 环境 → 中央智能体
    arrow_env_central = FancyArrowPatch((5, 9.5), (5, 8.7),
                                        arrowstyle='->', mutation_scale=20, 
                                        color='#2C3E50', linewidth=2, zorder=9)
    ax_arch.add_patch(arrow_env_central)
    ax_arch.text(5.5, 9.0, '状态', fontsize=9, color='#2C3E50')
    
    # 中央智能体 → 决策智能体
    for x, _, _, color in decision_agents:
        arrow = FancyArrowPatch((5, 7.5), (x, 6.5),
                                arrowstyle='->', mutation_scale=15, 
                                color=color, linewidth=1.5, zorder=9, alpha=0.6)
        ax_arch.add_patch(arrow)
    
    # 决策智能体 → TD3模块
    ax_arch.text(5, 4.6, '策略网络', fontsize=10, ha='center', color='#7F8C8D', style='italic')
    
    # TD3模块 → 优化目标
    for x, _, _, color in td3_modules:
        arrow = FancyArrowPatch((x, modules_y), (5, objective_y+1.2),
                                arrowstyle='->', mutation_scale=12, 
                                color=color, linewidth=1.2, zorder=9, alpha=0.5)
        ax_arch.add_patch(arrow)
    
    # 反馈回路
    feedback_arrow = FancyArrowPatch((9, objective_y+0.6), (9.5, 10.2),
                                     arrowstyle='->', mutation_scale=18, 
                                     color='#E74C3C', linewidth=2.5, 
                                     linestyle='dashed', zorder=9)
    ax_arch.add_patch(feedback_arrow)
    ax_arch.text(9.6, 6, '奖励反馈', fontsize=10, ha='left', 
                 weight='bold', color='#C0392B', rotation=90)
    
    # ========== 底部关键参数说明 ==========
    params_y = 0.3
    ax_arch.text(5, params_y, 
                 '关键参数：γ=0.995 | τ=0.005 | batch=256 | lr=3e-4 | 时延权重=1.0 | 能耗权重=1.2',
                 fontsize=9, ha='center', family='monospace', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ========== 总标题 ==========
    fig.suptitle('VEC边缘计算系统：场景拓扑与多智能体强化学习架构', 
                 fontsize=24, weight='bold', y=0.98, color='#2C3E50')
    
    # 添加底部说明
    fig.text(0.5, 0.02, 
             '场景配置：12车辆 | 4个RSU | 2个UAV | 100MHz带宽 | 50GHz总计算资源',
             ha='center', fontsize=11, style='italic', color='#7F8C8D')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 保存图片
    output_path = 'd:/VEC_mig_caching/results/scenario_architecture_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#F8F9FA', edgecolor='none')
    print(f"✅ 场景架构图已保存至: {output_path}")
    
    # 显示图片
    plt.show()
    
    return output_path

if __name__ == '__main__':
    create_scenario_architecture_diagram()
