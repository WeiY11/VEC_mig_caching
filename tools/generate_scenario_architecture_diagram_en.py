#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEC Edge Computing System - Scenario and Algorithm Architecture Diagram Generator (English Version)

Function: Generate academic-level visualization combining scenario topology and algorithm architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def create_scenario_architecture_diagram():
    """Create scenario and algorithm architecture diagram"""
    
    # Create large canvas: left for scenario, right for architecture
    fig = plt.figure(figsize=(24, 12))
    
    # Set light background
    fig.patch.set_facecolor('#F8F9FA')
    
    # ==================== Left: VEC Scenario Topology ====================
    ax_scene = plt.subplot(121)
    ax_scene.set_xlim(0, 12)
    ax_scene.set_ylim(0, 12)
    ax_scene.axis('off')
    ax_scene.set_aspect('equal')
    
    # Title
    ax_scene.text(6, 11.5, 'VEC Edge Computing Scenario Topology', 
                  fontsize=20, ha='center', weight='bold', color='#2C3E50')
    
    # Define node positions (4 intersection scenario)
    # RSU positions (4 intersections)
    rsu_positions = [
        (3, 9),   # Top-left intersection
        (9, 9),   # Top-right intersection
        (3, 3),   # Bottom-left intersection
        (9, 3),   # Bottom-right intersection
    ]
    
    # UAV positions (2 UAVs)
    uav_positions = [
        (2, 10.5),  # Top-left airspace
        (10, 2.5),  # Bottom-right airspace
    ]
    
    # Vehicle positions (12 vehicles on roads)
    vehicle_positions = [
        (1.5, 9), (2.5, 9), (3.5, 9),  # Around top-left
        (7.5, 9), (8.5, 9), (9.5, 9),  # Around top-right
        (1.5, 3), (2.5, 3), (3.5, 3),  # Around bottom-left
        (7.5, 3), (8.5, 3), (9.5, 3),  # Around bottom-right
    ]
    
    # Draw roads (gray rectangles)
    roads = [
        Rectangle((0.5, 8.5), 11, 1, color='#95A5A6', alpha=0.3, zorder=1),  # Top road
        Rectangle((0.5, 2.5), 11, 1, color='#95A5A6', alpha=0.3, zorder=1),  # Bottom road
        Rectangle((2.5, 0.5), 1, 11, color='#95A5A6', alpha=0.3, zorder=1),  # Left road
        Rectangle((8.5, 0.5), 1, 11, color='#95A5A6', alpha=0.3, zorder=1),  # Right road
    ]
    for road in roads:
        ax_scene.add_patch(road)
    
    # Draw RSU coverage (light blue circles)
    for pos in rsu_positions:
        coverage = Circle(pos, 2.5, color='#3498DB', alpha=0.15, zorder=2)
        ax_scene.add_patch(coverage)
    
    # Draw UAV coverage (light green circles)
    for pos in uav_positions:
        coverage = Circle(pos, 2.0, color='#2ECC71', alpha=0.15, zorder=2)
        ax_scene.add_patch(coverage)
    
    # Draw RSU nodes (tower shape)
    for i, pos in enumerate(rsu_positions):
        # Base
        base = Rectangle((pos[0]-0.15, pos[1]-0.3), 0.3, 0.2, 
                         color='#E74C3C', zorder=5)
        ax_scene.add_patch(base)
        # Tower body
        tower = Polygon([
            (pos[0]-0.1, pos[1]-0.1),
            (pos[0]+0.1, pos[1]-0.1),
            (pos[0]+0.05, pos[1]+0.3),
            (pos[0]-0.05, pos[1]+0.3),
        ], color='#C0392B', zorder=5)
        ax_scene.add_patch(tower)
        # Antenna
        ax_scene.plot([pos[0]-0.08, pos[0]+0.08], [pos[1]+0.3, pos[1]+0.3], 
                      'k-', linewidth=2, zorder=5)
        ax_scene.plot([pos[0]-0.06, pos[0]-0.06], [pos[1]+0.3, pos[1]+0.4], 
                      'k-', linewidth=1.5, zorder=5)
        ax_scene.plot([pos[0], pos[0]], [pos[1]+0.3, pos[1]+0.45], 
                      'k-', linewidth=1.5, zorder=5)
        ax_scene.plot([pos[0]+0.06, pos[0]+0.06], [pos[1]+0.3, pos[1]+0.4], 
                      'k-', linewidth=1.5, zorder=5)
        # Label
        ax_scene.text(pos[0], pos[1]-0.6, f'RSU{i+1}', 
                      fontsize=9, ha='center', weight='bold', color='#C0392B')
        # MEC server icon
        server = Rectangle((pos[0]+0.3, pos[1]-0.2), 0.3, 0.25, 
                           color='#34495E', alpha=0.7, zorder=4)
        ax_scene.add_patch(server)
        ax_scene.plot([pos[0]+0.33, pos[0]+0.57], [pos[1]-0.1, pos[1]-0.1], 
                      'w-', linewidth=1, zorder=4)
        ax_scene.plot([pos[0]+0.33, pos[0]+0.57], [pos[1], pos[1]], 
                      'w-', linewidth=1, zorder=4)
    
    # Draw UAV nodes (drone shape)
    for i, pos in enumerate(uav_positions):
        # Body
        body = Rectangle((pos[0]-0.15, pos[1]-0.08), 0.3, 0.16, 
                         color='#2ECC71', zorder=5)
        ax_scene.add_patch(body)
        # Rotors
        for dx, dy in [(-0.2, 0.15), (0.2, 0.15), (-0.2, -0.15), (0.2, -0.15)]:
            prop = Circle((pos[0]+dx, pos[1]+dy), 0.08, 
                          color='#27AE60', alpha=0.6, zorder=4)
            ax_scene.add_patch(prop)
        # Label
        ax_scene.text(pos[0], pos[1]-0.5, f'UAV{i+1}', 
                      fontsize=9, ha='center', weight='bold', color='#27AE60')
    
    # Draw vehicles (red cars)
    for i, pos in enumerate(vehicle_positions):
        # Car body
        car = Rectangle((pos[0]-0.12, pos[1]-0.08), 0.24, 0.16, 
                        color='#E74C3C', zorder=5)
        ax_scene.add_patch(car)
        # Window
        window = Rectangle((pos[0]-0.08, pos[1]-0.04), 0.16, 0.08, 
                           color='#ECF0F1', zorder=6)
        ax_scene.add_patch(window)
        # Wheels
        for dx in [-0.08, 0.08]:
            wheel = Circle((pos[0]+dx, pos[1]-0.1), 0.03, 
                          color='#2C3E50', zorder=6)
            ax_scene.add_patch(wheel)
    
    # Draw communication links (dashed lines)
    # V2R links (blue dashed)
    for veh_pos in vehicle_positions[:3]:
        ax_scene.plot([veh_pos[0], rsu_positions[0][0]], 
                      [veh_pos[1], rsu_positions[0][1]], 
                      'b--', alpha=0.3, linewidth=1, zorder=3)
    
    # V2U links (green dashed)
    ax_scene.plot([vehicle_positions[0][0], uav_positions[0][0]], 
                  [vehicle_positions[0][1], uav_positions[0][1]], 
                  'g--', alpha=0.3, linewidth=1, zorder=3)
    
    # U2R links (red dashed)
    ax_scene.plot([uav_positions[1][0], rsu_positions[3][0]], 
                  [uav_positions[1][1], rsu_positions[3][1]], 
                  'r--', alpha=0.3, linewidth=1.5, zorder=3)
    
    # Draw cache content flow (yellow arrows)
    arrow1 = FancyArrowPatch((1.5, 9), (3, 9),
                             arrowstyle='->', mutation_scale=15, 
                             color='#F39C12', linewidth=2, zorder=4)
    ax_scene.add_patch(arrow1)
    
    # Add legend
    legend_y = 1.0
    ax_scene.text(0.8, legend_y+0.5, 'Legend:', fontsize=10, weight='bold')
    ax_scene.plot([0.8, 1.2], [legend_y+0.2, legend_y+0.2], 'b--', linewidth=1.5)
    ax_scene.text(1.4, legend_y+0.2, 'V2R Link', fontsize=8, va='center')
    ax_scene.plot([0.8, 1.2], [legend_y, legend_y], 'g--', linewidth=1.5)
    ax_scene.text(1.4, legend_y, 'V2U Link', fontsize=8, va='center')
    ax_scene.plot([0.8, 1.2], [legend_y-0.2, legend_y-0.2], 'r--', linewidth=1.5)
    ax_scene.text(1.4, legend_y-0.2, 'U2R Relay', fontsize=8, va='center')
    ax_scene.arrow(0.8, legend_y-0.4, 0.3, 0, head_width=0.08, 
                   head_length=0.1, fc='#F39C12', ec='#F39C12')
    ax_scene.text(1.4, legend_y-0.4, 'Cache Flow', fontsize=8, va='center')
    
    # ==================== Right: Algorithm Architecture ====================
    ax_arch = plt.subplot(122)
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 12)
    ax_arch.axis('off')
    
    # Title
    ax_arch.text(5, 11.5, 'CAMTD3 Multi-Agent RL Architecture', 
                 fontsize=20, ha='center', weight='bold', color='#2C3E50')
    
    # ========== Layer 1: Environment ===========
    env_box = FancyBboxPatch((0.5, 9.5), 9, 1.5, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#34495E', facecolor='#ECF0F1', 
                             linewidth=2, zorder=10)
    ax_arch.add_patch(env_box)
    ax_arch.text(5, 10.6, 'VEC Environment Layer', fontsize=14, ha='center', weight='bold')
    ax_arch.text(5, 10.2, 'State: Vehicle Position, RSU Load, UAV Battery, Queue Length, Cache Status', 
                 fontsize=9, ha='center', style='italic')
    
    # ========== Layer 2: Agent Layer ==========
    # Central Agent
    central_box = FancyBboxPatch((3, 7.5), 4, 1.2, 
                                 boxstyle="round,pad=0.08", 
                                 edgecolor='#E74C3C', facecolor='#FADBD8', 
                                 linewidth=2, zorder=10)
    ax_arch.add_patch(central_box)
    ax_arch.text(5, 8.4, 'Central Resource Allocation Agent', fontsize=12, ha='center', weight='bold', color='#C0392B')
    ax_arch.text(5, 8.0, 'TD3 Actor-Critic Network', fontsize=9, ha='center')
    ax_arch.text(5, 7.7, 'Action: CPU Freq Allocation + Bandwidth Allocation', fontsize=8, ha='center', style='italic')
    
    # Decision layer agents
    decision_agents = [
        (1.5, 5.5, 'Caching', '#3498DB'),
        (5, 5.5, 'Offloading', '#2ECC71'),
        (8.5, 5.5, 'Migration', '#F39C12'),
    ]
    
    for x, y, name, color in decision_agents:
        box = FancyBboxPatch((x-0.9, y), 1.8, 1, 
                            boxstyle="round,pad=0.06", 
                            edgecolor=color, facecolor=color, 
                            alpha=0.3, linewidth=2, zorder=10)
        ax_arch.add_patch(box)
        ax_arch.text(x, y+0.7, f'{name} Agent', fontsize=11, ha='center', weight='bold')
        ax_arch.text(x, y+0.3, 'TD3 Network', fontsize=8, ha='center')
    
    # ========== Layer 3: Core Algorithm Modules ==========
    modules_y = 3.5
    
    # TD3 core modules
    td3_modules = [
        (2, modules_y, 'Actor Network\nPolicy Output', '#9B59B6'),
        (5, modules_y, 'Critic Network\nQ-Value Estimate', '#E67E22'),
        (8, modules_y, 'Target Network\nSoft Update', '#16A085'),
    ]
    
    for x, y, text, color in td3_modules:
        box = FancyBboxPatch((x-0.7, y), 1.4, 0.8, 
                            boxstyle="round,pad=0.05", 
                            edgecolor=color, facecolor=color, 
                            alpha=0.25, linewidth=1.5, zorder=10)
        ax_arch.add_patch(box)
        ax_arch.text(x, y+0.4, text, fontsize=9, ha='center', weight='bold')
    
    # ========== Layer 4: Optimization Objective ==========
    objective_y = 1.5
    objective_box = FancyBboxPatch((1, objective_y), 8, 1.2, 
                                   boxstyle="round,pad=0.08", 
                                   edgecolor='#C0392B', facecolor='#FADBD8', 
                                   linewidth=2, zorder=10)
    ax_arch.add_patch(objective_box)
    ax_arch.text(5, objective_y+0.8, 'Optimization Objective Function', fontsize=12, ha='center', weight='bold')
    ax_arch.text(5, objective_y+0.4, 'R = -(w_T × Delay + w_E × Energy) - 0.1 × Dropped Tasks', 
                 fontsize=10, ha='center', style='italic', family='monospace')
    
    # ========== Data Flow Arrows ==========
    # Environment → Central Agent
    arrow_env_central = FancyArrowPatch((5, 9.5), (5, 8.7),
                                        arrowstyle='->', mutation_scale=20, 
                                        color='#2C3E50', linewidth=2, zorder=9)
    ax_arch.add_patch(arrow_env_central)
    ax_arch.text(5.5, 9.0, 'State', fontsize=9, color='#2C3E50')
    
    # Central Agent → Decision Agents
    for x, _, _, color in decision_agents:
        arrow = FancyArrowPatch((5, 7.5), (x, 6.5),
                                arrowstyle='->', mutation_scale=15, 
                                color=color, linewidth=1.5, zorder=9, alpha=0.6)
        ax_arch.add_patch(arrow)
    
    # Decision Agents → TD3 Modules
    ax_arch.text(5, 4.6, 'Policy Network', fontsize=10, ha='center', color='#7F8C8D', style='italic')
    
    # TD3 Modules → Optimization Objective
    for x, _, _, color in td3_modules:
        arrow = FancyArrowPatch((x, modules_y), (5, objective_y+1.2),
                                arrowstyle='->', mutation_scale=12, 
                                color=color, linewidth=1.2, zorder=9, alpha=0.5)
        ax_arch.add_patch(arrow)
    
    # Feedback loop
    feedback_arrow = FancyArrowPatch((9, objective_y+0.6), (9.5, 10.2),
                                     arrowstyle='->', mutation_scale=18, 
                                     color='#E74C3C', linewidth=2.5, 
                                     linestyle='dashed', zorder=9)
    ax_arch.add_patch(feedback_arrow)
    ax_arch.text(9.6, 6, 'Reward\nFeedback', fontsize=10, ha='left', 
                 weight='bold', color='#C0392B', rotation=90)
    
    # ========== Bottom Key Parameters ==========
    params_y = 0.3
    ax_arch.text(5, params_y, 
                 'Key Params: gamma=0.995 | tau=0.005 | batch=256 | lr=3e-4 | w_delay=1.0 | w_energy=1.2',
                 fontsize=9, ha='center', family='monospace', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ========== Overall Title ==========
    fig.suptitle('VEC Edge Computing System: Scenario Topology & Multi-Agent RL Architecture', 
                 fontsize=24, weight='bold', y=0.98, color='#2C3E50')
    
    # Add bottom description
    fig.text(0.5, 0.02, 
             'Config: 12 Vehicles | 4 RSUs | 2 UAVs | 100MHz Bandwidth | 50GHz Total Computing Resource',
             ha='center', fontsize=11, style='italic', color='#7F8C8D')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    output_path = 'd:/VEC_mig_caching/results/scenario_architecture_diagram_en.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#F8F9FA', edgecolor='none')
    print(f"✅ Scenario architecture diagram saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    return output_path

if __name__ == '__main__':
    create_scenario_architecture_diagram()
