#!/usr/bin/env python3
"""
固定网络拓扑的参数优化器
保持4个RSU和2个UAV不变，只调整神经网络和训练参数
用于公平比较不同车辆数下的算法性能

作者: VEC-Migration-Caching系统
创建日期: 2025-10-10
"""

import os
import json
from typing import Dict

class FixedTopologyOptimizer:
    """
    固定拓扑的参数优化器
    保持RSU=4, UAV=2不变，根据车辆数调整其他参数
    """
    
    def __init__(self):
        """初始化优化参数表"""
        # 针对不同车辆数的优化参数
        # 核心思想：车辆少时降低网络容量防止过拟合，车辆多时增加容量提升学习能力
        self.optimizations = {
            8: {
                # 较小的网络，防止过拟合
                'hidden_dim': 256,
                'actor_lr': 5e-5,     # 降低学习率
                'critic_lr': 8e-5,    
                'batch_size': 128,    # 较小批次
                'tau': 0.001,         # 更保守的软更新
                'exploration_noise': 0.15,  # 略低的探索噪声
                'policy_delay': 2,
                'gradient_clip': 0.5,  # 更严格的梯度裁剪
                'note': '8辆车：减少网络容量，降低学习率，防止过拟合'
            },
            12: {
                # 标准配置（最优）
                'hidden_dim': 400,
                'actor_lr': 1e-4,
                'critic_lr': 8e-5,
                'batch_size': 256,
                'tau': 0.005,
                'exploration_noise': 0.2,
                'policy_delay': 2,
                'gradient_clip': 0.7,
                'note': '12辆车：标准配置，系统平衡最优'
            },
            16: {
                # 略微增加容量
                'hidden_dim': 512,
                'actor_lr': 1e-4,
                'critic_lr': 8e-5,
                'batch_size': 256,
                'tau': 0.005,
                'exploration_noise': 0.2,
                'policy_delay': 2,
                'gradient_clip': 0.7,
                'note': '16辆车：略微增加网络容量'
            },
            20: {
                # 增加容量和批次大小
                'hidden_dim': 512,
                'actor_lr': 8e-5,     # 稍微降低学习率保持稳定
                'critic_lr': 6e-5,
                'batch_size': 384,    # 增加批次大小
                'tau': 0.003,         # 略保守的更新
                'exploration_noise': 0.25,  # 增加探索
                'policy_delay': 1,    # 减少延迟，加快学习
                'gradient_clip': 1.0, # 放宽梯度裁剪
                'note': '20辆车：增加批次大小，调整学习率'
            },
            24: {
                # 最大容量配置
                'hidden_dim': 640,
                'actor_lr': 8e-5,
                'critic_lr': 6e-5,
                'batch_size': 512,    # 大批次
                'tau': 0.003,
                'exploration_noise': 0.3,  # 高探索噪声
                'policy_delay': 1,
                'gradient_clip': 1.0,
                'note': '24辆车：最大网络容量，大批次训练'
            }
        }
    
    def get_optimized_params(self, num_vehicles: int) -> Dict:
        """
        获取优化参数（不改变RSU/UAV数量）
        
        Args:
            num_vehicles: 车辆数量
            
        Returns:
            优化后的参数字典
        """
        # 如果有精确匹配，使用预设参数
        if num_vehicles in self.optimizations:
            params = self.optimizations[num_vehicles].copy()
        else:
            # 否则使用最接近的配置
            params = self._interpolate_params(num_vehicles)
        
        # 始终保持固定的网络拓扑
        params['num_rsus'] = 4  # 固定4个RSU
        params['num_uavs'] = 2  # 固定2个UAV
        
        return params
    
    def _interpolate_params(self, num_vehicles: int) -> Dict:
        """插值计算参数"""
        vehicle_counts = sorted(self.optimizations.keys())
        
        if num_vehicles < vehicle_counts[0]:
            # 小于最小值
            base_params = self.optimizations[vehicle_counts[0]].copy()
            base_params['note'] = f'{num_vehicles}辆车：基于8辆车配置调整'
            
        elif num_vehicles > vehicle_counts[-1]:
            # 大于最大值
            base_params = self.optimizations[vehicle_counts[-1]].copy()
            # 适当扩展
            ratio = num_vehicles / vehicle_counts[-1]
            base_params['hidden_dim'] = min(800, int(base_params['hidden_dim'] * ratio))
            base_params['batch_size'] = min(768, int(base_params['batch_size'] * ratio))
            base_params['note'] = f'{num_vehicles}辆车：基于24辆车配置扩展'
            
        else:
            # 在范围内，找最接近的
            closest = min(vehicle_counts, key=lambda x: abs(x - num_vehicles))
            base_params = self.optimizations[closest].copy()
            base_params['note'] = f'{num_vehicles}辆车：基于{closest}辆车配置'
        
        return base_params
    
    def apply_to_environment(self, num_vehicles: int):
        """
        将优化参数应用到环境变量
        
        Args:
            num_vehicles: 车辆数量
        """
        params = self.get_optimized_params(num_vehicles)
        
        # 设置固定的网络拓扑
        scenario = {
            "num_vehicles": num_vehicles,
            "num_rsus": 4,  # 固定
            "num_uavs": 2   # 固定
        }
        os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(scenario)
        
        # 设置TD3参数
        os.environ['TD3_HIDDEN_DIM'] = str(params['hidden_dim'])
        os.environ['TD3_ACTOR_LR'] = str(params['actor_lr'])
        os.environ['TD3_CRITIC_LR'] = str(params['critic_lr'])
        os.environ['TD3_BATCH_SIZE'] = str(params['batch_size'])
        os.environ['TD3_TAU'] = str(params['tau'])
        os.environ['TD3_EXPLORATION_NOISE'] = str(params['exploration_noise'])
        os.environ['TD3_POLICY_DELAY'] = str(params['policy_delay'])
        os.environ['TD3_GRADIENT_CLIP'] = str(params.get('gradient_clip', 0.7))
        
        return params
    
    def print_analysis(self, num_vehicles: int):
        """打印负载分析"""
        vehicle_per_rsu = num_vehicles / 4
        
        print(f"\n{'='*60}")
        print(f"车辆数: {num_vehicles} | RSU: 4 (固定) | UAV: 2 (固定)")
        print(f"{'='*60}")
        print(f"负载情况: {vehicle_per_rsu:.1f} 辆车/RSU")
        
        if vehicle_per_rsu < 2:
            print("[WARNING] 负载较轻：RSU利用率低")
            print("[TIP] 建议：降低学习率和网络容量，防止过拟合")
        elif 2 <= vehicle_per_rsu <= 4:
            print("[OK] 负载适中：系统平衡")
            print("[TIP] 建议：使用标准参数配置")
        elif 4 < vehicle_per_rsu <= 5:
            print("[WARNING] 负载较高：RSU压力大")
            print("[TIP] 建议：增加网络容量和探索噪声")
        else:
            print("[ERROR] 负载过高：RSU过载严重")
            print("[TIP] 建议：使用最大网络容量，增加批次大小")
        
        params = self.get_optimized_params(num_vehicles)
        print(f"\n优化参数:")
        print(f"  Hidden层: {params['hidden_dim']}")
        print(f"  Actor学习率: {params['actor_lr']:.1e}")
        print(f"  Critic学习率: {params['critic_lr']:.1e}")
        print(f"  批次大小: {params['batch_size']}")
        print(f"  软更新系数: {params['tau']}")
        print(f"  探索噪声: {params['exploration_noise']}")
        print(f"  策略延迟: {params['policy_delay']}")
        print(f"  说明: {params['note']}")


def main():
    """主函数：演示固定拓扑下的参数优化"""
    optimizer = FixedTopologyOptimizer()
    
    print("="*70)
    print("固定网络拓扑参数优化器")
    print("保持4个RSU + 2个UAV，通过调整算法参数适应不同车辆数")
    print("="*70)
    
    # 测试不同车辆数
    test_vehicles = [8, 12, 16, 20, 24]
    
    for num_vehicles in test_vehicles:
        optimizer.print_analysis(num_vehicles)
    
    print("\n" + "="*70)
    print("使用方法:")
    print("="*70)
    
    print("\n1. 使用环境变量（推荐）:")
    print("   python fixed_topology_optimizer.py --apply --vehicles 20")
    print("   python train_single_agent.py --algorithm TD3 --episodes 1600")
    
    print("\n2. 直接运行（使用默认参数）:")
    print("   python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 20")
    
    print("\n3. 查看优化建议:")
    print("   python fixed_topology_optimizer.py --analyze --vehicles 20")
    
    print("\n" + "="*70)
    print("核心优势:")
    print("  [OK] 保持网络拓扑不变（4 RSU + 2 UAV）")
    print("  [OK] 公平比较不同负载下的算法性能")
    print("  [OK] 通过参数调优适应不同场景")
    print("  [OK] 验证算法策略的有效性和鲁棒性")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='固定拓扑参数优化器')
    parser.add_argument('--apply', action='store_true',
                       help='应用优化参数到环境变量')
    parser.add_argument('--analyze', action='store_true',
                       help='分析负载情况和优化建议')
    parser.add_argument('--vehicles', type=int, default=12,
                       help='车辆数量')
    
    args = parser.parse_args()
    
    optimizer = FixedTopologyOptimizer()
    
    if args.apply:
        params = optimizer.apply_to_environment(args.vehicles)
        print(f"\n已为{args.vehicles}辆车应用优化参数（固定4 RSU + 2 UAV）:")
        print(f"  Hidden层: {params['hidden_dim']}")
        print(f"  学习率: Actor={params['actor_lr']:.1e}, Critic={params['critic_lr']:.1e}")
        print(f"  批次大小: {params['batch_size']}")
        print(f"\n建议运行命令:")
        print(f"  python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles {args.vehicles}")
    
    elif args.analyze:
        optimizer.print_analysis(args.vehicles)
    
    else:
        main()
