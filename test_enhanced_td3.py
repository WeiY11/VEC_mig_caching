"""
增强型TD3测试脚本

验证所有5项优化的功能：
1. 分布式Critic
2. 熵正则化
3. 模型化队列预测
4. 队列感知回放
5. GAT路由器

用法:
    # 测试基线配置（所有优化禁用）
    python test_enhanced_td3.py --config baseline
    
    # 测试完全增强配置（所有优化启用）
    python test_enhanced_td3.py --config full
    
    # 测试队列优化焦点配置
    python test_enhanced_td3.py --config queue_focused
"""

import numpy as np
import torch
import argparse
import sys
sys.path.insert(0, '.')

from single_agent.enhanced_td3_config import (
    create_baseline_config,
    create_full_enhanced_config,
    create_queue_focused_config,
)
from single_agent.enhanced_td3_agent import EnhancedTD3Agent


def test_basic_functionality(config_name='baseline'):
    """测试基本功能"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}\n")
    
    # 创建配置
    if config_name == 'baseline':
        config = create_baseline_config()
    elif config_name == 'full':
        config = create_full_enhanced_config()
    elif config_name == 'queue_focused':
        config = create_queue_focused_config()
    else:
        config = create_baseline_config()
    
    # 设置小规模参数用于测试
    config.device = 'cpu'
    config.buffer_size = 1000
    config.batch_size = 32
    config.warmup_steps = 100
    
    # 模拟参数
    num_vehicles = 4
    num_rsus = 2
    num_uavs = 1
    state_dim = num_vehicles * 5 + num_rsus * 5 + num_uavs * 5 + 8  # + global
    action_dim = 3 + num_rsus + num_uavs + 10  # offload + rsu_sel + uav_sel + control
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"拓扑: {num_vehicles}辆车, {num_rsus}个RSU, {num_uavs}个UAV\n")
    
    # 创建智能体
    print("初始化EnhancedTD3Agent...")
    agent = EnhancedTD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        num_vehicles=num_vehicles,
        num_rsus=num_rsus,
        num_uavs=num_uavs,
    )
    print("✓ 智能体初始化成功\n")
    
    # 测试动作选择
    print("测试动作选择...")
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state, training=True)
    print(f"  状态形状: {state.shape}")
    print(f"  动作形状: {action.shape}")
    print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
    print("✓ 动作选择成功\n")
    
    # 测试经验存储
    print("测试经验存储...")
    for i in range(150):  # 超过warmup_steps
        s = np.random.randn(state_dim).astype(np.float32)
        a = agent.select_action(s, training=True)
        r = np.random.randn()
        s_next = np.random.randn(state_dim).astype(np.float32)
        done = (i % 50 == 49)
        
        # 生成队列指标（模拟）
        queue_metrics = {
            'queue_occupancy': np.random.rand(),
            'packet_loss': np.random.rand() * 0.1,
            'migration_congestion': np.random.rand() * 0.2,
        }
        
        agent.store_experience(s, a, r, s_next, done, queue_metrics)
    
    print(f"  Replay buffer大小: {len(agent.replay_buffer)}")
    print("✓ 经验存储成功\n")
    
    # 测试网络更新
    print("测试网络更新...")
    update_info = agent.update()
    
    if update_info:
        print("  更新统计:")
        for key, value in update_info.items():
            print(f"    {key}: {value:.6f}")
        print("✓ 网络更新成功\n")
    else:
        print("  (预热期，未执行更新)\n")
    
    # 测试队列感知回放统计
    if config.use_queue_aware_replay:
        print("测试队列感知回放统计...")
        stats = agent.replay_buffer.get_queue_statistics()
        print("  队列统计:")
        for key, value in stats.items():
            print(f"    {key}: {value:.6f}")
        print("✓ 队列统计获取成功\n")
    
    # 测试模型保存和加载
    print("测试模型保存和加载...")
    save_path = f"test_models/enhanced_td3_{config_name}_test.pth"
    agent.save_model(save_path)
    
    # 创建新智能体并加载
    agent2 = EnhancedTD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        num_vehicles=num_vehicles,
        num_rsus=num_rsus,
        num_uavs=num_uavs,
    )
    agent2.load_model(save_path)
    
    # 验证动作一致性
    test_state = np.random.randn(state_dim).astype(np.float32)
    action1 = agent.select_action(test_state, training=False)
    action2 = agent2.select_action(test_state, training=False)
    
    diff = np.abs(action1 - action2).max()
    print(f"  动作差异: {diff:.8f}")
    
    if diff < 1e-5:
        print("✓ 模型保存/加载成功\n")
    else:
        print("✗ 模型保存/加载可能有问题\n")
    
    print(f"{'='*60}")
    print(f"配置 '{config_name}' 测试完成!")
    print(f"{'='*60}\n")


def test_all_configs():
    """测试所有配置"""
    configs = ['baseline', 'full', 'queue_focused']
    
    for config_name in configs:
        try:
            test_basic_functionality(config_name)
        except Exception as e:
            print(f"\n✗ 配置 '{config_name}' 测试失败: {e}\n")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='测试增强型TD3')
    parser.add_argument('--config', type=str, default='all', 
                        choices=['baseline', 'full', 'queue_focused', 'all'],
                        help='要测试的配置')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("增强型TD3测试脚本")
    print("="*60)
    
    if args.config == 'all':
        test_all_configs()
    else:
        test_basic_functionality(args.config)
    
    print("\n✓ 所有测试完成!")


if __name__ == '__main__':
    main()
