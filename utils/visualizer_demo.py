"""
🎨 训练可视化集成示例
展示如何在训练循环中使用高端可视化器

使用方法：
python utils/visualizer_demo.py
"""

import numpy as np
import time
from advanced_training_visualizer import create_visualizer


def demo_training_with_visualization():
    """演示带可视化的训练流程"""
    
    print("🎯 启动高端训练可视化演示...")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = create_visualizer(max_history=500)
    visualizer.start(interval=500)  # 500ms 刷新一次
    
    # 模拟训练循环
    num_episodes = 1000
    
    for episode in range(num_episodes):
        # ===== 模拟训练一个 episode =====
        
        # 模拟奖励逐渐提升
        base_reward = 50 + episode * 0.15
        noise = np.random.randn() * 10
        reward = base_reward + noise
        
        # 模拟损失逐渐下降
        loss = 1.0 / (1 + episode * 0.01) + np.random.randn() * 0.05
        loss = max(0.001, loss)  # 保证非负
        
        # 模拟缓存命中率逐渐提升
        hit_rate = min(0.92, 0.55 + episode * 0.0008) + np.random.randn() * 0.03
        hit_rate = np.clip(hit_rate, 0, 1)
        
        # 模拟延迟逐渐降低
        delay = max(20, 120 - episode * 0.08 + np.random.randn() * 8)
        
        # 模拟能耗逐渐降低
        energy = max(10, 55 - episode * 0.025 + np.random.randn() * 3)
        
        # 模拟成功率提升
        success_rate = min(0.98, 0.75 + episode * 0.0005)
        success_rate = np.clip(success_rate, 0, 1)
        
        # 模拟动作向量（10维连续动作）
        action = np.random.randn(10) * 0.5
        
        # 模拟梯度范数（逐渐收敛）
        gradient_norm = 2.0 / (1 + episode * 0.015) * (1 + np.random.randn() * 0.2)
        gradient_norm = max(0.001, gradient_norm)
        
        # ===== 更新可视化 =====
        metrics = {
            'reward': reward,
            'loss': loss,
            'hit_rate': hit_rate,
            'delay': delay,
            'energy': energy,
            'success_rate': success_rate,
            'action': action,
            'gradient_norm': gradient_norm
        }
        
        visualizer.update(episode, metrics)
        
        # 每10个episode打印一次进度
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {reward:7.2f} | "
                  f"Loss: {loss:6.4f} | "
                  f"Hit Rate: {hit_rate*100:5.1f}% | "
                  f"Delay: {delay:6.2f}ms")
        
        # 控制训练速度（实际训练中不需要）
        time.sleep(0.03)
        
        # 每100个episode自动保存一次
        if episode > 0 and episode % 100 == 0:
            visualizer.save(f"checkpoint_episode_{episode}.png")
    
    print("=" * 60)
    print("✅ 训练完成！")
    print("   可视化窗口将保持打开状态")
    print("   按 's' 保存最终图像，按 'q' 退出")
    
    # 保存最终结果
    visualizer.save("final_training_result.png")
    
    # 保持窗口打开
    import matplotlib.pyplot as plt
    plt.show()


def integration_example():
    """
    集成到实际训练代码的示例
    """
    code_example = '''
# ========== 在训练脚本中集成可视化 ==========

from utils.advanced_training_visualizer import create_visualizer

def train_agent():
    # 1. 创建可视化器
    visualizer = create_visualizer(max_history=500)
    visualizer.start(interval=1000)  # 每秒更新一次
    
    # 2. 训练循环
    for episode in range(num_episodes):
        # ... 你的训练代码 ...
        
        state = env.reset()
        episode_reward = 0
        losses = []
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # 训练agent
            loss = agent.train(state, action, reward, next_state, done)
            losses.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 3. 更新可视化（每个episode一次）
        metrics = {
            'reward': episode_reward,
            'loss': np.mean(losses) if losses else 0,
            'hit_rate': info.get('cache_hit_rate', 0),
            'delay': info.get('avg_delay', 0),
            'energy': info.get('total_energy', 0),
            'success_rate': info.get('success_rate', 0),
            'action': action,  # 最后一个动作
            'gradient_norm': agent.get_gradient_norm()  # 如果可用
        }
        visualizer.update(episode, metrics)
        
        # 4. 定期保存检查点
        if episode % 100 == 0:
            visualizer.save(f"checkpoint_{episode}.png")
    
    # 5. 保存最终结果
    visualizer.save("final_result.png")
    
    return visualizer

# ========== 使用示例 ==========
if __name__ == "__main__":
    visualizer = train_agent()
    
    # 训练结束后保持可视化窗口打开
    import matplotlib.pyplot as plt
    plt.show()
'''
    
    print("=" * 60)
    print("📚 集成示例代码：")
    print("=" * 60)
    print(code_example)
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'example':
        # 显示集成示例
        integration_example()
    else:
        # 运行演示
        demo_training_with_visualization()
