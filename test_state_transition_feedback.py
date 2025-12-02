"""测试状态转移透明度增强"""
import numpy as np
from train_single_agent import SingleAgentTrainingEnvironment

# 创建环境
env = SingleAgentTrainingEnvironment(algorithm='OPTIMIZED_TD3')
print("环境创建成功")

# 重置环境
state = env.reset_environment()
print(f"重置成功, 状态维度: {state.shape}")

# 执行多步
for step in range(5):
    action = np.random.randn(19)  # OPTIMIZED_TD3使用的是19维
    next_state, reward, done, info = env.step(action)
    
    fb = info.get('task_feedback', {})
    print(f"\n=== Step {step+1} ===")
    print(f"  生成:{fb.get('step_generated', 0)} | 完成:{fb.get('step_completed', 0)} | 丢弃:{fb.get('step_dropped', 0)} | 缓存命中:{fb.get('step_cache_hits', 0)}")
    print(f"  卸载分布: {fb.get('offload_distribution', {})}")
    details = fb.get('execution_details', [])
    print(f"  执行详情: {len(details)}个任务")
    for d in details[:2]:  # 显示前2个
        print(f"    - {d.get('task_id')} -> {d.get('target_type')} | {d.get('result')} | delay:{d.get('delay', 0):.4f}s")

print("\n状态转移透明度测试完成!")
