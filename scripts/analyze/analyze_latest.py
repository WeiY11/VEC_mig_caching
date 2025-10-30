"""
快速分析最新训练结果
"""
import json
import numpy as np
from pathlib import Path
import sys

def find_latest_result():
    """查找最新的训练结果文件"""
    results_dir = Path("results/single_agent/td3")
    if not results_dir.exists():
        print("Error: Results directory not found")
        return None
    
    json_files = sorted(results_dir.glob("training_results_*.json"))
    if not json_files:
        print("Error: No result files found")
        return None
    
    return json_files[-1]

def analyze_episode_rewards(rewards):
    """分析episode奖励趋势"""
    rewards = np.array(rewards)
    n = len(rewards)
    
    # 分段分析
    early_idx = int(n * 0.2)
    late_idx = int(n * 0.8)
    
    early_mean = np.mean(rewards[:early_idx])
    late_mean = np.mean(rewards[late_idx:])
    final_50 = np.mean(rewards[-50:])
    
    improvement = ((late_mean - early_mean) / abs(early_mean)) * 100
    
    # 收敛性分析
    window = min(50, n // 10)
    variances = []
    for i in range(window, len(rewards)):
        variances.append(np.var(rewards[i-window:i]))
    
    early_var = np.mean(variances[:len(variances)//2]) if variances else 0
    late_var = np.mean(variances[len(variances)//2:]) if variances else 0
    
    return {
        'early_mean': early_mean,
        'late_mean': late_mean,
        'final_50': final_50,
        'improvement_pct': improvement,
        'early_variance': early_var,
        'late_variance': late_var,
        'variance_reduction_pct': ((early_var - late_var) / early_var * 100) if early_var > 0 else 0
    }

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    result_file = find_latest_result()
    if not result_file:
        sys.exit(1)
    
    print_section("TD3 Training Results Analysis")
    print(f"Analyzing: {result_file.name}\n")
    
    # 读取数据
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # 基本信息
    print_section("Training Configuration")
    print(f"Algorithm:         {data.get('algorithm', 'N/A')}")
    print(f"Vehicles:          {data.get('network_topology', {}).get('num_vehicles', 'N/A')}")
    print(f"RSUs:              {data.get('network_topology', {}).get('num_rsus', 'N/A')}")
    print(f"UAVs:              {data.get('network_topology', {}).get('num_uavs', 'N/A')}")
    print(f"Total Episodes:    {data.get('training_config', {}).get('num_episodes', 'N/A')}")
    print(f"Training Time:     {data.get('training_config', {}).get('training_time_hours', 0):.2f} hours")
    
    # 奖励分析
    episode_rewards = data.get('episode_rewards', [])
    if episode_rewards:
        print_section("Reward Analysis")
        stats = analyze_episode_rewards(episode_rewards)
        
        print(f"Early Average (first 20%):     {stats['early_mean']:.2f}")
        print(f"Late Average (last 20%):       {stats['late_mean']:.2f}")
        print(f"Final 50 Episodes:             {stats['final_50']:.2f}")
        print(f"Improvement:                   {stats['improvement_pct']:+.2f}%")
        
        print(f"\nConvergence Analysis:")
        print(f"Early Variance:                {stats['early_variance']:.4f}")
        print(f"Late Variance:                 {stats['late_variance']:.4f}")
        print(f"Variance Reduction:            {stats['variance_reduction_pct']:+.1f}%")
        
        if stats['late_variance'] < stats['early_variance'] * 0.5:
            print(f"Status:                        CONVERGED")
        elif stats['late_variance'] < stats['early_variance']:
            print(f"Status:                        NEAR CONVERGENCE")
        else:
            print(f"Status:                        NOT CONVERGED")
    
    # 查找详细指标
    print_section("Performance Metrics")
    
    # 尝试提取step级别的指标
    metrics_found = False
    for key in ['step_delays', 'step_energies', 'step_completion_rates']:
        if key in data and len(data[key]) > 0:
            metrics_found = True
            break
    
    if not metrics_found:
        print("Note: Detailed step metrics not found in this result file.")
        print("Only episode-level rewards are available.")
    
    # 优化建议
    print_section("Optimization Recommendations")
    
    if episode_rewards:
        stats = analyze_episode_rewards(episode_rewards)
        
        issues = []
        
        # 检查收敛
        if stats['late_variance'] > stats['early_variance'] * 0.7:
            issues.append("CONVERGENCE")
            print("[!] Convergence Issue:")
            print(f"    Variance not sufficiently reduced ({stats['variance_reduction_pct']:.1f}%)")
            print(f"    Recommendation: Increase episodes to 1200-1500")
        
        # 检查改进幅度
        if stats['improvement_pct'] < 10:
            issues.append("IMPROVEMENT")
            print("[!] Limited Improvement:")
            print(f"    Only {stats['improvement_pct']:.1f}% improvement")
            print(f"    Recommendation: Adjust reward weights or increase training time")
        
        # 检查稳定性
        if stats['late_variance'] > 1000:
            issues.append("STABILITY")
            print("[!] Stability Issue:")
            print(f"    High late variance ({stats['late_variance']:.1f})")
            print(f"    Recommendation: Increase replay buffer size or adjust noise decay")
        
        if not issues:
            print("[OK] No major issues detected!")
            print("     Performance looks good. Consider running longer for better convergence.")
    
    # 配置建议
    print_section("Suggested Configuration Changes")
    
    if episode_rewards:
        stats = analyze_episode_rewards(episode_rewards)
        
        print("In config/system_config.py:")
        print()
        
        if stats['late_variance'] > stats['early_variance'] * 0.7:
            print("# Improve convergence:")
            print("config.experiment.num_episodes = 1200")
            print("config.rl.lr_decay_rate = 0.995")
            print("config.rl.noise_decay = 0.998")
            print()
        
        if stats['improvement_pct'] < 10:
            print("# Boost optimization:")
            print("config.rl.reward_weight_energy = 2.0  # Increase from 1.8")
            print("config.rl.batch_size = 512  # Increase from 256")
            print()
        
        if stats['late_variance'] > 1000:
            print("# Enhance stability:")
            print("config.rl.memory_size = 300000  # Increase from 200000")
            print("config.rl.min_noise = 0.01  # Decrease from current value")
            print()
    
    print_section("Next Steps")
    print("1. Review the training curves in results/single_agent/td3/")
    print("2. Apply recommended configuration changes if needed")
    print("3. Run optimized training:")
    print("   python run_optimized_training.py")
    print("4. Compare results with baseline algorithms:")
    print("   python run_academic_experiments.py --mode baseline")
    print()

if __name__ == "__main__":
    main()
