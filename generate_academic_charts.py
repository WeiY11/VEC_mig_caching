"""
学术论文图表自动生成脚本
从训练结果自动生成多种学术图表
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from utils.academic_chart_generator import AcademicChartGenerator

def load_training_result(json_path: str) -> Dict:
    """加载训练结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics(result: Dict) -> Dict[str, np.ndarray]:
    """提取关键指标"""
    metrics = {}
    
    # Episode奖励
    if 'episode_rewards' in result:
        metrics['rewards'] = np.array(result['episode_rewards'])
    
    # 时延
    if 'episode_metrics' in result and 'avg_delay' in result['episode_metrics']:
        metrics['delay'] = np.array(result['episode_metrics']['avg_delay'])
    
    # 能耗
    if 'episode_metrics' in result and 'total_energy' in result['episode_metrics']:
        metrics['energy'] = np.array(result['episode_metrics']['total_energy'])
    
    # 完成率
    if 'episode_metrics' in result and 'task_completion_rate' in result['episode_metrics']:
        metrics['completion_rate'] = np.array(result['episode_metrics']['task_completion_rate'])
    
    # 缓存命中率
    if 'episode_metrics' in result and 'cache_hit_rate' in result['episode_metrics']:
        metrics['cache_hit_rate'] = np.array(result['episode_metrics']['cache_hit_rate'])
    
    return metrics

def generate_single_algorithm_charts(result_path: str, output_dir: str):
    """
    为单个算法生成学术图表
    
    Args:
        result_path: 训练结果JSON路径
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"Processing: {result_path}")
    print(f"{'='*60}\n")
    
    # 加载数据
    result = load_training_result(result_path)
    algorithm = result.get('algorithm', 'Unknown')
    metrics = extract_metrics(result)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化生成器
    generator = AcademicChartGenerator(dpi=300)
    
    charts_generated = []
    
    # 1. 学习曲线（带方差）
    if 'rewards' in metrics:
        print("[1/6] Generating learning curve with variance...")
        rewards = metrics['rewards']
        episodes = np.arange(1, len(rewards) + 1)
        
        # 计算滚动均值和标准差
        window = min(20, len(rewards) // 5)
        if window > 1:
            mean_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            std_rewards = []
            for i in range(len(mean_rewards)):
                window_data = rewards[i:i+window]
                std_rewards.append(np.std(window_data))
            std_rewards = np.array(std_rewards)
            episodes_adj = episodes[window-1:]
            
            path = generator.generate_learning_curve_with_variance(
                episodes_adj, mean_rewards, std_rewards,
                f"{output_dir}/{algorithm}_learning_curve_variance.png",
                title=f"{algorithm} Learning Curve with Confidence Intervals"
            )
            charts_generated.append(('Learning Curve (±σ)', path))
            print(f"  [OK] Saved: {path}")
    
    # 2. CDF图（奖励分布）
    if 'rewards' in metrics:
        print("[2/6] Generating CDF plot...")
        path = generator.generate_cdf_plot(
            {algorithm: metrics['rewards'].tolist()},
            f"{output_dir}/{algorithm}_reward_cdf.png",
            title=f"{algorithm} Reward Cumulative Distribution"
        )
        charts_generated.append(('Reward CDF', path))
        print(f"  [OK] Saved: {path}")
    
    # 3. 箱线图（奖励分布统计）
    if 'rewards' in metrics:
        print("[3/6] Generating boxplot...")
        # 分段统计（前/中/后期）
        rewards = metrics['rewards']
        n = len(rewards)
        segments = {
            'Early (1-33%)': rewards[:n//3].tolist(),
            'Middle (34-66%)': rewards[n//3:2*n//3].tolist(),
            'Late (67-100%)': rewards[2*n//3:].tolist()
        }
        
        path = generator.generate_boxplot_comparison(
            segments,
            f"{output_dir}/{algorithm}_reward_boxplot_phases.png",
            title=f"{algorithm} Reward Distribution by Training Phase"
        )
        charts_generated.append(('Reward Boxplot (Phases)', path))
        print(f"  [OK] Saved: {path}")
    
    # 4. 相关性热力图
    if len(metrics) >= 3:
        print("[4/6] Generating correlation heatmap...")
        metric_names = list(metrics.keys())
        metric_arrays = [metrics[m] for m in metric_names]
        
        # 确保所有指标长度一致
        min_len = min(len(arr) for arr in metric_arrays)
        metric_arrays = [arr[:min_len] for arr in metric_arrays]
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(metric_arrays)
        
        path = generator.generate_heatmap(
            corr_matrix,
            metric_names,
            f"{output_dir}/{algorithm}_metric_correlation.png",
            title=f"{algorithm} Metric Correlation Analysis"
        )
        charts_generated.append(('Metric Correlation', path))
        print(f"  [OK] Saved: {path}")
    
    # 5. 散点图（时延 vs 能耗）
    if 'delay' in metrics and 'energy' in metrics:
        print("[5/6] Generating delay-energy scatter plot...")
        min_len = min(len(metrics['delay']), len(metrics['energy']))
        
        path = generator.generate_scatter_with_regression(
            metrics['delay'][:min_len],
            metrics['energy'][:min_len],
            f"{output_dir}/{algorithm}_delay_energy_scatter.png",
            title=f"{algorithm} Delay vs Energy Trade-off",
            xlabel="Average Delay (s)",
            ylabel="Total Energy Consumption (J)"
        )
        charts_generated.append(('Delay-Energy Scatter', path))
        print(f"  [OK] Saved: {path}")
    
    # 6. 雷达图（多维度性能）
    print("[6/6] Generating radar chart...")
    # 归一化指标到[0,1]
    radar_data = {}
    if 'completion_rate' in metrics:
        radar_data['Task Completion'] = np.mean(metrics['completion_rate'])
    if 'cache_hit_rate' in metrics:
        radar_data['Cache Hit Rate'] = np.mean(metrics['cache_hit_rate'])
    if 'rewards' in metrics:
        # 奖励归一化（负值，越大越好）
        rewards = metrics['rewards']
        reward_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        radar_data['Reward'] = np.mean(reward_norm[-50:])  # 最后50轮
    if 'delay' in metrics:
        # 时延归一化（越小越好，所以取反）
        delays = metrics['delay']
        delay_norm = 1 - (delays - delays.min()) / (delays.max() - delays.min() + 1e-8)
        radar_data['Low Delay'] = np.mean(delay_norm[-50:])
    
    # 计算稳定性（方差的倒数）
    if 'rewards' in metrics and len(metrics['rewards']) > 20:
        recent_rewards = metrics['rewards'][-50:]
        variance = np.var(recent_rewards)
        stability = 1 / (1 + variance / 100)  # 归一化
        radar_data['Stability'] = stability
    
    if radar_data:
        path = generator.generate_radar_chart(
            {algorithm: radar_data},
            f"{output_dir}/{algorithm}_performance_radar.png",
            title=f"{algorithm} Multi-dimensional Performance"
        )
        charts_generated.append(('Performance Radar', path))
        print(f"  [OK] Saved: {path}")
    
    # 生成总结
    print(f"\n{'='*60}")
    print(f"SUCCESS! Generated {len(charts_generated)} academic charts!")
    print(f"{'='*60}\n")
    
    for i, (name, path) in enumerate(charts_generated, 1):
        print(f"  {i}. {name}")
        print(f"     -> {path}")
    
    return charts_generated

def generate_multi_algorithm_comparison(result_paths: List[str], output_dir: str):
    """
    生成多算法对比图表
    
    Args:
        result_paths: 多个训练结果JSON路径
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"Multi-Algorithm Comparison")
    print(f"{'='*60}\n")
    
    # 加载所有结果
    algorithms_data = {}
    for path in result_paths:
        result = load_training_result(path)
        algorithm = result.get('algorithm', 'Unknown')
        if 'episode_rewards' in result:
            algorithms_data[algorithm] = result['episode_rewards']
    
    if len(algorithms_data) < 2:
        print("WARNING: Need at least 2 algorithms for comparison!")
        return []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化生成器
    generator = AcademicChartGenerator(dpi=300)
    
    charts_generated = []
    
    # 1. 收敛性对比
    print("[1/5] Generating convergence comparison...")
    path = generator.generate_convergence_comparison(
        algorithms_data,
        f"{output_dir}/algorithms_convergence_comparison.png",
        title="Algorithm Convergence Comparison"
    )
    charts_generated.append(('Convergence Comparison', path))
    print(f"  [OK] Saved: {path}")
    
    # 2. 箱线图对比
    print("[2/5] Generating boxplot comparison...")
    # 只取后50%的数据（训练后期）
    late_data = {}
    for algo, rewards in algorithms_data.items():
        n = len(rewards)
        late_data[algo] = rewards[n//2:]
    
    path = generator.generate_boxplot_comparison(
        late_data,
        f"{output_dir}/algorithms_boxplot_comparison.png",
        title="Algorithm Performance Distribution (Late Training)"
    )
    charts_generated.append(('Boxplot Comparison', path))
    print(f"  [OK] Saved: {path}")
    
    # 3. CDF对比
    print("[3/5] Generating CDF comparison...")
    path = generator.generate_cdf_plot(
        late_data,
        f"{output_dir}/algorithms_cdf_comparison.png",
        title="Cumulative Distribution of Rewards (Late Training)"
    )
    charts_generated.append(('CDF Comparison', path))
    print(f"  [OK] Saved: {path}")
    
    # 4. 小提琴图对比
    print("[4/5] Generating violin plot comparison...")
    path = generator.generate_violin_plot(
        late_data,
        f"{output_dir}/algorithms_violin_comparison.png",
        title="Algorithm Performance Distribution (Violin Plot)"
    )
    charts_generated.append(('Violin Plot Comparison', path))
    print(f"  [OK] Saved: {path}")
    
    # 5. 柱状图对比（平均性能）
    print("[5/5] Generating bar chart comparison...")
    mean_values = {algo: np.mean(rewards) for algo, rewards in late_data.items()}
    std_values = {algo: np.std(rewards) for algo, rewards in late_data.items()}
    
    path = generator.generate_bar_comparison(
        mean_values,
        f"{output_dir}/algorithms_bar_comparison.png",
        title="Algorithm Average Performance Comparison (Late Training)",
        error_bars=std_values
    )
    charts_generated.append(('Bar Chart Comparison', path))
    print(f"  [OK] Saved: {path}")
    
    # 生成总结
    print(f"\n{'='*60}")
    print(f"SUCCESS! Generated {len(charts_generated)} comparison charts!")
    print(f"{'='*60}\n")
    
    for i, (name, path) in enumerate(charts_generated, 1):
        print(f"  {i}. {name}")
        print(f"     -> {path}")
    
    return charts_generated

def main():
    parser = argparse.ArgumentParser(
        description='生成学术论文图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【单个算法】
  python generate_academic_charts.py results/single_agent/td3/training_results_xxx.json
  
【多算法对比】
  python generate_academic_charts.py results/single_agent/td3/training_results_xxx.json \\
                                    results/single_agent/ddpg/training_results_xxx.json \\
                                    results/single_agent/sac/training_results_xxx.json \\
                                    --compare
  
【指定输出目录】
  python generate_academic_charts.py input.json -o academic_figures/
  
【批量处理】
  python generate_academic_charts.py results/single_agent/*/training_results_*.json
        """
    )
    
    parser.add_argument('json_files', nargs='+', help='训练结果JSON文件路径（支持多个文件）')
    parser.add_argument('-o', '--output', default='academic_figures', 
                       help='输出目录（默认: academic_figures/）')
    parser.add_argument('--compare', action='store_true', 
                       help='生成多算法对比图表')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='图表分辨率（默认300 DPI）')
    
    args = parser.parse_args()
    
    # 展开通配符
    json_files = []
    for pattern in args.json_files:
        if '*' in pattern or '?' in pattern:
            import glob
            json_files.extend(glob.glob(pattern))
        else:
            json_files.append(pattern)
    
    if not json_files:
        print("ERROR: No matching files found")
        sys.exit(1)
    
    print(f"\n[Academic Chart Generator]")
    print(f"   Input files: {len(json_files)}")
    print(f"   Output dir: {args.output}")
    print(f"   DPI: {args.dpi}")
    print(f"   Comparison mode: {'Yes' if args.compare else 'No'}\n")
    
    all_charts = []
    
    # 单个算法图表
    if not args.compare:
        for json_file in json_files:
            result = load_training_result(json_file)
            algorithm = result.get('algorithm', 'Unknown')
            output_dir = os.path.join(args.output, algorithm.lower())
            
            charts = generate_single_algorithm_charts(json_file, output_dir)
            all_charts.extend(charts)
    
    # 多算法对比
    if args.compare and len(json_files) >= 2:
        output_dir = os.path.join(args.output, 'comparison')
        charts = generate_multi_algorithm_comparison(json_files, output_dir)
        all_charts.extend(charts)
    
    # 最终总结
    print(f"\n{'#'*60}")
    print(f"  SUCCESS! Generated {len(all_charts)} academic charts")
    print(f"{'#'*60}\n")
    
    print("All charts saved to:")
    print(f"   {os.path.abspath(args.output)}\n")
    
    print("Tips:")
    print("   - 300 DPI is suitable for paper submission")
    print("   - IEEE standard colors (color-blind friendly)")
    print("   - Includes confidence intervals and statistics")
    print("   - Ready for LaTeX documents\n")

if __name__ == "__main__":
    main()
