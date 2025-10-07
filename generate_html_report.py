"""
训练可视化工具 - 支持实时可视化和事后报告生成
- 实时模式：在训练过程中实时显示图表和指标
- 报告模式：从已有的JSON文件生成静态HTML报告
"""
import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime
from utils.html_report_generator import HTMLReportGenerator

# 导入实时可视化模块
try:
    from realtime_visualization import create_visualizer, RealtimeVisualizer
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("⚠️  实时可视化功能不可用，请安装依赖: pip install flask flask-socketio")


def load_training_results(json_path: str) -> dict:
    """加载训练结果JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {json_path} 不是有效的JSON格式")
        sys.exit(1)


def create_mock_training_env(results: dict):
    """从结果创建模拟训练环境对象"""
    class MockTrainingEnv:
        def __init__(self, results_data):
            self.episode_rewards = results_data.get('episode_rewards', [])
            self.episode_metrics = results_data.get('episode_metrics', {})
            self.performance_tracker = {
                'recent_rewards': type('obj', (), {'get_average': lambda: results_data.get('final_performance', {}).get('avg_reward', 0)})(),
                'recent_delays': type('obj', (), {'get_average': lambda: results_data.get('final_performance', {}).get('avg_delay', 0)})(),
                'recent_energy': type('obj', (), {'get_average': lambda: 0})(),
                'recent_completion': type('obj', (), {'get_average': lambda: results_data.get('final_performance', {}).get('avg_completion', 0)})()
            }

            # 模拟智能体环境
            class MockAgentEnv:
                def __init__(self):
                    self.state_dim = results_data.get('state_dim', 'N/A')
                    self.action_dim = results_data.get('environment_info', {}).get('action_dim', 'N/A')

                    # 模拟神经网络
                    class MockActor:
                        def __init__(self):
                            self.fc1 = type('obj', (), {'out_features': 256})()
                            self.fc2 = type('obj', (), {'out_features': 128})()

                    class MockCritic:
                        def __init__(self):
                            self.fc1 = type('obj', (), {'out_features': 256})()
                            self.fc2 = type('obj', (), {'out_features': 128})()

                    self.actor = MockActor()
                    self.critic = MockCritic()

                    # 模拟优化器
                    self.actor_optimizer = type('obj', (), {'param_groups': [{'lr': 0.0003}]})()
                    self.critic_optimizer = type('obj', (), {'param_groups': [{'lr': 0.0003}]})()

                    # 模拟超参数
                    self.gamma = 0.99
                    self.tau = 0.005
                    self.policy_noise = 0.1
                    self.noise_clip = 0.3
                    self.policy_delay = 2

            self.agent_env = MockAgentEnv()

            # 模拟仿真器
            class MockSimulator:
                def __init__(self):
                    self.vehicles = []
                    self.rsus = []
                    self.uavs = []

            self.simulator = MockSimulator()

            # 模拟自适应控制器
            class MockController:
                def get_cache_metrics(self):
                    return {'effectiveness': 0.85, 'utilization': 0.72, 'agent_params': {}}
                def get_migration_metrics(self):
                    return {'effectiveness': 0.78, 'decision_quality': 0.83, 'agent_params': {}}

            self.adaptive_cache_controller = MockController()
            self.adaptive_migration_controller = MockController()

    return MockTrainingEnv(results)


def generate_report_from_json(json_path: str, output_path: str = None, open_browser: bool = False):
    """从JSON文件生成HTML报告"""
    print(f"Reading training results: {json_path}")
    results = load_training_results(json_path)

    # 提取信息
    algorithm = results.get('algorithm', 'Unknown')
    training_time = results.get('training_config', {}).get('training_time_hours', 0) * 3600

    # 创建模拟环境
    training_env = create_mock_training_env(results)

    # 生成报告
    print("Generating HTML report...")
    generator = HTMLReportGenerator()

    html_content = generator.generate_full_report(
        algorithm=algorithm,
        training_env=training_env,
        training_time=training_time,
        results=results,
        simulator_stats={}  # 如果JSON中有，可以提取
    )

    # 确定输出路径
    if output_path is None:
        # 自动生成输出路径
        dir_name = os.path.dirname(json_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(dir_name, f"training_report_{timestamp}.html")

    # 保存报告
    print(f"Saving report to: {output_path}")
    if generator.save_report(html_content, output_path):
        print(f"Report saved successfully!")

        # 打开浏览器
        if open_browser:
            print("Opening report in browser...")
            abs_path = os.path.abspath(output_path)
            webbrowser.open(f'file://{abs_path}')
            print("Report opened in browser")
        else:
            print(f"Open file in browser to view: {output_path}")

        return True
    else:
        print("Failed to save report")
        return False


def start_realtime_mode(algorithm: str = "Unknown", total_episodes: int = 100, port: int = 5000):
    """启动实时可视化模式"""
    if not REALTIME_AVAILABLE:
        print("❌ 实时可视化功能不可用")
        print("请安装依赖: pip install flask flask-socketio")
        sys.exit(1)
    
    print(f"🚀 启动实时可视化模式")
    print(f"   算法: {algorithm}")
    print(f"   总轮次: {total_episodes}")
    print(f"   端口: {port}")
    print(f"\n📌 使用方法：")
    print(f"   在训练代码中导入：from realtime_visualization import create_visualizer")
    print(f"   创建可视化器：visualizer = create_visualizer('{algorithm}', {total_episodes})")
    print(f"   训练循环中更新：visualizer.update(episode, reward, metrics)")
    print(f"   训练完成：visualizer.complete()")
    print(f"\n🌐 访问 http://localhost:{port} 查看实时可视化")
    
    visualizer = create_visualizer(algorithm, total_episodes, port, auto_open=True)
    
    try:
        import time
        print("\n✅ 实时可视化服务器运行中... 按 Ctrl+C 退出")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 退出实时可视化服务器")


def main():
    parser = argparse.ArgumentParser(
        description='训练可视化工具 - 支持实时可视化和事后报告生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【实时可视化模式】
  # 启动实时监控服务器（在训练前运行）
  python generate_html_report.py --realtime --algorithm TD3 --episodes 200 --port 5000

【静态报告生成模式】
  # 从JSON文件生成报告（自动命名）
  python generate_html_report.py results/single_agent/ddpg/training_results_20250930_174833.json
  
  # 指定输出路径
  python generate_html_report.py input.json -o custom_report.html
  
  # 生成后自动打开浏览器
  python generate_html_report.py input.json --open
  
  # 批量生成报告（使用通配符）
  python generate_html_report.py results/single_agent/*/training_results_*.json
        """
    )
    
    # 实时模式参数
    parser.add_argument('--realtime', action='store_true', help='启动实时可视化模式')
    parser.add_argument('--algorithm', default='Unknown', help='算法名称（实时模式）')
    parser.add_argument('--episodes', type=int, default=100, help='总训练轮次（实时模式）')
    parser.add_argument('--port', type=int, default=5000, help='Web服务器端口（实时模式）')
    
    # 报告生成模式参数
    parser.add_argument('json_files', nargs='*', help='训练结果JSON文件路径（支持多个文件）')
    parser.add_argument('-o', '--output', help='输出HTML文件路径（仅单文件时有效）')
    parser.add_argument('--open', action='store_true', help='生成后在浏览器中打开')
    parser.add_argument('--quiet', action='store_true', help='静默模式，不显示详细信息')
    
    args = parser.parse_args()
    
    # 判断运行模式
    if args.realtime:
        # 实时可视化模式
        start_realtime_mode(args.algorithm, args.episodes, args.port)
    else:
        # 静态报告生成模式
        if not args.json_files:
            print("❌ 错误: 静态报告模式需要提供JSON文件路径")
            print("使用 --realtime 启动实时可视化模式，或提供JSON文件路径生成报告")
            parser.print_help()
            sys.exit(1)
        
        # 处理多个文件
        json_files = []
        for pattern in args.json_files:
            if '*' in pattern or '?' in pattern:
                import glob
                json_files.extend(glob.glob(pattern))
            else:
                json_files.append(pattern)
        
        if not json_files:
            print("❌ 没有找到匹配的文件")
            sys.exit(1)
        
        # 生成报告
        success_count = 0
        for i, json_file in enumerate(json_files, 1):
            if not args.quiet and len(json_files) > 1:
                print(f"\n{'='*60}")
                print(f"处理文件 {i}/{len(json_files)}")
            
            output_path = args.output if len(json_files) == 1 else None
            
            if generate_report_from_json(json_file, output_path, args.open and i == 1):
                success_count += 1
        
        # 总结
        if len(json_files) > 1:
            print(f"\n{'='*60}")
            print(f"📊 完成! 成功生成 {success_count}/{len(json_files)} 个报告")


if __name__ == "__main__":
    main()


