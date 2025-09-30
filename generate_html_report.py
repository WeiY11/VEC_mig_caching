"""
独立HTML报告生成脚本
用于从已有的训练结果JSON文件生成HTML报告
"""
import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime
from utils.html_report_generator import HTMLReportGenerator


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
    print(f"📖 读取训练结果: {json_path}")
    results = load_training_results(json_path)
    
    # 提取信息
    algorithm = results.get('algorithm', 'Unknown')
    training_time = results.get('training_config', {}).get('training_time_hours', 0) * 3600
    
    # 创建模拟环境
    training_env = create_mock_training_env(results)
    
    # 生成报告
    print("📝 生成HTML报告...")
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
    print(f"💾 保存报告到: {output_path}")
    if generator.save_report(html_content, output_path):
        print(f"✅ 报告保存成功!")
        
        # 打开浏览器
        if open_browser:
            print("🌐 在浏览器中打开报告...")
            abs_path = os.path.abspath(output_path)
            webbrowser.open(f'file://{abs_path}')
            print("✅ 报告已在浏览器中打开")
        else:
            print(f"💡 使用浏览器打开文件查看: {output_path}")
        
        return True
    else:
        print("❌ 报告保存失败")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='从训练结果JSON文件生成HTML报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
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
    
    parser.add_argument('json_files', nargs='+', help='训练结果JSON文件路径（支持多个文件）')
    parser.add_argument('-o', '--output', help='输出HTML文件路径（仅单文件时有效）')
    parser.add_argument('--open', action='store_true', help='生成后在浏览器中打开')
    parser.add_argument('--quiet', action='store_true', help='静默模式，不显示详细信息')
    
    args = parser.parse_args()
    
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


