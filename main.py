#!/usr/bin/env python3
"""
MATD3-MIG 主程序入口
车联网边缘缓存系统主控制程序
"""

import argparse
import sys
import os
from pathlib import Path

def print_banner():
    """打印程序横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🚗 MATD3-MIG 车联网边缘缓存系统                          ║
║                                                                              ║
║              Multi-Agent Twin Delayed DDPG for Vehicular Edge Caching       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def show_menu():
    """显示主菜单"""
    menu = """
🎯 请选择操作:

1. 🤖 多智能体训练 (MATD3/MADDPG)
2. 👤 单智能体训练 (DDPG/PPO)  
3. 🧪 运行完整实验
4. 📊 结果可视化
5. 📈 高级性能分析
6. 🔍 算法诊断
7. 🎬 系统演示
8. 🧹 项目清理
9. 📋 系统功能检查
0. 🚪 退出程序

请输入选项 (0-9): """
    
    return input(menu).strip()

def run_multi_agent_training():
    """运行多智能体训练"""
    print("🤖 启动多智能体训练...")
    os.system("python train_multi_agent.py")

def run_single_agent_training():
    """运行单智能体训练"""
    print("👤 启动单智能体训练...")
    os.system("python train_single_agent.py")

def run_full_experiment():
    """运行完整实验"""
    print("🧪 启动完整实验...")
    
    # 询问实验参数
    episodes = input("请输入训练轮次 (默认10): ").strip() or "10"
    runs = input("请输入运行次数 (默认3): ").strip() or "3"
    
    cmd = f"python run_full_experiment.py --episodes {episodes} --runs {runs}"
    os.system(cmd)

def run_visualization():
    """运行结果可视化"""
    print("📊 启动结果可视化...")
    os.system("python visualize_results.py")

def run_advanced_analysis():
    """运行高级分析"""
    print("📈 启动高级性能分析...")
    os.system("python advanced_analysis.py")

def run_algorithm_diagnostics():
    """运行算法诊断"""
    print("🔍 启动算法诊断...")
    os.system("python algorithm_diagnostics.py")

def run_system_demo():
    """运行系统演示"""
    print("🎬 启动系统演示...")
    os.system("python demo.py")

def run_project_cleanup():
    """运行项目清理"""
    print("🧹 启动项目清理...")
    confirm = input("确认要清理项目吗? (y/N): ").strip().lower()
    if confirm == 'y':
        os.system("python cleanup_project.py")
    else:
        print("❌ 清理操作已取消")

def run_system_check():
    """运行系统功能检查"""
    print("📋 显示系统功能检查...")
    
    if Path("system_functionality_check.md").exists():
        os.system("type system_functionality_check.md" if os.name == 'nt' else "cat system_functionality_check.md")
    else:
        print("❌ 系统功能检查文件不存在")

def show_help():
    """显示帮助信息"""
    help_text = """
🆘 MATD3-MIG 系统帮助

📚 主要功能:
  • 多智能体强化学习训练
  • 单智能体算法对比
  • 完整实验评估
  • 性能分析和可视化
  • 系统诊断和优化

🚀 快速开始:
  1. 首先运行完整实验: 选项 3
  2. 查看结果可视化: 选项 4
  3. 进行高级分析: 选项 5

📁 重要文件:
  • train_multi_agent.py - 多智能体训练
  • run_full_experiment.py - 完整实验
  • visualize_results.py - 结果可视化
  • demo.py - 系统演示

🔧 故障排除:
  • 如果训练失败，尝试算法诊断 (选项 6)
  • 如果结果异常，运行项目清理 (选项 8)
  • 查看系统功能检查 (选项 9)

💡 提示:
  • 建议先运行演示了解系统 (选项 7)
  • 训练前确保有足够的计算资源
  • 实验结果保存在 results/ 目录
"""
    print(help_text)

def main():
    """主函数"""
    print_banner()
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        sys.exit(1)
    
    # 检查必要的目录
    required_dirs = ['algorithms', 'models', 'environment', 'results']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    while True:
        try:
            choice = show_menu()
            
            if choice == '0':
                print("👋 感谢使用MATD3-MIG系统，再见！")
                break
            elif choice == '1':
                run_multi_agent_training()
            elif choice == '2':
                run_single_agent_training()
            elif choice == '3':
                run_full_experiment()
            elif choice == '4':
                run_visualization()
            elif choice == '5':
                run_advanced_analysis()
            elif choice == '6':
                run_algorithm_diagnostics()
            elif choice == '7':
                run_system_demo()
            elif choice == '8':
                run_project_cleanup()
            elif choice == '9':
                run_system_check()
            elif choice.lower() in ['h', 'help']:
                show_help()
            else:
                print("❌ 无效选项，请重新选择")
            
            # 等待用户确认继续
            if choice != '0':
                input("\n按Enter键继续...")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请检查系统配置或联系技术支持")

if __name__ == "__main__":
    # 支持命令行参数
    parser = argparse.ArgumentParser(description='MATD3-MIG 车联网边缘缓存系统')
    parser.add_argument('--demo', action='store_true', help='直接运行演示')
    parser.add_argument('--train', choices=['multi', 'single'], help='直接开始训练')
    parser.add_argument('--experiment', action='store_true', help='直接运行完整实验')
    parser.add_argument('--visualize', action='store_true', help='直接运行可视化')
    
    args = parser.parse_args()
    
    if args.demo:
        run_system_demo()
    elif args.train == 'multi':
        run_multi_agent_training()
    elif args.train == 'single':
        run_single_agent_training()
    elif args.experiment:
        run_full_experiment()
    elif args.visualize:
        run_visualization()
    else:
        main()