#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整对比实验（包含TD3训练和评估）
自动训练TD3并与所有基准算法对比
"""

import sys
import os

# 设置输出编码为UTF-8
import io
if sys.platform == 'win32':
    # Windows环境下设置输出编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    # 设置环境变量以支持UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
import json
import subprocess
import shutil
from pathlib import Path
import time

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def train_td3_if_needed(num_vehicles=12, episodes=200):
    """训练或验证TD3模型"""
    print("\n" + "="*70)
    print("步骤1: 检查/训练TD3模型")
    print("="*70)
    
    # 检查多个可能的模型路径
    model_paths = [
        parent_dir / f"results/single_agent/td3/{num_vehicles}/best_model.pth",
        parent_dir / f"models/td3/{num_vehicles}/best_model.pth",
        parent_dir / f"results/models/single_agent/td3/best_model_td3.pth",
        parent_dir / f"results/models/single_agent/td3/checkpoint_50_td3.pth",
    ]
    
    model_exists = False
    existing_model = None
    for path in model_paths:
        if path.exists():
            model_exists = True
            existing_model = path
            print(f"[FOUND] 找到已有TD3模型: {path}")
            break
    
    if not model_exists:
        print(f"[WARNING] 未找到TD3模型，开始训练...")
        
        # 训练TD3模型
        cmd = [
            sys.executable,
            "train_single_agent.py",
            "--algorithm", "TD3",
            "--num-vehicles", str(num_vehicles),
            "--episodes", str(episodes),
            "--save_interval", "50",
        ]
        
        try:
            print(f"\n运行命令: {' '.join(cmd)}")
            # 通过stdin自动输入'y'来保存HTML报告
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            result = subprocess.run(cmd, cwd=parent_dir, capture_output=False, 
                                  text=True, input='y\n', env=env)
            if result.returncode == 0:
                print("[SUCCESS] TD3训练完成!")
            else:
                print(f"[ERROR] TD3训练失败，错误代码: {result.returncode}")
                return False
        except Exception as e:
            print(f"[ERROR] 训练过程出错: {e}")
            return False
    
    # 确保模型在正确的位置（创建副本）
    target_dirs = [
        parent_dir / f"results/single_agent/td3/{num_vehicles}",
        parent_dir / f"models/td3/{num_vehicles}"
    ]
    
    # 找到最新的模型文件
    if existing_model:
        source_model = existing_model
    else:
        # 查找训练后的模型
        search_patterns = [
            parent_dir / "results/models/single_agent/td3/*.pth",
            parent_dir / "results/single_agent/td3/**/*.pth",
        ]
        
        import glob
        latest_model = None
        latest_time = 0
        
        for pattern in search_patterns:
            for model_file in glob.glob(str(pattern), recursive=True):
                if os.path.getmtime(model_file) > latest_time:
                    latest_time = os.path.getmtime(model_file)
                    latest_model = model_file
        
        if latest_model:
            source_model = Path(latest_model)
            print(f"找到最新训练的模型: {source_model}")
        else:
            print("[ERROR] 未找到任何TD3模型文件")
            return False
    
    # 复制到所有目标位置
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "best_model.pth"
        if not target_file.exists():
            try:
                shutil.copy2(source_model, target_file)
                print(f"[SUCCESS] 复制模型到: {target_file}")
            except Exception as e:
                print(f"[WARNING] 复制到 {target_file} 失败: {e}")
    
    return True

def run_comparison_experiment(episodes=50, vehicle_counts=[8, 12, 16]):
    """运行对比实验"""
    print("\n" + "="*70)
    print("步骤2: 运行对比实验")
    print("="*70)
    print(f"评估轮次: {episodes}")
    print(f"车辆数量: {vehicle_counts}")
    
    # 构建参数
    cmd = [
        sys.executable,
        "run_offloading_comparison.py",
        "--mode", "vehicle",
        "--episodes", str(episodes),
    ]
    
    # 如果是快速测试，修改车辆数量
    if len(vehicle_counts) < 5:
        # 需要修改默认的车辆数量配置
        import json
        config_file = Path(__file__).parent / "experiment_config.json"
        config = {
            "vehicle_counts": vehicle_counts,
            "episodes": episodes
        }
        with open(config_file, 'w') as f:
            json.dump(config, f)
        print(f"已保存实验配置到: {config_file}")
    
    try:
        print(f"\n运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=Path(__file__).parent, 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] 对比实验完成!")
            return True
        else:
            print(f"[ERROR] 实验失败，错误代码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"[ERROR] 实验过程出错: {e}")
        return False

def generate_visualizations():
    """生成可视化图表"""
    print("\n" + "="*70)
    print("步骤3: 生成可视化图表")  
    print("="*70)
    
    cmd = [sys.executable, "visualize_vehicle_comparison.py"]
    
    try:
        print("正在生成图表...")
        result = subprocess.run(cmd, cwd=Path(__file__).parent,
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] 可视化图表生成完成!")
            
            # 显示生成的文件
            figures_dir = Path(__file__).parent / "academic_figures/vehicle_comparison"
            if figures_dir.exists():
                files = list(figures_dir.glob("*"))
                if files:
                    print("\n[FILES] 生成的图表文件:")
                    for f in files:
                        print(f"  - {f.name}")
            
            return True
        else:
            print(f"[ERROR] 图表生成失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] 可视化过程出错: {e}")
        return False

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="完整对比实验")
    
    parser.add_argument('--quick', action='store_true',
                       help="快速测试模式")
    parser.add_argument('--skip-training', action='store_true',
                       help="跳过TD3训练（假设已有模型）")
    parser.add_argument('--train-episodes', type=int, default=200,
                       help="TD3训练轮次")
    parser.add_argument('--eval-episodes', type=int, default=50,
                       help="评估轮次")
    
    args = parser.parse_args()
    
    # 快速测试模式参数
    if args.quick:
        train_episodes = min(args.train_episodes, 50)
        eval_episodes = min(args.eval_episodes, 10)
        vehicle_counts = [8, 12, 16]
        print("[QUICK MODE] 快速测试模式")
    else:
        train_episodes = args.train_episodes
        eval_episodes = args.eval_episodes
        vehicle_counts = [8, 12, 16, 20, 24]
        print("[STANDARD MODE] 标准实验模式")
    
    print("\n" + "="*70)
    print("一键运行完整对比实验")
    print("="*70)
    print(f"TD3训练轮次: {train_episodes}")
    print(f"评估轮次: {eval_episodes}")
    print(f"车辆数量: {vehicle_counts}")
    print(f"跳过训练: {args.skip_training}")
    
    start_time = time.time()
    
    # 步骤1: 训练/验证TD3
    if not args.skip_training:
        if not train_td3_if_needed(12, train_episodes):
            print("\n[ERROR] TD3模型准备失败，退出实验")
            return 1
    else:
        print("\n[SKIP] 跳过TD3训练步骤")
    
    # 步骤2: 运行对比实验
    if not run_comparison_experiment(eval_episodes, vehicle_counts):
        print("\n[ERROR] 对比实验失败")
        return 1
    
    # 步骤3: 生成可视化
    if not generate_visualizations():
        print("\n[WARNING] 可视化生成失败，但实验数据已保存")
    
    # 完成总结
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("[COMPLETE] 实验完成总结")
    print("="*70)
    print(f"[TIME] 总耗时: {int(elapsed//60)}分{int(elapsed%60)}秒")
    
    print(f"""
[OUTPUT FILES] 输出文件:
  结果数据: results/offloading_comparison/vehicle_sweep_*.json
  对比图表: academic_figures/vehicle_comparison/
    - vehicle_comparison_main.pdf/png (4子图对比)
    - weighted_cost_highlight.pdf/png (成本突出图)
    - performance_table.md (性能表格)

[STRATEGIES] 对比策略:
  1. LocalOnly   - 纯本地计算（基准）
  2. RSUOnly     - 仅RSU卸载（传统MEC）
  3. LoadBalance - 负载均衡（启发式）
  4. Random      - 随机策略（对照组）
  5. TD3         - 完整TD3（主要贡献）
  6. TD3-NoMig   - 无迁移TD3（消融实验）
""")
    
    print("="*70)
    
    # 自动打开结果目录（Windows）
    if sys.platform == 'win32':
        try:
            import subprocess
            result_dir = Path(__file__).parent / "academic_figures/vehicle_comparison"
            if result_dir.exists():
                subprocess.Popen(f'explorer "{result_dir}"')
                print("[OPEN] 已打开结果目录")
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
