#!/usr/bin/env python3
"""
🔬 消融实验运行脚本
简化版 - 直接调用train_single_agent.py运行7组实验
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SimpleAblationRunner:
    """简化的消融实验运行器"""
    
    def __init__(self, base_dir: str = "results/ablation_study"):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_dir / f"run_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义7组实验配置
        self.experiments = {
            "baseline_td3": {
                "name": "TD3 Baseline",
                "algorithm": "TD3",
                "env_vars": {}
            },
            "opt1_distributional": {
                "name": "TD3 + Distributional",
                "algorithm": "ENHANCED_TD3",
                "env_vars": {
                    "ENHANCED_TD3_USE_DISTRIBUTIONAL": "1",
                    "ENHANCED_TD3_USE_ENTROPY": "0",
                    "ENHANCED_TD3_USE_MODEL": "0",
                    "ENHANCED_TD3_USE_QUEUE": "0",
                    "ENHANCED_TD3_USE_GNN": "0",
                }
            },
            "opt2_entropy": {
                "name": "TD3 + Entropy",
                "algorithm": "ENHANCED_TD3",
                "env_vars": {
                    "ENHANCED_TD3_USE_DISTRIBUTIONAL": "0",
                    "ENHANCED_TD3_USE_ENTROPY": "1",
                    "ENHANCED_TD3_USE_MODEL": "0",
                    "ENHANCED_TD3_USE_QUEUE": "0",
                    "ENHANCED_TD3_USE_GNN": "0",
                }
            },
            "opt3_model": {
                "name": "TD3 + Model",
                "algorithm": "ENHANCED_TD3",
                "env_vars": {
                    "ENHANCED_TD3_USE_DISTRIBUTIONAL": "0",
                    "ENHANCED_TD3_USE_ENTROPY": "0",
                    "ENHANCED_TD3_USE_MODEL": "1",
                    "ENHANCED_TD3_USE_QUEUE": "0",
                    "ENHANCED_TD3_USE_GNN": "0",
                }
            },
            "opt4_queue": {
                "name": "TD3 + Queue-aware",
                "algorithm": "ENHANCED_TD3",
                "env_vars": {
                    "ENHANCED_TD3_USE_DISTRIBUTIONAL": "0",
                    "ENHANCED_TD3_USE_ENTROPY": "0",
                    "ENHANCED_TD3_USE_MODEL": "0",
                    "ENHANCED_TD3_USE_QUEUE": "1",
                    "ENHANCED_TD3_USE_GNN": "0",
                }
            },
            "opt5_gnn": {
                "name": "TD3 + GNN",
                "algorithm": "ENHANCED_TD3",
                "env_vars": {
                    "ENHANCED_TD3_USE_DISTRIBUTIONAL": "0",
                    "ENHANCED_TD3_USE_ENTROPY": "0",
                    "ENHANCED_TD3_USE_MODEL": "0",
                    "ENHANCED_TD3_USE_QUEUE": "0",
                    "ENHANCED_TD3_USE_GNN": "1",
                }
            },
            "full_optimizations": {
                "name": "TD3 + All Optimizations",
                "algorithm": "ENHANCED_TD3",
                "env_vars": {
                    "ENHANCED_TD3_USE_DISTRIBUTIONAL": "1",
                    "ENHANCED_TD3_USE_ENTROPY": "1",
                    "ENHANCED_TD3_USE_MODEL": "1",
                    "ENHANCED_TD3_USE_QUEUE": "1",
                    "ENHANCED_TD3_USE_GNN": "1",
                }
            },
        }
    
    def run_single_experiment(
        self,
        exp_id: str,
        episodes: int = 1500,
        num_vehicles: int = 12,
        seed: int = 42
    ) -> Dict:
        """运行单个实验"""
        exp_config = self.experiments[exp_id]
        
        print(f"\n{'='*70}")
        print(f"🔬 实验: {exp_config['name']}")
        print(f"{'='*70}")
        
        # 设置环境变量
        env = os.environ.copy()
        env.update(exp_config['env_vars'])
        
        # 保存配置
        config_file = self.results_dir / f"{exp_id}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                "exp_id": exp_id,
                "name": exp_config['name'],
                "algorithm": exp_config['algorithm'],
                "env_vars": exp_config['env_vars'],
                "episodes": episodes,
                "num_vehicles": num_vehicles,
                "seed": seed,
            }, f, indent=2, ensure_ascii=False)
        
        # 构建命令
        output_dir = self.results_dir / exp_id
        cmd = [
            sys.executable,
            "train_single_agent.py",
            "--algorithm", exp_config['algorithm'],
            "--episodes", str(episodes),
            "--num-vehicles", str(num_vehicles),
            "--seed", str(seed),
            "--output-dir", str(output_dir),
        ]
        
        print(f"命令: {' '.join(cmd)}")
        print(f"环境变量: {exp_config['env_vars']}")
        print()
        
        # 运行训练
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            elapsed_time = time.time() - start_time
            success = result.returncode == 0
            
            # 保存结果
            result_data = {
                "success": success,
                "elapsed_time": elapsed_time,
                "returncode": result.returncode,
            }
            
            # 尝试提取训练结果
            if success and output_dir.exists():
                training_file = output_dir / "training_results.json"
                if training_file.exists():
                    with open(training_file, 'r', encoding='utf-8') as f:
                        training_data = json.load(f)
                        result_data['training_data'] = training_data
            
            # 保存结果
            result_file = self.results_dir / f"{exp_id}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            if success:
                print(f"✅ {exp_config['name']} 完成 ({elapsed_time/60:.1f}分钟)")
            else:
                print(f"❌ {exp_config['name']} 失败 (返回码: {result.returncode})")
                print(f"错误输出: {result.stderr[-500:]}")
            
            return result_data
            
        except Exception as e:
            print(f"❌ 实验异常: {e}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    def run_all(
        self,
        episodes: int = 1500,
        num_vehicles: int = 12,
        seed: int = 42,
        skip_baseline: bool = False
    ):
        """运行所有实验"""
            if skip_baseline and exp_id == "baseline_td3":
                print(f"⏭️  跳过baseline (使用已有结果)")
                continue
            
            result = self.run_single_experiment(exp_id, episodes, num_vehicles, seed)
            summary[exp_id] = result
            
            # 保存中间进度
            progress_file = self.results_dir / "progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成汇总报告
        self.generate_summary(summary)
        
        print(f"\n✅ 所有实验完成！")
        print(f"📊 结果目录: {self.results_dir}")
        print(f"\n下一步: 运行可视化分析")
        print(f"python visualize_ablation_results.py --results-dir {self.results_dir}")
    
    def generate_summary(self, results: Dict):
        """生成简要总结"""
        summary_file = self.results_dir / "summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("消融实验总结\n")
            f.write("=" * 60 + "\n\n")
            
            for exp_id, result in results.items():
                exp_name = self.experiments[exp_id]['name']
                status = "✅ 成功" if result.get('success') else "❌ 失败"
                elapsed = result.get('elapsed_time', 0) / 60
                
                f.write(f"{exp_name}:\n")
                f.write(f"  状态: {status}\n")
                f.write(f"  用时: {elapsed:.1f} 分钟\n")
                
                if 'training_data' in result:
                    training = result['training_data']
                    f.write(f"  最终奖励: {training.get('final_reward', 'N/A')}\n")
                    f.write(f"  平均延迟: {training.get('avg_delay', 'N/A')}\n")
                    f.write(f"  缓存命中率: {training.get('cache_hit_rate', 'N/A')}\n")
                
                f.write("\n")
        
        print(f"✅ 总结已保存: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='消融实验运行器')
    parser.add_argument('--episodes', type=int, default=1500, help='训练轮次')
    parser.add_argument('--num-vehicles', type=int, default=12, help='车辆数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--skip-baseline', action='store_true', help='跳过baseline实验')
    parser.add_argument('--experiment', type=str, default='all', 
                       help='运行特定实验 (baseline_td3, opt1_distributional, etc., all)')
    
    args = parser.parse_args()
    
    runner = SimpleAblationRunner()
    
    if args.experiment == 'all':
        runner.run_all(
            episodes=args.episodes,
            num_vehicles=args.num_vehicles,
            seed=args.seed,
            skip_baseline=args.skip_baseline
        )
    else:
        if args.experiment in runner.experiments:
            runner.run_single_experiment(
                args.experiment,
                episodes=args.episodes,
                num_vehicles=args.num_vehicles,
                seed=args.seed
            )
        else:
            print(f"❌ 未知实验: {args.experiment}")
            print(f"可用实验: {', '.join(runner.experiments.keys())}, all")


if __name__ == '__main__':
    main()
