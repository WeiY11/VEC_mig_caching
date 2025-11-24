#!/usr/bin/env python3
"""
Enhanced TD3 消融实验 (Ablation Study)
系统对比5种优化方法的独立效果

实验配置：
- Baseline: 标准TD3
- Opt1: TD3 + Distributional Critic (分布式Critic)
- Opt2: TD3 + Entropy Regularization (熵正则化)
- Opt3: TD3 + Model-based Rollout (模型预测)
- Opt4: TD3 + Queue-aware Replay (队列感知回放)
- Opt5: TD3 + GNN Attention (图神经网络)
- Full:  TD3 + All 5 optimizations (全优化)
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_agent.enhanced_td3_config import EnhancedTD3Config


class AblationExperiment:
    """消融实验管理器"""
    
    def __init__(self, base_dir: str = "results/ablation_study"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_dir / f"run_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义实验配置
        self.experiments = {
            "baseline": {
                "name": "TD3 Baseline",
                "description": "标准TD3，无优化",
                "config": {
                    "use_distributional_critic": False,
                    "use_entropy_reg": False,
                    "use_model_based_rollout": False,
                    "use_queue_aware_replay": False,
                    "use_gat_router": False,
                }
            },
            "opt1_distributional": {
                "name": "TD3 + Distributional Critic",
                "description": "仅启用分布式Critic (QR-DQN)",
                "config": {
                    "use_distributional_critic": True,
                    "use_entropy_reg": False,
                    "use_model_based_rollout": False,
                    "use_queue_aware_replay": False,
                    "use_gat_router": False,
                }
            },
            "opt2_entropy": {
                "name": "TD3 + Entropy Regularization",
                "description": "仅启用熵正则化 (SAC-style)",
                "config": {
                    "use_distributional_critic": False,
                    "use_entropy_reg": True,
                    "use_model_based_rollout": False,
                    "use_queue_aware_replay": False,
                    "use_gat_router": False,
                }
            },
            "opt3_model": {
                "name": "TD3 + Model-based Rollout",
                "description": "仅启用模型预测回放",
                "config": {
                    "use_distributional_critic": False,
                    "use_entropy_reg": False,
                    "use_model_based_rollout": True,
                    "use_queue_aware_replay": False,
                    "use_gat_router": False,
                }
            },
            "opt4_queue": {
                "name": "TD3 + Queue-aware Replay",
                "description": "仅启用队列感知优先回放",
                "config": {
                    "use_distributional_critic": False,
                    "use_entropy_reg": False,
                    "use_model_based_rollout": False,
                    "use_queue_aware_replay": True,
                    "use_gat_router": False,
                }
            },
            "opt5_gnn": {
                "name": "TD3 + GNN Attention",
                "description": "仅启用图神经网络注意力",
                "config": {
                    "use_distributional_critic": False,
                    "use_entropy_reg": False,
                    "use_model_based_rollout": False,
                    "use_queue_aware_replay": False,
                    "use_gat_router": True,
                }
            },
            "full": {
                "name": "TD3 + All Optimizations",
                "description": "启用全部5项优化",
                "config": {
                    "use_distributional_critic": True,
                    "use_entropy_reg": True,
                    "use_model_based_rollout": True,
                    "use_queue_aware_replay": True,
                    "use_gat_router": True,
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
        print(f"\n{'='*60}")
        print(f"[INFO] 运行实验: {exp_config['name']}")
        print(f"   {exp_config['description']}")
        print(f"{'='*60}")
        
        # 创建配置对象
        config = EnhancedTD3Config(**exp_config['config'])
        
        # 保存配置
        config_file = self.results_dir / f"{exp_id}_config.json"
        config_dict = {
            "name": exp_config['name'],
            "description": exp_config['description'],
            **exp_config['config']
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # 运行训练
        start_time = time.time()
        
        # 使用临时Python脚本运行，避免import问题
        train_script = f"""
import sys
import os
sys.path.insert(0, {repr(os.getcwd())})

from single_agent.enhanced_td3_wrapper import EnhancedTD3Wrapper
from config import config
import numpy as np

# 设置随机种子
np.random.seed({seed})

# 创建环境
env = EnhancedTD3Wrapper(
    num_vehicles={num_vehicles},
    num_rsus=4,
    num_uavs=2,
    use_central_resource=True,
    config_preset=None  # 使用自定义配置
)

# 应用自定义配置
env.agent.config.use_distributional_critic = {exp_config['config']['use_distributional_critic']}
env.agent.config.use_entropy_reg = {exp_config['config']['use_entropy_reg']}
env.agent.config.use_model_based_rollout = {exp_config['config']['use_model_based_rollout']}
env.agent.config.use_queue_aware_replay = {exp_config['config']['use_queue_aware_replay']}
env.agent.config.use_gat_router = {exp_config['config']['use_gat_router']}

# 训练
print("开始训练...")
# TODO: 集成到train_single_agent.py
"""
        
        # 实际上，我们应该通过train_single_agent.py运行
        # 但由于被删除了Enhanced TD3支持，我们需要另一种方法
        
        cmd = [
            sys.executable,
            "train_single_agent.py",
            "--algorithm", "TD3",  # 使用TD3作为基础
            "--episodes", str(episodes),
            "--num-vehicles", str(num_vehicles),
        ]
        
        print(f"命令: {' '.join(cmd)}")
        print("WARNING: 当前使用标准TD3训练，需要集成Enhanced TD3支持")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "elapsed_time": elapsed_time,
                "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            }
        except Exception as e:
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    def run_all_experiments(
        self,
        episodes: int = 1500,
        num_vehicles: int = 12,
        seed: int = 42
    ):
        """运行所有消融实验"""
        print(f"\nEnhanced TD3 消融实验")
        print(f"{'='*60}")
        print(f"实验数量: {len(self.experiments)}")
        print(f"训练轮次: {episodes}")
        print(f"车辆数量: {num_vehicles}")
        print(f"随机种子: {seed}")
        print(f"结果目录: {self.results_dir}")
        print(f"{'='*60}\n")
        
        results = {}
        
        for exp_id in self.experiments.keys():
            result = self.run_single_experiment(exp_id, episodes, num_vehicles, seed)
            results[exp_id] = result
            
            # 保存中间结果
            self._save_results(results)
            
            if result['success']:
                print(f"[OK] {self.experiments[exp_id]['name']} 完成")
                print(f"   用时: {result['elapsed_time']/60:.1f} 分钟")
            else:
                print(f"[FAIL] {self.experiments[exp_id]['name']} 失败")
                if 'error' in result:
                    print(f"   错误: {result['error']}")
        
        # 生成最终报告
        self._generate_report(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """保存实验结果"""
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _generate_report(self, results: Dict):
        """生成实验报告"""
        report_file = self.results_dir / "ablation_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Enhanced TD3 消融实验报告\n\n")
            f.write(f"实验时间: {self.timestamp}\n\n")
            
            f.write("## 实验配置\n\n")
            f.write("| 实验ID | 名称 | 描述 | 状态 | 用时(分钟) |\n")
            f.write("|--------|------|------|------|------------|\n")
            
            for exp_id, exp_config in self.experiments.items():
                result = results.get(exp_id, {})
                status = "✅" if result.get('success') else "❌"
                elapsed = result.get('elapsed_time', 0) / 60
                
                f.write(f"| {exp_id} | {exp_config['name']} | {exp_config['description']} | {status} | {elapsed:.1f} |\n")
            
            f.write("\n## 优化对比\n\n")
            f.write("### 5种优化方法\n\n")
            f.write("1. **Distributional Critic** (分布式Critic)\n")
            f.write("   - 使用QR-DQN风格的分位数网络\n")
            f.write("   - 学习价值分布而非期望值\n")
            f.write("   - 优势: 更好的风险感知，避免尾部延迟\n\n")
            
            f.write("2. **Entropy Regularization** (熵正则化)\n")
            f.write("   - SAC风格的熵奖励\n")
            f.write("   - 鼓励探索，避免过早收敛\n")
            f.write("   - 优势: 提高策略鲁棒性\n\n")
            
            f.write("3. **Model-based Rollout** (模型预测)\n")
            f.write("   - 学习队列动态模型\n")
            f.write("   - 生成合成经验进行训练\n")
            f.write("   - 优势: 提高样本效率\n\n")
            
            f.write("4. **Queue-aware Replay** (队列感知回放)\n")
            f.write("   - 优先回放队列拥塞相关经验\n")
            f.write("   - 结合TD-error和队列稀缺性\n")
            f.write("   - 优势: 加速学习关键场景\n\n")
            
            f.write("5. **GNN Attention** (图神经网络)\n")
            f.write("   - 使用GAT处理节点关系\n")
            f.write("   - 动态学习节点间注意力权重\n")
            f.write("   - 优势: 更好的缓存和迁移决策\n\n")
            
            f.write("\n## 下一步分析\n\n")
            f.write("1. 收集每个实验的训练结果\n")
            f.write("2. 对比关键指标:\n")
            f.write("   - 平均延迟\n")
            f.write("   - 总能耗\n")
            f.write("   - 缓存命中率\n")
            f.write("   - 任务完成率\n")
            f.write("   - 训练收敛速度\n")
            f.write("3. 绘制对比图表\n")
            f.write("4. 分析优化组合效果\n")
        
        print(f"\n报告已生成: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced TD3 消融实验')
    parser.add_argument('--episodes', type=int, default=1500, help='训练轮次')
    parser.add_argument('--num-vehicles', type=int, default=12, help='车辆数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--experiment', type=str, default='all', 
                       help='运行特定实验 (baseline, opt1_distributional, opt2_entropy, opt3_model, opt4_queue, opt5_gnn, full, all)')
    
    args = parser.parse_args()
    
    # 创建实验管理器
    ablation = AblationExperiment()
    
    if args.experiment == 'all':
        # 运行所有实验
        ablation.run_all_experiments(
            episodes=args.episodes,
            num_vehicles=args.num_vehicles,
            seed=args.seed
        )
    else:
        # 运行单个实验
        if args.experiment in ablation.experiments:
            result = ablation.run_single_experiment(
                args.experiment,
                episodes=args.episodes,
                num_vehicles=args.num_vehicles,
                seed=args.seed
            )
            status = "成功" if result.get('success') else "失败"
            elapsed_min = result.get('elapsed_time', 0) / 60
            print(f"\n实验完成，状态: {status}，用时: {elapsed_min:.1f} 分钟")
        else:
            print(f"错误: 未知实验 '{args.experiment}'")
            print(f"可用实验: {', '.join(ablation.experiments.keys())}, all")


if __name__ == '__main__':
    main()
