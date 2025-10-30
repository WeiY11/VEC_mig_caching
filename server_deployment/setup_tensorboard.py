#!/usr/bin/env python
"""
为VEC批量实验配置TensorBoard
"""
import os
import json
from pathlib import Path
from datetime import datetime

def create_tensorboard_logs():
    """从现有日志创建TensorBoard可视化数据"""
    
    # 检查是否安装tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard已安装")
    except ImportError:
        print("❌ TensorBoard未安装，正在安装...")
        os.system("pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple")
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard安装完成")
    
    # 创建TensorBoard日志目录
    tb_dir = Path("runs/batch_experiments")
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描结果目录
    results_dir = Path("results/parameter_sensitivity")
    if not results_dir.exists():
        print("⚠️  结果目录不存在，等待实验生成结果...")
        return
    
    print(f"扫描结果目录: {results_dir}")
    
    # 查找所有summary文件
    summary_files = list(results_dir.glob("**/summary.json"))
    
    if not summary_files:
        print("⚠️  未找到summary文件，实验可能刚开始...")
        print("💡 可以先启动TensorBoard，它会自动更新")
        return
    
    print(f"找到 {len(summary_files)} 个实验结果")
    
    # 为每个实验创建TensorBoard记录
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            exp_name = summary_file.parent.name
            writer = SummaryWriter(log_dir=str(tb_dir / exp_name))
            
            # 记录配置信息
            if 'configurations' in data:
                for idx, config in enumerate(data['configurations']):
                    config_name = config.get('name', f'config_{idx}')
                    metrics = config.get('metrics', {})
                    
                    # 写入关键指标
                    if 'avg_delay' in metrics:
                        writer.add_scalar(f'{config_name}/avg_delay', metrics['avg_delay'], idx)
                    if 'total_energy' in metrics:
                        writer.add_scalar(f'{config_name}/total_energy', metrics['total_energy'], idx)
                    if 'normalized_cost' in metrics:
                        writer.add_scalar(f'{config_name}/normalized_cost', metrics['normalized_cost'], idx)
                    if 'completion_rate' in metrics:
                        writer.add_scalar(f'{config_name}/completion_rate', metrics['completion_rate'], idx)
            
            writer.close()
            print(f"  ✅ {exp_name}")
            
        except Exception as e:
            print(f"  ⚠️  {exp_name}: {e}")
    
    print()
    print("=" * 50)
    print("✅ TensorBoard日志创建完成！")
    print()
    print("启动TensorBoard:")
    print(f"  tensorboard --logdir={tb_dir} --port=6006 --bind_all")
    print()

if __name__ == "__main__":
    create_tensorboard_logs()

