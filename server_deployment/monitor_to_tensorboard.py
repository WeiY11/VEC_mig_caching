#!/usr/bin/env python
"""
实时监控训练进度并写入TensorBoard
"""
import json
import time
from pathlib import Path
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("安装tensorboard...")
    import os
    os.system("pip install tensorboard -q")
    from torch.utils.tensorboard import SummaryWriter

def monitor_experiments():
    """监控实验结果并实时更新TensorBoard"""
    
    results_dir = Path("results/parameter_sensitivity")
    tb_dir = Path("runs/batch_experiments")
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VEC批量实验 - TensorBoard实时监控")
    print("=" * 60)
    print(f"监控目录: {results_dir}")
    print(f"TensorBoard日志: {tb_dir}")
    print(f"启动时间: {datetime.now()}")
    print()
    print("提示: 在AutoDL控制台点击'TensorBoard'按钮即可查看")
    print("      或访问: http://localhost:6007")
    print()
    print("按 Ctrl+C 停止监控")
    print("=" * 60)
    print()
    
    processed_files = set()
    writers = {}
    
    try:
        while True:
            # 扫描所有summary文件
            summary_files = list(results_dir.glob("**/summary.json"))
            
            for summary_file in summary_files:
                if summary_file in processed_files:
                    continue
                
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 获取实验名称
                    exp_name = summary_file.parent.name
                    
                    # 创建或获取writer
                    if exp_name not in writers:
                        writers[exp_name] = SummaryWriter(log_dir=str(tb_dir / exp_name))
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 新实验: {exp_name}")
                    
                    writer = writers[exp_name]
                    
                    # 写入配置数据
                    if 'configurations' in data:
                        configs = data['configurations']
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 {exp_name}: {len(configs)}个配置")
                        
                        for idx, config in enumerate(configs):
                            config_name = config.get('name', f'config_{idx}')
                            metrics = config.get('metrics', {})
                            
                            # 写入所有可用指标
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    writer.add_scalar(
                                        f'{exp_name}/{config_name}/{metric_name}',
                                        value,
                                        idx
                                    )
                            
                            # 特别标注归一化成本
                            if 'normalized_cost' in metrics:
                                writer.add_scalar(
                                    f'Summary/normalized_cost',
                                    metrics['normalized_cost'],
                                    idx
                                )
                    
                    writer.flush()
                    processed_files.add(summary_file)
                    
                except Exception as e:
                    print(f"⚠️  处理 {summary_file.name} 时出错: {e}")
            
            # 每30秒检查一次
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n停止监控...")
        for writer in writers.values():
            writer.close()
        print("✅ TensorBoard日志已保存")

if __name__ == "__main__":
    monitor_experiments()

