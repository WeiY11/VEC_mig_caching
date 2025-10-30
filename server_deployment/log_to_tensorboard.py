#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控批量实验日志并写入 TensorBoard

【功能】
实时解析 batch_experiments.log，提取训练指标，写入 TensorBoard

【使用方法】
python log_to_tensorboard.py [--log-file PATH] [--tensorboard-dir PATH]

【监控指标】
- 平均奖励 (Average Reward)
- 平均时延 (Average Delay)  
- 总能耗 (Total Energy)
- 任务完成率 (Task Completion Rate)
- 实验进度
"""

import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class LogMonitor:
    """日志监控器"""
    
    def __init__(self, log_file: str, tensorboard_dir: str):
        self.log_file = Path(log_file)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.writers = {}  # 每个实验一个writer
        self.current_experiment = None
        self.experiment_step = defaultdict(int)
        
        # 确保TensorBoard目录存在
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 正则表达式模式
        self.patterns = {
            'experiment_start': re.compile(r'开始实验 (\d+)/\d+: (.+)'),
            'episode': re.compile(r'Episode (\d+)/\d+'),
            'avg_reward': re.compile(r'平均奖励[：:]\s*([-\d.]+)'),
            'avg_delay': re.compile(r'平均时延[：:]\s*([\d.]+)'),
            'total_energy': re.compile(r'总能耗[：:]\s*([\d.]+)'),
            'completion_rate': re.compile(r'完成率[：:]\s*([\d.]+)'),
            'experiment_complete': re.compile(r'实验完成'),
        }
    
    def get_writer(self, experiment_name: str) -> SummaryWriter:
        """获取或创建 TensorBoard writer"""
        if experiment_name not in self.writers:
            log_dir = self.tensorboard_dir / experiment_name
            self.writers[experiment_name] = SummaryWriter(log_dir=str(log_dir))
            print(f"[TensorBoard] 创建writer: {experiment_name}")
        return self.writers[experiment_name]
    
    def parse_line(self, line: str):
        """解析日志行"""
        # 检测实验开始
        match = self.patterns['experiment_start'].search(line)
        if match:
            exp_num = match.group(1)
            exp_name = match.group(2).strip()
            self.current_experiment = f"{exp_num}_{exp_name}"
            print(f"[监控] 开始跟踪实验: {self.current_experiment}")
            return
        
        # 如果没有当前实验，跳过
        if not self.current_experiment:
            return
        
        writer = self.get_writer(self.current_experiment)
        
        # 检测episode
        match = self.patterns['episode'].search(line)
        if match:
            episode = int(match.group(1))
            # 更新步数（以episode为单位）
            self.experiment_step[self.current_experiment] = episode
        
        # 提取指标
        step = self.experiment_step[self.current_experiment]
        
        # 平均奖励
        match = self.patterns['avg_reward'].search(line)
        if match and step > 0:
            value = float(match.group(1))
            writer.add_scalar('Reward/Average', value, step)
            print(f"[TensorBoard] {self.current_experiment} - Episode {step}: Avg Reward = {value:.2f}")
        
        # 平均时延
        match = self.patterns['avg_delay'].search(line)
        if match and step > 0:
            value = float(match.group(1))
            writer.add_scalar('Performance/Average_Delay', value, step)
            print(f"[TensorBoard] {self.current_experiment} - Episode {step}: Avg Delay = {value:.4f}")
        
        # 总能耗
        match = self.patterns['total_energy'].search(line)
        if match and step > 0:
            value = float(match.group(1))
            writer.add_scalar('Performance/Total_Energy', value, step)
            print(f"[TensorBoard] {self.current_experiment} - Episode {step}: Total Energy = {value:.2f}")
        
        # 完成率
        match = self.patterns['completion_rate'].search(line)
        if match and step > 0:
            value = float(match.group(1))
            writer.add_scalar('Performance/Completion_Rate', value, step)
            print(f"[TensorBoard] {self.current_experiment} - Episode {step}: Completion = {value:.2%}")
        
        # 实验完成
        if self.patterns['experiment_complete'].search(line):
            print(f"[监控] 实验完成: {self.current_experiment}")
            if self.current_experiment in self.writers:
                self.writers[self.current_experiment].flush()
    
    def monitor(self):
        """监控日志文件"""
        print(f"[监控] 开始监控日志文件: {self.log_file}")
        print(f"[监控] TensorBoard 日志目录: {self.tensorboard_dir}")
        print("[监控] 按 Ctrl+C 停止监控\n")
        
        # 如果文件不存在，等待创建
        while not self.log_file.exists():
            print(f"[监控] 等待日志文件创建: {self.log_file}")
            time.sleep(5)
        
        # 打开文件并跟踪
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 先读取已有内容
            for line in f:
                self.parse_line(line.strip())
            
            # 然后持续监控新内容
            while True:
                line = f.readline()
                if line:
                    self.parse_line(line.strip())
                else:
                    # 刷新所有writer
                    for writer in self.writers.values():
                        writer.flush()
                    time.sleep(1)  # 没有新行时等待
    
    def close(self):
        """关闭所有writer"""
        for writer in self.writers.values():
            writer.close()
        print("[监控] 所有 TensorBoard writer 已关闭")


def main():
    parser = argparse.ArgumentParser(description="监控实验日志并写入TensorBoard")
    parser.add_argument(
        "--log-file",
        type=str,
        default="/root/VEC_mig_caching/batch_experiments.log",
        help="日志文件路径"
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="/root/VEC_mig_caching/runs/batch_experiments",
        help="TensorBoard 日志目录"
    )
    args = parser.parse_args()
    
    monitor = LogMonitor(args.log_file, args.tensorboard_dir)
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n[监控] 收到停止信号")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()

