#!/usr/bin/env python
"""
在服务器上启动批量实验的脚本
"""
import subprocess
import sys
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"batch_experiments_{timestamp}.log"
    
    print("=" * 50)
    print("VEC批量参数敏感性实验")
    print(f"开始时间: {datetime.now()}")
    print("=" * 50)
    print()
    
    # 构建命令
    cmd = [
        "nohup",
        "python",
        "experiments/camtd3_strategy_suite/run_batch_experiments.py",
        "--mode", "full",
        "--all",
        "--non-interactive",
    ]
    
    # 启动进程
    with open(log_file, 'w') as log:
        log.write(f"VEC批量实验启动\n")
        log.write(f"时间: {datetime.now()}\n")
        log.write(f"命令: {' '.join(cmd)}\n")
        log.write("=" * 50 + "\n\n")
        
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd="/root/VEC_mig_caching"
        )
    
    # 保存PID
    with open("batch_experiments.pid", "w") as f:
        f.write(str(process.pid))
    
    print(f"✅ 批量实验已启动！")
    print(f"   进程ID: {process.pid}")
    print(f"   日志文件: {log_file}")
    print()
    print("监控命令:")
    print(f"  tail -f {log_file}")
    print(f"  nvidia-smi")
    print(f"  ps -p {process.pid}")
    print()
    print("停止实验:")
    print(f"  kill {process.pid}")
    print()

if __name__ == "__main__":
    main()

