#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新启动实验（带 TensorBoard 支持）- Python 版本

【功能】
1. 停止旧的实验进程
2. 清理旧日志和 TensorBoard 数据（可选）
3. 启动 TensorBoard 服务
4. 启动新的批量实验
5. 提供监控命令

【使用方法】
python restart_with_tensorboard.py [--clean]

【参数】
--clean: 清理旧的日志和 TensorBoard 数据
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ========== 服务器配置 ==========
SERVER_HOST = "region-9.autodl.pro"
SERVER_PORT = "47042"
SERVER_USER = "root"
SERVER_PASSWORD = "dfUJkmli0mHk"
REMOTE_DIR = "/root/VEC_mig_caching"

# ========== 颜色输出 ==========
class Color:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_colored(color, text):
    print(f"{color}{text}{Color.NC}")

def run_ssh_command(command, description="执行命令"):
    """执行SSH命令"""
    full_cmd = [
        "ssh",
        "-p", SERVER_PORT,
        f"{SERVER_USER}@{SERVER_HOST}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        command
    ]
    
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={"SSHPASS": SERVER_PASSWORD}
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"{description} 超时"
    except Exception as e:
        return False, "", str(e)

def main():
    parser = argparse.ArgumentParser(description="重新启动实验（带 TensorBoard）")
    parser.add_argument("--clean", action="store_true", help="清理旧的日志和TensorBoard数据")
    args = parser.parse_args()
    
    print("")
    print_colored(Color.YELLOW, "=" * 50)
    print_colored(Color.YELLOW, "   重新启动实验（带 TensorBoard 支持）")
    print_colored(Color.YELLOW, "=" * 50)
    print("")
    
    # ========== 步骤1: 停止旧进程 ==========
    print_colored(Color.YELLOW, "[步骤 1/5] 停止旧的实验进程...")
    success, stdout, stderr = run_ssh_command(
        "pkill -f run_batch_experiments || true",
        "停止进程"
    )
    if success:
        print_colored(Color.GREEN, "[OK] 旧进程已停止")
    else:
        print_colored(Color.RED, f"[FAIL] 停止进程失败: {stderr}")
    
    # ========== 步骤2: 清理旧数据（可选）==========
    if args.clean:
        print_colored(Color.YELLOW, "[步骤 2/5] 清理旧日志和 TensorBoard 数据...")
        success, stdout, stderr = run_ssh_command(
            f"cd {REMOTE_DIR} && rm -f batch_experiments.log && rm -rf runs/batch_experiments/*",
            "清理数据"
        )
        if success:
            print_colored(Color.GREEN, "[OK] 旧数据已清理")
        else:
            print_colored(Color.RED, f"[FAIL] 清理数据失败: {stderr}")
    else:
        print_colored(Color.YELLOW, "[步骤 2/5] 跳过清理旧数据（使用 --clean 参数启用）")
    
    # ========== 步骤3: 设置 TensorBoard ==========
    print_colored(Color.YELLOW, "[步骤 3/5] 设置 TensorBoard...")
    success, stdout, stderr = run_ssh_command(
        f"cd {REMOTE_DIR} && bash server_deployment/setup_autodl_tensorboard.sh",
        "设置TensorBoard"
    )
    if success:
        print_colored(Color.GREEN, "[OK] TensorBoard 设置完成")
    else:
        print_colored(Color.RED, f"[FAIL] TensorBoard 设置失败: {stderr}")
    
    # ========== 步骤4: 启动 TensorBoard ==========
    print_colored(Color.YELLOW, "[步骤 4/5] 启动 TensorBoard 服务...")
    success, stdout, stderr = run_ssh_command(
        f"cd {REMOTE_DIR} && bash server_deployment/start_tensorboard.sh",
        "启动TensorBoard"
    )
    if success:
        print_colored(Color.GREEN, "[OK] TensorBoard 服务已启动（端口 6006）")
    else:
        print_colored(Color.RED, f"[FAIL] TensorBoard 启动失败: {stderr}")
    
    # ========== 步骤5: 启动新实验 ==========
    print_colored(Color.YELLOW, "[步骤 5/5] 启动新的批量实验...")
    success, stdout, stderr = run_ssh_command(
        f"cd {REMOTE_DIR} && bash server_deployment/remote_start.sh",
        "启动实验"
    )
    if success:
        print_colored(Color.GREEN, "[OK] 批量实验已启动")
    else:
        print_colored(Color.RED, f"[FAIL] 实验启动失败: {stderr}")
        sys.exit(1)
    
    # ========== 显示监控信息 ==========
    print("")
    print_colored(Color.GREEN, "=" * 50)
    print_colored(Color.GREEN, "   实验重启成功！")
    print_colored(Color.GREEN, "=" * 50)
    print("")
    print_colored(Color.YELLOW, "[TensorBoard 访问方式]")
    print("1. AutoDL 控制台 -> 自定义服务 -> TensorBoard (端口 6006)")
    print(f"2. SSH 隧道: ssh -p {SERVER_PORT} -L 6006:localhost:6006 {SERVER_USER}@{SERVER_HOST}")
    print("   然后访问: http://localhost:6006")
    print("")
    print_colored(Color.YELLOW, "[监控实验进度]")
    print("查看日志:")
    print(f"  ssh -p {SERVER_PORT} {SERVER_USER}@{SERVER_HOST} 'tail -f {REMOTE_DIR}/batch_experiments.log'")
    print("")
    print("运行监控脚本:")
    print(f"  ssh -p {SERVER_PORT} {SERVER_USER}@{SERVER_HOST} 'cd {REMOTE_DIR} && bash server_deployment/remote_monitor.sh'")
    print("")
    print("检查进程状态:")
    print(f"  ssh -p {SERVER_PORT} {SERVER_USER}@{SERVER_HOST} 'ps aux | grep run_batch'")
    print("")
    print("检查 GPU 使用:")
    print(f"  ssh -p {SERVER_PORT} {SERVER_USER}@{SERVER_HOST} 'nvidia-smi'")
    print("")

if __name__ == "__main__":
    main()

