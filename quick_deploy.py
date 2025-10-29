#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VEC项目快速部署到远程服务器（Python版本）
适用于Windows系统，无需安装sshpass

使用方法:
    python quick_deploy.py

需要依赖:
    pip install paramiko tqdm
"""

import os
import sys
import paramiko
from pathlib import Path
from tqdm import tqdm
import stat

# ========== 服务器配置 ==========
SERVER_CONFIG = {
    'hostname': 'region-9.autodl.pro',
    'port': 19287,
    'username': 'root',
    'password': 'dfUJkmli0mHk',
    'remote_dir': '/root/VEC_mig_caching'
}

# 需要排除的目录和文件
EXCLUDE_PATTERNS = [
    '__pycache__',
    '.git',
    'results',
    'models',
    'academic_figures',
    'test_results',
    '.pyc',
    '.png',
    '.pdf',
    '.jpg',
    '.jpeg'
]


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def should_exclude(path):
    """判断是否应该排除该路径"""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def connect_to_server(config):
    """连接到服务器"""
    print_header("[1/5] 连接到服务器")
    print(f"主机: {config['username']}@{config['hostname']}:{config['port']}")
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=config['hostname'],
            port=config['port'],
            username=config['username'],
            password=config['password'],
            timeout=10
        )
        print("✅ 连接成功！")
        return ssh
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        sys.exit(1)


def create_remote_directory(ssh, remote_dir):
    """创建远程目录"""
    print_header("[2/5] 创建远程项目目录")
    try:
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_dir}")
        stdout.channel.recv_exit_status()
        print(f"✅ 目录已创建: {remote_dir}")
    except Exception as e:
        print(f"❌ 创建目录失败: {e}")
        sys.exit(1)


def upload_files(ssh, local_dir, remote_dir):
    """上传项目文件"""
    print_header("[3/5] 上传项目文件")
    
    sftp = ssh.open_sftp()
    local_path = Path(local_dir)
    
    # 收集所有需要上传的文件
    files_to_upload = []
    for file_path in local_path.rglob('*'):
        if file_path.is_file() and not should_exclude(file_path):
            relative_path = file_path.relative_to(local_path)
            files_to_upload.append((file_path, relative_path))
    
    print(f"共需上传 {len(files_to_upload)} 个文件...\n")
    
    # 上传文件
    for local_file, relative_path in tqdm(files_to_upload, desc="上传进度"):
        remote_file = f"{remote_dir}/{relative_path}".replace('\\', '/')
        remote_file_dir = os.path.dirname(remote_file)
        
        # 创建远程目录
        try:
            sftp.stat(remote_file_dir)
        except FileNotFoundError:
            try:
                # 递归创建目录
                dirs = []
                current_dir = remote_file_dir
                while current_dir != remote_dir:
                    dirs.insert(0, current_dir)
                    current_dir = os.path.dirname(current_dir)
                
                for d in dirs:
                    try:
                        sftp.stat(d)
                    except FileNotFoundError:
                        sftp.mkdir(d)
            except Exception as e:
                print(f"\n⚠️  创建目录失败 {remote_file_dir}: {e}")
                continue
        
        # 上传文件
        try:
            sftp.put(str(local_file), remote_file)
        except Exception as e:
            print(f"\n⚠️  上传失败 {relative_path}: {e}")
    
    sftp.close()
    print("\n✅ 文件上传完成！")


def setup_environment(ssh, remote_dir):
    """配置服务器环境"""
    print_header("[4/5] 配置服务器环境")
    
    commands = [
        f"cd {remote_dir}",
        "echo '检查Python和CUDA环境...'",
        "python --version",
        "nvcc --version 2>/dev/null || echo '⚠️  CUDA未安装'",
        "echo ''",
        "echo '检查GPU...'",
        "nvidia-smi 2>/dev/null || echo '⚠️  无法检测GPU'",
        "echo ''",
        "echo '安装Python依赖...'",
        "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "echo ''",
        "echo '验证PyTorch和CUDA...'",
        "python -c \"import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')\""
    ]
    
    full_command = " && ".join(commands)
    
    try:
        stdin, stdout, stderr = ssh.exec_command(full_command)
        
        # 实时打印输出
        for line in stdout:
            print(line.strip())
        
        for line in stderr:
            error_line = line.strip()
            if error_line and not error_line.startswith("WARNING"):
                print(f"⚠️  {error_line}")
        
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("\n✅ 环境配置完成！")
        else:
            print(f"\n⚠️  环境配置可能存在问题（退出码: {exit_status}）")
    except Exception as e:
        print(f"❌ 环境配置失败: {e}")


def create_training_scripts(ssh, remote_dir):
    """创建训练启动脚本"""
    print_header("[5/5] 创建训练脚本")
    
    # 训练启动脚本
    start_script = """#!/bin/bash
# 远程训练启动脚本

echo "=========================================="
echo "VEC项目 - 训练启动"
echo "时间: $(date)"
echo "=========================================="

# 训练参数
ALGORITHM=${1:-TD3}
EPISODES=${2:-200}
DEVICE="cuda"

echo ""
echo "训练配置:"
echo "  算法: ${ALGORITHM}"
echo "  轮次: ${EPISODES}"
echo "  设备: ${DEVICE}"
echo ""

# 启动训练（后台运行，输出到日志）
nohup python train_single_agent.py \\
    --algorithm ${ALGORITHM} \\
    --episodes ${EPISODES} \\
    --device ${DEVICE} \\
    > training_${ALGORITHM}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ 训练已在后台启动！"
echo "   进程ID: ${PID}"
echo "   日志文件: training_${ALGORITHM}_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f training_${ALGORITHM}_*.log"
echo "  查看进程: ps aux | grep train_single_agent"
echo "  停止训练: kill ${PID}"
echo ""
"""

    # 监控脚本
    monitor_script = """#!/bin/bash
# 训练监控脚本

echo "=========================================="
echo "VEC训练监控"
echo "=========================================="

echo ""
echo "运行中的训练进程:"
ps aux | grep -E "(train_single_agent|train_multi_agent)" | grep -v grep

echo ""
echo "GPU使用情况:"
nvidia-smi

echo ""
echo "最新训练日志 (最后20行):"
if ls training_*.log 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t training_*.log | head -1)
    echo "日志文件: ${LATEST_LOG}"
    echo "----------------------------------------"
    tail -20 ${LATEST_LOG}
else
    echo "未找到训练日志"
fi
"""

    try:
        sftp = ssh.open_sftp()
        
        # 创建start_training.sh
        with sftp.open(f"{remote_dir}/start_training.sh", 'w') as f:
            f.write(start_script)
        sftp.chmod(f"{remote_dir}/start_training.sh", stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        print("✅ start_training.sh 创建完成")
        
        # 创建monitor_training.sh
        with sftp.open(f"{remote_dir}/monitor_training.sh", 'w') as f:
            f.write(monitor_script)
        sftp.chmod(f"{remote_dir}/monitor_training.sh", stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        print("✅ monitor_training.sh 创建完成")
        
        sftp.close()
    except Exception as e:
        print(f"❌ 脚本创建失败: {e}")


def print_usage_guide(config):
    """打印使用指南"""
    print_header("✅ 部署完成！")
    
    print("📝 下一步操作：\n")
    
    print("1️⃣  连接到服务器:")
    print(f"   ssh -p {config['port']} {config['username']}@{config['hostname']}")
    print(f"   密码: {config['password']}\n")
    
    print("2️⃣  进入项目目录:")
    print(f"   cd {config['remote_dir']}\n")
    
    print("3️⃣  启动训练（后台运行）:")
    print("   ./start_training.sh TD3 200        # 训练TD3算法200轮")
    print("   ./start_training.sh SAC 200        # 训练SAC算法200轮\n")
    
    print("4️⃣  监控训练进度:")
    print("   ./monitor_training.sh              # 查看训练状态")
    print("   tail -f training_*.log             # 实时查看日志\n")
    
    print("5️⃣  下载训练结果（在本地执行）:")
    print(f"   scp -P {config['port']} -r {config['username']}@{config['hostname']}:{config['remote_dir']}/results ./results_from_server\n")
    
    print("=" * 60)


def main():
    """主函数"""
    print_header("🚀 VEC项目远程服务器部署")
    print(f"目标服务器: {SERVER_CONFIG['username']}@{SERVER_CONFIG['hostname']}:{SERVER_CONFIG['port']}")
    
    # 检查paramiko
    try:
        import paramiko
    except ImportError:
        print("❌ 缺少paramiko库，正在安装...")
        os.system("pip install paramiko tqdm")
        print("请重新运行此脚本")
        sys.exit(1)
    
    # 获取当前目录
    local_dir = os.getcwd()
    print(f"本地项目目录: {local_dir}\n")
    
    # 确认部署
    response = input("是否继续部署？(y/n): ")
    if response.lower() != 'y':
        print("部署已取消")
        sys.exit(0)
    
    try:
        # 连接服务器
        ssh = connect_to_server(SERVER_CONFIG)
        
        # 创建远程目录
        create_remote_directory(ssh, SERVER_CONFIG['remote_dir'])
        
        # 上传文件
        upload_files(ssh, local_dir, SERVER_CONFIG['remote_dir'])
        
        # 配置环境
        setup_environment(ssh, SERVER_CONFIG['remote_dir'])
        
        # 创建训练脚本
        create_training_scripts(ssh, SERVER_CONFIG['remote_dir'])
        
        # 关闭连接
        ssh.close()
        
        # 打印使用指南
        print_usage_guide(SERVER_CONFIG)
        
    except KeyboardInterrupt:
        print("\n\n❌ 部署被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 部署失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

