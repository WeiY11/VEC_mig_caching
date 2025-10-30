#!/bin/bash
# Kaggle环境快速设置脚本
# 在Kaggle Notebook中运行此脚本来准备训练环境

echo "🚀 开始配置VEC边缘计算训练环境..."

# 1. 安装依赖（Kaggle已预装大部分，只需补充缺失的）
echo "📦 检查并安装依赖..."
pip install flask-socketio -q 2>/dev/null || echo "flask-socketio跳过（可选）"

# 2. 设置PyTorch使用GPU
echo "🔧 配置GPU环境..."
python -c "import torch; print(f'✅ PyTorch版本: {torch.__version__}'); print(f'✅ CUDA可用: {torch.cuda.is_available()}'); print(f'✅ GPU数量: {torch.cuda.device_count()}' if torch.cuda.is_available() else '⚠️  未检测到GPU')"

# 3. 创建结果目录
echo "📁 创建输出目录..."
mkdir -p results/single_agent/{td3,ddpg,sac,ppo,dqn}
mkdir -p results/multi_agent
mkdir -p academic_figures

# 4. 验证项目结构
echo "🔍 验证项目结构..."
python -c "
import sys
required_modules = ['config', 'evaluation', 'single_agent', 'utils']
missing = []
for mod in required_modules:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f'❌ 缺少模块: {missing}')
    sys.exit(1)
else:
    print('✅ 所有核心模块已就绪')
"

echo "✨ 环境配置完成！可以开始训练了。"
echo ""
echo "快速训练命令："
echo "  python train_single_agent.py --algorithm TD3 --episodes 100"
echo "  python train_single_agent.py --algorithm SAC --episodes 100"

