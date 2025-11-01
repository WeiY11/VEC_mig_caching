#!/bin/bash
# GPU优化配置脚本
# 用于提高强化学习训练的GPU利用率

echo "=========================================="
echo "VEC GPU训练优化配置"
echo "=========================================="
echo ""

# 1. 增加batch size
echo "优化1: 增加batch size（从64->256）"
sed -i 's/batch_size = 64/batch_size = 256/g' config/system_config.py

# 2. 增加更新频率
echo "优化2: 增加更新频率"
sed -i 's/policy_delay = 2/policy_delay = 1/g' config/system_config.py

# 3. 设置GPU优化环境变量
echo "优化3: 设置GPU环境变量"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo ""
echo "=========================================="
echo "优化完成！预期GPU利用率: 15-30%"
echo "=========================================="
echo ""
echo "说明:"
echo "  - 强化学习训练GPU利用率5-15%是正常的"
echo "  - 大部分时间在CPU上模拟环境"
echo "  - 只有神经网络更新时使用GPU"
echo ""

