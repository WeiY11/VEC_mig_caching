#!/bin/bash
# 批量训练不同车辆数的脚本

ALGORITHM="TD3"
EPISODES=1200
VEHICLES=(8 10 12 14 16)

echo "========================================"
echo "批量训练任务启动"
echo "========================================"
echo "算法: $ALGORITHM"
echo "训练轮次: $EPISODES"
echo "车辆数: ${VEHICLES[@]}"
echo "========================================"
echo ""

for NUM_VEHICLES in "${VEHICLES[@]}"
do
    echo ""
    echo "========================================"
    echo "开始训练: ${NUM_VEHICLES}辆车"
    echo "========================================"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    python train_single_agent.py \
        --algorithm $ALGORITHM \
        --episodes $EPISODES \
        --num-vehicles $NUM_VEHICLES
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ ${NUM_VEHICLES}辆车训练完成"
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo ""
        echo "❌ ${NUM_VEHICLES}辆车训练失败"
        echo "继续下一个配置..."
    fi
    echo ""
done

echo ""
echo "========================================"
echo "✅ 所有训练任务完成"
echo "========================================"
