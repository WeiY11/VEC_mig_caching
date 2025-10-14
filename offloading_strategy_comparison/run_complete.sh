#!/bin/bash
# 一键运行完整对比实验（Linux/Mac版）

echo "========================================"
echo "一键运行完整对比实验"
echo "========================================"
echo ""
echo "本脚本将自动完成以下任务："
echo "1. 检查TD3模型（如不存在则自动训练）"
echo "2. 运行6种策略对比实验"
echo "3. 生成学术论文图表"
echo ""
echo "策略列表："
echo "- LocalOnly（纯本地）"
echo "- RSUOnly（纯RSU）"
echo "- LoadBalance（负载均衡）"
echo "- Random（随机）"
echo "- TD3（深度强化学习）"
echo "- TD3-NoMig（无迁移TD3）"
echo ""
echo "========================================"
echo ""

# 询问用户选择
echo "请选择运行模式："
echo "1. 快速测试（约5-10分钟）"
echo "2. 标准实验（约30-60分钟）"
echo "3. 完整实验（约1-2小时）"
echo ""
read -p "请输入选项 (1/2/3): " mode

case $mode in
    1)
        echo ""
        echo "运行快速测试模式..."
        python run_complete_comparison.py --quick --train-episodes 50 --eval-episodes 20
        ;;
    2)
        echo ""
        echo "运行标准实验模式..."
        python run_complete_comparison.py --train-episodes 200 --eval-episodes 50
        ;;
    3)
        echo ""
        echo "运行完整实验模式..."
        python run_complete_comparison.py --train-episodes 500 --eval-episodes 100
        ;;
    *)
        echo ""
        echo "无效选项，使用默认标准模式..."
        python run_complete_comparison.py --train-episodes 200 --eval-episodes 50
        ;;
esac

echo ""
echo "========================================"
echo "实验完成！"
echo ""
echo "请查看以下文件："
echo "- 结果数据: results/offloading_comparison/"
echo "- 对比图表: academic_figures/vehicle_comparison/"
echo "========================================"
echo ""
