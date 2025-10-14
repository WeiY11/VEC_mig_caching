# 实验结果分析报告
生成时间: 2025-10-14T04:10:49

## 1. 数据概览

### 参数扫描实验
- **Bandwidth**: 参数 bandwidth_mhz, 取值范围 [10, 20, 30, 40, 50], 策略数 8
- **Cpu Frequency**: 参数 cpu_frequency_ghz, 取值范围 [1.5, 2.0, 2.5, 3.0, 3.5], 策略数 8
- **Data Size**: 参数 data_size_mb, 取值范围 [0.5, 1.0, 1.5, 2.0, 2.5], 策略数 8
- **Vehicle**: 参数 num_vehicles, 取值范围 [8, 12, 16, 20, 24], 策略数 8
- **Task Rate**: 参数 task_arrival_rate, 取值范围 [0.3, 0.5, 0.7, 0.9, 1.1], 策略数 8

### TD3训练结果
- 训练轮数: 1600
- 车辆数: 12
- 初始奖励: -281.85
- 最终奖励: -119.21
- 提升幅度: 57.7%

## 2. 关键发现

### 车辆数量影响分析
- **LocalOnly**: 成本上升 2.0%
- **RSUOnly**: 成本下降 0.0%
- **UAVOnly**: 成本下降 0.0%
- **Random**: 成本下降 0.0%
- **RoundRobin**: 成本下降 0.0%
- **NearestNode**: 成本下降 0.0%
- **LoadBalance**: 成本下降 0.0%
- **MinDelay**: 成本下降 0.0%

## 3. 策略对比

### 在16辆车时的性能对比

| 策略 | 加权成本 | 延迟(s) | 能耗(J) | 完成率(%) |
|------|----------|---------|---------|----------|
| LocalOnly | 24.63 | 0.446 | 1602.3 | 87.1 |
| RSUOnly | 25.00 | 0.361 | 3442.2 | 81.2 |
| UAVOnly | 25.00 | 0.271 | 3928.4 | 74.0 |
| Random | 25.00 | 0.254 | 4903.2 | 70.1 |
| RoundRobin | 25.00 | 0.265 | 6374.7 | 68.0 |
| NearestNode | 25.00 | 0.257 | 7398.3 | 67.2 |
| LoadBalance | 25.00 | 0.265 | 9281.4 | 68.6 |
| MinDelay | 25.00 | 0.295 | 12474.9 | 71.1 |

## 4. 图表生成

所有图表已保存到: `academic_figures\comparison_analysis`

生成的图表包括:
- vehicle_sweep_comparison.pdf/png - 车辆数量扫描对比
- task_rate_sweep_comparison.pdf/png - 任务到达率扫描对比
- bandwidth_sweep_comparison.pdf/png - 带宽扫描对比
- data_size_sweep_comparison.pdf/png - 数据大小扫描对比
- cpu_frequency_sweep_comparison.pdf/png - CPU频率扫描对比
- td3_convergence.pdf/png - TD3训练收敛曲线
