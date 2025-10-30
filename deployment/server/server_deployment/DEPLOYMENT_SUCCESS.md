# ✅ VEC批量实验部署成功！

## 🎯 部署摘要

**服务器**: region-9.autodl.pro:47042  
**项目目录**: /root/VEC_mig_caching  
**实验进程ID**: 1597  
**日志文件**: batch_experiments.log  
**部署时间**: 2025-10-30 01:05

---

## 📊 实验配置

- **模式**: Full (500轮/配置)
- **实验数量**: 8个参数对比
- **预计时间**: 2-5天
- **GPU**: Tesla T4 (15.6 GB)
- **PyTorch**: 2.0.0+cu118

### 8个参数对比实验

1. ✅ 数据大小对比 (5个配置)
2. ✅ 车辆数量对比 (5个配置)
3. ✅ 本地资源对卸载影响 (5个配置)
4. ✅ 本地资源对成本影响 (7个配置)
5. ✅ 带宽对成本影响 (7个配置)
6. ✅ 边缘节点配置对比 (6个配置)
7. ✅ 任务到达率对比 (6个配置)
8. ✅ 移动速度对比 (6个配置)

**总计**: 47个配置 × 500轮 = 23,500轮训练

---

## 📱 如何监控实验

### 方法1：使用监控脚本（推荐）

```bash
# 连接到服务器
ssh -p 47042 root@region-9.autodl.pro
# 密码: dfUJkmli0mHk

# 进入项目目录
cd /root/VEC_mig_caching

# 运行监控脚本
./remote_monitor.sh
```

### 方法2：查看实时日志

```bash
# 连接到服务器后
cd /root/VEC_mig_caching

# 实时查看日志
tail -f batch_experiments.log

# 或查看最后100行
tail -100 batch_experiments.log
```

### 方法3：检查进程状态

```bash
# 查看进程是否在运行
ps -p 1597

# 或查看进程详情
ps -p 1597 -f

# 查看所有Python进程
ps aux | grep python
```

### 方法4：查看GPU使用

```bash
# 实时GPU监控（每2秒刷新）
watch -n 2 nvidia-smi

# 或单次查看
nvidia-smi
```

### 方法5：检查结果目录

```bash
# 查看生成的结果
ls -lth results/parameter_sensitivity/

# 查看摘要文件
cat results/parameter_sensitivity/batch_*_summary.json
```

---

## 📥 下载结果

### 实验完成后（2-5天）

在**本地计算机**的PowerShell中运行：

```powershell
# 下载所有结果
scp -P 47042 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results/parameter_sensitivity D:\VEC_results

# 下载日志
scp -P 47042 root@region-9.autodl.pro:/root/VEC_mig_caching/batch_experiments.log D:\VEC_logs\
```

### 结果文件结构

```
results/parameter_sensitivity/
└── batch_full_20251030_*/
    ├── batch_full_20251030_*_batch_summary.json  # 总摘要
    ├── 1_数据大小对比/
    │   ├── summary.json
    │   ├── data_size_comparison.png
    │   └── ...
    ├── 2_车辆数量对比/
    ├── 3_本地资源对卸载影响/
    ├── 4_本地资源对成本影响/
    ├── 5_带宽对成本影响/
    ├── 6_边缘节点配置对比/
    ├── 7_任务到达率对比/
    └── 8_移动速度对比/
```

---

## 🛑 如何停止实验

### 优雅停止

```bash
# 连接到服务器
ssh -p 47042 root@region-9.autodl.pro

cd /root/VEC_mig_caching

# 停止实验
kill 1597

# 或使用PID文件
kill $(cat batch_experiments.pid)
```

### 强制停止

```bash
# 强制终止
kill -9 1597

# 或杀死所有相关进程
pkill -f run_batch_experiments
```

---

## 📋 快速命令备忘录

### 连接服务器

```bash
ssh -p 47042 root@region-9.autodl.pro
# 密码: dfUJkmli0mHk
```

### 常用命令

```bash
cd /root/VEC_mig_caching              # 进入项目目录
./remote_monitor.sh                   # 查看状态
tail -f batch_experiments.log         # 实时日志
nvidia-smi                           # GPU状态
ps -p 1597                           # 检查进程
```

### 下载结果（在本地运行）

```powershell
scp -P 47042 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results/parameter_sensitivity D:\VEC_results
```

---

## ⏱️ 预计时间表

| 阶段 | 时间 | 说明 |
|------|------|------|
| 初始化 | 5-10分钟 | 加载环境、配置实验 |
| 实验1-2 | 12-16小时 | 数据大小、车辆数量 (10个配置) |
| 实验3-5 | 18-24小时 | 本地资源、带宽 (19个配置) |
| 实验6-8 | 16-20小时 | 边缘节点、任务到达、移动速度 (18个配置) |
| **总计** | **2-3天** | 取决于GPU性能和负载 |

---

## 🔔 监控建议

1. ✅ **首次检查**（启动后10-30分钟）
   - 确认日志有输出
   - 检查GPU开始工作
   - 验证没有错误

2. ✅ **定期检查**（每天1-2次）
   - 查看训练进度
   - 检查GPU温度（应<85°C）
   - 确认进程仍在运行

3. ✅ **实验完成前**（第2-3天）
   - 检查结果目录
   - 确认所有实验完成
   - 准备下载结果

---

## 🔧 故障排除

### 问题1：进程停止了

```bash
# 查看日志末尾找错误
tail -50 batch_experiments.log

# 重新启动
./remote_start.sh
```

### 问题2：GPU未使用

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 查看GPU进程
nvidia-smi
```

### 问题3：磁盘空间不足

```bash
# 查看磁盘使用
df -h

# 清理缓存
rm -rf __pycache__
rm -rf .cache
```

### 问题4：内存不足

```bash
# 查看内存
free -h

# 如果内存不足，可以减少batch size或降低模式
# 修改为medium模式（100轮）
```

---

## 📞 紧急联系

如遇到服务器问题：
1. 检查AutoDL平台实例状态
2. 查看日志文件获取错误信息
3. 保存重要的中间结果
4. 必要时重新启动实验

---

## 🎉 成功标志

实验成功完成的标志：

1. ✅ 日志文件中显示所有8个实验完成
2. ✅ `results/parameter_sensitivity/` 有完整的8个子目录
3. ✅ 每个子目录都有 `summary.json` 和对应的PNG图表
4. ✅ 总摘要文件 `batch_*_summary.json` 存在
5. ✅ 日志末尾显示成功消息

---

**部署完成！实验正在服务器后台运行中...**

**可以断开SSH连接，实验会继续运行。建议每天登录检查一次进度。**

🚀 **祝实验顺利！**

