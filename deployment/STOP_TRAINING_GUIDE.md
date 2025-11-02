# 中止训练指南

## 🎯 目标：在第5个实验完成后停止训练

---

## 方法1：手动监控并停止（最简单）

### 步骤1：监控进度

每隔1-2小时检查一次当前实验：

```powershell
# 在本地PowerShell运行
ssh -p 21960 root@connect.westc.gpuhub.com "ps aux | grep 'run_.*\.py' | grep -v grep"
```

### 步骤2：发现第5个实验开始时

当看到 `run_bandwidth_cost_comparison.py` (第5个) 开始运行时，记录时间，**等待2-3小时**。

### 步骤3：停止训练

```bash
# SSH连接服务器
ssh -p 21960 root@connect.westc.gpuhub.com

# 停止训练
pkill -f run_batch_experiments.py

# 取消自动关机
shutdown -c

# 确认已停止
ps aux | grep python
```

**优点**：简单直接  
**缺点**：需要手动监控

---

## 方法2：使用自动监控脚本（推荐）

### 步骤1：创建监控脚本

SSH连接服务器：
```bash
ssh -p 21960 root@connect.westc.gpuhub.com
cd /root/VEC_mig_caching
```

创建脚本（复制全部）：
```bash
cat > deployment/stop_after_5.sh << 'EOF'
#!/bin/bash
echo "监控中...将在第5个实验完成后自动停止"
echo "开始时间: $(date)"

while true; do
    # 检查结果文件数量（每个实验约产生5-8个文件）
    RESULT_COUNT=$(find /root/VEC_mig_caching/results/camtd3_strategy_suite -name "*.png" 2>/dev/null | wc -l)
    
    # 第5个实验完成大约有40个文件（5个实验 × 8个文件）
    if [ $RESULT_COUNT -ge 40 ]; then
        echo ""
        echo "检测到第5个实验已完成！"
        echo "停止训练..."
        pkill -f "run_batch_experiments.py"
        shutdown -c
        echo "✅ 训练已停止，自动关机已取消"
        exit 0
    fi
    
    echo "[$(date '+%H:%M:%S')] 结果文件: $RESULT_COUNT (目标: ≥40)"
    sleep 60  # 每分钟检查一次
done
EOF

chmod +x deployment/stop_after_5.sh
```

### 步骤2：在后台运行监控

```bash
# 在tmux新窗口运行监控
tmux new-session -d -s monitor "cd /root/VEC_mig_caching && bash deployment/stop_after_5.sh"

# 确认监控运行
tmux ls
```

### 步骤3：查看监控状态（可选）

```bash
# 进入监控会话查看
tmux attach -t monitor

# 退出: Ctrl+B 然后按 D
```

**优点**：全自动，无需人工监控  
**缺点**：需要创建脚本

---

## 方法3：进入tmux手动停止（最精确）

### 步骤1：进入训练会话

```bash
ssh -p 21960 root@connect.westc.gpuhub.com
tmux attach -t vec_training
```

### 步骤2：观察输出

等待看到类似输出：
```
========================================
实验 5/10: 带宽对成本影响
========================================
...
✅ 实验完成！
```

### 步骤3：立即停止

看到第5个完成后，**立即按 `Ctrl+C`** 停止训练。

### 步骤4：取消关机

```bash
shutdown -c
```

**优点**：最精确，能在第5个完成瞬间停止  
**缺点**：需要持续监控屏幕

---

## 方法4：定时停止（不太精确）

如果您知道大概时间（比如10小时后第5个应该完成）：

### 在服务器上设置定时停止

```bash
ssh -p 21960 root@connect.westc.gpuhub.com

# 设置10小时后停止训练
echo "pkill -f run_batch_experiments.py && shutdown -c" | at now + 10 hours

# 或使用sleep
nohup bash -c "sleep 36000 && pkill -f run_batch_experiments.py && shutdown -c" &
```

**优点**：设置后不用管  
**缺点**：时间不好估计

---

## 📊 实验时间估算

| 实验序号 | 实验名称 | 累计时间 |
|---------|---------|---------|
| 1-3 | 前3个（已完成） | 0小时 |
| 4 | 本地资源对成本影响（进行中） | +2-3小时 |
| 5 | 带宽对成本影响 | +2-3小时 |
| **总计** | **到第5个完成** | **约4-6小时** |

**建议**：
- 如果现在是中午12点，预计下午4-6点完成第5个
- 可以设置下午5点左右手动检查

---

## 🎯 推荐方案

根据您的情况：

### 如果您在电脑旁边
→ **方法3**（进入tmux手动停止）- 最精确

### 如果您不方便一直盯着
→ **方法2**（自动监控脚本）- 最省心

### 如果您只是偶尔检查
→ **方法1**（手动监控）- 最简单

---

## 🆘 停止后的操作

### 1. 确认训练已停止
```bash
ps aux | grep python
# 应该看不到 run_batch 相关进程
```

### 2. 取消自动关机
```bash
shutdown -c
```

### 3. 下载结果（在本地运行）
```powershell
cd D:\VEC_mig_caching
scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/results/camtd3_strategy_suite ./results_partial
```

### 4. 关闭服务器（可选）
```bash
# 如果不再需要服务器，立即关机节省费用
shutdown -h now
```

---

## 💡 注意事项

1. ✅ **确保第5个完成**：不要在第5个进行中停止
2. ✅ **取消关机**：停止后记得 `shutdown -c`
3. ✅ **下载结果**：停机前先下载结果
4. ⚠️ **谨慎操作**：错误停止可能丢失当前实验数据

---

## 📞 快速命令参考

```bash
# 查看当前实验
ps aux | grep run_ | grep -v grep

# 停止训练
pkill -f run_batch_experiments.py

# 取消关机
shutdown -c

# 查看已完成实验数量
find /root/VEC_mig_caching/results/camtd3_strategy_suite -name "*.png" | wc -l

# 立即关机
shutdown -h now
```

---

**需要我帮您设置哪种方案？** 😊

