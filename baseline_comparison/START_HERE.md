# 🚀 Baseline对比实验 - 快速开始

## ⚡ 3步开始

### 步骤1️⃣: 测试环境
```bash
cd baseline_comparison
python test_baseline_env.py
```

### 步骤2️⃣: 快速对比（验证流程，1小时）
```bash
python run_baseline_comparison.py --episodes 50 --quick
```

### 步骤3️⃣: 标准对比（论文数据，4-5小时）
```bash
python run_baseline_comparison.py --episodes 200
```

---

## 📊 对比算法

### DRL算法（真实训练）
- **TD3** - 我们的方法 ⭐
- **DDPG** - 对比方法1
- **SAC** - 对比方法2
- **PPO** - 对比方法3
- **DQN** - 对比方法4

### 启发式算法（策略执行）
- **Random** - 随机选择
- **Greedy** - 贪心最小负载
- **RoundRobin** - 轮询分配
- **LocalFirst** - 本地优先
- **NearestNode** - 最近节点

---

## 🎯 实验目标

证明 **TD3 > 其他DRL > 启发式算法**

---

## 📁 结果位置

- **训练数据**: `results/` 文件夹
- **对比图表**: `analysis/` 文件夹

---

## 📝 注意

**这是真实训练，不是模拟数据！**
- DRL算法需要完整训练（200轮约4-5小时）
- 启发式算法直接执行策略（200轮约10-15分钟）

---

更多信息请查看 `README.md`


