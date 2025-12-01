"""
🔍 OPTIMIZED_TD3训练问题诊断脚本

用于快速测试和诊断训练问题:
1. 验证奖励目标调整后的效果
2. 监控卸载决策分布
3. 测试强制远程卸载模式

使用方法:
python diagnose_training.py
"""

import subprocess
import sys
from pathlib import Path

def run_short_test(mode="normal", episodes=50):
    """运行短期测试训练"""
    
    cmd = [
        sys.executable,
        "train_single_agent.py",
        "--algorithm", "OPTIMIZED_TD3",
        "--episodes", str(episodes),
        "--num-vehicles", "12",
        "--seed", "42"
    ]
    
    print(f"\n{'='*70}")
    print(f"🧪 测试模式: {mode}")
    print(f"训练轮次: {episodes}")
    print(f"{'='*70}\n")
    
    # 根据模式设置环境变量
    env = None
    if mode == "remote_only":
        print("⚙️  强制模式: 仅使用RSU/UAV (Remote-Only)")
        cmd.extend(["--enforce-offload-mode", "remote_only"])
    elif mode == "local_only":
        print("⚙️  强制模式: 仅使用本地计算 (Local-Only)")
        cmd.extend(["--enforce-offload-mode", "local_only"])
    else:
        print("⚙️  正常模式: 智能体自主学习卸载决策")
    
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0

def main():
    """主诊断流程"""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║        OPTIMIZED_TD3 训练诊断工具 v1.0                        ║
║                                                                ║
║  本工具将执行以下诊断测试:                                     ║
║  1. 正常模式训练 (50 episodes)                                ║
║  2. 强制远程卸载模式 (50 episodes)                            ║
║  3. 强制本地计算模式 (50 episodes)                            ║
║                                                                ║
║  预计耗时: ~15-20分钟                                          ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    input("按Enter键开始诊断...")
    
    # 测试1: 正常模式
    print("\n" + "="*70)
    print("📋 测试 1/3: 正常模式训练")
    print("="*70)
    success_normal = run_short_test("normal", episodes=50)
    
    # 测试2: 强制远程卸载
    print("\n" + "="*70)
    print("📋 测试 2/3: 强制远程卸载模式")
    print("="*70)
    success_remote = run_short_test("remote_only", episodes=50)
    
    # 测试3: 强制本地计算
    print("\n" + "="*70)
    print("📋 测试 3/3: 强制本地计算模式")
    print("="*70)
    success_local = run_short_test("local_only", episodes=50)
    
    # 总结
    print("\n" + "="*70)
    print("📊 诊断测试总结")
    print("="*70)
    print(f"正常模式:       {'✅ 成功' if success_normal else '❌ 失败'}")
    print(f"强制远程模式:   {'✅ 成功' if success_remote else '❌ 失败'}")
    print(f"强制本地模式:   {'✅ 成功' if success_local else '❌ 失败'}")
    
    print("\n💡 下一步建议:")
    print("1. 检查 results/single_agent/optimized_td3/ 目录下的训练报告")
    print("2. 对比三种模式下的:")
    print("   - 平均奖励值")
    print("   - RSU/UAV利用率")
    print("   - 平均时延和能耗")
    print("3. 查看训练日志中的 '🔍 [Step X] 卸载偏好' 输出")
    
    results_dir = Path("results/single_agent/optimized_td3")
    if results_dir.exists():
        print(f"\n📁 结果目录: {results_dir.absolute()}")
        html_files = list(results_dir.glob("training_report_*.html"))
        if html_files:
            latest_report = max(html_files, key=lambda p: p.stat().st_mtime)
            print(f"📄 最新报告: {latest_report.name}")

if __name__ == "__main__":
    main()
