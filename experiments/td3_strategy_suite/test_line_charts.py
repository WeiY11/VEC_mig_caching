#!/usr/bin/env python3
"""
快速测试离散折线图功能
======================

运行一个最简单的实验，验证离散折线图是否正常生成
"""

import subprocess
import sys
from pathlib import Path

def test_experiment(script_name: str, episodes: int = 10) -> bool:
    """测试单个实验脚本"""
    
    print(f"\n{'='*70}")
    print(f"测试: {script_name}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        f"experiments/td3_strategy_suite/{script_name}",
        "--episodes", str(episodes),
        "--seed", "42",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
        )
        
        if result.returncode == 0:
            print("✅ 实验运行成功")
            
            # 检查输出中是否包含新增图表
            if "_delay_line.png" in result.stdout:
                print("✅ 检测到时延折线图")
            if "_energy_line.png" in result.stdout:
                print("✅ 检测到能耗折线图")
            if "_cost_line.png" in result.stdout:
                print("✅ 检测到成本折线图")
            if "_completion_line.png" in result.stdout:
                print("✅ 检测到完成率折线图")
            if "_multiline.png" in result.stdout:
                print("✅ 检测到多指标综合图")
            
            return True
        else:
            print(f"❌ 实验运行失败")
            print(f"错误输出:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 实验超时")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    
    print("="*70)
    print("离散折线图功能测试")
    print("="*70)
    print("说明: 运行一个快速实验（10轮），验证新功能是否正常工作")
    print("预计耗时: 3-5分钟")
    print("="*70)
    
    # 选择一个简单的实验进行测试
    test_script = "run_vehicle_count_comparison.py"
    
    print(f"\n选择测试脚本: {test_script}")
    print("按 Enter 继续，或 Ctrl+C 取消...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n测试已取消")
        return
    
    success = test_experiment(test_script, episodes=10)
    
    print(f"\n{'='*70}")
    if success:
        print("✅ 测试通过！离散折线图功能正常工作")
        print("\n下一步:")
        print("1. 查看生成的图表文件")
        print("2. 运行其他实验脚本验证")
        print("3. 执行完整的500轮实验")
    else:
        print("❌ 测试失败！请检查错误信息")
    print("="*70)


if __name__ == "__main__":
    main()

