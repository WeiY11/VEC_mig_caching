#!/usr/bin/env python3
"""
🎯 快速测试Enhanced TD3恢复情况
验证train_single_agent.py集成是否成功
"""

import subprocess
import sys

def test_enhanced_td3():
    """测试Enhanced TD3是否可以正常运行"""
    print("🧪 测试Enhanced TD3集成...")
    print("=" * 60)
    
    # 运行短期测试（10个episode）
    cmd = [
        sys.executable,
        "train_single_agent.py",
        "--algorithm", "ENHANCED_TD3",
        "--episodes", "10",
        "--num-vehicles", "8",
        "--seed", "42"
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Enhanced TD3集成成功！")
        print("   可以开始运行消融实验了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Enhanced TD3运行失败")
        print(f"   错误码: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False


if __name__ == '__main__':
    success = test_enhanced_td3()
    sys.exit(0 if success else 1)
