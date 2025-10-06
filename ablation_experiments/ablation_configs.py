#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3消融实验配置
定义7种不同的消融配置，用于验证各模块有效性

【配置设计】
1. Full-System: 完整系统（基准）
2. No-Cache: 禁用边缘缓存
3. No-Migration: 禁用任务迁移
4. No-Priority: 禁用任务优先级
5. No-Adaptive: 禁用自适应控制
6. No-Collaboration: 禁用RSU协作
7. Minimal-System: 最小系统
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AblationConfig:
    """
    消融实验配置
    
    【参数说明】
    - name: 配置名称
    - description: 配置描述
    - enable_cache: 是否启用缓存模块
    - enable_migration: 是否启用迁移模块
    - enable_priority: 是否启用优先级队列
    - enable_adaptive: 是否启用自适应控制
    - enable_collaboration: 是否启用RSU协作
    """
    name: str
    description: str
    enable_cache: bool = True
    enable_migration: bool = True
    enable_priority: bool = True
    enable_adaptive: bool = True
    enable_collaboration: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'enable_cache': self.enable_cache,
            'enable_migration': self.enable_migration,
            'enable_priority': self.enable_priority,
            'enable_adaptive': self.enable_adaptive,
            'enable_collaboration': self.enable_collaboration
        }
    
    def apply_to_system(self):
        """
        应用配置到系统
        修改全局配置以禁用相应模块
        """
        from config import config
        
        # 创建ablation配置节点（如果不存在）
        if not hasattr(config, 'ablation'):
            class AblationSettings:
                pass
            config.ablation = AblationSettings()
        
        # 应用配置
        config.ablation.enable_cache = self.enable_cache
        config.ablation.enable_migration = self.enable_migration
        config.ablation.enable_priority = self.enable_priority
        config.ablation.enable_adaptive = self.enable_adaptive
        config.ablation.enable_collaboration = self.enable_collaboration
        
        print(f"\n{'='*60}")
        print(f"📋 应用消融配置: {self.name}")
        print(f"{'='*60}")
        print(f"  描述: {self.description}")
        print(f"  缓存模块:   {'✓ 启用' if self.enable_cache else '✗ 禁用'}")
        print(f"  迁移模块:   {'✓ 启用' if self.enable_migration else '✗ 禁用'}")
        print(f"  优先级队列: {'✓ 启用' if self.enable_priority else '✗ 禁用'}")
        print(f"  自适应控制: {'✓ 启用' if self.enable_adaptive else '✗ 禁用'}")
        print(f"  RSU协作:    {'✓ 启用' if self.enable_collaboration else '✗ 禁用'}")
        print(f"{'='*60}\n")


def get_all_ablation_configs() -> List[AblationConfig]:
    """
    获取所有消融实验配置
    
    【返回】7种配置，涵盖所有消融场景
    """
    configs = []
    
    # ========== 1. 完整系统（基准） ==========
    configs.append(AblationConfig(
        name="Full-System",
        description="完整系统（所有模块启用）- 基准配置",
        enable_cache=True,
        enable_migration=True,
        enable_priority=True,
        enable_adaptive=True,
        enable_collaboration=True
    ))
    
    # ========== 2. 无缓存 ==========
    configs.append(AblationConfig(
        name="No-Cache",
        description="禁用边缘缓存模块",
        enable_cache=False,
        enable_migration=True,
        enable_priority=True,
        enable_adaptive=True,
        enable_collaboration=True
    ))
    
    # ========== 3. 无迁移 ==========
    configs.append(AblationConfig(
        name="No-Migration",
        description="禁用任务迁移模块",
        enable_cache=True,
        enable_migration=False,
        enable_priority=True,
        enable_adaptive=True,
        enable_collaboration=True
    ))
    
    # ========== 4. 无优先级 ==========
    configs.append(AblationConfig(
        name="No-Priority",
        description="禁用任务优先级队列",
        enable_cache=True,
        enable_migration=True,
        enable_priority=False,
        enable_adaptive=True,
        enable_collaboration=True
    ))
    
    # ========== 5. 无自适应控制 ==========
    configs.append(AblationConfig(
        name="No-Adaptive",
        description="禁用自适应缓存和迁移控制",
        enable_cache=True,
        enable_migration=True,
        enable_priority=True,
        enable_adaptive=False,
        enable_collaboration=True
    ))
    
    # ========== 6. 无协作 ==========
    configs.append(AblationConfig(
        name="No-Collaboration",
        description="禁用RSU间协作缓存",
        enable_cache=True,
        enable_migration=True,
        enable_priority=True,
        enable_adaptive=True,
        enable_collaboration=False
    ))
    
    # ========== 7. 最小系统 ==========
    configs.append(AblationConfig(
        name="Minimal-System",
        description="最小系统（仅基础功能）",
        enable_cache=False,
        enable_migration=False,
        enable_priority=False,
        enable_adaptive=False,
        enable_collaboration=False
    ))
    
    return configs


def get_config_by_name(name: str) -> AblationConfig:
    """
    根据名称获取配置
    
    【参数】
    - name: 配置名称
    
    【返回】对应的配置对象，若不存在则返回Full-System
    """
    configs = get_all_ablation_configs()
    for config in configs:
        if config.name == name:
            return config
    
    print(f"⚠️ 配置 '{name}' 不存在，返回Full-System配置")
    return configs[0]  # 返回Full-System


if __name__ == "__main__":
    # 测试：打印所有配置
    print("🔬 TD3消融实验配置列表")
    print("="*80)
    
    configs = get_all_ablation_configs()
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config.name}")
        print(f"   {config.description}")
        print(f"   Cache={config.enable_cache}, "
              f"Migration={config.enable_migration}, "
              f"Priority={config.enable_priority}")
        print(f"   Adaptive={config.enable_adaptive}, "
              f"Collaboration={config.enable_collaboration}")
    
    print("\n" + "="*80)
    print(f"✓ 共 {len(configs)} 种消融配置")

