#!/usr/bin/env python3
"""
数据处理工具
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
import json
from pathlib import Path

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.normalization_params = {}
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """数据归一化"""
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val == 0:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return np.zeros_like(data)
            return (data - mean_val) / std_val
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denormalize_data(self, normalized_data: np.ndarray, 
                        original_min: float, original_max: float) -> np.ndarray:
        """反归一化"""
        return normalized_data * (original_max - original_min) + original_min
    
    def smooth_data(self, data: List[float], window_size: int = 10) -> List[float]:
        """数据平滑"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """移除异常值"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # 使用Z-score方法
        z_scores = np.abs((data - mean_val) / std_val)
        return data[z_scores < threshold]
    
    def interpolate_missing(self, data: List[Optional[float]]) -> List[float]:
        """插值填补缺失值"""
        result = []
        last_valid = None
        
        for i, value in enumerate(data):
            if value is not None:
                result.append(value)
                last_valid = value
            else:
                # 寻找下一个有效值
                next_valid = None
                for j in range(i + 1, len(data)):
                    if data[j] is not None:
                        next_valid = data[j]
                        break
                
                # 插值
                if last_valid is not None and next_valid is not None:
                    # 线性插值
                    interpolated = last_valid + (next_valid - last_valid) * 0.5
                elif last_valid is not None:
                    interpolated = last_valid
                elif next_valid is not None:
                    interpolated = next_valid
                else:
                    interpolated = 0.0
                
                result.append(interpolated)
        
        return result
    
    def batch_data(self, data: List[Any], batch_size: int) -> List[List[Any]]:
        """数据分批"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def shuffle_data(self, *arrays) -> Tuple:
        """随机打乱数据"""
        if not arrays:
            return ()
        
        indices = np.random.permutation(len(arrays[0]))
        return tuple(array[indices] for array in arrays)
    
    def split_data(self, data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """数据分割"""
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
    def convert_to_tensor(self, data: Any, device: str = 'cpu') -> torch.Tensor:
        """转换为PyTorch张量"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        elif isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def save_processed_data(self, data: Dict[str, Any], filepath: str):
        """保存处理后的数据"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换numpy数组为列表以便JSON序列化
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                json_data[key] = value.cpu().numpy().tolist()
            else:
                json_data[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def load_processed_data(self, filepath: str) -> Dict[str, Any]:
        """加载处理后的数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换回numpy数组
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = np.array(value)
        
        return data

def test_data_processor():
    """测试数据处理器"""
    print("🧪 测试数据处理器...")
    
    processor = DataProcessor()
    
    # 测试数据
    test_data = np.random.randn(100) * 10 + 50
    
    # 归一化测试
    normalized = processor.normalize_data(test_data, 'minmax')
    print(f"原始数据范围: [{np.min(test_data):.2f}, {np.max(test_data):.2f}]")
    print(f"归一化后范围: [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
    
    # 平滑测试
    noisy_data = [i + np.random.randn() for i in range(50)]
    smoothed = processor.smooth_data(noisy_data, window_size=5)
    print(f"平滑前方差: {np.var(noisy_data):.3f}")
    print(f"平滑后方差: {np.var(smoothed):.3f}")
    
    # 异常值移除测试
    data_with_outliers = np.concatenate([
        np.random.randn(90),
        np.array([10, -10, 15, -15])  # 异常值
    ])
    cleaned_data = processor.remove_outliers(data_with_outliers)
    print(f"移除异常值前: {len(data_with_outliers)} 个数据点")
    print(f"移除异常值后: {len(cleaned_data)} 个数据点")
    
    print("✅ 数据处理器测试完成")

if __name__ == "__main__":
    test_data_processor()