"""
基于LSTM的内容流行度预测器
预测未来哪些内容会变得热门，实现主动缓存
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import pickle
import os

class LSTMPopularityPredictor:
    """
    LSTM内容流行度预测器
    基于历史访问模式预测内容未来的流行度
    """
    
    def __init__(self, 
                 content_dim: int = 64,      # 减小维度，加快训练
                 hidden_dim: int = 32,       # 更小的隐藏层，防止过拟合
                 sequence_length: int = 10,  # 较短序列，快速响应
                 prediction_horizon: int = 3, # 预测较近的未来
                 learning_rate: float = 0.01):
        """
        初始化LSTM预测器
        
        Args:
            content_dim: 内容特征维度
            hidden_dim: LSTM隐藏层维度
            sequence_length: 输入序列长度
            prediction_horizon: 预测未来多少个时间步
            learning_rate: 学习率
        """
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 构建模型
        self.model = PopularityLSTM(
            input_dim=content_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 历史数据缓存
        self.access_history = deque(maxlen=1000)
        self.content_features = {}  # 内容ID到特征向量的映射
        self.popularity_scores = defaultdict(list)  # 内容流行度历史
        
        # 训练相关
        self.training_data = []
        self.is_trained = False
        self.min_training_samples = 100
        
    def extract_features(self, content_id: str, metadata: Dict = None) -> np.ndarray:
        """
        提取内容特征向量
        
        Args:
            content_id: 内容标识
            metadata: 内容元数据（大小、类型、创建时间等）
            
        Returns:
            特征向量
        """
        if content_id in self.content_features:
            return self.content_features[content_id]
        
        # 简单的特征提取（实际应用中可以更复杂）
        features = np.zeros(self.content_dim)
        
        # 基于content_id的哈希特征
        hash_val = hash(content_id)
        for i in range(min(10, self.content_dim)):
            features[i] = ((hash_val >> (i * 4)) & 0xF) / 15.0
        
        # 如果有元数据，提取额外特征
        if metadata:
            # 文件大小特征
            if 'size' in metadata:
                size_feature = np.log(metadata['size'] + 1) / 20.0
                features[10] = min(1.0, size_feature)
            
            # 文件类型特征
            if 'type' in metadata:
                type_hash = hash(metadata['type'])
                features[11] = (type_hash & 0xFF) / 255.0
            
            # 访问时间特征（一天中的时间）
            if 'timestamp' in metadata:
                hour = (metadata['timestamp'] // 3600) % 24
                features[12] = hour / 23.0
                
                # 星期几
                day = (metadata['timestamp'] // 86400) % 7
                features[13] = day / 6.0
        
        # 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        self.content_features[content_id] = features
        return features
    
    def record_access(self, content_id: str, timestamp: float, metadata: Dict = None):
        """
        记录内容访问
        
        Args:
            content_id: 内容标识
            timestamp: 访问时间戳
            metadata: 内容元数据
        """
        # 提取特征
        features = self.extract_features(content_id, metadata)
        
        # 记录访问
        self.access_history.append({
            'content_id': content_id,
            'timestamp': timestamp,
            'features': features
        })
        
        # 更新流行度分数（简单计数）
        time_window = 3600  # 1小时窗口
        current_window = int(timestamp // time_window)
        self.popularity_scores[content_id].append((current_window, 1))
        
        # 准备训练数据
        if len(self.access_history) >= self.sequence_length + self.prediction_horizon:
            self._prepare_training_sample()
    
    def _prepare_training_sample(self):
        """准备训练样本"""
        # 从历史中提取序列
        sequence = []
        for i in range(self.sequence_length):
            idx = -(self.sequence_length + self.prediction_horizon) + i
            access = self.access_history[idx]
            sequence.append(access['features'])
        
        # 目标值：未来的访问频率
        future_access_count = 0
        for i in range(self.prediction_horizon):
            idx = -self.prediction_horizon + i
            if idx < len(self.access_history):
                future_access_count += 1
        
        # 归一化目标值
        target = future_access_count / self.prediction_horizon
        
        self.training_data.append((np.array(sequence), target))
        
        # 限制训练数据大小
        if len(self.training_data) > 500:
            self.training_data.pop(0)
    
    def train(self, epochs: int = 10, batch_size: int = 32):
        """
        训练LSTM模型
        
        Args:
            epochs: 训练轮次
            batch_size: 批次大小
        """
        if len(self.training_data) < self.min_training_samples:
            return
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 随机打乱数据
            np.random.shuffle(self.training_data)
            
            # 批次训练
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i+batch_size]
                
                # 准备批次数据
                sequences = torch.FloatTensor([s for s, _ in batch]).to(self.device)
                targets = torch.FloatTensor([t for _, t in batch]).to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(sequences).squeeze()
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
        self.is_trained = True
    
    def predict(self, content_ids: List[str], current_time: float) -> Dict[str, float]:
        """
        预测内容流行度
        
        Args:
            content_ids: 要预测的内容ID列表
            current_time: 当前时间戳
            
        Returns:
            内容ID到预测流行度的映射
        """
        if not self.is_trained and len(self.training_data) >= self.min_training_samples:
            self.train()
        
        predictions = {}
        
        self.model.eval()
        with torch.no_grad():
            for content_id in content_ids:
                # 获取内容特征
                features = self.extract_features(content_id, {'timestamp': current_time})
                
                # 构造输入序列（使用最近的访问历史）
                sequence = []
                for access in list(self.access_history)[-self.sequence_length:]:
                    if access['content_id'] == content_id:
                        sequence.append(access['features'])
                    else:
                        sequence.append(features * 0.1)  # 弱化其他内容的影响
                
                # 如果序列不够长，用零填充
                while len(sequence) < self.sequence_length:
                    sequence.insert(0, np.zeros(self.content_dim))
                
                # 预测
                input_tensor = torch.FloatTensor([sequence]).to(self.device)
                prediction = self.model(input_tensor).item()
                
                # 结合历史流行度
                historical_score = self._calculate_historical_popularity(content_id)
                
                # 加权组合
                final_score = 0.7 * prediction + 0.3 * historical_score
                predictions[content_id] = max(0.0, min(1.0, final_score))
        
        return predictions
    
    def _calculate_historical_popularity(self, content_id: str) -> float:
        """计算历史流行度"""
        if content_id not in self.popularity_scores:
            return 0.0
        
        scores = self.popularity_scores[content_id]
        if not scores:
            return 0.0
        
        # 时间衰减的流行度
        current_window = max(s[0] for s in scores) if scores else 0
        weighted_sum = 0
        weight_total = 0
        
        for window, count in scores:
            age = max(0, current_window - window)
            weight = np.exp(-age * 0.1)  # 指数衰减
            weighted_sum += count * weight
            weight_total += weight
        
        if weight_total > 0:
            return min(1.0, weighted_sum / weight_total / 10)  # 归一化
        return 0.0
    
    def get_top_predictions(self, n: int = 10, current_time: float = None) -> List[Tuple[str, float]]:
        """
        获取预测流行度最高的内容
        
        Args:
            n: 返回前n个
            current_time: 当前时间戳
            
        Returns:
            (内容ID, 预测流行度)列表
        """
        if current_time is None:
            current_time = self.access_history[-1]['timestamp'] if self.access_history else 0
        
        # 获取所有内容的预测
        all_content_ids = list(self.content_features.keys())
        predictions = self.predict(all_content_ids, current_time)
        
        # 排序并返回前n个
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:n]
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'content_features': self.content_features,
            'popularity_scores': dict(self.popularity_scores),
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.content_features = checkpoint['content_features']
            self.popularity_scores = defaultdict(list, checkpoint['popularity_scores'])
            self.is_trained = checkpoint['is_trained']


class PopularityLSTM(nn.Module):
    """LSTM模型架构"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super(PopularityLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # 激活和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # 输出0-1之间的流行度分数
        
        return out
