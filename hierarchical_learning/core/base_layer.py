from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """分层学习架构中所有层的抽象基类。"""

    def __init__(self, config):
        """初始化基础层。"""
        self.config = config

    @abstractmethod
    def process_state(self, state):
        """处理来自环境或上层的状态信息。"""
        pass

    @abstractmethod
    def get_action(self, processed_state):
        """根据处理后的状态生成动作。"""
        pass

    @abstractmethod
    def train(self, replay_buffer):
        """使用经验回放缓冲区中的数据训练模型。"""
        pass

    @abstractmethod
    def save_model(self, path):
        """将模型权重保存到指定路径。"""
        pass

    @abstractmethod
    def load_model(self, path):
        """从指定路径加载模型权重。"""
        pass