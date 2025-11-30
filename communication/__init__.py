"""
通信模块
实现车联网通信协议和网络管理
"""

try:
    from .v2v_communication import V2VCommunication
    from .v2i_communication import V2ICommunication
    from .network_manager import NetworkManager
    __all__ = ['V2VCommunication', 'V2ICommunication', 'NetworkManager']
except ImportError:
    # 兼容模式：仅导入核心模型
    __all__ = []