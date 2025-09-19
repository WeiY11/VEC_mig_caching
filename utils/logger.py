#!/usr/bin/env python3
"""
日志记录器
提供统一的日志记录功能
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class Logger:
    """日志记录器类"""
    
    def __init__(self, name: str = "MATD3", level: str = "INFO", 
                 log_file: Optional[str] = None):
        """初始化日志记录器"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # 文件处理器
            if log_file:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """一般信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """错误信息"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """严重错误"""
        self.logger.critical(message)

# 创建默认日志记录器
default_logger = Logger()

def get_logger(name: str = "MATD3", level: str = "INFO", 
               log_file: Optional[str] = None) -> Logger:
    """获取日志记录器实例"""
    return Logger(name, level, log_file)