# config/config.py

import os
import logging

class Config:
    # JWT 配置
    JWT_SECRET_KEY = '8271d4b0317467d8a191d8c4f415f0155660a7840f3e7d7cd55ce5c68e4dc0c9'
    SECRET_KEY = '5c2dcbd5802a024dab0f399707c2b2537c13c73cd9c40ae5ce70a150ce602bd9'
    
    # 日志配置
    LOG_PATH = './logs/app.log'
    LOG_LEVEL = 'DEBUG'  # 设置为 DEBUG 以输出所有信息
    
    # 线程池和任务管理器配置
    EXECUTOR_MAX_WORKERS = 8
    TASK_STATUS = {}  # 可以在应用启动时初始化

    @staticmethod
    def init_app(app):
        """初始化应用配置"""
        # 创建必要的目录
        os.makedirs(os.path.dirname(Config.LOG_PATH), exist_ok=True)

        # 配置日志
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),  # 设置日志级别
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_PATH),  # 输出到文件
                logging.StreamHandler()  # 输出到终端
            ]
        )
