from celery import Celery

# 创建 Celery 实例
celery = Celery('tasks',
                backend='redis://localhost:6379/0',
                broker='redis://localhost:6379/0')

# 可选：配置 Celery
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
)
