from celery import Celery
from app import app  # 确保导入 Flask 应用

def make_celery(app):
    celery = Celery(app.import_name, backend='redis://localhost:6379/0', broker='redis://localhost:6379/0')
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)
