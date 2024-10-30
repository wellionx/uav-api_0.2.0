import os
from flask import Flask
from flask_jwt_extended import JWTManager
from resources.login_resource import login_bp # 导入登录蓝图
from resources.metashape_resource import metashape_bp
from resources.plot_seg_resource import plot_seg_bp  # 导入新的蓝图
from resources.plot_image_mask_resource import plot_image_mask_bp  # 导入新的图像掩膜蓝图
from resources.image_predict_resource import image_predict_bp, ModelManager  # 导入图像推理蓝图
from resources.result_show_resource import result_show_bp  # 导入结果可视化蓝图
from config.config import Config
import logging

app = Flask(__name__)

# 初始化 JWT
jwt = JWTManager(app)
app.config.from_object(Config)

# 添加 Celery 配置
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6380/0'

# 注册蓝图
app.register_blueprint(login_bp)  # 注册登录蓝图
app.register_blueprint(metashape_bp)  # 注册拼图蓝图
app.register_blueprint(plot_seg_bp)  # 注册网格绘制蓝图
app.register_blueprint(plot_image_mask_bp)  # 注册图像掩膜蓝图
app.register_blueprint(image_predict_bp)  # 注册图像推理蓝图
app.register_blueprint(result_show_bp)  # 注册结果可视化蓝图

# 在应用启动时预加载模型
@app.before_first_request
def initialize_models():
    try:
        # 预加载常用模型
        ModelManager.get_model('IntegrateNet', 'maize', 'tassel_count')
        logging.info("Models initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")

if __name__ == '__main__':
    # 启动 Celery worker
    os.system('celery -A celery_worker worker --loglevel=info &')
    
    # 启动 Flask 应用
    #app.run(host='0.0.0.0', port=5000)  # 测试环境
    app.run(host='0.0.0.0', port=8081) #对外监听
