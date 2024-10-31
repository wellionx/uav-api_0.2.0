import os
from flask import Flask
from flask_jwt_extended import JWTManager
from resources.login_resource import login_bp  # 导入登录蓝图
from resources.metashape_resource import metashape_bp # 导入拼图蓝图
from resources.plot_seg_resource import plot_seg_bp  # 导入新的蓝图
from resources.plot_image_mask_resource import plot_image_mask_bp  # 导入新的图像掩膜蓝图
from resources.image_predict_resource import image_predict_bp, ModelManager  # 导入图像推理蓝图
from resources.result_show_resource import result_show_bp  # 导入结果可视化蓝图
from config.config import Config
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager

# 创建必要的目录
os.makedirs('./logs', exist_ok=True)
os.makedirs('./data/out', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/app.log'),
        logging.StreamHandler()
    ]
)

# 创建线程池和任务管理器（全局）
executor = ThreadPoolExecutor(max_workers=8)
manager = Manager()
task_status = manager.dict()

app = Flask(__name__)

# 初始化 JWT
jwt = JWTManager(app)
app.config.from_object(Config)

# 将线程池和任务管理器添加到 app 配置中
app.config['EXECUTOR'] = executor
app.config['TASK_STATUS'] = task_status

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

@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f"Unhandled error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081)
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
    finally:
        executor.shutdown(wait=True)
        logging.info("Application shutdown complete")
