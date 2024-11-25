import os
import logging
from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from resources.login_resource import login_bp  # 导入登录蓝图
from resources.metashape_resource import metashape_bp  # 导入拼图蓝图
from resources.plot_seg_resource import plot_seg_bp  # 导入新的蓝图
from resources.plot_image_mask_resource import plot_image_mask_bp  # 导入新的图像掩膜蓝图
from resources.image_predict_resource import image_predict_bp, ModelManager  # 导入图像推理蓝图
from resources.result_show_resource import result_show_bp  # 导入结果可视化蓝图
from config.config import Config
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from controller.image_predict import load_model  # 导入 load_model 函数

app = Flask(__name__)
app.config.from_object(Config)

# 添加文件上传配置
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024  # 限制上传文件大小为 6MB
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化 JWTManager
jwt = JWTManager(app)  # 确保在创建 Flask 应用后初始化 JWTManager

# 创建线程池和任务管理器（全局）
executor = ThreadPoolExecutor(max_workers=app.config['EXECUTOR_MAX_WORKERS'])
manager = Manager()
task_status = manager.dict()

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
        # 只加载 seedling_count 模型
        logging.info("Loading seedling_count model (IntegrateNet)...")
        seedling_model = load_model('IntegrateNet', 'maize', 'seedling_count')
        ModelManager._models['IntegrateNet_maize_seedling_count'] = seedling_model
        logging.info("Seedling_count model loaded successfully")

        # TODO: 后续添加 tassel_count 模型
        # logging.info("Loading tassel_count model (V3segplus)...")
        # tassel_model = load_model('V3segplus', 'maize', 'tassel_count')
        # ModelManager._models['V3segplus_maize_tassel_count'] = tassel_model
        # logging.info("Tassel_count model loaded successfully")

        logging.info("Model initialized successfully")
        return {
            'seedling': seedling_model
            # 'tassel': tassel_model  # 暂时注释掉
        }
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        ModelManager._models.clear()
        raise

@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f"Unhandled error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# 清理临时文件的函数
def cleanup_temp_files():
    """清理上传文件夹中的临时文件"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        logging.error(f"Error cleaning up temporary files: {str(e)}")

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081)
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
    finally:
        cleanup_temp_files()  # 清理临时文件
        executor.shutdown(wait=True)
        logging.info("Application shutdown complete")
