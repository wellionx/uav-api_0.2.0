from flask import Flask
from flask_jwt_extended import JWTManager
from resources.login_resource import login_bp
from resources.metashape_resource import metashape_bp  # 导入拼图蓝图
from resources.image_predict2_resource import image_predict_bp
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from config.config import Config  # 导入配置类

app = Flask(__name__)
app.config.from_object(Config)  # 加载配置

# 初始化配置
Config.init_app(app)

# 初始化 JWTManager
jwt = JWTManager(app)
# 创建线程池和任务管理器（全局）
executor = ThreadPoolExecutor(max_workers=Config.EXECUTOR_MAX_WORKERS)
manager = Manager()
task_status = manager.dict()

# 将线程池和任务管理器添加到 app 配置中
app.config['EXECUTOR'] = executor
app.config['TASK_STATUS'] = task_status

# 注册蓝图
app.register_blueprint(login_bp)
app.register_blueprint(metashape_bp)  # 注册无人机拼图蓝图
app.register_blueprint(image_predict_bp)    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
