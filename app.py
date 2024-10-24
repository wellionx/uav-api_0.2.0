from flask import Flask
from flask_jwt_extended import JWTManager
from resources.login_resource import login_bp
#from resources.metashape_resource import metashape_bp
from resources.plot_seg_resource import plot_seg_bp  # 导入新的蓝图
from resources.plot_image_mask_resource import plot_image_mask_bp  # 导入新的图像掩膜蓝图
from resources.image_predict_resource import image_predict_bp  # 导入图像推理蓝图
from resources.result_show_resource import result_show_bp  # 导入结果可视化蓝图
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# 初始化 JWT
jwt = JWTManager(app)

# 注册蓝图
app.register_blueprint(login_bp)  # 注册登录蓝图
#app.register_blueprint(metashape_bp)  # 注册拼图蓝图
app.register_blueprint(plot_seg_bp)  # 注册网格绘制蓝图
app.register_blueprint(plot_image_mask_bp)  # 注册图像掩膜蓝图
app.register_blueprint(image_predict_bp)  # 注册图像推理蓝图
app.register_blueprint(result_show_bp)  # 注册结果可视化蓝图

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 测试环境
