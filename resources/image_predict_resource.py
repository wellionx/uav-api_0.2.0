from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
import os
import logging
import torch
from controller.image_predict import predict

# 设置日志记录
logging.basicConfig(level=logging.INFO)

image_predict_bp = Blueprint('image_predict', __name__)

class ModelManager:
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls, model_name, crop, trait):
        """获取模型实例，如果不存在则创建"""
        key = f"{model_name}_{crop}_{trait}"
        if key not in cls._models:
            try:
                from controller.image_predict import load_model
                model = load_model(model_name, crop, trait)
                cls._models[key] = model
                logging.info(f"Model {key} loaded successfully")
            except Exception as e:
                logging.error(f"Error loading model {key}: {str(e)}")
                raise
        return cls._models[key]

@image_predict_bp.route('/predict', methods=['POST'])
@jwt_required()  # 增加保护路由
def image_predict():
    data = request.json

    # 输入验证
    model_name = data.get('model_name')
    imgfile = data.get('imgfile', 'data/image/')  # 默认图像文件夹
    crop = data.get('crop', 'maize')  # 默认作物
    trait = data.get('trait', 'seedling_count')  # 默认性状

    if not model_name:
        return jsonify({"error": "Model name is required."}), 400

    # 检查模型名称是否有效
    valid_models = ['IntegrateNet', 'V3liteNet', 'V3segnet', 'V3segplus', 'linear_regression']
    if model_name not in valid_models:
        return jsonify({"error": f"Invalid model name: {model_name}. Valid options are: {valid_models}."}), 400

    try:
        # 使用 ModelManager 获取模型
        model = ModelManager.get_model(model_name, crop, trait)
        
        # 调用预测函数，传入已加载的模型
        with torch.no_grad():
            results = predict(model_name, imgfile, crop, trait, model)
        return jsonify({"results": results}), 200
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500
