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
    
    @classmethod
    def get_model(cls, model_name, crop, trait):
        """获取预加载的模型实例"""
        key = f"{model_name}_{crop}_{trait}"
        if key not in cls._models:
            logging.error(f"Model {key} not found in preloaded models")
            raise ValueError(f"Model {key} not found. Please ensure the model is preloaded.")
        return cls._models[key]

@image_predict_bp.route('/predict', methods=['POST'])
@jwt_required()  # 增加保护路由
def image_predict():
    data = request.json

    # 输入验证
    model_name = data.get('model_name', 'IntegrateNet')  # 设置默认模型为 IntegrateNet
    imgfile = data.get('imgfile')  # 可以是单个文件路径或目录
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
        with torch.no_grad():
            results = predict(model_name, imgfile, crop, trait, model)  # 处理单张图片或目录
        return jsonify({"results": results}), 200
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500
