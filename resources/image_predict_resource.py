from flask import Blueprint, request, jsonify
from controller.image_predict import predict
import os

image_predict_bp = Blueprint('image_predict', __name__)

@image_predict_bp.route('/predict', methods=['POST'])
def image_predict():
    data = request.json
    model_name = data.get('model_name')
    imgfile = data.get('imgfile', 'data/image/')  # 默认图像文件夹
    crop = data.get('crop', 'maize')  # 默认作物
    trait = data.get('trait', 'seedling_count')  # 默认性状

    # 检查模型名称是否有效
    valid_models = ['IntegrateNet', 'V3liteNet', 'V3segnet', 'V3segplus', 'linear_regression']
    if model_name not in valid_models:
        return jsonify({"error": f"Invalid model name: {model_name}. Valid options are: {valid_models}."}), 400

    try:
        # 调用预测函数
        results = predict(model_name, imgfile, crop, trait)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500