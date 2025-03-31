from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
import os
import logging
import torch
from controller.image_predict2 import predict2
from controller.predict2_yolo import yolo_predict
from controller.image_predict_pre import (
    setup_task_directories,
    crop_geotiff_with_shapefile,
    clear_output_images)
    
# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 创建 Flask 蓝图
image_predict_bp = Blueprint('image_predict', __name__)

@image_predict_bp.route('/predict', methods=['POST'])
@jwt_required()  # 接口保护
def predict():
    """处理图像预测请求"""
    try:
        # 从请求中获取参数
        imgfile = request.json.get('imgfile')  # 图片文件或目录路径
        crop = request.json.get('crop')          # 作物类型
        trait = request.json.get('trait')        # 性状类型

        # 参数验证
        if not imgfile or not crop or not trait:
            return jsonify({"status": "error", "message": "imgfile, crop, and trait are required."}), 400

        # 根据 trait 选择预测方法
        if trait == 'tassel_count':
            # 使用 yolo_predict 进行预测
            results = yolo_predict(imgfile, crop, trait)
        else:
            # 使用原有的 predict2 进行预测
            results = predict2(imgfile, crop, trait)

        return jsonify({
            "status": "success",
            "data": results
        }), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error during prediction: {str(e)}"
        }), 500

@image_predict_bp.route('/plot/predict', methods=['POST'])
@jwt_required()  # 接口保护
def plot_predict():
    """Process plot prediction requests with file upload"""
    try:
        # 从请求中获取文件和其他参数
        crop = request.form.get('crop', 'maize')  # 默认作物
        trait = request.form.get('trait', 'seedling_count')  # 默认性状
        logging.info(f"Received plot prediction request - crop: {crop}, trait: {trait}")

        # 定义允许的作物类型
        valid_crops = ['maize', 'rice', 'wheat', 'canola']
        
        # 检查 crop 参数
        if crop not in valid_crops:
            logging.warning(f"Invalid crop type received: {crop}")
            return jsonify({
                "status": "error",
                "message": f"Invalid crop type: {crop}. Valid options are: {valid_crops}."
            }), 400

        # 定义允许的性状类型
        valid_traits = ['seedling_count', 'tassel_count', 'flower_count']
        
        # 检查 trait 参数
        if trait not in valid_traits:
            logging.warning(f"Invalid trait type received: {trait}")
            return jsonify({
                "status": "error",
                "message": f"Invalid trait type: {trait}. Valid options are: {valid_traits}."
            }), 400

        # 处理文件上传
        shapefile = request.files.get('shapefile')  # 获取上传的 shapefile
        if not shapefile:
            logging.error("No shapefile uploaded")
            return jsonify({"status": "error", "message": "No shapefile uploaded."}), 400

        # 设置任务目录结构
        task_id = request.form.get('task_id')  # 从表单中获取 task_id
        if not task_id:
            logging.error("No task ID provided")
            return jsonify({"status": "error", "message": "Task ID is required."}), 400

        logging.info(f"Setting up task directories for task_id: {task_id}")
        try:
            paths = setup_task_directories(task_id)
            shapefile_path = os.path.join(paths['task_dir'], 'input.geojson')
            shapefile.save(shapefile_path)
            logging.info(f"Shapefile saved successfully to: {shapefile_path}")
        except OSError as e:
            logging.error(f"File system error: {str(e)}")
            return jsonify({"status": "error", "message": "File system error occurred."}), 500
        except Exception as e:
            logging.error(f"Unexpected error during file processing: {str(e)}")
            return jsonify({"status": "error", "message": "Unexpected error occurred."}), 500

        # 清空之前的输出图像
        logging.info(f"Clearing previous output images from: {paths['output_image_dir']}")
        clear_output_images(paths['output_image_dir'])

        # 步骤1：裁剪图像并转换为JPG
        logging.info("Starting image cropping and conversion process")
        crop_geotiff_with_shapefile(task_id, shapefile_path, paths)
        logging.info("Image processing completed")

        # 步骤2：使用预加载的模型进行预测
        imgfile = paths['output_image_dir']
        if not os.path.exists(imgfile):
            logging.error(f"Image directory not found after processing: {imgfile}")
            return jsonify({"status": "error", "message": "Image directory not found after processing."}), 400

        logging.info("Starting prediction process")
        with torch.no_grad():
            if trait == 'tassel_count':
                # 使用 yolo_predict 进行预测
                results = yolo_predict(imgfile, crop, trait)
            else:
                # 使用原有的 predict2 进行预测
                results = predict2(imgfile, crop, trait)
        logging.info("Prediction completed successfully")

        return jsonify({
            "status": "success",
            "data": results
        }), 200

    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }), 500

    finally:
        # 清理临时文件
        if 'shapefile_path' in locals() and os.path.exists(shapefile_path):
            try:
                os.remove(shapefile_path)
                logging.info(f"Temporary file cleaned up: {shapefile_path}")
            except OSError:
                logging.warning(f"Failed to clean up temporary file: {shapefile_path}")        