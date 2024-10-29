# resources/metashape_resource.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from controller.metashape import process_images

metashape_bp = Blueprint('metashape', __name__)


@metashape_bp.route('/metashape', methods=['POST'])
@jwt_required()  # 保护此路由
def metashape():
    if 'input_path' not in request.json:
        return jsonify({'error': 'No input path provided'}), 400

    # 从请求中获取输入路径
    input_path = request.json['input_path']  # 从请求中获取输入路径
    task = process_images.apply_async(args=[input_path])  # 异步执行任务
    return jsonify({'task_id': task.id, 'status': 'Processing started'}), 202


@metashape_bp.route('/task_status/<task_id>', methods=['GET'])
@jwt_required()  # 保护此路由
def task_status(task_id):
    task = process_images.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'error': str(task.info),  # 任务失败的原因
        }
    return jsonify(response)