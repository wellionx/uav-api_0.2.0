# resources/metashape_resource.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from controller.metashape import start_processing, get_task_status
from multiprocessing import Manager

metashape_bp = Blueprint('metashape', __name__)

# 创建任务状态管理器
manager = Manager()
task_status = manager.dict()

@metashape_bp.route('/metashape', methods=['POST'])
@jwt_required()
def metashape():
    if 'input_path' not in request.json:
        return jsonify({'error': 'No input path provided'}), 400

    input_path = request.json['input_path']
    task_id = start_processing(input_path, task_status)  # 获取任务 ID
    return jsonify({'task_id': task_id, 'status': 'Processing started'}), 202  # 返回任务 ID

@metashape_bp.route('/task_status/<task_id>', methods=['GET'])
@jwt_required()
def task_status_route(task_id):
    status = get_task_status(task_id, task_status)
    return jsonify(status)