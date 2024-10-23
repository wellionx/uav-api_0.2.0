# resources/plot_seg_resource.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from controller.plot_grid_segment import plot_grid

plot_seg_bp = Blueprint('plot_seg', __name__)

@plot_seg_bp.route('/plot_seg', methods=['POST'])
@jwt_required()  # 保护此路由
def plot_seg():
    # 检查请求体中是否包含必要的参数
    if 'num_rows' not in request.json or 'num_cols' not in request.json or 'input_pts' not in request.json:
        return jsonify({'error': 'Missing parameters'}), 400

    num_rows = request.json['num_rows']
    num_cols = request.json['num_cols']
    input_pts = request.json['input_pts']  # 从请求中获取 input_pts

    # 调用绘制网格的函数
    result = plot_grid(num_rows, num_cols, input_pts)
    return jsonify(result), 200
