# resources/result_show_resource.py

from flask import Blueprint, request, jsonify
from controller.result_show import run_r_script

result_show_bp = Blueprint('result_show', __name__)

@result_show_bp.route('/result_show', methods=['POST'])
def result_show():
    data = request.json
    shapefile_path = data.get('shapefile')
    countfile_path = data.get('countfile')
    output_dir = data.get('output_dir', 'data/out/')  # 默认输出目录

    if not shapefile_path or not countfile_path:
        return jsonify({"error": "shapefile and countfile paths are required."}), 400

    try:
        result = run_r_script(shapefile_path, countfile_path, output_dir)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
