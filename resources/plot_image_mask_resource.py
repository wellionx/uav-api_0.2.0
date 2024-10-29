from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required  # 导入 jwt_required 装饰器
from controller.plot_image_mask import crop_geotiff_with_shapefile

plot_image_mask_bp = Blueprint('plot_image_mask', __name__)

@plot_image_mask_bp.route('/plot_image_mask', methods=['POST'])
@jwt_required()  # 增加保护路由
def plot_image_mask():
    data = request.json
    shapefile_path = data.get('shapefile')
    tif_path = data.get('tif')

    try:
        # 调用 crop_geotiff_with_shapefile 函数
        crop_geotiff_with_shapefile(tif_path, shapefile_path)
        return jsonify({"message": "Processing completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
