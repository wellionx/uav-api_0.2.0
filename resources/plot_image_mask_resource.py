from flask import Blueprint, request, jsonify
from controller.plot_image_mask import crop_geotiff_with_shapefile

plot_image_mask_bp = Blueprint('plot_image_mask', __name__)

@plot_image_mask_bp.route('/plot_image_mask', methods=['POST'])
def plot_image_mask():
    data = request.json
    shapefile_path = data.get('shapefile')
    tif_path = data.get('tif')
    output_dir = data.get('output_dir', 'image/')

    try:
        crop_geotiff_with_shapefile(tif_path, shapefile_path, output_dir)
        return jsonify({"message": "Processing completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
