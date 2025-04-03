import os
import cv2
import logging
import rasterio
import geopandas as gpd
from rasterio.mask import mask

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义全局常量
BASE_DIR = "data"

def setup_task_directories(task_id):
    """设置任务相关的目录结构并返回重要路径"""
    task_dir = os.path.join(BASE_DIR, str(task_id))
    output_image_dir = os.path.join(task_dir, "image")
    input_tiff_path = os.path.join(task_dir, "output.tif")

    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    logging.info(f"Created directory structure for task {task_id}")

    return {
        'task_dir': task_dir,
        'output_image_dir': output_image_dir,
        'input_tiff_path': input_tiff_path
    }

def clear_output_images(output_image_dir):
    """清空输出图像目录中的所有JPG文件"""
    try:
        for file in os.listdir(output_image_dir):
            if file.endswith('.jpg'):
                os.remove(os.path.join(output_image_dir, file))
                logging.info(f"Deleted file: {file}")
    except Exception as e:
        logging.error(f"Error clearing output images: {str(e)}")

def convert_tiff_to_jpg(tiff_path, output_image_dir):
    """将 TIFF 图像转换为 JPG 格式并保存"""
    try:
        img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            jpg_file_name = os.path.splitext(os.path.basename(tiff_path))[0] + '.jpg'
            jpg_path = os.path.join(output_image_dir, jpg_file_name)
            cv2.imwrite(jpg_path, img)
            logging.info(f"Converted and saved: {jpg_path}")
        else:
            logging.warning(f"Failed to read image: {tiff_path}")
    except Exception as e:
        logging.error(f"Error converting {tiff_path} to JPG: {e}")
        
def crop_geotiff_with_shapefile(task_id, shapefile_path, paths):
    """基于shapefile裁剪GeoTIFF文件"""
    try:
        shapes = gpd.read_file(shapefile_path)
        shapes = shapes.to_crs(crs="EPSG:4326")

        with rasterio.open(paths['input_tiff_path']) as src:
            for index, shape in shapes.iterrows():
                out_image, out_transform = mask(src, [shape['geometry']], crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # 保存裁剪后的TIFF文件
                file_id = shape['ID']
                tiff_file_name = f"{file_id}.tif"
                cropped_tiff_path = os.path.join(paths['output_image_dir'], tiff_file_name)
                
                with rasterio.open(cropped_tiff_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                logging.info(f"Cropped and saved: {cropped_tiff_path}")

                # 转换为JPG
                convert_tiff_to_jpg(cropped_tiff_path, paths['output_image_dir'])

                # 删除临时TIFF文件
                if os.path.exists(cropped_tiff_path):
                    os.remove(cropped_tiff_path)
                    logging.info(f"Deleted TIFF file: {cropped_tiff_path}")

    except Exception as e:
        logging.error(f"Error cropping GeoTIFF with shapefile: {e}")
        raise