import os
import cv2
import argparse
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建它"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def convert_tiff_to_jpg(tiff_path, output_dir):
    """将 TIFF 图像转换为 JPG 格式并保存"""
    try:
        img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            jpg_file_name = os.path.splitext(os.path.basename(tiff_path))[0] + '.jpg'
            jpg_path = os.path.join(output_dir, jpg_file_name)
            cv2.imwrite(jpg_path, img)
            logging.info(f"Converted and saved: {jpg_path}")
        else:
            logging.warning(f"Failed to read image: {tiff_path}")
    except Exception as e:
        logging.error(f"Error converting {tiff_path} to JPG: {e}")

def crop_geotiff_with_shapefile(tiff_path, shapefile_path, output_dir):
    """基于shapefile裁剪GeoTIFF文件"""
    try:
        # 读取shapefile
        shapes = gpd.read_file(shapefile_path)
        shapes = shapes.to_crs(crs="EPSG:4326")

        # 读取GeoTIFF文件
        with rasterio.open(tiff_path) as src:
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
                tiff_path = os.path.join(output_dir, tiff_file_name)
                with rasterio.open(tiff_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                logging.info(f"Cropped and saved: {tiff_path}")

                # 转换为JPG格式
                convert_tiff_to_jpg(tiff_path, output_dir)

                # 删除 TIFF 文件
                os.remove(tiff_path)
                logging.info(f"Deleted TIFF file: {tiff_path}")

    except Exception as e:
        logging.error(f"Error cropping GeoTIFF with shapefile: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--shapefile', type=str, default='data/out/plot_grids.shp', help='Path to the shapefile.')
    parser.add_argument('--tif', type=str, default='data/output.tif', help='Path to the GeoTiff image.')
    parser.add_argument('--output_dir', type=str, default='data/image', help='Path to the output directory.')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    crop_geotiff_with_shapefile(args.tif, args.shapefile, args.output_dir)

if __name__ == "__main__":
    main()
