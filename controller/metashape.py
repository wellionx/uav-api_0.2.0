# controller/metashape.py

import os
import logging
from pathlib import Path
import Metashape
from celery_worker import celery  # 导入 Celery 实例
from flask import current_app  # 导入 Flask 当前应用上下文

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_files(folder, types):
    """在给定文件夹中找到所有指定类型的文件"""
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        logging.error(f"Folder {folder} does not exist or is not a directory.")
        return []
    return [str(file) for file in folder_path.glob('**/*') if file.suffix.lower() in types]

def ensure_output_dir(output_dir):
    """确保输出目录存在，如果不存在则创建它"""
    if not output_dir.exists():
        logging.info(f"Creating output directory {output_dir}")
        output_dir.mkdir(parents=True)

@celery.task
def process_images(input_path):
    """处理图像并生成正射影像"""
    with current_app.app_context():  # 确保使用 Flask 应用上下文
        output_dir = Path("./data/out")  # 更新输出目录
        ensure_output_dir(output_dir)

        photos = find_files(input_path, [".jpg", ".jpeg", ".tif", ".tiff"])
        if not photos:
            logging.error("No photos found to process.")
            return {'error': 'No photos found to process.'}

        doc = Metashape.app.document
        if not doc:
            doc = Metashape.Document()

        chunk = doc.addChunk()
        chunk.addPhotos(photos)
        chunk.matchPhotos(generic_preselection=True, reference_preselection=False)
        chunk.alignCameras()
        chunk.buildDepthMaps()
        chunk.buildPointCloud()
        doc.save(os.path.join(input_path, "project.psx"))

        chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
                       interpolation=Metashape.Interpolation.EnabledInterpolation)
        doc.save()

        chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ElevationData,
                               blending_mode=Metashape.BlendingMode.MosaicBlending)
        doc.save()

        output_tif_path = output_dir / "output.tif"
        chunk.exportRaster(str(output_tif_path))
        logging.info(f"Processing complete. Output file saved to {output_tif_path}")

        doc.close()
        Metashape.app.quit()

        return {'message': 'Processing complete', 'output_file': str(output_tif_path)}