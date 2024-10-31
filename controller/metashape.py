# controller/metashape.py

import os
import logging
from pathlib import Path
import Metashape
from flask import current_app
from concurrent.futures import ThreadPoolExecutor
import uuid

# Disable CUDA
Metashape.app.settings.setValue("main/gpu_enable_cuda", "0")

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建线程池
executor = ThreadPoolExecutor(max_workers=8)

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

def process_images(input_path, task_status, task_id):
    """处理图像并生成正射影像"""
    output_dir = Path("./data/out")
    ensure_output_dir(output_dir)

    # 更新任务状态
    task_status[task_id] = {'status': 'PROCESSING', 'message': 'Processing images...'}

    photos = find_files(input_path, [".jpg", ".jpeg", ".tif", ".tiff"])
    if not photos:
        logging.error("No photos found to process.")
        task_status[task_id] = {'status': 'FAILURE', 'error': 'No photos found to process.'}
        return

    try:
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

        output_tif_path = output_dir / f"output_{task_id}.tif"
        chunk.exportRaster(str(output_tif_path))
        logging.info(f"Processing complete. Output file saved to {output_tif_path}")

        task_status[task_id] = {'status': 'SUCCESS', 'result': str(output_tif_path)}

    except Exception as e:
        logging.error(f"Error processing images: {str(e)}")
        task_status[task_id] = {'status': 'FAILURE', 'error': str(e)}

def start_processing(input_path, task_status):
    """启动异步处理任务"""
    task_id = str(uuid.uuid4())  # 生成唯一的任务 ID
    executor.submit(process_images, input_path, task_status, task_id)  # 将 task_id 传递给处理函数
    return task_id  # 返回任务 ID

def get_task_status(task_id, task_status):
    """获取任务状态"""
    return task_status.get(task_id, {'status': 'PENDING', 'message': 'Task not found'})