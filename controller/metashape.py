# controller/metashape.py

import os
import logging
from pathlib import Path
import Metashape
from flask import current_app
from concurrent.futures import ThreadPoolExecutor
import uuid
import subprocess
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

# Disable CUDA
#Metashape.app.settings.setValue("main/gpu_enable_cuda", "0")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a thread pool
executor = ThreadPoolExecutor(max_workers=8)

def find_files(folder, types):
    """Find all specified types of files in the given folder"""
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        logging.error(f"Folder {folder} does not exist or is not a directory.")
        return []
    return [str(file) for file in folder_path.glob('**/*') if file.suffix.lower() in types]

def ensure_output_dir(output_dir):
    """Ensure the output directory exists; create it if it does not"""
    if not output_dir.exists():
        logging.info(f"Creating output directory {output_dir}")
        output_dir.mkdir(parents=True)

def extract_zip_files(input_path):
    """
    使用Linux unzip命令解压指定路径下的所有ZIP文件
    
    Args:
        input_path: 包含ZIP文件的目录路径
        
    Returns:
        bool: 解压是否成功
    """
    try:
        input_dir = Path(input_path)
        if not input_dir.exists():
            logging.error(f"Input directory {input_path} does not exist")
            return False

        # 查找所有zip文件
        zip_files = list(input_dir.glob('*.zip'))
        if not zip_files:
            logging.info("No ZIP files found in the input directory")
            return False

        for zip_file in zip_files:
            logging.info(f"Extracting {zip_file}")
            try:
                # 使用unzip命令解压文件
                # -o: 覆盖已存在的文件
                # -q: 安静模式，减少输出
                # -j: 不保留目录结构
                # -d: 指定解压目录
                # 仅解压文件，不改变目录结构
                # cmd = ['unzip', '-o', '-q', str(zip_file), '-d', str(input_dir)]
                # 解压文件，但不保留目录结构 
                cmd = ['unzip', '-o', '-j', '-d', str(input_dir), str(zip_file)]
                # 执行解压命令
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode == 0:
                    logging.info(f"Successfully extracted {zip_file}")
                    # 可选删除已解压的zip文件
                    # zip_file.unlink()
                else:
                    logging.error(f"Error extracting {zip_file}: {result.stderr}")

            except Exception as e:
                logging.error(f"Error processing {zip_file}: {str(e)}")
                continue

        return True

    except Exception as e:
        logging.error(f"Error in extract_zip_files: {str(e)}")
        return False

def get_photo_date(photo_path):
    """
    读取照片的拍摄时间
    
    Args:
        photo_path: 照片文件路径
        
    Returns:
        str: 格式化的日期 (YYYY-MM-DD)，如果无法读取则返回None
    """
    try:
        with Image.open(photo_path) as img:
            exif = img._getexif()
            if exif is None:
                return None
                
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    date_str = exif[tag_id]
                    # 转换日期格式 (通常格式为 "YYYY:MM:DD HH:MM:SS")
                    date_obj = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                    return date_obj.strftime('%Y-%m-%d')
    except Exception as e:
        logging.error(f"Error reading date from {photo_path}: {str(e)}")
        return None
    return None

def process_images(input_path, task_status, task_id):
    """Process images and generate orthomosaic"""
    # Set the output directory to data/task_id
    output_dir = Path("./data") / str(task_id)
    ensure_output_dir(output_dir)  # Ensure the output directory exists

    # 首先尝试解压任何ZIP文件
    extract_zip_files(input_path)

    # 查找照片文件
    photos = find_files(input_path, [".jpg", ".jpeg", ".tif", ".tiff"])
    if not photos:
        logging.error("No photos found to process.")
        task_status[task_id] = {'status': 'FAILURE', 'error': 'No photos found to process.'}
        return

    # 读取第一张照片的拍摄日期
    photo_date = get_photo_date(photos[0]) if photos else None
    # 更新任务状态，包含照片信息
    task_status[task_id] = {
        'status': 'PROCESSING', 
        'message': 'Processing images...',
        'photo_date': photo_date,
        'total_photos': len(photos),
    }

    try:
        doc = Metashape.app.document
        if not doc:
            doc = Metashape.Document()
        # Save the project to the output directory
        project_path = output_dir / "project.psx"
        doc.save(str(project_path))
        chunk = doc.addChunk()
        chunk.addPhotos(photos)
        doc.save()

        chunk.matchPhotos(generic_preselection=True, reference_preselection=False)
        chunk.alignCameras()
        doc.save()
        chunk.buildDepthMaps()
        doc.save()
        chunk.buildPointCloud()
        doc.save()

        chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
                      interpolation=Metashape.Interpolation.EnabledInterpolation)
        doc.save()

        chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ElevationData,
                              blending_mode=Metashape.BlendingMode.MosaicBlending)
        doc.save()

        # Set the output file path to output.tif
        output_tif_path = output_dir / "output.tif"  # Use fixed file name output.tif
        chunk.exportRaster(str(output_tif_path))
        logging.info(f"Processing complete. Output file saved to {output_tif_path}")

        # 删除项目文件和文件夹
        try:
            # 删除 project.psx 文件
            if project_path.exists():
                subprocess.run(['rm', str(project_path)], check=True)
                logging.info(f"Deleted project file: {project_path}")

            # 删除 project.files/ 文件夹
            project_files_dir = output_dir / "project.files"
            if project_files_dir.exists():
                subprocess.run(['rm', '-r', str(project_files_dir)], check=True)
                logging.info(f"Deleted project files directory: {project_files_dir}")

        except Exception as e:
            logging.error(f"Error deleting project files: {str(e)}")

        task_status[task_id] = {
            'status': 'SUCCESS', 
            'result': str(output_tif_path),
            'photo_date': photo_date,
            'total_photos': len(photos)
        }

    except Exception as e:
        logging.error(f"Error processing images: {str(e)}")
        task_status[task_id] = {
            'status': 'FAILURE', 
            'error': str(e),
            'photo_date': photo_date,
            'total_photos': len(photos)
        }

def start_processing(input_path, task_status):
    """Start asynchronous processing task"""
    task_id = str(uuid.uuid4())  # Generate a unique task ID
    executor.submit(process_images, input_path, task_status, task_id)  # Pass task_id to the processing function
    return task_id  # Return the task ID

def get_task_status(task_id, task_status):
    """Get the status of the task"""
    return task_status.get(task_id, {'status': 'PENDING', 'message': 'Task not found'})