# controller/result_show.py

import subprocess
import logging
import os

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_r_script(shapefile_path, countfile_path, output_dir='out/'):
    """运行 R 脚本以生成可视化结果"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # R 脚本路径
    r_script_path = "controller/result_show.R"  # R 脚本的相对路径

    # 构建命令
    command = [
        "Rscript",
        r_script_path,
        shapefile_path,
        countfile_path,
        output_dir
    ]

    # 执行 R 脚本
    try:
        subprocess.run(command, check=True)
        logging.info("R script executed successfully.")
        return {'message': 'Processing completed successfully.'}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in R script: {str(e)}")
        return {'error': str(e)}
