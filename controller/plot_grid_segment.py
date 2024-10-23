# controller/plot_grid_segment.py

import subprocess
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_grid(num_rows, num_cols, input_pts):
    """绘制网格并生成 shapefile"""
    command = [
        "Rscript",
        "controller/plot_grid_segment.R",  # 更新为相对路径
        str(num_rows),
        str(num_cols),
        input_pts
    ]

    # 执行 R 脚本
    try:
        subprocess.run(command, check=True)
        logging.info("Grid plotted successfully.")
        return {'message': 'Grid plotted successfully'}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in R script: {str(e)}")
        return {'error': str(e)}
