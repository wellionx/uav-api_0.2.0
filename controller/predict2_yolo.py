import scipy
import numpy as np
import cv2 as cv
import argparse
import os
import sys
from pathlib import Path
import torch
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression
import logging
from controller.model_manager import orchestrator  # 导入模型管理器

def yolo_predict(imgfile, crop, trait):
    # 从模型管理器加载模型
    logging.info(f"Loading model for crop: {crop}, trait: {trait}")
    model = orchestrator.load_model(crop, trait)  # 使用 load_model 获取模型
    logging.info("Model loaded successfully.")

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)
    dataset = LoadImages(imgfile, img_size=imgsz, stride=stride, auto=pt)
    logging.info(f"Loaded {len(dataset)} images from {imgfile}")

    df_count = []
    for path, im, im0s, vid_cap, s in dataset:
        logging.info(f"Processing image: basename{Path(path).name}")
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255

        if len(im.shape) == 3:
            im = im.unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)

        count = 0
        for det in pred:
            if len(det):
                count += len(det)
                logging.info(f"Detected {count} objects")

        # 将结果存储为字典格式
        df_count.append({
            'imageID': Path(path).name,
            'pdcount': count
        })

    return df_count

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgfile', type=str, required=True, help='directory containing images')
    parser.add_argument('--crop', type=str, required=True, help='Crop type (e.g., maize)')
    parser.add_argument('--trait', type=str, required=True, help='Trait type (e.g., tassel_count)')
    return parser.parse_args()

def main(opt):
    results = yolo_predict(opt.imgfile, opt.crop, opt.trait)
    # 输出每张图片对应的检测到的数量
    for result in results:
        print(f"{result['imageID']}: {result['pdcount']}")  # 输出格式为 "图片名称: 检测数量"

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)