import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import argparse
import logging
from models.IntegrateNet import *
from models.V3liteNet import *
from flask import current_app  # 导入 current_app

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_image(x):
    try:
        img_arr = np.array(Image.open(x))
        return img_arr
    except Exception as e:
        logging.error(f"Error reading image {x}: {e}")
        return None

def Data_list(path):
    return [filename[:-4] for filename in os.listdir(path) if filename.endswith('.jpg')]

class MaizeDataset(Dataset):
    def __init__(self, imgfile, transform=None):
        self.imgfile = imgfile
        self.data_list = Data_list(imgfile)  # 使用当前目录
        self.transform = transform
        self.images = defaultdict(lambda: None)  # 使用 defaultdict 缓存图像
        self.image_list = self.data_list  # 初始化 image_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        if self.images[file_name] is None:
            image = read_image(os.path.join(self.imgfile, file_name + ".jpg"))  # 使用当前目录
            if image is None:
                raise ValueError(f"Image {file_name} could not be loaded.")
            self.images[file_name] = image.astype('float32')
        sample = {'image': self.images[file_name]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Normalize(object):
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image = (self.scale * image - self.mean) / self.std
        return {'image': image}

class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        h, w = image.shape[-2:]
        ph, pw = (self.psize - h % self.psize), (self.psize - w % self.psize)
        (pl, pr) = (pw // 2, pw - pw // 2) if pw != self.psize else (0, 0)
        (pt, pb) = (ph // 2, ph - ph // 2) if ph != self.psize else (0, 0)
        if (ph != self.psize) or (pw != self.psize):
            tmp_pad = [pl, pr, pt, pb]
            image = F.pad(torch.from_numpy(image), tmp_pad)  # 确保 image 是 Tensor
        return {'image': image}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))  # H x W x C -> C x H x W
        return {'image': torch.from_numpy(image).float()}  # 确保转换为 float32

def load_model(model_name, crop, trait):
    """加载模型并返回"""
    try:
        if model_name == 'IntegrateNet':
            weights_path = os.path.join('weights', crop, trait, f'{model_name}_model_best.pth')
            net = IntegrateNet().cuda()
        else:
            weights_path = os.path.join('weights', crop, trait, f'{model_name}_model_best.pth.tar')
            if model_name == 'V3liteNet':
                net = V3lite().cuda()
                net = nn.DataParallel(net)
            elif model_name == 'V3segnet':
                net = V3seg().cuda()
                net = nn.DataParallel(net)
            elif model_name == 'V3segplus':
                net = V3segplus().cuda()
                net = nn.DataParallel(net)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        checkpoint = torch.load(weights_path)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        net.eval()
        
        # 确保模型参数不需要梯度
        for param in net.parameters():
            param.requires_grad = False
            
        return net
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def predict(model_name, imgfile, crop, trait, model=None):
    """修改预测函数以接受预加载的模型"""
    image_scale = 1. / 255
    image_mean = np.array([0.0210, 0.0193, 0.0181]).reshape((1, 1, 3))
    image_std = np.array([1, 1, 1]).reshape((1, 1, 3))
    output_stride = 8

    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean),
        ToTensor(),
        ZeroPadding(output_stride)
    ])

    valset = MaizeDataset(imgfile=imgfile, transform=val_transforms)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 使用传入的模型或加载新模型
    net = model if model is not None else load_model(model_name, crop, trait)
    net.eval()  # 确保模型处于评估模式

    pd_counts = []
    for i, sample in enumerate(val_loader):
        image = sample['image'].cuda()

        with torch.no_grad():  # 禁用梯度计算
            if model_name == 'IntegrateNet':
                dic = net(image)
                output = dic['density']  # predicted density map
                output = output.squeeze().cpu().detach().numpy()
                output = np.clip(output, 0, None)  # eliminate < 0 values
                pdcount = math.ceil(output.sum())  # 取整数
            else:
                dic = net(image, is_normalize=False)  # 其他模型的推理
                R = dic['R']
                R = R.cpu().numpy()  # 将 Tensor 移动到 CPU 并转换为 NumPy 数组
                R = np.clip(R, 0, None)  # 使用 NumPy 的 clip 函数
                pdcount = math.ceil(R.sum())  # 取整数

        pd_counts.append({'image_list': valset.image_list[i], 'pdcount': pdcount})

    # 保存预测结果为 CSV 文件
    output_dir = current_app.config['OUTPUT_DIR']  # 从配置中获取输出目录
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    output_file = os.path.join(output_dir, 'predictions.csv')
    pd.DataFrame(pd_counts).to_csv(output_file, index=False)  # 保存为 CSV 文件

    return pd_counts

def main():
    parser = argparse.ArgumentParser(description='Image Prediction using specified model.')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., IntegrateNet, V3liteNet).')
    parser.add_argument('--imgfile', type=str, required=True, help='Path to the image file directory.')
    parser.add_argument('--crop', type=str, required=True, help='Crop type.')
    parser.add_argument('--trait', type=str, required=True, help='Trait type.')
    args = parser.parse_args()

    logging.info(f"Starting prediction with model: {args.model}, images from: {args.imgfile}, crop: {args.crop}, trait: {args.trait}")
    results = predict(args.model, args.imgfile, args.crop, args.trait)
    logging.info("Prediction results: %s", results)

if __name__ == "__main__":
    main()
