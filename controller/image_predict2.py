# the 2nd version update for counting plants and flowers
# crop-trait: rice seedling, canola flower
import os
import argparse
import logging
import torch
import pandas as pd
import numpy as np  # 确保导入 numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from controller.model_manager import orchestrator  # 导入orchestrator实例
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F


# 设置日志记录
logging.basicConfig(level=logging.INFO)

def read_image(x):
    img_arr = np.array(Image.open(x))
    if img_arr.ndim == 2:  # grayscale
        img_arr = np.stack((img_arr,) * 3, axis=-1)  # 将灰度图转换为RGB
    return img_arr

def Data_list(path):
    return [[filename[:-4]] for filename in os.listdir(path) if filename.endswith('.jpg')]  # 只获取jpg文件

class Normalize(object):
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image = sample['image'].astype('float32')
        image = (self.scale * image - self.mean) / self.std
        return {'image': image}

class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize
    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[-2:]
        ph, pw = (self.psize - h % self.psize), (self.psize - w % self.psize)
        padding = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
        if ph or pw:
            image = F.pad(image, padding)
        return {'image': image}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))  # H x W x C -> C x H x W
        return {'image': torch.from_numpy(image).float()}  # 确保转换为 float32

class CounterNetDataset(Dataset):
    def __init__(self, imgfile, transform=None):
        self.imgfile = imgfile
        self.data_list = Data_list(imgfile)
        self.transform = transform
        self.image_list = []
        self.images = {}
    def readfile(self,data):
        lines = data.readlines()
        X = []
        X_row = 0
        for line in lines:
            linedata = line.strip('\n')
            if len(linedata) == 0:
                break
            X.append(line.strip('\n').split(','))
            X_row += 1
        return X, X_row
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        file_name = self.data_list[idx][0]  # 获取文件名字符串
        self.image_list.append(file_name)
        if file_name not in self.images:
            image = read_image(os.path.join(self.imgfile, file_name + ".jpg"))  # 使用当前目录
            self.images.update({file_name: image.astype('float32')})
        sample = {
            'image': self.images[file_name],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_preprocessing_params(model_type):
    if model_type == "CounterNet":
        # 使用 CounterNet 的图像预处理参数
        return {
            "image_mean": np.array([0.3859, 0.4905, 0.2895]).reshape((1, 1, 3)),
            "image_std": np.array([0.1718, 0.1712, 0.1518]).reshape((1, 1, 3))
        }
    elif model_type=="IntegrateNet":
        # 使用 IntegrateNet的图像预处理参数
        return {
            "image_mean": np.array([0.0210, 0.0193, 0.0181]).reshape((1, 1, 3)),
            "image_std": np.array([1, 1, 1]).reshape((1, 1, 3))
        }
    else:
        raise ValueError("Unsupported model type")
        
def predict2(imgfile, crop, trait):
    """预测函数，支持目录下图片文件的识别
    
    Args:
        imgfile (str): 图片文件或目录路径
        crop (str): 作物类型
        trait (str): 性状类型
        模型名称根据性状和作物自动从字典中匹配
        
    Returns:
        Union[int, List[Dict]]: 单文件返回计数值，目录返回预测结果列表
    """
    # 获取模型架构
    model_arch = orchestrator.get_model_arch(crop, trait)  # 从字典中获取模型架构
    model_name = model_arch.__name__  # 获取模型名称

    # 图像预处理参数--根据模型名称匹配
    params = get_preprocessing_params(model_name)
    image_scale = 1. / 255
    image_mean = params["image_mean"]
    image_std = params["image_std"]
    output_stride = 8 

    # 基础图像转换
    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean),
        ToTensor(),
        ZeroPadding(output_stride)
    ])
    
    # 确保模型已加载并处于评估模式
    net = orchestrator.load_model(crop, trait)  # 使用orchestrator加载模型
    net.eval()

    # 图片文件读取与转换
    valset = CounterNetDataset(imgfile=imgfile, transform=val_transforms)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    pd_counts = []
    total_images = len(valset)
    logging.info(f"Found {total_images} images to process")

    for i, sample in enumerate(val_loader):
        logging.info(f"Processing image: {valset.data_list[i]}")
        image = sample['image'].cuda()
        result = orchestrator.process_prediction(net, image, model_name)  # 使用获取的模型名称
        logging.info(f"Detected {result['pdcount']} objects")
        pd_counts.append({
            'imageID': valset.image_list[i],
            'pdcount': result['pdcount']
        })
    return pd_counts

def main():
    parser = argparse.ArgumentParser(description='Image Prediction using specified model.')
    parser.add_argument('--imgfile', type=str, required=True, help='Path to the image file or directory.')
    parser.add_argument('--crop', type=str, required=True, help='Crop type.')
    parser.add_argument('--trait', type=str, required=True, help='Trait type.')
    args = parser.parse_args()

    logging.info(f"Starting prediction with images from: {args.imgfile}, crop: {args.crop}, trait: {args.trait}")
    results = predict2(args.imgfile, args.crop, args.trait)
    logging.info("Prediction results: %s", results)

if __name__ == "__main__":
    main()



