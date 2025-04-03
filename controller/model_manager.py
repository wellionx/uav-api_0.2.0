# resources/model_manager.py

import logging
import multiprocessing
from collections import defaultdict
import torch  # 确保导入torch库
import os
import torch.nn as nn
import numpy as np
import math

# 导入模型类
from models.CounterNet import *  # 导入 CounterNet
from models.IntegrateNet import *  # 导入 IntegrateNet
from models.common import DetectMultiBackend # 导入 DetectMultiBackend--yolov9
from utils.torch_utils import select_device
# 如果有其他模型，也需要在这里导入

# 架构设计
class ModelOrchestrator:
    def __init__(self):
        # 三维字典：作物类型 -> 性状类型 -> 模型实例
        self.models = defaultdict(lambda: defaultdict(dict))
        self.locks = defaultdict(multiprocessing.Lock)  # 使用进程锁
        self.loaders = {
            ('maize', 'seedling_count'): {
                'arch': IntegrateNet,
                'weights': 'models/IntegrateNet_model_best.pth'
            },
            ('rice', 'seedling_count'): {
                'arch': CounterNet,
                'weights': 'models/CounterNet_rice_seedling_model_best.pth.tar'
            },
            ('canola', 'flower_count'): {
                'arch': CounterNet,
                'weights': 'models/CounterNet_canola_flower_model_best.pth.tar'
            },
            ('maize', 'tassel_count'): {
                'arch': DetectMultiBackend, # 导入 DetectMultiBackend--yolov9
                'weights': 'models/yolov9_maize_tassel_best_model.pt'
            }
        }

    def get_model_arch(self, crop, trait):
        """根据作物和性状获取对应的模型架构"""
        if (crop, trait) in self.loaders:
            return self.loaders[(crop, trait)]['arch']
        else:
            raise ValueError(f"Unknown crop and trait combination: {crop}, {trait}")

    def load_model(self, crop, trait):
        """在单独的进程中加载模型"""
        with self.locks[(crop, trait)]:
            if not self.models[crop][trait]:
                config = self.loaders[(crop, trait)]
                model_name = config['arch'].__name__  # 获取模型名称
                weights_path = config['weights']  # 统一获取权重路径
                
                # 根据模型名称创建模型实例
                if model_name == 'DetectMultiBackend': # DetectMultiBackend即yolov9的模型
                    # 选择设备
                    device = select_device('')
                    net = DetectMultiBackend(weights_path, device=device)
                elif model_name == 'IntegrateNet':
                    net = IntegrateNet().cuda()  # 特殊处理IntegrateNet
                elif model_name == 'CounterNet':
                    net = CounterNet()  # 创建CounterNet实例
                else:
                    net = config['arch']()  # 创建其他模型实例
                
                # 对于除IntegrateNet和YOLOv9外的所有模型，使用DataParallel
                if model_name not in ['IntegrateNet', 'DetectMultiBackend'] and not isinstance(net, nn.DataParallel):
                    net = nn.DataParallel(net).cuda()
                
                # 统一加载权重（排除YOLOv9）
                if model_name != 'DetectMultiBackend':
                    try:
                        checkpoint = torch.load(weights_path)
                        net.load_state_dict(checkpoint['state_dict'], strict=False)
                        logging.info(f"成功加载模型 {model_name} 权重: {weights_path}")
                    except Exception as e:
                        logging.error(f"加载模型 {model_name} 权重失败: {e}")
                
                # 设置为评估模式
                net.eval()
                self.models[crop][trait] = net
            
            return self.models[crop][trait]

    def get_model(self, crop, trait):
        """获取模型，使用进程锁确保进程安全"""
        with self.locks[(crop, trait)]:
            if not self.models[crop][trait]:
                # 在单独的进程中加载模型
                process = multiprocessing.Process(target=self.load_model, args=(crop, trait))
                process.start()
                process.join()  # 等待进程完成
            return self.models[crop][trait]

    def process_prediction(self, net, image, model_name):
        """处理单个预测"""
        try:
            if model_name == 'IntegrateNet':
                dic = net(image)
                output = dic['density']
                output = output.squeeze().cpu().detach().numpy()
                output = np.clip(output, 0, None)
                pdcount = math.ceil(output.sum())
            elif model_name == 'CounterNet':
                output = net(image)
                output = np.clip(output, 0, None)
                pdcount = math.ceil(output.sum())
            else:
                dic = net(image, is_normalize=False)
                R = dic['R']
                R = R.cpu().detach().numpy()  # 添加 detach() 方法
                R = np.clip(R, 0, None)
                pdcount = math.ceil(R.sum())
            
            return {'pdcount': pdcount}
        
        except Exception as e:
            logging.error(f"Error in prediction processing: {str(e)}")
            raise

orchestrator = ModelOrchestrator()