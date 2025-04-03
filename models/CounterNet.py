import torch
import numpy as np
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from models.loss import *
from models.coordatt import *

def BaseConv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class PyConv4(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = BaseConv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                                stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = BaseConv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                                stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = BaseConv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                                stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = BaseConv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                                stride=stride, groups=pyconv_groups[3])
    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class PyConv3(nn.Module):
    def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = BaseConv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                                stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = BaseConv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                                stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = BaseConv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                                stride=stride, groups=pyconv_groups[2])
    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)

class PyConv2(nn.Module):
    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = BaseConv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                                stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = BaseConv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                                stride=stride, groups=pyconv_groups[1])
    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)

def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return BaseConv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            get_pyconv(3, 16, [3], 1, [1]),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.layer2 = nn.Sequential(
            get_pyconv(16, 32, [3, 5], 1, [1, 4]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.ca2 = CoordAtt(32, 32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(128)
        self.layer3 = nn.Sequential(
            get_pyconv(32, 64, [3, 5, 7, 9], 1, [1, 4, 8, 16]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.ca3 = CoordAtt(64, 64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(256)
        self.layer4 = nn.Sequential(
            get_pyconv(64, 128, [3, 5, 7], 1, [1, 4, 8]),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            get_pyconv(128, 256, [3, 5], 1, [1, 4]),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer6 = nn.Sequential(
            get_pyconv(256, 256, [3], 1, [1]),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.reg_layer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        ca2 = self.ca2(x)
        ca2 = self.conv2(ca2)
        ca2 = F.max_pool2d(ca2, kernel_size=2, stride=2)
        ca2 = self.BN2(ca2)
        x = self.layer3(x)
        ca3 = self.ca3(x)
        ca3 = self.conv3(ca3)
        ca3 = self.BN3(ca3)
        x = self.layer4(x)
        x = x.clone() + ca2
        x = self.layer5(x)
        x = x.clone() + ca3
        x = self.layer6(x)
        x = self.reg_layer(x)
        return x

class Counter(nn.Module):
    def __init__(self, input_size=64, output_stride=8):
        super(Counter, self).__init__()
        k = int(input_size / 8)
        avg_pool_stride = int(output_stride / 8)
        self.counter = nn.Sequential(
                nn.AvgPool2d((k, k), stride=avg_pool_stride),
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1)
            )
    def forward(self, x):
        x = self.counter(x)
        return x

class Normalizer:
    @staticmethod
    def gpu_normalizer(x, imh, imw, insz, os):
        _, _, h, w = x.size()
        accm = torch.tensor(1, dtype=torch.float32, device='cuda').expand(1, insz*insz, h*w)
        accm = F.fold(accm, (imh, imw), kernel_size=insz, stride=os)
        accm = 1 / accm
        accm /= insz**2
        accm = F.unfold(accm, kernel_size=insz, stride=os).sum(1).view(1, 1, h, w)
        x *= accm
        return x.squeeze().cpu().detach().numpy()

class CounterNet(nn.Module):
    def __init__(self, input_size=64, output_stride=8):
        super(CounterNet, self).__init__()
        self.input_size = input_size
        self.output_stride = output_stride
        self.encoder = Encoder()
        self.counter = Counter(input_size, output_stride)
        self.normalizer = Normalizer.gpu_normalizer
        self.weight_init()
    def forward(self, x, is_normalize=True):
        imh, imw = x.size()[2:]
        x = self.encoder(x)
        x = self.counter(x)
        if is_normalize:
            x = self.normalizer(x, imh, imw, self.input_size, self.output_stride)
        return x
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    insz, os = 64, 8
    imH, imW = 1080, 1920
    net = CountingModels(input_size=insz, output_stride=os).cuda()
    with torch.no_grad():
        net.eval()
        x = torch.randn(1, 3, imH, imW).cuda()
        y = net(x)
        print(y.shape)
    with torch.no_grad():
        frame_rate = np.zeros((100, 1))
        for i in range(100):
            x = torch.randn(1, 3, imH, imW).cuda()
            torch.cuda.synchronize()
            start = time()
            y = net(x)
            torch.cuda.synchronize()
            end = time()
            running_frame_rate = 1 * float(1 / (end - start))
            frame_rate[i] = running_frame_rate
        print(np.mean(frame_rate))