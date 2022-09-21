import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utilsa import *
from resnet_dilated import ResnetDilated
from aspp import DeepLabHead1
import torch.optim as optim
import torchvision.transforms as transforms
from options import Options
import pickle
import math
import torchvision.models as models
from torch.autograd.function import Function

class GradReverse(Function):

    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        #　其实就是传入dict{'lambd' = lambd} 
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        # 传入的是tuple，我们只需要第一个
        return grad_output[0] * -ctx.lambd, None

    # 这样写是没有warning，看起来很舒服，但是显然是多此一举咯，所以也可以改写成

    def backward(ctx, grad_output):
        # 直接传入一格数
        return grad_output * -ctx.lambd, None

class MBSNET(nn.Module):
    def __init__(self):
        super(MBSNET, self).__init__()

        resnet = models.resnet18(True)       
        checkpoint = torch.load('emrnet/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        backbone = ResnetDilated(resnet)
        self.tasks = ['br1', 'br2', 'br3', 'mainbr']
        self.num_out_classes = {'br1': 8, 'br2': 8, 'br3': 8, 'mainbr': 3}
        num_classes = 8

        self.decoders = nn.ModuleList([DeepLabHead1(512, self.num_out_classes[t]) for t in self.tasks])

        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        self.avp = nn.AdaptiveAvgPool2d(1)
        self.arrangement = nn.PixelShuffle(16)
        self.fc = nn.Linear(512, num_classes)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        self.shared_layer4_b1 = backbone.layer4[:-1]
        self.shared_layer4_t1 = backbone.layer4[-1]

        self.shared_layer4_b2 = backbone.layer4[:-1]
        self.shared_layer4_t2 = backbone.layer4[-1]

        self.shared_layer4_b3 = backbone.layer4[:-1]
        self.shared_layer4_t3 = backbone.layer4[-1]

        
    def forward(self, x):
        x = self.shared_conv(x)
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)
        u_4_b1 = self.shared_layer4_b1(u_3_t)
        u_4_t1 = self.shared_layer4_t1(u_4_b1)
        u_4_b2 = self.shared_layer4_b2(u_3_t)
        u_4_t2 = self.shared_layer4_t2(u_4_b2)
        u_4_b3 = self.shared_layer4_b3(u_3_t)
        u_4_t3 = self.shared_layer4_t3(u_4_b3)

        u_4_t = u_4_t - 0.5 * u_4_t3
        u_4_t1 = u_4_t1 - 0.5 * u_4_t3
        u_4_t2 = u_4_t2 - 0.5 * u_4_t3
        #u_4_t3 = u_4_t3 - 0.15*u - 0.15*u1 - 0.15*u2

        u_4_t = self.avp(u_4_t)
        u_4_t1 = self.avp(u_4_t1)
        u_4_t2 = self.avp(u_4_t2)
        u_4_t3 = self.avp(u_4_t3)
        #feature = [0 for _ in self.tasks]
        u_4_t = u_4_t.view(u_4_t.size(0), -1)
        u_4_t1 = u_4_t1.view(u_4_t1.size(0), -1)
        u_4_t2 = u_4_t2.view(u_4_t2.size(0), -1)
        feature = u_4_t3 = u_4_t3.view(u_4_t3.size(0), -1)
        # Task specific decoders
        out = [0 for _ in self.tasks]
        out[0] = self.fc(u_4_t)
        out[1] = self.fc1(u_4_t1)
        out[2] = self.fc2(u_4_t2)
        out[3] = self.fc3(u_4_t3)
        return feature, out