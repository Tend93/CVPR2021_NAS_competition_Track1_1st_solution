import sys

import math
import torch
import torch.nn as nn
import numpy as np
from .op_dynamic import DynamicConv2d, DynamicBatchNorm2d, DynamicLinear
import pdb
import random

__all__ = ['ResNet20']

def Conv(in_planes, planes, kernel_size, stride, padding):
    return nn.Sequential(
        DynamicConv2d(in_channels=in_planes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        DynamicBatchNorm2d(planes))

##########   Original_Module   ##########
class Block_Conv1(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Block_Conv1, self).__init__()
        self.conv1_input_channel = in_planes
        self.output_channel = planes

        #defining conv1
        self.conv1 = Conv(self.conv1_input_channel, self.output_channel, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()

    def set_mask(self, channel_width):
        self.conv1[0].set_mask(channel_width)

    def reset_mask(self):
        self.conv1[0].reset_mask()

    def forward(self, x):
        out = self.conv1(x)
        return self.relu(out)

class BasicBolock(nn.Module):
    def __init__(self, len_list ,stride=1, group=1, downsampling=False):
        super(BasicBolock,self).__init__()

        global IND

        self.downsampling = downsampling
        self.adaptive_pooling = False
        self.len_list = len_list

        self.conv1 = Conv(self.len_list[IND - 1], self.len_list[IND], kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv(self.len_list[IND], self.len_list[IND + 1], kernel_size=3, stride=1, padding=1)

        # if self.downsampling :
        #     self.downsample = Conv(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0)
        # elif not self.downsampling and (self.len_list[IND - 1] != self.len_list[IND + 1]):
        self.downsample = Conv(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0)
        self.downsampling = True
        self.relu = nn.ReLU()
        self.p_relu = nn.PReLU()
        IND += 2

    def set_mask(self, channel_list):
        self.conv1[0].set_mask(channel_list[0])
        self.conv2[0].set_mask(channel_list[1])
        self.downsample[0].set_mask(channel_list[1])

    def reset_mask(self):
        self.conv1[0].reset_mask()
        self.conv2[0].reset_mask()
        self.downsample[0].reset_mask()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        out = self.conv2(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.p_relu(out)
        return out


def _calculate_fan_in_and_fan_out(tensor, op):
    op = op.lower()
    valid_modes = ['linear', 'conv']
    if op not in valid_modes:
        raise ValueError("op {} not supported, please use one of {}".format(op, valid_modes))
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if op == 'linear':
        num_input_fmaps = tensor.shape[0]
        num_output_fmaps = tensor.shape[1]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    #pdb.set_trace()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, op, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, op)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_(tensor, op='linear', a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, op, mode)
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    # with paddle.no_grad():
    #     return paddle.assign(paddle.uniform(tensor.shape, min=-bound, max=bound), tensor)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class ResNet(nn.Module):
    def __init__(self, blocks, len_list, module_type=BasicBolock, num_classes=10, expansion=1):
        super(ResNet,self).__init__()
        self.block = module_type
        self.len_list = len_list
        self.expansion = expansion
        global IND
        self.conv1 = Block_Conv1(in_planes=3, planes=self.len_list[0])
        IND = 1
        self.layer1 = self.make_layer(self.len_list, block=blocks[0], block_type=self.block, stride=1)
        self.layer2 = self.make_layer(self.len_list, block=blocks[1], block_type=self.block, stride=2)
        self.layer3 = self.make_layer(self.len_list, block=blocks[2], block_type=self.block, stride=2)

        #print('IND is {}'.format(IND))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = DynamicLinear(self.len_list[-2], num_classes)

        for m in self.modules():
            if isinstance(m, DynamicConv2d) or isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, op='conv', mode='fan_out', nonlinearity='relu')
            elif isinstance(m, DynamicBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                with torch.no_grad():
                    m.weight.fill_(1)
                    m.bias.zero_()
            elif isinstance(m, DynamicLinear) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight, op='linear', mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    m.bias.zero_()

    def make_layer(self, len_list, block, block_type, stride):
        layers = []
        layers.append(block_type(len_list, stride, downsampling=True))
        for i in range(1, block):
            layers.append(block_type(len_list))
        return nn.Sequential(*layers)

    def set_random_path(self):
        channel_list = []
        for i in range(1, 21):
            if 0 < i <=7:
                channel_list.append(random.choice([ 4, 8, 12, 16]))
            elif 7 < i <= 13:
                channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32]))
            elif 13 < i <= 19:
                channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
            else:
                channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
        #print(channel_list)
        self.set_mask(channel_list)

    def get_random_path(self, level=0):
        channel_list = []
        channel_table = [ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        for i in range(1, 21):
            if 0 < i <=7:
                select_num = 3
                channel_list.append(channel_table[random.randint(level, select_num)])
            elif 7 < i <= 13:
                select_num = 7
                channel_list.append(channel_table[random.randint(level * 2, select_num)])
            elif 13 < i <= 19:
                select_num = 15
                channel_list.append(channel_table[random.randint(level * 4, select_num)])
            else:
                select_num = 15
                channel_list.append(channel_table[random.randint(level * 4, select_num)])
        return channel_list

    def get_constraint_path(self, model_code, const_num, const_values=[4, 8]):
        const_indices = [i for i in range(6)]
        const_indices = random.sample(const_indices, const_num)
        for i in const_indices:
            model_code[i + 13] = random.choice(const_values) 
        return model_code

    def set_min_path(self):
        channel_list = []
        for i in range(1, 21):
            if 0 < i <=7:
                channel_list.append(min([ 4, 8, 12, 16]))
            elif 7 < i <= 13:
                channel_list.append(min([ 4, 8, 12, 16, 20, 24, 28, 32]))
            elif 13 < i <= 19:
                channel_list.append(min([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
            else:
                channel_list.append(min([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
        #print(channel_list)
        self.set_mask(channel_list)

    def set_max_path(self):
        channel_list = []
        for i in range(1, 21):
            if 0 < i <=7:
                channel_list.append(max([ 4, 8, 12, 16]))
            elif 7 < i <= 13:
                channel_list.append(max([ 4, 8, 12, 16, 20, 24, 28, 32]))
            elif 13 < i <= 19:
                channel_list.append(max([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
            else:
                channel_list.append(max([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
        #print(channel_list)
        self.set_mask(channel_list)

    def set_mask(self, channel_list):
        channel_list = [c+1 if c==4 else c for c in channel_list]
        channel_list = [c+1 if c==8 else c for c in channel_list]

        self.conv1.set_mask(channel_list[0])
        idx = 1
        for i in range(len(self.layer1)):
            self.layer1[i].set_mask(channel_list[idx:idx+2])
            idx += 2
        for i in range(len(self.layer2)):
            self.layer2[i].set_mask(channel_list[idx:idx+2])
            idx += 2
        for i in range(len(self.layer3)):
            self.layer3[i].set_mask(channel_list[idx:idx+2])
            idx += 2
        assert idx == 19
        #print(channel_list)
        #self.fc.set_mask(channel_list[-2])

    def finalize_net(self, data):
        self.forward(data)

        module_dict = {}
        for name, m in self.named_modules():
            if hasattr(m, "finalize"):
                m = m.finalize()
                module_dict[name] = m
        for name, m in module_dict.items():
            name_strs = name.split('.')
            self_module = self
            for st in name_strs[:-1]:
                self_module = getattr(self_module, st)
            setattr(self_module, name_strs[-1], m)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.avgpool(out3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

##########   ResNet Model   ##########
#default block type --- BasicBolock for ResNet20;

def ResNet20_SPN(CLASS, len_list=[16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]):
    return ResNet([3, 3, 3], len_list=len_list, num_classes=CLASS, module_type=BasicBolock)

# from paddle.vision.transforms import (
#         ToTensor, RandomHorizontalFlip, RandomResizedCrop, SaturationTransform,
#         HueTransform, BrightnessTransform, ContrastTransform
#     )
# import random
# channel_list = []
# for i in range(1, 21):
#     if 0 < i <=7:
#         channel_list.append(random.choice([ 4, 8, 12, 16]))
#     elif 7 < i <= 13:
#         channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32]))
#     elif 13 < i <= 19:
#         channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64]))
#     else:
#         channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64]))
        
# resnet20 = ResNet20(100, channel_list)
# model = paddle.Model(resnet20)
# model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
#             paddle.nn.CrossEntropyLoss(),
#             paddle.metric.Accuracy())
# data_file = './data/data76994/cifar-100-python.tar.gz'
# transforms = paddle.vision.transforms.Compose([
#     RandomHorizontalFlip(),
#     RandomResizedCrop((32, 32)),
#     SaturationTransform(0.2),
#     BrightnessTransform(0.2), ContrastTransform(0.2),
#     HueTransform(0.2), ToTensor()
# ])
# train_dataset = paddle.vision.datasets.Cifar100(data_file, mode='train', transform=transforms)
# test_dataset = paddle.vision.datasets.Cifar100(data_file, mode='test', transform=ToTensor())
# model.fit(train_dataset, test_dataset, epochs=100, batch_size=64, verbose=1)