import sys

import math
import torch
import torch.nn as nn
import numpy as np
from .op_dynamic import DynamicConv2d, DynamicBatchNorm2d, DynamicLinear
from torch.nn.parameter import Parameter
import pdb
import random
# for multi channel branch only partly
__all__ = ['ResNet20']

def Conv(in_planes, planes, kernel_size, stride, padding):
    return nn.Sequential(
        DynamicConv2d(in_channels=in_planes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        DynamicBatchNorm2d(planes))

def get_branch_id(branch_channels, current_channels, b_num, use_branch):
    if not use_branch:
        return b_num
    idx = (current_channels-1) // branch_channels
    return idx

##########   Original_Module   ##########
class Block_Conv1(nn.Module):
    def __init__(self, in_planes, planes, stride=1, use_branch=False, b_num=2):
        super(Block_Conv1, self).__init__()
        self.conv1_input_channel = in_planes
        self.output_channel = planes
        self.use_branch = use_branch
        self.b_num = b_num
        self.b_channels = planes // b_num

        #defining conv1
        self.conv1 = Conv(self.conv1_input_channel, self.output_channel, kernel_size=3, stride=stride, padding=1)
        if self.use_branch:
            self.conv1_twin = nn.ModuleList()
            for i in range(b_num):
                self.conv1_twin.add_module('b_{}'.format(i), Conv(self.conv1_input_channel, self.b_channels * (i + 1), kernel_size=3, stride=stride, padding=1))
        self.relu = nn.ReLU()

    def set_mask(self, channel_width):
        self.channel_width = channel_width
        b_idx = get_branch_id(self.b_channels, channel_width, self.b_num, self.use_branch)
        self.b_idx = b_idx
        if b_idx == self.b_num:
            self.conv1[0].set_mask(channel_width)
        else:
            getattr(self.conv1_twin, 'b_{}'.format(b_idx))[0].set_mask(channel_width)

    # def reset_mask(self):
    #     self.conv1[0].reset_mask()
    #     if self.use_branch:
    #         self.conv1_twin[0].reset_mask()

    def copy_weight2twin(self, requires_grad=True):
        if self.use_branch:
            for i in range(self.b_num):
                conv_layer = getattr(self.conv1_twin, 'b_{}'.format(i))
                copy_conv_weight(self.conv1[0],  conv_layer[0])
                copy_bn_weight(self.conv1[1],  conv_layer[1])

    def load_weight_from_previous_stage(self, pre_model):
        copy_conv_weight(pre_model.conv1[0], self.conv1[0])
        copy_bn_weight(pre_model.conv1[1], self.conv1[1])
        for i in range(self.b_num):
            pre_m = getattr(pre_model.conv1_twin, 'b_{}'.format(i))
            cur_m1 = getattr(self.conv1_twin, 'b_{}'.format(i))
            copy_conv_weight(pre_m[0], cur_m1[0])
            copy_bn_weight(pre_m[1], cur_m1[1])

    def forward(self, x):
        if self.b_idx == self.b_num:
            out = self.conv1(x)
        else:
            out = getattr(self.conv1_twin, 'b_{}'.format(self.b_idx))(x)
        return self.relu(out)

class BasicBolock(nn.Module):
    def __init__(self, len_list ,stride=1, group=1, downsampling=False, use_branch=False, b_num=2):
        super(BasicBolock,self).__init__()

        global IND

        self.downsampling = downsampling
        self.adaptive_pooling = False
        self.len_list = len_list
        #self.max_channels = [self.len_list[IND - 1], self.len_list[IND], self.len_list[IND + 1]]
        self.use_branch = use_branch
        self.b_num = b_num
        self.b_channels = [self.len_list[IND] // b_num, self.len_list[IND + 1] // b_num]

        self.conv1 = Conv(self.len_list[IND - 1], self.len_list[IND], kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv(self.len_list[IND], self.len_list[IND + 1], kernel_size=3, stride=1, padding=1)
        if self.use_branch:
            self.conv1_twin = nn.ModuleList()
            self.conv2_twin = nn.ModuleList()
            self.downsample_twin = nn.ModuleList()
            for i in range(b_num):
                self.conv1_twin.add_module('b_{}'.format(i), Conv(self.len_list[IND - 1], self.b_channels[0] * (i + 1), kernel_size=3, stride=stride, padding=1))
                self.conv2_twin.add_module('b_{}'.format(i), Conv(self.len_list[IND], self.b_channels[1] * (i + 1), kernel_size=3, stride=1, padding=1))
                self.downsample_twin.add_module('b_{}'.format(i), Conv(self.len_list[IND - 1], self.b_channels[1] * (i + 1), kernel_size=1, stride=stride, padding=0))

        # if self.downsampling :
        #     self.downsample = Conv(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0)
        # elif not self.downsampling and (self.len_list[IND - 1] != self.len_list[IND + 1]):
        self.downsample = Conv(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0)    
        self.downsampling = True
        self.relu = nn.ReLU()
        self.p_relu = nn.PReLU()
        IND += 2

    def set_mask(self, channel_list):
        self.channel_list = channel_list
        b_idx = []
        b_idx.append(get_branch_id(self.b_channels[0], channel_list[0], self.b_num, self.use_branch))
        b_idx.append(get_branch_id(self.b_channels[1], channel_list[1], self.b_num, self.use_branch))
        self.b_idx = b_idx
        if b_idx[0] == self.b_num: 
            self.conv1[0].set_mask(channel_list[0])
        else:
            getattr(self.conv1_twin, 'b_{}'.format(b_idx[0]))[0].set_mask(channel_list[0])

        if b_idx[1] == self.b_num: 
            self.conv2[0].set_mask(channel_list[1])
            self.downsample[0].set_mask(channel_list[1])
        else:
            getattr(self.conv2_twin, 'b_{}'.format(b_idx[1]))[0].set_mask(channel_list[1])
            getattr(self.downsample_twin, 'b_{}'.format(b_idx[1]))[0].set_mask(channel_list[1])

    def copy_weight2twin(self, requires_grad=True):
        if self.use_branch:
            for i in range(self.b_num):
                conv1_layer = getattr(self.conv1_twin, 'b_{}'.format(i))
                conv2_layer = getattr(self.conv2_twin, 'b_{}'.format(i))
                down_layer = getattr(self.downsample_twin, 'b_{}'.format(i))
                copy_conv_weight(self.conv1[0], conv1_layer[0])
                copy_bn_weight(self.conv1[1], conv1_layer[1])
                copy_conv_weight(self.conv2[0], conv2_layer[0])
                copy_bn_weight(self.conv2[1], conv2_layer[1])
                copy_conv_weight(self.downsample[0], down_layer[0])
                copy_bn_weight(self.downsample[1], down_layer[1])

    def load_weight_from_previous_stage(self, pre_model):
        copy_conv_weight(pre_model.conv1[0], self.conv1[0])
        copy_bn_weight(pre_model.conv1[1], self.conv1[1])
        copy_conv_weight(pre_model.conv2[0], self.conv2[0])
        copy_bn_weight(pre_model.conv2[1], self.conv2[1])
        copy_conv_weight(pre_model.downsample[0], self.downsample[0])
        copy_bn_weight(pre_model.downsample[1], self.downsample[1])
        twin_times = self.b_num // pre_model.b_num
        for i in range(pre_model.b_num):
            pre_conv1 = getattr(pre_model.conv1_twin, 'b_{}'.format(i))
            pre_conv2 = getattr(pre_model.conv2_twin, 'b_{}'.format(i))
            pre_downsample = getattr(pre_model.downsample_twin, 'b_{}'.format(i))
            for j in range(twin_times):
                cur_conv1 = getattr(self.conv1_twin, 'b_{}'.format(i*twin_times + j))
                copy_conv_weight(pre_conv1[0], cur_conv1[0])
                copy_bn_weight(pre_conv1[1], cur_conv1[1])

                cur_conv2 = getattr(self.conv2_twin, 'b_{}'.format(i*twin_times + j))
                copy_conv_weight(pre_conv2[0], cur_conv2[0])
                copy_bn_weight(pre_conv2[1], cur_conv2[1])

                cur_downsample = getattr(self.downsample_twin, 'b_{}'.format(i*twin_times + j))
                copy_conv_weight(pre_downsample[0], cur_downsample[0])
                copy_bn_weight(pre_downsample[1], cur_downsample[1])

        self.p_relu.weight.data.copy_(pre_model.p_relu.weight[...])
        #pdb.set_trace()
        print('load done')

    def forward(self, x):
        residual = x
        if self.b_idx[0] == self.b_num:
            x = self.relu(self.conv1(x))
        else:
            x = getattr(self.conv1_twin, 'b_{}'.format(self.b_idx[0]))(x)
            x = self.relu(x)

        if self.b_idx[1] == self.b_num:
            out = self.conv2(x)
        else:
            out = getattr(self.conv2_twin, 'b_{}'.format(self.b_idx[1]))(x)
            #out = self.conv2_twin(x)
        if self.downsampling:
            if self.b_idx[1] == self.b_num:
                residual = self.downsample(residual)
            else:
                residual = getattr(self.downsample_twin, 'b_{}'.format(self.b_idx[1]))(residual)
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


def copy_conv_weight(conv1, conv_twin):
    twin_out = conv_twin.out_channels
    conv_twin.weight.data.copy_(conv1.weight[:twin_out,:,:,:])
    if conv_twin.bias is not None:
        conv_twin.bias.data.copy_(conv1.bias[:twin_out])

def copy_fc_weight(fc, fc_twin):
    twin_in = fc_twin.in_features
    fc_twin.weight.data.copy_(fc.weight[:,:twin_in])
    if fc_twin.bias is not None:
        fc_twin.bias.data.copy_(fc.bias[:])

def copy_bn_weight(bn1, bn1_twin):
    twin_out = bn1_twin.num_features
    bn1_twin.running_mean.data.copy_(bn1.running_mean[:twin_out]) if bn1.running_mean is not None else None
    bn1_twin.running_var.data.copy_(bn1.running_var[:twin_out]) if bn1.running_var is not None else None
    bn1_twin.weight.data.copy_(bn1.weight[:twin_out]) if bn1.weight is not None else None
    bn1_twin.bias.data.copy_(bn1.bias[:twin_out]) if bn1.bias is not None else None

class ResNet(nn.Module):
    def __init__(self, blocks, len_list, use_branch=[True, True, True, True], 
                       module_type=BasicBolock, num_classes=10, expansion=1):
        super(ResNet,self).__init__()
        self.block = module_type
        self.len_list = len_list
        self.expansion = expansion
        self.fc_use_branch = True
        global IND
        self.conv1 = Block_Conv1(in_planes=3, planes=self.len_list[0], use_branch=use_branch[0], b_num=4)
        IND = 1
        self.layer1 = self.make_layer(self.len_list, block=blocks[0], block_type=self.block, stride=1, use_branch=use_branch[1], b_num=4)
        self.layer2 = self.make_layer(self.len_list, block=blocks[1], block_type=self.block, stride=2, use_branch=use_branch[2], b_num=8)
        self.layer3 = self.make_layer(self.len_list, block=blocks[2], block_type=self.block, stride=2, use_branch=use_branch[3], b_num=16)

        #print('IND is {}'.format(IND))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = DynamicLinear(self.len_list[-2], num_classes)
        self.fc_b_num = 16
        self.fc_b_channels = 64 // self.fc_b_num
        self.fc_twin = nn.ModuleList()
        for i in range(self.fc_b_num):
            self.fc_twin.add_module('fc_b_{}'.format(i), DynamicLinear(self.fc_b_channels * (i+1), num_classes))

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

    def make_layer(self, len_list, block, block_type, stride, use_branch=False, b_num=2):
        layers = []
        layers.append(block_type(len_list, stride, downsampling=True, use_branch=use_branch, b_num=b_num))
        for i in range(1, block):
            layers.append(block_type(len_list, use_branch=use_branch, b_num=b_num))
        return nn.Sequential(*layers)

    def copy_weight2twin(self, requires_grad=True):
        for m in self.modules():
            if isinstance(m, Block_Conv1) or isinstance(m, BasicBolock):
                m.copy_weight2twin(requires_grad=requires_grad)
        for i in range(self.fc_b_num):
            fc_twin_b = getattr(self.fc_twin, 'fc_b_{}'.format(i))
            copy_fc_weight(self.fc, fc_twin_b)
            # if not requires_grad:
            #     self.fc.weight.requires_grad = False
            #     self.fc.bias.requires_grad = False

    def load_weight_from_previous_stage(self, pre_model):
        self.conv1.load_weight_from_previous_stage(pre_model.conv1)
        for i in range(len(self.layer1)):
            self.layer1[i].load_weight_from_previous_stage(pre_model.layer1[i])

        for i in range(len(self.layer2)):
            self.layer2[i].load_weight_from_previous_stage(pre_model.layer2[i])

        for i in range(len(self.layer3)):
            self.layer3[i].load_weight_from_previous_stage(pre_model.layer3[i])

        copy_fc_weight(pre_model.fc, self.fc)
        fc_times = self.fc_b_num // pre_model.fc_b_num
        for i in range(pre_model.fc_b_num):
            #pdb.set_trace()
            pre_fc = getattr(pre_model.fc_twin, 'fc_b_{}'.format(i))
            for j in range(fc_times):
                cur_fc = getattr(self.fc_twin, 'fc_b_{}'.format(i*fc_times + j))
                copy_fc_weight(pre_fc, cur_fc)

    def set_use_branch(self, use_branch):
        for m in self.modules():
            if isinstance(m, Block_Conv1) or isinstance(m, BasicBolock):
                m.use_branch = use_branch
        self.fc_use_branch = use_branch

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
        idx = get_branch_id(self.fc_b_channels, x.size(1), self.fc_b_num, self.fc_use_branch)
        if idx == self.fc_b_num:
            x = self.fc(x)
        else:
            x = getattr(self.fc_twin, 'fc_b_{}'.format(idx))(x)
        return x

##########   ResNet Model   ##########
#default block type --- BasicBolock for ResNet20;

def ResNet20_SPN(CLASS, len_list=[16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]):
    return ResNet([3, 3, 3], len_list=len_list, num_classes=CLASS, module_type=BasicBolock)