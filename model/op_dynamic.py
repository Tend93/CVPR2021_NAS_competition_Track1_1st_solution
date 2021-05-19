# -*- coding:utf-8 -*-
# @yangziwei5 2021-01-08
# dynamic version of Conv2d/Linear/BatchNorm2d
# for NAS, searching channel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from collections import OrderedDict
from functools import partial
import pdb

DYNAMIC_MODULES = {
    'Conv2d':  nn.Conv2d,
    'Linear': nn.Linear,
    'BatchNorm2d': nn.BatchNorm2d,
}

#  --- base class for dynamic layers ---
class DynamicBaseLayer(object):
    def __init__(self):
        self.reset_mask()

    def set_mask(self, *args, **kwargs):
        raise NotImplementedError()

    def reset_mask(self):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class DynamicConv2d(DYNAMIC_MODULES['Conv2d'], DynamicBaseLayer):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, **kwargs):
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, 
                                            groups=groups, **kwargs)
        DynamicBaseLayer.__init__(self)

    def _select_channels(self, channels_width):
        assert len(channels_width) == 2, "The width information for input channels and output channels, but not get 2 elements."
        return self.weight[:channels_width[1], :channels_width[0], :, :].contiguous(), \
                              self.bias[:channels_width[1]].contiguous() if self.bias is not None else None
    def set_mask(self, channels_width):
        if self.channels_width is not None:
            self.channels_width = [self.in_channels, channels_width] if isinstance(channels_width, int) else channels_width 

    def reset_mask(self):
        self.channels_width = [self.in_channels, self.out_channels]

    def forward(self, inputs):
        # adjust the input mask adptively.
        self.channels_width[0] = inputs.shape[1]
        filters, bias = self._select_channels(self.channels_width)
        return F.conv2d(inputs, filters, bias=bias, padding=self.padding, 
                         stride=self.stride, dilation=self.dilation, groups=self.groups)

class DynamicLinear(DYNAMIC_MODULES['Linear'], DynamicBaseLayer):
    """
    dynamic implementation of nn.Linear, for searching channels
    usage:
        #do set_mask before forward() when searching channels
    
    """
    def __init__(self, in_features, out_features, bias=True):
        super(DynamicLinear, self).__init__(
            in_features, out_features, bias=bias
        )

        # channels width of input and output features
        DynamicBaseLayer.__init__(self)

    def _select_channels(self, channels_width):
        assert len(channels_width) == 2, "The width information for input channels and output channels, but not get 2 elements."
        return self.weight[:channels_width[1], :channels_width[0]].contiguous(), self.bias[:channels_width[1]].contiguous() if self.bias is not None else None

    def set_mask(self, channels_width):
        if channels_width is not None:
            self.channels_width = [self.in_features, channels_width] if isinstance(channels_width, int) else channels_width

    def reset_mask(self):
        self.channels_width = [self.in_features, self.out_features]

    def forward(self, inputs):
        # adjust the input mask adptively.
        self.channels_width[0] = inputs.shape[1]
        filters, bias = self._select_channels(self.channels_width)
        return F.linear(inputs, filters, bias)

class DynamicBatchNorm2d(DYNAMIC_MODULES['BatchNorm2d'], DynamicBaseLayer):
    """
    dynamic implementation of nn.BatchNorm2d, for searching channels
    usage:
        
        #do with_arithmetic_mean() to change to arithmetic mean.
    
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True):
        super(DynamicBatchNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        
        DynamicBaseLayer.__init__(self)

    def _select_channels(self, channels_width):
        assert channels_width > 0, "channels_width must be larger than 0."
        assert channels_width <= self.num_features, "channels_width must be smaller than self.num_features."
        running_mean = self.running_mean[:channels_width] if self.running_mean is not None else None
        running_var = self.running_var[:channels_width] if self.running_var is not None else None
        weight = self.weight[:channels_width] if self.weight is not None else None
        bias = self.bias[:channels_width] if self.bias is not None else None

        return running_mean, running_var, weight, bias

    def set_mask(self, channels_width):
        pass

    def reset_mask(self):
        self.channels_width = self.num_features

    def forward_mask(self, inputs, channels_width=None):

        running_mean, running_var, weight, bias = self._select_channels(channels_width)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            inputs, running_mean, running_var, weight, bias, self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def forward(self, inputs):
        # adjust the input mask adptively.
        self.channels_width = inputs.shape[1]
        return self.forward_mask(inputs, self.channels_width)