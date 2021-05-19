import numpy as np
import torch
import random
import os, sys
from torchvision import datasets
from torchvision import transforms as T
from . import transforms_extra as ET

class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        img = torch.from_numpy(img.astype('float32'))
        return img

class RandomApply(object):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.p = p
        self.transform = transform

    def __call__(self, img):
        if self.p < random.random():
            return img
        img = self.transform(img)
        return img

def get_dataset(data_name):
    train_set, val_set, test_set = None, None, None
    if data_name == 'cifar100':
        root = './data/'
        normalize = T.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                std=[0.1942, 0.1918, 0.1958])
        train_set = datasets.CIFAR100(root, True, T.Compose([
            ET.RandomCrop(32, padding=4),
            RandomApply(ET.BrightnessTransform(0.1)),
            RandomApply(ET.ContrastTransform(0.1)),
            ET.RandomHorizontalFlip(),
            ET.RandomRotation(15),
            ToArray(),
            normalize,
        ]))

        val_set = datasets.CIFAR100(root, False, T.Compose([
            ToArray(),
            normalize,
        ]))
    else:
        print('Dataset type wrong. Only cifar10, cifar100, imagenet, simagenet are available now.')
        return
    return train_set, val_set, test_set