import torch
import math
import random
import numpy as np
import numbers
import types
import collections
import warnings
import pdb

from PIL import Image, ImageOps, ImageEnhance
from . import functional_extra as F
from torchvision.transforms import functional as F_ori

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

__all__ = ["SaturationTransform", "HueTransform", "BrightnessTransform", 
           "ContrastTransform", "RandomResizedCrop", 'RandomCrop', 'RandomRotation',
           'RandomHorizontalFlip']

def _check_input(value,
                 name,
                 center=1,
                 bound=(0, float('inf')),
                 clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(
                    name))
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name,
                                                                     bound))
    else:
        raise TypeError(
            "{} should be a single number or a list/tuple with lenght 2.".
            format(name))

    if value[0] == value[1] == center:
        value = None
    return value

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size) if isinstance(size, int) else size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size[0], img.size[1]
        area = height * width

        for attempt in range(10):
            target_area = np.random.uniform(*scale) * area
            log_ratio = tuple(math.log(x) for x in ratio)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            # return whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_ori.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class SaturationTransform(object):

    def __init__(self, value):
        self.value = _check_input(value, 'saturation')

    def __call__(self, img):
        if self.value is None:
            return img

        saturation_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_saturation(img, saturation_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(value={0})'.format(self.value)


class HueTransform(object):

    def __init__(self, value):
        self.value = _check_input(
            value, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def __call__(self, img):
        if self.value is None:
            return img

        hue_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_hue(img, hue_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(value={0})'.format(self.value)

class BrightnessTransform(object):

    def __init__(self, value):
        self.value = _check_input(value, 'brightness')

    def __call__(self, img):
        if self.value is None:
            return img

        brightness_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_brightness(img, brightness_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(value={0})'.format(self.value)

class ContrastTransform(object):

    def __init__(self, value):
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = _check_input(value, 'contrast')

    def __call__(self, img):
        if self.value is None:
            return img

        contrast_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_contrast(img, contrast_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(value={0})'.format(self.value)

class RandomCrop(object):

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_param(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        w, h = img.size

        # pad the width if needed
        if self.pad_if_needed and w < self.size[1]:
            img = F.pad(img, (self.size[1] - w, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and h < self.size[0]:
            img = F.pad(img, (0, self.size[0] - h), self.fill,
                        self.padding_mode)

        i, j, h, w = self._get_param(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, interpolation='nearest', expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string