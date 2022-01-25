from __future__ import division
import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from imgaug import augmenters as iaa  # for bbox augmentation
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from . import functional as F

class RandomMask(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, max_patch_size=10, max_patch_num=5, p=0.2):
        self.max_patch_size = 10
        self.max_patch_num = 5
        self.shapes= [(10, 20), (20, 10), (15, 15)]
        self.fillvalues = np.array([0.485, 0.456, 0.406]) * 255.
        self.p = p

    def __call__(self, img, bbox, fillvalue=None):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        if np.random.uniform() < self.p:
            return img, bbox
        
        img = np.array(img) # h, w, 3 instead of W, H
        
        shape = np.random.choice(self.shapes)
        h, w = shape
        H, W = img.shape[:2]
        l, t = np.random.randint(0, W-w), np.random.randint(0, H-h)
        img[t:t+h, l:l+w] = fillvalue if fillvalue is not None else self.fillvalue
        img = img.astype(np.uint8)
        img = Image.fromarray(img)

        return img, bbox



    def __repr__(self):
        format_string = self.__class__.__name__ + '(' + f"p={self.p}" ')'
        return format_string
    
    
class RandomAffine(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
            
        # NOTE: degree is suggested to be in [-10, 10]

        self.resample = resample
        self.expand = expand
        self.center = center
        self.rot = None


    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)
        rot = iaa.Affine(rotate=angle)

        img = np.array(img)

        img_aug, bbox_aug = rot(image=img, bounding_boxes=bbox)

        img_aug = img_aug.astype(np.uint8)
        img_aug = Image.fromarray(img_aug)
        # return F.rotate(img, angle, self.resample, self.expand, self.center)
        return img_aug, bbox_aug 
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string