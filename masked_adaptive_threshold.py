# coding=utf-8
"""
adapted from
https://stackoverflow.com/questions/9842127/using-a-mask-with-an-adaptive-threshold
"""

import numpy as np
from scipy import signal

import cfg


def thresh(a, b, max_value, c):
    return max_value if a > b - c else 0


def mask(a, b):
    return a if b > 100 else 0


def unmask(a, b, c):
    return b if c > 100 else a


v_unmask = np.vectorize(unmask)
v_mask = np.vectorize(mask)
v_thresh = np.vectorize(thresh)


def block_size(size):
    block = np.ones((size, size), dtype='d')
    block[(size - 1) // 2, (size - 1) // 2] = 0
    return block


def get_number_neighbours(mask, block):
    """returns number of unmasked neighbours of every element within block"""
    mask //= 255
    return signal.convolve2d(mask, block, mode='same', boundary='symm')


def masked_adaptive_threshold(image, mask, size, c):
    """thresholds only using the unmasked elements"""
    block = block_size(size)
    conv = signal.convolve2d(image, block, mode='same', boundary='symm')
    mean_conv = conv / get_number_neighbours(mask, block)
    return v_thresh(image, mean_conv, cfg.max_intensity, c)