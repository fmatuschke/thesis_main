#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" From Schilling et al. 10.1016/j.neuroimage.2017.10.046
"""

import numpy as np


def angular_correlation_coefficient(sh0, sh1):

    divisor = (np.sqrt(np.sum(np.multiply(sh0, sh0))) *
               np.sqrt(np.sum(np.multiply(sh1, sh1))))

    return np.sum(np.multiply(sh0, sh1)) / divisor if divisor else 0


def jensen_shannon_divergence(sh0, sh1):
    return (kullback_leibler_divergence(p, m) +
            kullback_leibler_divergence(q, m)) / 2


def kullback_leibler_divergence(p, q):
    return np.sum(np.multiply(np.log(np.divide(p, q))))
