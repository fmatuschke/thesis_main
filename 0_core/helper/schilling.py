#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" From Schilling et al. 10.1016/j.neuroimage.2017.10.046
"""

import numpy as np

# def linear_sh_coeff_index(l, m):
#     # num_coeff = ((l // 2) + 1) * (2 * (l // 2) + 1)
#     return l**2 + m # not pli index


def angular_correlation_coefficient(sh0, sh1):
    return np.sum(np.multiply(sh0, sh1)) / (np.sqrt(
        np.sum(np.multiply(sh0, sh0))) * np.sqrt(np.sum(np.multiply(sh1, sh1))))


def jensen_shannon_divergence(sh0, sh1):
    pass
