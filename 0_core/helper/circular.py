import numpy as np
from scipy.stats import circmean
from fastpli.analysis.orientation import remap_half_sphere_z


def remap(data, high=2 * np.pi, low=0):
    high, low = max(high, low), min(high, low)
    data = np.array(data, copy=True)
    data -= low
    data %= (high - low)
    data[data < 0] += (high - low)
    data += low
    return data


def center_mean(data, high=2 * np.pi, low=0, margin=0):
    cm = circmean(data, high, low)
    data = remap(data, cm - (high - low) / 2, cm + (high - low) / 2)
    return data
