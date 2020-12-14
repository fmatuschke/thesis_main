import numpy as np
from scipy.stats import circmean
from fastpli.analysis.orientation import remap_orientation


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


# TODO: check
# def orientation_sph_plot(phi, theta):
#     # rotate phi [0,360), theta [0,90) to [-90, 90), [0,180)
#     phi, theta = remap_orientation(phi, theta)

#     # phi > 180 to [-180,180)
#     mask = phi >= np.pi
#     phi[mask] -= 2 * np.pi

#     # phi > 90, phi < 90 to theta > 90
#     mask = phi >= np.pi / 2
#     theta[mask] = np.pi - theta[mask]
#     phi[mask] -= np.pi
#     mask = phi < -np.pi / 2
#     theta[mask] = np.pi - theta[mask]
#     phi[mask] += np.pi
#     theta[theta >= np.pi] -= np.pi

#     phi_theta = np.pi - (np.pi / 2 - theta)

#     return phi, phi_theta
