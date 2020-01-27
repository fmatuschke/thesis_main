import numpy as np
import h5py
import os
import sys
import glob

import matplotlib.pyplot as plt
import scipy.stats

from tqdm import tqdm

# reproducability
np.random.seed(42)


def calcMeanStdAngle(angles):
    msin = 1 / angles.size * np.sum(np.sin(angles.flatten()))
    mcos = 1 / angles.size * np.sum(np.cos(angles.flatten()))

    mean = np.arctan2(msin, mcos)
    std = np.sqrt(-2 * np.log(msin * msin + mcos * mcos))

    return mean, std


def calcMeanStdHalfAngle(angles):
    mean, std = calcMeanStdAngle(2 * angles)
    return mean / 2, std / 2


# /data/PLI-Group/felix/data/judac/felix/project/2_cube_factory/output
if len(sys.argv) > 1:
    FILE_PATH = sys.argv[1]
else:
    raise ValueError("need file output path")

version = 'v0'
os.makedirs(os.path.join(FILE_PATH, 'analysis'), exist_ok=True)
file_list = sorted(
    glob.glob(os.path.join(FILE_PATH, 'simulations', '*.' + version + '.h5')))

# get resolution:
with h5py.File(file_list[0], 'r') as h5f:
    for name in ['PM']:
        h5f_sub = h5f[name + '/p']
        res = h5f_sub['analysis/rofl/direction'][...].shape
shape = (len(file_list), 2, res[0], res[1])

if not os.path.isfile(os.path.join(FILE_PATH, 'analysis', version + '.npz')):

    print('collect rofl data')

    rofl_dire = np.empty(shape)
    rofl_incl = np.empty(shape)
    rofl_trel = np.empty(shape)

    rofl_dire_mean = np.empty(shape[:2])
    rofl_incl_mean = np.empty(shape[:2])
    rofl_trel_mean = np.empty(shape[:2])

    rofl_dire_std = np.empty(shape[:2])
    rofl_incl_std = np.empty(shape[:2])
    rofl_trel_std = np.empty(shape[:2])

    dphi_list = []
    psi_list = []
    phi_list = []

    for f, file in enumerate(tqdm(file_list)):
        # TODO: werte mit in h5 abspeichern !
        dphi = float(file.partition('_dphi_')[-1].partition('_psi')[0])
        psi = float(file.partition('_psi_')[-1].partition('.v')[0])
        phi = float(file.partition('_phi_')[-1].partition('.v')[0])

        dphi_list.append(dphi)
        psi_list.append(psi)
        phi_list.append(phi)

        with h5py.File(file, 'r') as h5f:
            for name in ['PM']:
                for m, model in enumerate(['p', 'r']):
                    h5f_sub = h5f[name + '/' + model]

                    rofl_dire[f, m] = h5f_sub['analysis/rofl/direction'][...]
                    rofl_incl[f, m] = h5f_sub['analysis/rofl/inclination'][...]
                    rofl_trel[f, m] = h5f_sub['analysis/rofl/t_rel'][...]

                    # mean, std = calcMeanStdAngle(rofl_dire[f, m].flatten())
                    mean = scipy.stats.circmean(rofl_dire[f, m].flatten(),
                                                np.pi / 2, -np.pi / 2)
                    std = scipy.stats.circstd(rofl_dire[f, m].flatten(),
                                              np.pi / 2, -np.pi / 2)
                    rofl_dire_mean[f, m] = mean
                    rofl_dire_std[f, m] = std

                    # mean, std = calcMeanStdAngle(rofl_incl[f, m].flatten())
                    mean = scipy.stats.circmean(rofl_incl[f, m].flatten(),
                                                np.pi / 2, -np.pi / 2)
                    std = scipy.stats.circstd(rofl_incl[f, m].flatten(),
                                              np.pi / 2, -np.pi / 2)
                    rofl_incl_mean[f, m] = mean
                    rofl_incl_std[f, m] = std

                    rofl_trel_mean[f, m] = np.mean(rofl_trel[f, m].flatten())
                    rofl_trel_std[f, m] = np.std(rofl_trel[f, m].flatten())

    dphi_list = np.array(dphi_list)
    psi_list = np.array(psi_list)
    phi_list = np.array(phi_list)

    np.savez(os.path.join(FILE_PATH, 'analysis', version),
             rofl_dire=rofl_dire,
             rofl_incl=rofl_incl,
             rofl_trel=rofl_trel,
             rofl_dire_mean=rofl_dire_mean,
             rofl_dire_std=rofl_dire_std,
             rofl_incl_mean=rofl_incl_mean,
             rofl_incl_std=rofl_incl_std,
             rofl_trel_mean=rofl_trel_mean,
             rofl_trel_std=rofl_trel_std,
             dphi_list=dphi_list,
             psi_list=psi_list,
             phi_list=phi_list)

npzfile = np.load(os.path.join(FILE_PATH, 'analysis', version + '.npz'))

# print(npzfile.files)

psi_list = npzfile['psi_list']
phi_list = npzfile['phi_list']

print(np.unique(psi_list))

for p in np.unique(psi_list):
    ind = np.where(np.logical_and(psi_list == p, phi_list == 0))[0]
    p = p * 5

    # print(ind)

    m = 0
    # for m in [0, 1]:
    print(npzfile['dphi_list'][ind])
    # print(np.rad2deg(npzfile['rofl_dire_mean'][ind, m]))
    # print(np.rad2deg(npzfile['rofl_dire_std'][ind, m]))

    plt.subplot(311)
    plt.errorbar(npzfile['dphi_list'][ind] + m + p - 0.5 * 5,
                 np.rad2deg(npzfile['rofl_dire_mean'][ind, m]),
                 np.rad2deg(npzfile['rofl_dire_std'][ind, m]),
                 linestyle='None',
                 marker='^')

    plt.subplot(312)
    plt.errorbar(npzfile['dphi_list'][ind] + m + p - 0.5 * 5,
                 np.rad2deg(npzfile['rofl_incl_mean'][ind, m]),
                 np.rad2deg(npzfile['rofl_incl_std'][ind, m]),
                 linestyle='None',
                 marker='^')

    plt.subplot(313)
    plt.errorbar(npzfile['dphi_list'][ind] + m + p - 0.5 * 5,
                 npzfile['rofl_trel_mean'][ind, m],
                 npzfile['rofl_trel_std'][ind, m],
                 linestyle='None',
                 marker='^')
plt.show()
