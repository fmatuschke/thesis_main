#%% prepair data
import numpy as np
import h5py
import os
import sys
import glob

import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import pandas as pd

from tqdm import tqdm

# reproducability
np.random.seed(42)

FILE_PATH = '/data/PLI-Group/felix/data/judac/felix/project/2_cube_factory/cube_2pop/simulations'
if len(sys.argv) > 1:
    FILE_PATH = sys.argv[1]
else:
    raise ValueError("need file output path")

#%% prepair data
os.makedirs(os.path.join(FILE_PATH, 'analysis'), exist_ok=True)
file_list = sorted(glob.glob(os.path.join(FILE_PATH, '*.h5')))

# get resolution:
with h5py.File(file_list[0], 'r') as h5f:
    for name in ['PM']:
        h5f_sub = h5f[name + '/p']
        res = h5f_sub['analysis/rofl/direction'].shape
shape = (len(file_list), 2, res[0] * res[1])

dataframe = pd.DataFrame(
    columns=["phi", "dphi", "psi", "rofl_dir", "rofl_inc", "rofl_trel"])

if not os.path.isfile(os.path.join(
        FILE_PATH, 'analysis', 'data.npz')) or not os.path.isfile(
            os.path.join(FILE_PATH, 'analysis', 'data.pkl')):

    print('collect rofl data')

    rofl_dir = np.empty(shape)
    rofl_inc = np.empty(shape)
    rofl_trel = np.empty(shape)

    rofl_dir_mean = np.empty(shape[:2])
    rofl_inc_mean = np.empty(shape[:2])
    rofl_trel_mean = np.empty(shape[:2])

    rofl_dir_std = np.empty(shape[:2])
    rofl_inc_std = np.empty(shape[:2])
    rofl_trel_std = np.empty(shape[:2])

    dphi_list = []
    psi_list = []
    phi_list = []

    for f, file in enumerate(tqdm(file_list)):
        with h5py.File(file, 'r') as h5f:
            for name in ['PM']:
                for m, model in enumerate(['p', 'r']):
                    h5f_sub = h5f[name + '/' + model]

                    rofl_dir[f, m] = h5f_sub['analysis/rofl/direction'][
                        ...].flatten()
                    rofl_inc[f, m] = h5f_sub['analysis/rofl/inclination'][
                        ...].flatten()
                    rofl_trel[f, m] = h5f_sub['analysis/rofl/t_rel'][
                        ...].flatten()

                    rofl_dir[f, m][rofl_dir[f, m] > np.pi / 2] -= np.pi
                    # rofl_inc[f, m][rofl_inc[f, m] > np.pi / 2] -= np.pi

                    # mean, std = calcMeanStdAngle(rofl_dir[f, m].flatten())
                    mean = scipy.stats.circmean(rofl_dir[f, m], np.pi / 2,
                                                -np.pi / 2)
                    std = scipy.stats.circstd(rofl_dir[f, m], np.pi / 2,
                                              -np.pi / 2)
                    rofl_dir_mean[f, m] = mean
                    rofl_dir_std[f, m] = std

                    # mean, std = calcMeanStdAngle(rofl_inc[f, m].flatten())
                    mean = scipy.stats.circmean(rofl_inc[f, m], np.pi / 2,
                                                -np.pi / 2)
                    std = scipy.stats.circstd(rofl_inc[f, m], np.pi / 2,
                                              -np.pi / 2)
                    rofl_inc_mean[f, m] = mean
                    rofl_inc_std[f, m] = std

                    rofl_trel_mean[f, m] = np.mean(rofl_trel[f, m])
                    rofl_trel_std[f, m] = np.std(rofl_trel[f, m])

                    psi = h5f_sub.attrs['parameter/model/psi']
                    dphi = h5f_sub.attrs['parameter/model/dphi']
                    phi = h5f_sub.attrs['parameter/model/phi']

                    dphi_list.append(psi)
                    psi_list.append(dphi)
                    phi_list.append(phi)

                    df = pd.DataFrame({
                        "phi": [phi] * rofl_inc[f, m].size,
                        "dphi": [dphi] * rofl_inc[f, m].size,
                        "psi": [psi] * rofl_inc[f, m].size,
                        "rofl_dir":
                            np.rad2deg(rofl_dir[f, m].flatten()).tolist(),
                        "rofl_inc":
                            np.rad2deg(rofl_inc[f, m].flatten()).tolist(),
                        "rofl_trel":
                            np.rad2deg(rofl_trel[f, m].flatten()).tolist()
                    })

                    dataframe = dataframe.append(df)

    dphi_list = np.array(dphi_list)
    psi_list = np.array(psi_list)
    phi_list = np.array(phi_list)

    np.savez(os.path.join(FILE_PATH, 'analysis', 'data'),
             rofl_dir=rofl_dir,
             rofl_inc=rofl_inc,
             rofl_trel=rofl_trel,
             rofl_dir_mean=rofl_dir_mean,
             rofl_dir_std=rofl_dir_std,
             rofl_inc_mean=rofl_inc_mean,
             rofl_inc_std=rofl_inc_std,
             rofl_trel_mean=rofl_trel_mean,
             rofl_trel_std=rofl_trel_std,
             dphi_list=dphi_list,
             psi_list=psi_list,
             phi_list=phi_list)

    species = [
        str(i) + ',' + str(j) + ',' + str(k)
        for i, j, k in zip(dataframe['psi'].to_numpy(), dataframe['phi'].
                           to_numpy(), dataframe['dphi'].to_numpy())
    ]
    dataframe.insert(0, "species", species)
    dataframe.to_pickle(os.path.join(FILE_PATH, 'analysis', 'data.pkl'))

#%% visualize data
npzfile = np.load(os.path.join(FILE_PATH, 'analysis', 'data.npz'))
dataframe = pd.read_pickle(os.path.join(FILE_PATH, 'analysis', 'data.pkl'))

sns.set(style="ticks")
df = dataframe.loc[dataframe['phi'] == 0.0]
# print(df.shape)
print(list(set(df["species"].to_list())))
print(len(list(set(df["species"].to_list()))))
print(list(set(df["dphi"])))
df = df[["dphi", "rofl_dir", "rofl_trel", "species"]]
g = sns.pairplot(df, hue="species", diag_kind=None)
g.axes[0, 0].set_xlim((-5, 95))
g.axes[1, 0].set_xlim((-5, 95))
g.axes[2, 0].set_xlim((-5, 95))
# g.axes[2, 0].set_xlim((-5, 10))
# g.axes[2, 1].set_xlim((-5, 10))
# g.axes[2, 2].set_xlim((-5, 10))
plt.show()

# df = dataframe.loc[dataframe['psi'] == 0.5]
# print(list(set(df["species"].to_list())))

if False:
    psi_list = npzfile['psi_list']
    phi_list = npzfile['phi_list']
    dphi_list = npzfile['dphi_list']

    print(np.unique(psi_list))

    for p in np.unique(psi_list):
        ind = np.where(np.logical_and(psi_list == p, phi_list == 0))[0]

        dp = 5
        p = p * dp

        m = 0
        plt.subplot(311)
        plt.errorbar(npzfile['dphi_list'][ind] + m + p - 0.5 * dp,
                     np.rad2deg(npzfile['rofl_dir_mean'][ind, m]),
                     np.rad2deg(npzfile['rofl_dir_std'][ind, m]),
                     linestyle='None',
                     marker='^')

        plt.subplot(312)
        plt.errorbar(npzfile['dphi_list'][ind] + m + p - 0.5 * dp,
                     np.rad2deg(npzfile['rofl_inc_mean'][ind, m]),
                     np.rad2deg(npzfile['rofl_inc_std'][ind, m]),
                     linestyle='None',
                     marker='^')

        plt.subplot(313)
        plt.errorbar(npzfile['dphi_list'][ind] + m + p - 0.5 * dp,
                     npzfile['rofl_trel_mean'][ind, m],
                     npzfile['rofl_trel_std'][ind, m],
                     linestyle='None',
                     marker='^')
    plt.show()
