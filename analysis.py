import numpy as np
import h5py
import glob
import sys
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistic
import scipy.stats
import scipy.optimize
import tikzplotlib
# import astropy.stats

df = pd.read_pickle(
    "output/" +
    "cube_2pop_psi_0.5_omega_0.0_vref_0.025_length_12.5.analysis.pkl")

df_analyse = pd.DataFrame()

ref = 'r'

for m, m_group in df.groupby('model'):
    for o, o_group in m_group.groupby('omega'):
        for r, r_group in o_group.groupby('resolution'):
            df_ = pd.DataFrame()
            for vs, vs_group in r_group.groupby('voxel_size'):

                dirc = statistic._remap_direction_sym(
                    np.array(vs_group['direction_ref_' + ref].iloc[0]) -
                    np.array(vs_group['direction'].iloc[0]))

                n = dirc.size
                df_ = df_.append(
                    pd.DataFrame({
                        "voxel_size": [float(vs)] * n,
                        "resolution": [float(r)] * n,
                        "model": [m] * n,
                        "omega": [float(o)] * n,
                        "transmittance":
                            np.array(vs_group['transmittance'].iloc[0]),
                        "direction":
                            np.array(vs_group['direction'].iloc[0]),
                        "retardation":
                            np.array(vs_group['retardation'].iloc[0]),
                        "diff_dir":
                            np.rad2deg(dirc),
                        "diff_ret":
                            np.array(vs_group['retardation_ref_' + ref].iloc[0])
                            - np.array(vs_group['retardation'].iloc[0]),
                        "diff_trans":
                            np.array(
                                vs_group['transmittance_ref_' + ref].iloc[0]) -
                            np.array(vs_group['transmittance'].iloc[0])
                    }))

            df_analyse = df_analyse.append(df_)

# df_analyse.to_csv(
#     "cube_2pop_psi_0.5_omega_0.0_vref_0.025_length_12.5.analysis.csv")

fig, axs = plt.subplots(2, 1)
sns.boxplot(x='voxel_size',
            y='diff_dir',
            hue='omega',
            data=df_analyse[(df_analyse.model == 'r') &
                            (df_analyse.resolution == 1.25)],
            ax=axs[0])

sns.boxplot(x='voxel_size',
            y='retardation',
            hue='omega',
            data=df_analyse[(df_analyse.model == 'r') &
                            (df_analyse.resolution == 1.25)],
            ax=axs[1])

tikzplotlib.save("test.tex")

# plt.show()
