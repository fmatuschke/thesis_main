import numpy as np
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

df = pd.read_pickle("output/data.pkl")

ref = 'r'
df_analyse = pd.DataFrame()
for m, m_group in df.groupby('model'):
    for o, o_group in m_group.groupby('omega'):
        for r, r_group in o_group.groupby('resolution'):
            df_ = pd.DataFrame()
            for vs, vs_group in r_group.groupby('voxel_size'):

                dirc = statistic._remap_direction_sym(
                    np.array(vs_group['direction_ref_' + ref].iloc[0]) -
                    np.array(vs_group['direction'].iloc[0]))

                df_analyse = df_analyse.append({
                        "voxel_size":
                            float(vs),
                        "resolution":
                            float(r),
                        "model":
                            m,
                        "omega":
                            float(o),
                        "transmittance":
                            vs_group['transmittance'].iloc[0],
                        "direction":
                            vs_group['direction'].iloc[0],
                        "retardation":
                            vs_group['retardation'].iloc[0],
                        "diff_dir": (np.rad2deg(dirc)).tolist(),
                        "diff_ret":
                            (np.array(
                                vs_group['retardation_ref_' + ref].iloc[0]) -
                             np.array(
                                 vs_group['retardation'].iloc[0])).tolist(),
                        "diff_trans": (
                            np.array(
                                vs_group['transmittance_ref_' + ref].iloc[0]) -
                            np.array(vs_group['transmittance'].iloc[0])).tolist()
                    }, ignore_index=True)

# fig, axs = plt.subplots(2, 1)
sns.boxplot(
    x='voxel_size',
    y='diff_dir',
    hue='omega',
    data=df_analyse[(df_analyse.model == 'r') &
                    (df_analyse.resolution == 1.25)].explode('diff_dir'),
    # ax=axs[0]
)
tikzplotlib.save("voxel_size_vs_diff_dir.tex")

sns.boxplot(
    x='voxel_size',
    y='retardation',
    hue='omega',
    data=df_analyse[(df_analyse.model == 'r') &
                    (df_analyse.resolution == 1.25)].explode('retardation'),
    # ax=axs[1]
)

tikzplotlib.save("voxel_size_vs_retardation.tex")
# plt.show()
