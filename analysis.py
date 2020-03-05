import numpy as np
import glob
import sys
import os
import time

from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistic
import scipy.stats
import scipy.optimize
import tikzplotlib

df = pd.read_pickle("output/data.pkl")

df_analyse = pd.DataFrame()
for i in tqdm(range(df.shape[0])):
    dirc = statistic._remap_direction_sym(
        np.array(df['direction_ref'].iloc[i]) -
        np.array(df['direction'].iloc[i]))
    df_analyse = df_analyse.append(
        {
            "voxel_size":
                df['voxel_size'].iloc[i],
            "resolution":
                df['resolution'].iloc[i],
            "model":
                df['model'].iloc[i],
            "omega":
                df['omega'].iloc[i],
            'f0':
                df['f0'].iloc[i],
            'f1':
                df['f1'].iloc[i],
            "transmittance":
                df['transmittance'].iloc[i],
            "direction":
                df['direction'].iloc[i],
            "retardation":
                df['retardation'].iloc[i],
            "diff_trans": (np.array(df['transmittance_ref'].iloc[i]) -
                           np.array(df['transmittance'].iloc[i])).tolist(),
            "diff_dir": (np.rad2deg(dirc)).tolist(),
            "diff_ret": (np.array(df['retardation_ref'].iloc[i]) -
                         np.array(df['retardation'].iloc[i])).tolist(),
        },
        ignore_index=True)

df_analyse.to_pickle("output/data_.pkl")

# # fig, axs = plt.subplots(2, 1)
# sns.boxplot(
#     x='voxel_size',
#     y='diff_dir',
#     hue='omega',
#     data=df_analyse[(df_analyse.model == 'r') &
#                     (df_analyse.resolution == 1.25)].explode('diff_dir'),
#     # ax=axs[0]
# )
# tikzplotlib.save("voxel_size_vs_diff_dir.tex")

# sns.boxplot(
#     x='voxel_size',
#     y='retardation',
#     hue='omega',
#     data=df_analyse[(df_analyse.model == 'r') &
#                     (df_analyse.resolution == 1.25)].explode('retardation'),
#     # ax=axs[1]
# )

# tikzplotlib.save("voxel_size_vs_retardation.tex")
# # plt.show()
