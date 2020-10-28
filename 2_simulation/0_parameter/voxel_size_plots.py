import numpy as np
import itertools
import h5py
import os
import sys
import glob
import argparse

import pandas as pd

import helper.circular

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of file.")
args = parser.parse_args()
os.makedirs(os.path.join(args.input, "results"), exist_ok=True)

df = pd.read_pickle(
    os.path.join(os.path.join(args.input, "voxel_size_post_1.pkl")))
df = df[df.f1_rot == 0]
df['epa_dir_diff'] = np.rad2deg(df['epa_dir_diff'])

parameters = list(df[["f0_inc", "omega", "psi",
                      "model"]].drop_duplicates().iterrows())

for _, p in parameters:
    f0_inc = p['f0_inc']
    omega = p['omega']
    psi = p['psi']
    model = p['model']
    sub = (df.f0_inc == f0_inc) & (df.omega == omega) & (df.psi == psi) & (
        df.model == model) & (df.m > 0)

    df_ = df[sub]
    for col in df_.columns:
        if len(df_[col].unique()) == 1:
            df_ = df_.drop(col, axis=1)

    string = [str(elm) for elm in p]
    string = '-'.join(string)

    for vs in df_.voxel_size.unique():
        for r in df_.radius.unique():
            # for n in df_.n.unique():
            #     for m in df_.m.unique():
            df__ = df_[(df_.voxel_size == vs) & (df_.radius == r)]
            df__.to_csv(os.path.join(args.input, "results",
                                     f"vs_stats_vs_{vs}_r_{r}.csv"),
                        index=False)

    # fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    # fig.suptitle(string, fontsize=21)

    # sns.boxplot(x="voxel_size", y="epa_trans_diff_rel", hue='radius', data=df[sub], ax=axs[0])
    # sns.stripplot(x="voxel_size", y="epa_dir_diff", hue='radius', data=df[sub], ax=axs[1], dodge=True)
    # ax = sns.boxplot(x="voxel_size", y="epa_ret_diff_rel", hue='radius', data=df[sub], ax=axs[2])
    # ax.set_ylim(0, 1)
