import numpy as np
import itertools
import h5py
import os
import sys
import glob
import argparse

import pandas as pd
import tqdm

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
# df = df.apply(pd.Series.explode).reset_index()
# print(df['epa_dir_diff'].iloc[0])
# df['epa_dir_diff'] = np.rad2deg(df['epa_dir_diff'])

parameters = list(df[["f0_inc", "omega", "psi", "model", "setup",
                      "species"]].drop_duplicates().iterrows())

for _, p in tqdm.tqdm(parameters):
    f0_inc = p['f0_inc']
    omega = p['omega']
    psi = p['psi']
    model = p['model']
    setup = p['setup']
    species = p['species']

    for m in [0, 1]:
        sub = (df.f0_inc == f0_inc) & (df.omega == omega) & (df.psi == psi) & (
            df.model == model) & (df.setup == setup) & (df.species == species)
        if m:
            sub = sub & (df.m > 0)
        else:
            sub = sub & (df.m == 0)

        df_ = df[sub].copy()
        for col in df_.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if len(df_[col].unique()) == 1:
                    df_ = df_.drop(col, axis=1)

        for vs in tqdm.tqdm(df_.voxel_size.unique(), leave=False):
            for r in df_.radius.unique():
                df__ = df_[(df_.voxel_size == vs) & (df_.radius == r)].copy()
                df__ = df__.apply(pd.Series.explode).reset_index(
                    drop=True).apply(pd.Series.explode).reset_index(drop=True)

                # for col in df__.columns:
                #     if len(df__[col].unique()) == 1:
                #         df__ = df__.drop(col, axis=1)

                # for col in df__.columns:
                #     if pd.api.types.is_object_dtype(df__[col]):
                #         df__[col] = df__[col].astype(float)

                if df__.isna().any().any():
                    print("FOOO missing")

                # # because of pgfplots/tikz
                # data = df__.epa_ret_diff_rel.to_numpy().ravel()
                # data[data > 4.2] = 4.2
                # df__.epa_ret_diff_rel = data

                df__.loc[df__.epa_ret_diff_rel == 0,
                         "epa_ret_diff_rel"] = np.nan
                df__.loc[df__.data_diff == 0, "data_diff"] = np.nan
                df__.drop(["voxel_size", "radius", "setup", "species", "model"],
                          inplace=True,
                          axis=1)

                # for col in df__.columns:
                #     if len(df__[col].unique()) == 1:
                #         df__.drop(col, inplace=True, axis=1)

                df__.to_csv(
                    os.path.join(
                        args.input, "results",
                        f"vs_stats_omega_{omega}_psi_{psi}_f0_inc_{f0_inc}_mode_{model}_species_{species}_setup_{setup}_vs_{vs}_r_{r}_m_{m}.csv"
                    ),
                    index=False,
                    # float_format='%.9f',
                    na_rep="nan")

    # fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    # fig.suptitle(string, fontsize=21)

    # sns.boxplot(x="voxel_size", y="epa_trans_diff_rel", hue='radius', data=df[sub], ax=axs[0])
    # sns.stripplot(x="voxel_size", y="epa_dir_diff", hue='radius', data=df[sub], ax=axs[1], dodge=True)
    # ax = sns.boxplot(x="voxel_size", y="epa_ret_diff_rel", hue='radius', data=df[sub], ax=axs[2])
    # ax.set_ylim(0, 1)
