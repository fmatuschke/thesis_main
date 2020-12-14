#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import itertools
import argparse
import h5py
import os
import sys
import glob

import pandas as pd
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
parser.add_argument("-p",
                    "--num_proc",
                    required=True,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()


def mod(a, n):
    return ((a % n) + n) % n


def circ_diff(a, b):
    diff = a - b
    return mod((diff + np.pi / 2), np.pi) - np.pi / 2


def run(row):
    _, row = row
    df_res = []

    voxel_size = row["voxel_size"]
    model = row["model"]
    setup = row["setup"]
    species = row["species"]
    omega = row["omega"]
    psi = row["psi"]
    f0_inc = row["f0_inc"]
    f1_rot = row["f1_rot"]
    # n = row["n"]
    # m = row["m"]

    sub = (df.setup == setup) & (df.species == species) & (
        df.model == model) & (df.omega == omega) & (df.psi == psi) & (
            df.f0_inc == f0_inc) & (df.f1_rot == f1_rot)
    #  & (df.n == n) & (df.m == m)

    df_ = df[sub]
    # print(len(df_))
    # print(df_.n.unique())
    # print(df_.m.unique())
    # print(df_.voxel_size.unique())
    # print(df_.radius.unique())
    # ref = df[sub]
    # ref = ref[ref.m == 0]
    # ref = ref.sort_values(by=['voxel_size'])
    # ref = df_.iloc[0]

    # print(df_.columns)
    ref = df_[(df_.radius == min(df_.radius.unique())) & (df_.m == 0)]

    if len(ref) != 1:
        print("FOOOO ref", len(ref))
        sys.exit()

    ref = ref.squeeze()
    for radius in df_.radius.unique():
        for m in df_.m.unique():
            df__ = df_[(df_.radius == radius) & (df_.m == m)]
            if len(df__) != 1:
                print("FOOOO df__", len(df__))
                sys.exit()

            df__ = df__.squeeze()

            df_res.append({
                "voxel_size":
                    voxel_size,
                "radius":
                    radius,
                "setup":
                    setup,
                "species":
                    species,
                "model":
                    model,
                "omega":
                    omega,
                "psi":
                    psi,
                "f0_inc":
                    f0_inc,
                "f1_rot":
                    f1_rot,
                "m":
                    m,
                "epa_trans_diff_rel":
                    np.abs((df__.epa_trans - ref.epa_trans)) / ref.epa_trans,
                "epa_dir_diff":
                    np.abs(circ_diff(
                        df__.epa_dir,
                        ref.epa_dir,
                    )),
                "epa_ret_diff":
                    np.abs((df__.epa_ret - ref.epa_ret)),
                "epa_ret_diff_rel":
                    np.abs((df__.epa_ret - ref.epa_ret)) / (ref.epa_ret + 1e-6),
                "data_diff":
                    np.mean(np.abs((df__.optic - ref.optic))) / ref.epa_trans,
                "data_diff_sqr":
                    np.mean((df__.optic - ref.optic)**2) / ref.epa_trans**2,
            })
    return df_res


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(args.input, "fiber_radii_post_0.pkl"))
    # df = df[df.m == 0].copy()

    df_res = []

    parameters = list(df[[
        "voxel_size", "model", "setup", "species", "omega", "psi", "f0_inc",
        "f1_rot"
    ]].drop_duplicates().iterrows())

    with mp.Pool(processes=args.num_proc) as pool:
        df_res = [
            sub for sub in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                     total=len(parameters))
        ]
    df_res = [item for sublist in df_res for item in sublist]

    df_res = pd.DataFrame(df_res)
    df_res.to_pickle(os.path.join(args.input, "fiber_radii_post_1.pkl"))
