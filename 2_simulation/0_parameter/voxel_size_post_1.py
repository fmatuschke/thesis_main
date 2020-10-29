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

    radius = row["radius"]
    model = row["model"]
    omega = row["omega"]
    psi = row["psi"]
    f0_inc = row["f0_inc"]
    f1_rot = row["f1_rot"]
    # n = row["n"]
    # m = row["m"]

    sub = (df.radius == radius) & (df.model == model) & (df.omega == omega) & (
        df.psi == psi) & (df.f0_inc == f0_inc) & (df.f1_rot == f1_rot)
    #  & (df.n == n) & (df.m == m)

    df_ = df[sub]
    # ref = df[sub]
    # ref = ref[ref.m == 0]
    # ref = ref.sort_values(by=['voxel_size'])
    # ref = df_.iloc[0]

    for vs in df_.voxel_size.unique():
        for n in df_.n.unique():
            ref = df_[(df_.voxel_size == min(df_.voxel_size.unique())) &
                      (df_.n == n) & (df_.m == 0)]

            if len(ref) != 1:
                print("FOOOO ref", len(ref))
                sys.exit()

            ref = ref.squeeze()

            for m in df_.m.unique():
                df__ = df_[(df_.voxel_size == vs) & (df_.n == n) & (df_.m == m)]
                if len(df__) != 1:
                    print("FOOOO df__", len(df__))
                    sys.exit()

                df__ = df__.squeeze()

                df_res.append({
                    "voxel_size":
                        vs,
                    "radius":
                        radius,
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
                    "n":
                        n,
                    "m":
                        m,
                    "epa_trans_diff_rel":
                        np.abs(
                            (df__.epa_trans - ref.epa_trans)) / ref.epa_trans,
                    "epa_dir_diff":
                        np.abs(circ_diff(
                            df__.epa_dir,
                            ref.epa_dir,
                        )),
                    "epa_ret_diff":
                        np.abs((df__.epa_ret - ref.epa_ret)
                              ),  # / (ref.epa_ret + 1e-6),
                    "epa_ret_diff_rel":
                        np.abs((df__.epa_ret - ref.epa_ret)) /
                        (ref.epa_ret + 1e-6),
                    "data_diff":
                        np.mean(np.abs(
                            (df__.optic - ref.optic))) / ref.epa_trans,
                    "data_diff_sqr":
                        np.mean((df__.optic - ref.optic)**2) / ref.epa_trans**2,
                })
    return df_res


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(args.input, "voxel_size_post_0.pkl"))
    # df = df[df.m == 0].copy()

    df_res = []

    parameters = list(
        df[["radius", "model", "omega", "psi", "f0_inc",
            "f1_rot"]].drop_duplicates().iterrows())

    with mp.Pool(processes=args.num_proc) as pool:
        df_res = [
            sub for sub in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                     total=len(parameters))
        ]
    df_res = [item for sublist in df_res for item in sublist]

    df_res = pd.DataFrame(df_res)
    df_res.to_pickle(os.path.join(args.input, "voxel_size_post_1.pkl"))
