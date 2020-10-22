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
    n = row["n"]
    m = row["m"]

    sub = (df.radius == radius) & (df.model == model) & (df.omega == omega) & (
        df.psi == psi) & (df.f0_inc == f0_inc) & (df.f1_rot == f1_rot) & (
            df.n == n) & (df.m == m)

    df_ = df[sub].sort_values(by=['voxel_size'])
    ref = df_.iloc[0]

    for vs in df_.voxel_size.unique():

        df_res.append({
            "radius": radius,
            "model": model,
            "omega": omega,
            "psi": psi,
            "f0_inc": f0_inc,
            "f1_rot": f1_rot,
            "n": n,
            "m": m,
            "vs": vs,
            "d_trans": ref.epa_trans - df_[df_.voxel_size == vs].epa_trans,
            "d_dir": circ_diff(ref.epa_dir, df_[df_.voxel_size == vs].epa_dir),
            "d_ret": ref.epa_ret - df_[df_.voxel_size == vs].epa_ret,
        })
    return df_res


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(args.input, "voxel_size_post_0.pkl"))

    df_res = []

    parameters = list(
        df[["radius", "model", "omega", "psi", "f0_inc", "f1_rot", "n",
            "m"]].drop_duplicates().iterrows())

    with mp.Pool(processes=args.num_proc) as pool:
        df_res = [
            sub for sub in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                     total=len(parameters))
        ]
    df_res = [item for sublist in df_res for item in sublist]

    df_res = pd.DataFrame(df_res)
    df_res.to_pickle(os.path.join(args.input, "voxel_size_post_1.pkl"))
