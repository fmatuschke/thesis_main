#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import argparse
import os
import sys
import glob
import warnings

import tqdm
import pandas as pd

import fibers

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input path.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()

os.makedirs(os.path.join(args.input, "hist"), exist_ok=True)

hist_polar_bin = lambda n: np.linspace(
    -np.pi / 2, np.pi / 2, n + 1, endpoint=True)


def polar_hist(phi, theta):
    incl = np.pi / 2 - theta

    # remap to phi->[-np.pi/2, np.pi/2], incl->[-np.pi/2, np.pi/2]
    phi[phi > np.pi * 3 / 2] -= 2 * np.pi
    incl[phi > np.pi / 2] = -incl[phi > np.pi / 2]
    phi[phi > np.pi / 2] -= np.pi

    if np.any(incl < -np.pi / 2) or np.any(incl > np.pi / 2):
        raise ValueError("FOOO incl")
    if np.any(phi < -np.pi / 2) or np.any(phi > np.pi / 2):
        raise ValueError("FOOO phi")

    # direction
    h, x = np.histogram(phi, hist_polar_bin(180), density=True)
    phi_x = x[:-1] + (x[1] - x[0]) / 2
    phi_h = h / np.amax(h)

    # inclination
    h, x = np.histogram(incl, hist_polar_bin(180), density=True)
    incl_x = x[:-1] + (x[1] - x[0]) / 2
    incl_h = h / np.amax(h)

    return phi_x, phi_h, incl_x, incl_h


def run(df):

    file = df["fiber"]
    phi, theta = fibers.ori_from_file(file, 0, 0, 60)
    phi_x, phi_h, incl_x, incl_h = polar_hist(phi, theta)

    # INIT
    file = file.replace("solved", "init")
    file = file.replace("tmp", "init")
    phi, theta = fibers.ori_from_file(file, 0, 0, 60)
    phi_init_x, phi_init_h, incl_init_x, incl_init_h = polar_hist(phi, theta)

    return pd.DataFrame([[
        df.omega,
        df.psi,
        df.r,
        phi_x,
        phi_h,
        incl_x,
        incl_h,
        phi_init_x,
        phi_init_h,
        incl_init_x,
        incl_init_h,
    ]],
                        columns=[
                            "omega",
                            "psi",
                            "r",
                            "phi_x",
                            "phi_h",
                            "incl_x",
                            "incl_h",
                            "phi_init_x",
                            "phi_init_h",
                            "incl_init_x",
                            "incl_init_h",
                        ])


if __name__ == "__main__":

    if not os.path.isfile(os.path.join(args.input, "hist", "cube_2pop.pkl")):
        df = pd.read_pickle(os.path.join(args.input, "cube_2pop.pkl"))
        df = df[df.state != "init"]
        df = [df.iloc[i] for i in range(df.shape[0])]

        with mp.Pool(processes=args.num_proc) as pool:
            df = [
                _ for _ in tqdm.tqdm(
                    pool.imap_unordered(run, df), total=len(df), smoothing=0.1)
            ]

        df = pd.concat(df, ignore_index=True)
        df.to_pickle(os.path.join(args.input, "hist", "cube_2pop.pkl"))
    else:
        df = pd.read_pickle(os.path.join(args.input, "hist", "cube_2pop.pkl"))

    for omega in df.omega.unique():
        for psi in df[df.omega == omega].psi.unique():
            df_ = pd.DataFrame()
            for r in df.r.unique():

                df_sub = df.query("r == @r & omega == @omega & psi == @psi")
                if len(df_sub) != 1:
                    warnings.warn(f"len(df_sub) != 1: {r} {omega} {psi}")
                    continue

                df_[f"x"] = np.rad2deg(df_sub.phi_x.iloc[0])
                df_[f"r_{r:.2f}_s_solved_phi_h"] = df_sub.phi_h.iloc[0]
                df_[f"r_{r:.2f}_s_solved_incl_h"] = df_sub.incl_h.iloc[0]
                df_[f"r_{r:.2f}_s_init_phi_h"] = df_sub.phi_init_h.iloc[0]
                df_[f"r_{r:.2f}_s_init_incl_h"] = df_sub.incl_init_h.iloc[0]

            df_.to_csv(os.path.join(args.input, "hist",
                                    f"cube_stats_p_{psi}_o_{omega}_.csv"),
                       index=False,
                       float_format='%.4f')
