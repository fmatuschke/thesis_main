#! /usr/bin/env python3

import argparse
import glob
import multiprocessing as mp
import os
import sys
import warnings

import fastpli.analysis
import numpy as np
import pandas as pd
import tqdm
import yaml

import models
import parameter

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

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

# FIXME parameter.py
CONFIG = parameter.get_tupleware()

hist_polar_bin = lambda n: np.linspace(
    -np.pi / 2, np.pi / 2, n + 1, endpoint=True)


def run(df):

    file = df["fiber"]
    phi, theta = models.ori_from_file(file, 0, 0, CONFIG.simulation.voi)
    # phi_x, phi_h, incl_x, incl_h = polar_hist(phi, theta)

    # 2d hist
    h, x, y, _ = fastpli.analysis.orientation.histogram(phi,
                                                        theta,
                                                        n_phi=36 * 2,
                                                        n_theta=18,
                                                        weight_area=True)

    return pd.DataFrame([[df.omega, df.psi, df.radius, h, x, y]],
                        columns=[
                            "omega", "psi", "radius", "hist_2d_h", "hist_2d_x",
                            "hist_2d_y"
                        ])


if __name__ == "__main__":

    if True or not os.path.isfile(
            os.path.join(args.input, "hist", "cube_2pop.pkl")):
        print("calculating data")
        df = pd.read_pickle(os.path.join(args.input, "cube_2pop.pkl"))
        df = df[df.state != "init"]
        df = [df.iloc[i] for i in range(df.shape[0])]
        # df = [df[0]]  # debug

        with mp.Pool(processes=args.num_proc) as pool:
            df = [
                _ for _ in tqdm.tqdm(
                    pool.imap_unordered(run, df), total=len(df), smoothing=0)
            ]

        df = pd.concat(df, ignore_index=True)
        df.to_pickle(os.path.join(args.input, "hist", "cube_2pop.pkl"))
    else:
        print("loading data")
        df = pd.read_pickle(os.path.join(args.input, "hist", "cube_2pop.pkl"))

    for omega in df.omega.unique():
        for psi in df[df.omega == omega].psi.unique():
            df_ = pd.DataFrame()
            for r in df.radius.unique():

                df_sub = df.query(
                    "radius == @r & omega == @omega & psi == @psi")
                if len(df_sub) != 1:
                    warnings.warn(f"len(df_sub) != 1: {r} {omega} {psi}")
                    continue

                # 2d hist
                with open(
                        os.path.join(args.input, "hist",
                                     f"cube_stats_p_{psi}_o_{omega}_r_{r}.dat"),
                        "w") as f:

                    H = df_sub.hist_2d_h.iloc[0]
                    x_axis = df_sub.hist_2d_x.iloc[0]
                    y_axis = df_sub.hist_2d_y.iloc[0]

                    # for pgfplots matrix plot*
                    x_axis = x_axis[:-1] + (x_axis[1] - x_axis[0]) / 2
                    y_axis = y_axis[:-1] + (y_axis[1] - y_axis[0]) / 2
                    H = H.T / np.sum(H.ravel())

                    X, Y = np.meshgrid(np.rad2deg(x_axis), np.rad2deg(y_axis))

                    # print(X.shape)
                    # print(Y.shape)
                    # print(H.shape)
                    for h_array, x_array, y_array in zip(H, X, Y):
                        for h, x, y in zip(h_array, x_array, y_array):
                            if y <= 90:
                                f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
                        f.write('\n')
