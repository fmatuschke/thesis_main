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

import models
# import esag
import acg
import fastpli.analysis

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

os.makedirs(os.path.join(args.input, "esag"), exist_ok=True)


def run(df):

    file = df["fiber"]
    # print(file)

    # calculate acg from file
    vecs = models.vec_from_file(file, 0, 0, [65, 65, 60])
    vec = vecs[0]
    if vec.shape[0] >= 3:
        vec = np.divide(vec, np.linalg.norm(vec, axis=1)[:, None])
        cov_mat = acg.fit(vec)
        w0, v0 = np.linalg.eig(cov_mat)
    else:
        w0 = np.zeros(3)
        v0 = np.zeros((3, 3))

    vec = vecs[1]
    if vec.shape[0] >= 3:
        vec = np.divide(vec, np.linalg.norm(vec, axis=1)[:, None])
        cov_mat = acg.fit(vec)
        w1, v1 = np.linalg.eig(cov_mat)
    else:
        w1 = np.zeros(3)
        v1 = np.zeros((3, 3))

    # vec = np.concatenate((vec, -vec))
    # cov_mat = acg.fit(vec)
    # w, v = np.linalg.eig(cov_mat)
    # print(w, v)

    # calculate acg from init
    file = file.replace("solved", "init")
    file = file.replace("tmp", "init")
    vec = models.vec_from_file(file, 0, 0, [65, 65, 60])

    vec = vecs[0]
    if vec.shape[0] >= 3:
        vec = np.divide(vec, np.linalg.norm(vec, axis=1)[:, None])
        cov_mat = acg.fit(vec)
        w0_init, v0_init = np.linalg.eig(cov_mat)
    else:
        w0_init = np.zeros(3)
        v0_init = np.zeros((3, 3))

    vec = vecs[1]

    if vec.shape[0] >= 3:
        vec = np.divide(vec, np.linalg.norm(vec, axis=1)[:, None])
        cov_mat = acg.fit(vec)
        w1_init, v1_init = np.linalg.eig(cov_mat)
    else:
        w1_init = np.zeros(3)
        v1_init = np.zeros((3, 3))

    return pd.DataFrame([[
        df.omega, df.psi, df.radius, w0, v0, w1, v1, w0_init, v0_init, w1_init,
        v1_init
    ]],
                        columns=[
                            "omega", "psi", "radius", "w0", "v0", "w1", "v1",
                            "w0_init", "v0_init", "w1_init", "v1_init"
                        ])


if __name__ == "__main__":

    if True or not os.path.isfile(
            os.path.join(args.input, "esag", "cube_2pop.pkl")):
        df = pd.read_pickle(os.path.join(args.input, "cube_2pop.pkl"))
        df = df[df.state != "init"]
        df = [df.iloc[i] for i in range(df.shape[0])]  # -> row iterator

        # df = [df[22]]  # debug
        # [run(d) for d in tqdm.tqdm(df)]

        with mp.Pool(processes=args.num_proc) as pool:
            df = [
                _ for _ in tqdm.tqdm(
                    pool.imap_unordered(run, df), total=len(df), smoothing=0.1)
            ]

        df = pd.concat(df, ignore_index=True)
        df.to_pickle(os.path.join(args.input, "esag", "cube_2pop.pkl"))
    else:
        df = pd.read_pickle(os.path.join(args.input, "esag", "cube_2pop.pkl"))

    # for omega in df.omega.unique():
    #     for psi in df[df.omega == omega].psi.unique():
    #         df_ = pd.DataFrame()
    #         for r in df.r.unique():

    #             df_sub = df.query("r == @r & omega == @omega & psi == @psi")
    #             if len(df_sub) != 1:
    #                 warnings.warn(f"len(df_sub) != 1: {r} {omega} {psi}")
    #                 continue

    #             # 2d hist
    #             with open(
    #                     os.path.join(args.input, "hist",
    #                                  f"cube_stats_p_{psi}_o_{omega}_r_{r}.dat"),
    #                     "w") as f:

    #                 H = df_sub.hist_2d_h.iloc[0]
    #                 x_axis = df_sub.hist_2d_x.iloc[0]
    #                 y_axis = df_sub.hist_2d_y.iloc[0]

    #                 x_axis = x_axis + (x_axis[1] - x_axis[0]) / 2
    #                 y_axis = y_axis[1:] + (y_axis[1] - y_axis[0]) / 2
    #                 # y_axis = y_axis + (y_axis[1] - y_axis[0]) / 2
    #                 H = np.vstack([H, H[0, :]])
    #                 # H = np.hstack(
    #                 #     [H, np.roll(H[:, 0], H.shape[1] // 2)[:, None]])
    #                 H = H / np.sum(H.ravel())

    #                 X, Y = np.meshgrid(np.rad2deg(x_axis), np.rad2deg(y_axis))

    #                 # print(X.shape)
    #                 # print(Y.shape)
    #                 # print(H.shape)
    #                 for h_array, x_array, y_array in zip(H.T, X, Y):
    #                     for h, x, y in zip(h_array, x_array, y_array):
    #                         f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
    #                     f.write('\n')

    #             # polar hist
    #             df_[f"x"] = np.rad2deg(df_sub.phi_x.iloc[0])
    #             df_[f"r_{r:.2f}_s_solved_phi_h"] = df_sub.phi_h.iloc[0]
    #             df_[f"r_{r:.2f}_s_solved_incl_h"] = df_sub.incl_h.iloc[0]
    #             df_[f"r_{r:.2f}_s_init_phi_h"] = df_sub.phi_init_h.iloc[0]
    #             df_[f"r_{r:.2f}_s_init_incl_h"] = df_sub.incl_init_h.iloc[0]

    #         df_.to_csv(os.path.join(args.input, "hist",
    #                                 f"cube_stats_p_{psi}_o_{omega}_.csv"),
    #                    index=False,
    #                    float_format='%.4f')
