#! /usr/bin/env python3

import argparse
import glob
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import tqdm

import fastpli.analysis
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
CONFIG = parameter.get_tupleware()

hist_polar_bin = lambda n: np.linspace(
    -np.pi / 2, np.pi / 2, n + 1, endpoint=True)


def polar_hist(phi, theta):
    phi = np.array(phi)
    theta = np.array(theta)

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


def calc_omega_stat(p, t):

    v0 = np.array([np.cos(p) * np.sin(t), np.sin(p) * np.sin(t), np.cos(t)])
    v1 = v0[:, 0].copy()

    for v in v0.T[1:, :]:
        s = np.dot(v1, v)

        if s > 0:
            v1 += v
        else:
            v1 -= v
    v1 /= np.linalg.norm(v1)

    # print(v1)

    data = np.empty(v0.shape[1])

    for i in range(v0.shape[1]):
        d = np.abs(np.dot(v0[:, i], v1))  # because orientation
        data[i] = np.arccos(d)

    return v1, np.mean(data), np.std(data), np.quantile(data, [0.25, 0.5, 0.75])


def run(df):

    file = df["fiber"]

    phis, thetas = models.fb_ori_from_file(file, 0, 0, CONFIG.simulation.voi)
    domegas = [calc_omega_stat(p, t) for p, t in zip(phis, thetas)]
    if len(domegas) == 1:
        domegas.append([np.array([0, 0, 0]), 0, 0, [0, 0, 0]])

    phi, theta = models.ori_from_file(file, 0, 0, CONFIG.simulation.voi)
    phi_x, phi_h, incl_x, incl_h = polar_hist(phi, theta)

    # 2d hist
    h, x, y, _ = fastpli.analysis.orientation.histogram(phi,
                                                        theta,
                                                        n_phi=36 * 2,
                                                        n_theta=18,
                                                        weight_area=True)

    # INIT
    file = file.replace("solved", "init")
    file = file.replace("tmp", "init")

    phis, thetas = models.fb_ori_from_file(file, 0, 0, CONFIG.simulation.voi)
    domegas_init = [calc_omega_stat(p, t) for p, t in zip(phis, thetas)]
    if len(domegas_init) == 1:
        domegas_init.append([np.array([0, 0, 0]), 0, 0, [0, 0, 0]])

    phi, theta = models.ori_from_file(file, 0, 0, CONFIG.simulation.voi)
    phi_init_x, phi_init_h, incl_init_x, incl_init_h = polar_hist(phi, theta)

    # print(domegas)
    # print(domegas_init)

    return pd.DataFrame([[
        df.omega, df.psi, df.radius, phi_x, phi_h, incl_x, incl_h, phi_init_x,
        phi_init_h, incl_init_x, incl_init_h, h, x, y, df.rep_n % 24,
        domegas[0][0], domegas[0][1], domegas[0][2], domegas[0][3][0],
        domegas[0][3][1], domegas[0][3][2], domegas[1][0], domegas[1][1],
        domegas[1][2], domegas[1][3][0], domegas[1][3][1], domegas[1][3][2],
        domegas_init[0][0], domegas_init[0][1], domegas_init[0][2],
        domegas_init[0][3][0], domegas_init[0][3][1], domegas_init[0][3][2],
        domegas_init[1][0], domegas_init[1][1], domegas_init[1][2],
        domegas_init[1][3][0], domegas_init[1][3][1], domegas_init[1][3][2]
    ]],
                        columns=[
                            "omega", "psi", "radius", "phi_x", "phi_h",
                            "incl_x", "incl_h", "phi_init_x", "phi_init_h",
                            "incl_init_x", "incl_init_h", "hist_2d_h",
                            "hist_2d_x", "hist_2d_y", "rep_n", "omega_0",
                            "omega_mean_0", "omega_std_0", "omega_25_0",
                            "omega_50_0", "omega_75_0", "omega_1",
                            "omega_mean_1", "omega_std_1", "omega_25_1",
                            "omega_50_1", "omega_75_1", "omega_init_0",
                            "omega_init_mean_0", "omega_init_std_0",
                            "omega_init_25_0", "omega_init_50_0",
                            "omega_init_75_0", "omega_init_1",
                            "omega_init_mean_1", "omega_init_std_1",
                            "omega_init_25_1", "omega_init_50_1",
                            "omega_init_75_1"
                        ])


if __name__ == "__main__":

    if False or not os.path.isfile(
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

        # exit(1)

        df = pd.concat(df, ignore_index=True)
        df.to_pickle(os.path.join(args.input, "hist", "cube_2pop.pkl"))
    else:
        print("loading data")
        df = pd.read_pickle(os.path.join(args.input, "hist", "cube_2pop.pkl"))

    # print(df.columns)
    # print(df.dtypes)
    df_ = df[[
        'omega',
        'psi',
        'radius',
        'rep_n',
        'omega_mean_0',
        'omega_std_0',
        'omega_25_0',
        'omega_50_0',
        'omega_75_0',
        'omega_mean_1',
        'omega_std_1',
        'omega_25_1',
        'omega_50_1',
        'omega_75_1',
        'omega_init_mean_0',
        'omega_init_std_0',
        'omega_init_25_0',
        'omega_init_50_0',
        'omega_init_75_0',
        'omega_init_mean_1',
        'omega_init_std_1',
        'omega_init_25_1',
        'omega_init_50_1',
        'omega_init_75_1',
    ]]
    # print(df_)
    df_.to_csv(os.path.join(args.input, "hist", 'omegas.csv'))

    for o in df.omega.unique():
        df_ = df[df.omega == o]
        print(o)
        print(
            # df_.omega_init_0,
            f'{np.rad2deg(np.mean(df_.omega_init_mean_0)):.1f} +- {np.rad2deg(np.std(df_.omega_init_mean_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_std_0)):.1f} +- {np.rad2deg(np.std(df_.omega_init_std_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_25_0)):.1f} +- {np.rad2deg(np.std(df_.omega_init_25_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_50_0)):.1f} +- {np.rad2deg(np.std(df_.omega_init_50_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_75_0)):.1f} +- {np.rad2deg(np.std(df_.omega_init_75_0)):.1f},'
        )
        print(
            # df_.omega_0,
            f'{np.rad2deg(np.mean(df_.omega_mean_0)):.1f} +- {np.rad2deg(np.std(df_.omega_mean_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_std_0)):.1f} +- {np.rad2deg(np.std(df_.omega_std_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_25_0)):.1f} +- {np.rad2deg(np.std(df_.omega_25_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_50_0)):.1f} +- {np.rad2deg(np.std(df_.omega_50_0)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_75_0)):.1f} +- {np.rad2deg(np.std(df_.omega_75_0)):.1f},'
        )
        print(
            # df_.omega_init_1,
            f'{np.rad2deg(np.mean(df_.omega_init_mean_1)):.1f} +- {np.rad2deg(np.std(df_.omega_init_mean_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_std_1)):.1f} +- {np.rad2deg(np.std(df_.omega_init_std_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_25_1)):.1f} +- {np.rad2deg(np.std(df_.omega_init_25_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_50_1)):.1f} +- {np.rad2deg(np.std(df_.omega_init_50_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_init_75_1)):.1f} +- {np.rad2deg(np.std(df_.omega_init_75_1)):.1f},'
        )
        print(
            # df_.omega_1,
            f'{np.rad2deg(np.mean(df_.omega_mean_1)):.1f} +- {np.rad2deg(np.std(df_.omega_mean_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_std_1)):.1f} +- {np.rad2deg(np.std(df_.omega_std_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_25_1)):.1f} +- {np.rad2deg(np.std(df_.omega_25_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_50_1)):.1f} +- {np.rad2deg(np.std(df_.omega_50_1)):.1f},',
            f'{np.rad2deg(np.mean(df_.omega_75_1)):.1f} +- {np.rad2deg(np.std(df_.omega_75_1)):.1f},'
        )

    # df_ = df[[
    #     'omega',
    #     'psi',
    #     'radius',
    #     'rep_n',
    # ]]
    # df_.to_csv('test.csv')
    exit(1)

    for omega in df.omega.unique():
        for psi in df[df.omega == omega].psi.unique():
            for rep_n in df.rep_n.unique():
                df_ = pd.DataFrame()
                for r in df.radius.unique():

                    print(rep_n)

                    df_sub = df.query(
                        "radius == @r & omega == @omega & psi == @psi & rep_n == @rep_n"
                    )
                    if len(df_sub) != 1:
                        warnings.warn(
                            f"len(df_sub) != 1: {r} {omega} {psi} {rep_n}")
                        print(df_sub)
                        continue

                    # 2d hist
                    with open(
                            os.path.join(
                                args.input, "hist",
                                f"cube_stats_p_{psi}_o_{omega}_r_{r}_rep_{rep_n}_.dat"
                            ), "w") as f:

                        H = df_sub.hist_2d_h.iloc[0]
                        x_axis = df_sub.hist_2d_x.iloc[0]
                        y_axis = df_sub.hist_2d_y.iloc[0]

                        x_axis = x_axis[:-1] + (x_axis[1] - x_axis[0]) / 2
                        y_axis = y_axis[:-1] + (y_axis[1] - y_axis[0]) / 2
                        H = H.T / np.sum(H.ravel())

                        X, Y = np.meshgrid(np.rad2deg(x_axis),
                                           np.rad2deg(y_axis))

                        for h_array, x_array, y_array in zip(H, X, Y):
                            for h, x, y in zip(h_array, x_array, y_array):
                                if y <= 90:
                                    f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
                            f.write('\n')

                    with open(
                            os.path.join(
                                args.input, "hist",
                                f"cube_stats_p_{psi}_o_{omega}_r_{r}_rep_{rep_n}_.dat"
                            ), "w") as f:

                        H = df_sub.hist_2d_h.iloc[0]
                        x_axis = df_sub.hist_2d_x.iloc[0]
                        y_axis = df_sub.hist_2d_y.iloc[0]

                        x_axis = x_axis[:-1] + (x_axis[1] - x_axis[0]) / 2
                        y_axis = y_axis[:-1] + (y_axis[1] - y_axis[0]) / 2
                        H = H.T / np.sum(H.ravel())

                        X, Y = np.meshgrid(np.rad2deg(x_axis),
                                           np.rad2deg(y_axis))

                        # print(X.shape)
                        # print(Y.shape)
                        # print(H.shape)
                        H /= np.amax(H)
                        for h_array, x_array, y_array in zip(H, X, Y):
                            for h, x, y in zip(h_array, x_array, y_array):
                                if y <= 90:
                                    f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
                            f.write('\n')

                    # polar hist
                    df_[f"x"] = np.rad2deg(df_sub.phi_x.iloc[0])
                    df_[f"r_{r:.2f}_s_solved_phi_h"] = df_sub.phi_h.iloc[0]
                    df_[f"r_{r:.2f}_s_solved_incl_h"] = df_sub.incl_h.iloc[0]
                    df_[f"r_{r:.2f}_s_init_phi_h"] = df_sub.phi_init_h.iloc[0]
                    df_[f"r_{r:.2f}_s_init_incl_h"] = df_sub.incl_init_h.iloc[0]

                df_.to_csv(os.path.join(
                    args.input, "hist",
                    f"cube_stats_p_{psi}_o_{omega}_rep_{rep_n}_.csv"),
                           index=False,
                           float_format='%.4f')
