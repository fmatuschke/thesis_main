import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import tqdm
import fastpli.tools

import models
import parameter

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

CONFIG = parameter.get_tupleware()
# MODEL = 'cube_2pop_135_rc1'

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

        if d > 1 and d < 1.0001:
            d = 1

        if d < -1 and d > -1.0001:
            d = -1

        data[i] = np.arccos(d)

    return v1, np.mean(data), np.std(data), np.quantile(data, [0.25, 0.5, 0.75])


def run(df):
    df = df[1]

    domegas = calc_omega_stat(df.rofl_dir, df.rofl_inc)

    return pd.DataFrame([[
        df.omega, df.psi, df.radius, df.f0_inc, df.f1_rot, df.rep_n, domegas[0],
        domegas[1], domegas[2], domegas[3][0], domegas[3][1], domegas[3][2]
    ]],
                        columns=[
                            "omega", "psi", "radius", "f0_inc", "f1_rot",
                            "rep_n", "omega_0", "omega_mean_0", "omega_std_0",
                            "omega_25_0", "omega_50_0", "omega_75_0"
                        ])


def main():
    df = pd.read_pickle(
        os.path.join(args.input, "analysis", f"cube_2pop_simulation.pkl"))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, df.iterrows()),
                                 total=len(df),
                                 smoothing=0.1)
        ]

    df = pd.concat(df, ignore_index=True)

    # df.to_pickle(
    #     os.path.join(args.input, "analysis", "cube_2pop_simulation_domega.pkl"))

    df_ = df[[
        'omega',
        'psi',
        'radius',
        'f0_inc',
        'f1_rot',
        'rep_n',
        'omega_mean_0',
        'omega_std_0',
        'omega_25_0',
        'omega_50_0',
        'omega_75_0',
    ]]

    print("f0")
    for f0 in sorted(list(df.f0_inc.unique())):
        df_ = df[df.f0_inc == f0]
        print(
            # df_.omega_0,
            f'{f0}: \t {np.rad2deg(np.mean(df_.omega_mean_0)):.2f} +- {np.rad2deg(np.std(df_.omega_mean_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_std_0)):.2f} +- {np.rad2deg(np.std(df_.omega_std_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_25_0)):.2f} +- {np.rad2deg(np.std(df_.omega_25_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_50_0)):.2f} +- {np.rad2deg(np.std(df_.omega_50_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_75_0)):.2f} +- {np.rad2deg(np.std(df_.omega_75_0)):.2f},'
        )

    print("f1")
    for f1 in sorted(list(df.f1_rot.unique())):
        df_ = df[df.f1_rot == f1]
        print(
            # df_.omega_0,
            f'{f1}: \t {np.rad2deg(np.mean(df_.omega_mean_0)):.2f} +- {np.rad2deg(np.std(df_.omega_mean_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_std_0)):.2f} +- {np.rad2deg(np.std(df_.omega_std_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_25_0)):.2f} +- {np.rad2deg(np.std(df_.omega_25_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_50_0)):.2f} +- {np.rad2deg(np.std(df_.omega_50_0)):.2f},',
            f'{np.rad2deg(np.mean(df_.omega_75_0)):.2f} +- {np.rad2deg(np.std(df_.omega_75_0)):.2f},'
        )


if __name__ == "__main__":
    main()
