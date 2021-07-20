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

os.makedirs(os.path.join(args.input, "analysis"), exist_ok=True)

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
CONFIG = parameter.get_tupleware()


def calc_omega_stat(p, t):

    if p.size == 0:
        return np.array([0, 0, 0]), 0, 0, [0, 0, 0]

    v0 = np.array([np.cos(p) * np.sin(t), np.sin(p) * np.sin(t), np.cos(t)])
    v1 = v0[:, 0].copy()

    for v in v0.T[1:, :]:
        s = np.dot(v1, v)

        if s > 0:
            v1 += v
        else:
            v1 -= v
    v1 /= np.linalg.norm(v1)

    data = np.empty(v0.shape[1])

    for i in range(v0.shape[1]):
        d = np.abs(np.dot(v0[:, i], v1))  # because orientation
        data[i] = np.rad2deg(np.arccos(d))

    return v1, np.mean(data), np.std(data), np.quantile(data, [0.25, 0.5, 0.75])


def run(df):

    file = df["fiber"]

    phis, thetas = models.fb_ori_from_file(file, 0, 0, CONFIG.simulation.voi)
    domegas = [calc_omega_stat(p, t) for p, t in zip(phis, thetas)]
    if len(domegas) == 1:
        domegas.append([np.array([0, 0, 0]), 0, 0, [0, 0, 0]])

    return pd.DataFrame([[
        df.omega,
        df.psi,
        df.radius,
        phis,
        thetas,
        domegas[0][0],
        domegas[0][1],
        domegas[0][2],
        domegas[0][3][0],
        domegas[0][3][1],
        domegas[0][3][2],
        domegas[1][0],
        domegas[1][1],
        domegas[1][2],
        domegas[1][3][0],
        domegas[1][3][1],
        domegas[1][3][2],
    ]],
                        columns=[
                            "omega", "psi", "radius", "phis", "thetas",
                            "omega_0", "omega_mean_0", "omega_std_0",
                            "omega_25_0", "omega_50_0", "omega_75_0", "omega_1",
                            "omega_mean_1", "omega_std_1", "omega_25_1",
                            "omega_50_1", "omega_75_1"
                        ])


def main():

    if False or not os.path.isfile(
            os.path.join(args.input, "analysis", "cube_2pop_domega.pkl")):
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
        df.to_pickle(
            os.path.join(args.input, "analysis", "cube_2pop_domega.pkl"))
    else:
        print("loading data")
        df = pd.read_pickle(
            os.path.join(args.input, "analysis", "cube_2pop_domega.pkl"))

    # print(df.columns)
    # print(df.dtypes)
    df_ = df[[
        'omega', 'psi', 'phis', 'thetas', 'radius', 'omega_mean_0',
        'omega_std_0', 'omega_25_0', 'omega_50_0', 'omega_75_0', 'omega_mean_1',
        'omega_std_1', 'omega_25_1', 'omega_50_1', 'omega_75_1'
    ]]
    # print(df_)
    df_.to_csv(os.path.join(args.input, "analysis", 'omegas.csv'), index=False)

    df___ = []
    for o in df.omega.unique():
        for p in df.psi.unique():
            for r in df.radius.unique():
                df_ = df[(df.omega == o) & (df.psi == p) & (df.radius == r)]

                if len(df_) == 0:
                    continue

                if len(df_) != 1:
                    print('FOOOO')

                # print(len(df_))

                # print(o, p, r)
                # print(f'{np.mean(df_.omega_mean_0):.1f}, '
                #       f'{np.mean(df_.omega_std_0):.1f}, '
                #       f'{np.mean(df_.omega_25_0):.1f}, '
                #       f'{np.mean(df_.omega_50_0):.1f}, '
                #       f'{np.mean(df_.omega_75_0):.1f}, ')

                thetas = df_.iloc[0]['thetas']
                phis = df_.iloc[0]['phis']

                for i, (tt, pp) in enumerate(zip(thetas, phis)):
                    tt[:] = 90 - np.rad2deg(tt)
                    pp[:] = np.rad2deg(pp)

                    # center on fiber main population orientation
                    if i == 0:
                        tt[np.logical_and(pp > 90, pp < 3 * 90)] *= -1
                        pp[np.logical_and(pp > 90, pp < 3 * 90)] -= 180
                        pp[pp >= 3 * 90] -= 360
                    if i == 1:
                        tt[np.logical_and(pp > o + 90, pp < o + 3 * 90)] *= -1
                        pp[np.logical_and(pp > o + 90, pp < o + 3 * 90)] -= 180
                        pp[pp >= o + 3 * 90] -= 360

                for i, tt in enumerate(thetas):
                    if tt.size == 0:
                        thetas[i] = np.array([np.nan])
                        phis[i] = np.array([np.nan])

                if len(thetas) == 1:
                    thetas.append(np.array([np.nan]))
                    phis.append(np.array([np.nan]))

                # np.set_printoptions(suppress=True)
                # print(phis[0])

                df__ = {}
                df__['omega'] = o
                df__['psi'] = p
                df__['radius'] = r
                df__['state'] = 'solved'
                df__['pop'] = 0

                df__['incl_mean'] = np.mean(thetas[0])
                df__['incl_std'] = np.std(thetas[0])
                df__['incl_25'] = np.quantile(thetas[0], 0.25)
                df__['incl_50'] = np.quantile(thetas[0], 0.5)
                df__['incl_75'] = np.quantile(thetas[0], 0.75)
                df__['phi_mean'] = np.mean(phis[0])
                df__['phi_std'] = np.std(phis[0])
                df__['phi_25'] = np.quantile(phis[0], 0.25)
                df__['phi_50'] = np.quantile(phis[0], 0.5)
                df__['phi_75'] = np.quantile(phis[0], 0.75)
                df__['mean'] = df_.iloc[0]['omega_mean_0']
                df__['std'] = df_.iloc[0]['omega_std_0']
                df__['25'] = df_.iloc[0]['omega_25_0']
                df__['50'] = df_.iloc[0]['omega_50_0']
                df__['75'] = df_.iloc[0]['omega_75_0']
                df___.append(df__)

                if p != 1:
                    df__ = {}
                    df__['omega'] = o
                    df__['psi'] = p
                    df__['radius'] = r
                    df__['state'] = 'solved'
                    df__['pop'] = 1
                    df__['incl_mean'] = np.mean(thetas[1])
                    df__['incl_std'] = np.std(thetas[1])
                    df__['incl_25'] = np.quantile(thetas[1], 0.25)
                    df__['incl_50'] = np.quantile(thetas[1], 0.5)
                    df__['incl_75'] = np.quantile(thetas[1], 0.75)
                    df__['phi_mean'] = np.mean(phis[1])
                    df__['phi_std'] = np.std(phis[1])
                    df__['phi_25'] = np.quantile(phis[1], 0.25)
                    df__['phi_50'] = np.quantile(phis[1], 0.5)
                    df__['phi_75'] = np.quantile(phis[1], 0.75)
                    df__['mean'] = df_.iloc[0]['omega_mean_1']
                    df__['std'] = df_.iloc[0]['omega_std_1']
                    df__['25'] = df_.iloc[0]['omega_25_1']
                    df__['50'] = df_.iloc[0]['omega_50_1']
                    df__['75'] = df_.iloc[0]['omega_75_1']
                    df___.append(df__)

    df___ = pd.DataFrame(df___)

    df___.sort_values(['omega', 'psi', 'radius', 'pop', 'state'], inplace=True)

    df___.to_pickle(os.path.join(args.input, "analysis", 'omegas_ms.pkl'))
    df___.to_csv(os.path.join(args.input, "analysis", 'omegas_ms.csv'),
                 index=False)

    df___ = df___[(df___.omega == 30) | (df___.omega == 60) |
                  (df___.omega == 90)]
    df___ = df___[(df___.psi == 0.3) | (df___.psi == 0.6) | (df___.psi == 0.9)]
    df___ = df___[(df___.radius == 0.5)]
    df___.to_csv(os.path.join(args.input, "analysis", 'omegas_ms_2pop.csv'),
                 index=False)


if __name__ == "__main__":
    main()
