#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import itertools
import argparse
import warnings
import h5py
import glob
import os

import pandas as pd
from tqdm import tqdm

import fastpli.io
import fastpli.objects
import fastpli.analysis
import fastpli.simulation

import helper.circular

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()
os.makedirs(os.path.join(args.input, "time_evolve"), exist_ok=True)


def run(file):
    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])
    r = float(file.split("_r_")[1].split("_")[0])
    v0 = float(file.split("_v0_")[1].split("_")[0])
    fl = float(file.split("_fl_")[1].split("_")[0])
    fr = float(file.split("_fr_")[1].split("_")[0])
    n = int(file.split("_n_")[1].split("_")[0])
    state = file.split(".")[-2].split(".")[-1]

    if state == "init":
        return pd.DataFrame()

    with h5py.File(file, "r") as h5f:
        times = h5f['/'].attrs['times']
        steps = h5f['/'].attrs['steps']
        overlaps = h5f['/'].attrs['overlaps']
        num_objs = h5f['/'].attrs['num_objs']
        num_col_objs = h5f['/'].attrs['num_col_objs']

    return pd.DataFrame([[
        omega,
        psi,
        r,
        v0,
        fl,
        fr,
        n,
        state,
        times,
        steps,
        overlaps,
        num_objs,
        num_col_objs,
    ]],
                        columns=[
                            "omega",
                            "psi",
                            "r",
                            "v0",
                            "fl",
                            "fr",
                            "n",
                            "state",
                            "times",
                            "steps",
                            "overlaps",
                            "num_objs",
                            "num_col_objs",
                        ])


def run_mean_std(parameter):
    r, omega = parameter[0], parameter[1]
    for psi in df[df.omega == omega].psi.unique():
        for fr in df.fr.unique():
            for fl in df.fl.unique():
                df_ = pd.DataFrame()
                names = []
                for n in df.n.unique():
                    times = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
                    ).times.iloc[0]
                    steps = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
                    ).steps.iloc[0]
                    overlaps = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
                    ).overlaps.iloc[0]
                    num_objs = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
                    ).num_objs.iloc[0]

                    num_col_objs = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
                    ).num_col_objs.iloc[0]

                    num_col_objs_ = num_col_objs.copy()
                    num_col_objs_[num_col_objs_ == 0] = 1
                    overlaps_frac = overlaps / num_col_objs_

                    # length between different "n" can vary
                    df__ = pd.DataFrame({
                        f'steps_{n}': steps,
                        f'times_{n}': times,
                        f'overlaps_{n}': overlaps,
                        f'num_objs_{n}': num_objs,
                        f'num_col_objs_{n}': num_col_objs,
                        f'overlaps_frac_{n}': overlaps_frac,
                    })

                    names.append(f"steps_{n}")
                    names.append(f"times_{n}")
                    names.append(f"overlaps_{n}")
                    names.append(f"num_objs_{n}")
                    names.append(f"num_col_objs_{n}")
                    names.append(f"overlaps_frac_{n}")

                    df_ = pd.concat([df_, df__], ignore_index=True, axis=1)
                df_.columns = names

                df__ = pd.DataFrame()
                for name in [
                        "steps", "times", "overlaps", "num_objs",
                        "num_col_objs", "overlaps_frac"
                ]:
                    df_tmp = pd.DataFrame()
                    for n in df.n.unique():
                        df_tmp[f"{name}_{n}"] = df_[f"{name}_{n}"]

                    df__[f"{name}_mean"] = df_tmp.T.mean(axis=0)
                    df__[f"{name}_std"] = df_tmp.T.std(axis=0)

                df__.to_csv(
                    os.path.join(
                        args.input, "time_evolve",
                        f'cube_stat_time_evolve_r_{r}_psi_{psi}_fr_{fr}_fl_{fl}_.csv'
                    ),
                    index=False,
                )


if __name__ == "__main__":
    files = glob.glob(os.path.join(args.input, "*.solved.h5"))

    run.flag = False
    run.num_p = args.num_proc
    with mp.Pool(processes=run.num_p) as pool:
        df = [
            d for d in tqdm(pool.imap_unordered(run, files),
                            total=len(files),
                            smoothing=0.1)
        ]
        df = pd.concat(df, ignore_index=True)

    df.to_pickle(os.path.join(args.input,
                              "parameter_statistic_time_evolve.pkl"))

    parameters = list(itertools.product(df.r.unique(), df.omega.unique()))
    with mp.Pool(processes=run.num_p) as pool:
        [
            _ for _ in tqdm(pool.imap_unordered(run_mean_std, parameters),
                            total=len(parameters),
                            smoothing=0.1)
        ]
