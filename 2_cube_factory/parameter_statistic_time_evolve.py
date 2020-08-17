''' 
'''

import numpy as np
import multiprocessing as mp
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


def run(file):
    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])
    r = float(file.split("_r_")[1].split("_")[0])
    v0 = float(file.split("_v0_")[1].split("_")[0])
    fl = float(file.split("_fl_")[1].split("_")[0])
    fr = float(file.split("_fr_")[1].split("_")[0])
    n = float(file.split("_n_")[1].split("_")[0])
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
                            "num_col_objs",
                        ])


files = glob.glob(os.path.join(args.input, "*.solved.h5"))

run.flag = False
run.num_p = args.num_proc
with mp.Pool(processes=run.num_p) as pool:
    df = [
        d for d in tqdm(
            pool.imap_unordered(run, files), total=len(files), smoothing=0)
    ]
    df = pd.concat(df, ignore_index=True)

df.to_pickle(os.path.join(args.input, "cube_stat_evolve.pkl"))

ys = ["times", "steps", "overlaps", "num_col_objs"]
for r in df.r.unique():
    for omega in df.omega.unique():
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
                        num_col_objs = df.query(
                            "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
                        ).num_col_objs.iloc[0]

                        df__ = pd.DataFrame({
                            f'steps_{n}': steps,
                            f'times_{n}': times,
                            f'overlaps_{n}': overlaps,
                            f'num_col_objs_{n}': num_col_objs,
                        })

                        names.append(f"steps_{int(n)}")
                        names.append(f"times_{int(n)}")
                        names.append(f"overlaps_{int(n)}")
                        names.append(f"num_col_objs_{int(n)}")

                        df_ = pd.concat([df_, df__], ignore_index=True, axis=1)

                    df_.columns = names

                    df_.to_csv(
                        os.path.join(
                            args.input,
                            f'cube_stat_time_evol_r_{r}_psi_{psi}_fr_{fr}_fl_{fl}_.csv'
                        ),
                        index=False,
                    )
