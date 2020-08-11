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
