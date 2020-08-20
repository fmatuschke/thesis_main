#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import gc
import os

import pandas as pd
from tqdm import tqdm

import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

import helper.file
import helper.circular
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


def hist_bin(n):
    return np.linspace(0, np.pi, n + 1, endpoint=True)


def run(file):
    fbs = fastpli.io.fiber_bundles.load(file)
    r = helper.file.value(file, "r")
    v0 = helper.file.value(file, "v0")
    with h5py.File(file, "r") as h5f:
        omega = h5f['/'].attrs['omega']
        psi = h5f['/'].attrs['psi']

        if h5f['/'].attrs['num_steps'] == 0:
            state = "init"
        elif h5f['/'].attrs['overlap'] == 0:
            state = "solved"
        else:
            state = "solved"
            warnings.warn("not solved")

        data = {
            "obj_mean_length": h5f['/'].attrs['obj_mean_length'],
            "obj_min_radius": h5f['/'].attrs['obj_min_radius']
        }
        if state != "init":
            data = {
                **data,
                **{
                    "overlap": h5f['/'].attrs['overlap'],
                    "num_col_obj": h5f['/'].attrs['num_col_obj'],
                    "num_obj": h5f['/'].attrs['num_obj'],
                    "num_steps": h5f['/'].attrs['num_steps'],
                }
            }

    data = {**data, **{"v0": v0, "r": r, "state": state, "file": file}}

    return data


if __name__ == "__main__":

    files = glob.glob(os.path.join(args.input, "*.h5"))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm(
                pool.imap_unordered(run, files), total=len(files), smoothing=0)
        ]
    # df = pd.concat(df, ignore_index=True)
    df = pd.DataFrame(df)
    df.to_pickle(os.path.join(args.input, "cube_2pop.pkl"))
