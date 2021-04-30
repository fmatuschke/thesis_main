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
import fastpli.analysis

import helper.file
import helper.circular

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


def hist_polar_bin(n):
    return np.linspace(-np.pi / 2, np.pi / 2, n + 1, endpoint=True)


def run(file):
    # fbs = fastpli.io.fiber_bundles.load(file)
    # state_ = file.split(f".h5")[0].split(".")[-1]
    r = helper.file.value(file, "r")
    v0 = helper.file.value(file, "v0")
    with h5py.File(file, "r") as h5f:

        if h5f['/'].attrs['num_steps'] == 0:
            state = "init"
        elif h5f['/'].attrs['overlap'] == 0 and h5f['/'].attrs[
                'num_col_obj'] == 0:
            state = "solved"
        else:
            state = "not_solved"
            warnings.warn("not solved")

        # if state != state_:
        #     print(state, state_)
        #     raise ValueError("state != state_")

        # TODO: only take h5 values
        omega = h5f['/'].attrs['omega']
        psi = h5f['/'].attrs['psi']
        step = -1
        overlap = -1
        num_obj = -1
        num_col_obj = -1
        num_steps = -1
        obj_mean_length = -1
        obj_min_radius = -1
        time = -1
        if state != "init":
            omega = h5f['/'].attrs['omega']
            psi = h5f['/'].attrs['psi']
            step = h5f['/'].attrs['step']
            overlap = h5f['/'].attrs['overlap']
            num_obj = h5f['/'].attrs['num_obj']
            num_col_obj = h5f['/'].attrs['num_col_obj']
            num_steps = h5f['/'].attrs['num_steps']
            obj_mean_length = h5f['/'].attrs['obj_mean_length']
            obj_min_radius = h5f['/'].attrs['obj_min_radius']
            time = h5f['/'].attrs['time']

    return pd.DataFrame([[
        omega, psi, v0, r, obj_mean_length, obj_min_radius, overlap,
        num_col_obj, num_obj, num_steps, step, time, state, file
    ]],
                        columns=[
                            "omega",
                            "psi",
                            "v0",
                            "radius",
                            "obj_mean_length",
                            "obj_min_radius",
                            "overlap",
                            "num_col_obj",
                            "num_obj",
                            "num_steps",
                            "step",
                            "time",
                            "state",
                            "fiber",
                        ])


if __name__ == "__main__":

    files = glob.glob(os.path.join(args.input, "*.h5"))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm(pool.imap_unordered(run, files),
                            total=len(files),
                            smoothing=0.1)
        ]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(os.path.join(args.input, "cube_2pop.pkl"))
