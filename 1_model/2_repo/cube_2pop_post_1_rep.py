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
parser.add_argument("-N",
                    "--num_repo",
                    default=1,
                    type=int,
                    help="Number of processes.")

args = parser.parse_args()


def run(file):
    fbs = fastpli.io.fiber_bundles.load(file)
    r = helper.file.value(file, "r")
    v0 = helper.file.value(file, "v0")
    omega = helper.file.value(file, "omega")
    psi = helper.file.value(file, "psi")
    rep_n = int(helper.file.value(file, "n"))

    return pd.DataFrame([[omega, psi, v0, r, rep_n % args.num_repo]],
                        columns=[
                            "omega",
                            "psi",
                            "v0",
                            "radius",
                            "rep_n",
                        ])


if __name__ == "__main__":

    files = glob.glob(os.path.join(args.input, "*solved.h5"))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm(pool.imap_unordered(run, files),
                            total=len(files),
                            smoothing=0.1)
        ]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(os.path.join(args.input, "cube_2pop_radii_dist.pkl"))
