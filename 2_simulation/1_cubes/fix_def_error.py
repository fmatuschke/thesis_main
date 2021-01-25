#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import itertools
import argparse
import h5py
import os
import glob
import numba

import helper.file
import pandas as pd
import tqdm

import fastpli.tools

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


def run(file):

    with h5py.File(file, 'a') as h5f:
        for microscope, species, model in list(
                itertools.product(["PM", "LAP"], ["Roden", "Vervet", "Human"],
                                  ["r", "p"])):
            h5f_sub = h5f[f"/{microscope}/{species}/{model}/"]

            for t in range(5):
                data = h5f_sub['analysis/epa/' + str(t) + '/direction']
                data[...] = np.deg2rad(data[...])
            data = h5f_sub['analysis/rofl/direction']
            data[...] = np.deg2rad(data[...])
            data = h5f_sub['analysis/rofl/inclination']
            data[...] = np.deg2rad(data[...])


if __name__ == "__main__":
    os.makedirs(os.path.join(args.input, "analysis"), exist_ok=True)
    files = glob.glob(os.path.join(args.input, "*.h5"))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(run, files),
                                 total=len(files),
                                 smoothing=0.1)
        ]
