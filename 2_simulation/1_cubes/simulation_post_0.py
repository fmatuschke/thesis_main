#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import itertools
import argparse
import warnings
import h5py
import os
import glob

import pandas as pd
from tqdm import tqdm

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
    df = [None] * 4
    with h5py.File(file, 'r') as h5f:
        for i, (microscope, model) in enumerate(
                list(itertools.product(["PM", "LAP"], ["r", "p"]))):
            h5f_sub = h5f[microscope + '/' + model]

            radius = 1  # FIXME: !!!

            df[i] = pd.DataFrame(
                [[
                    microscope,
                    model,
                    radius,
                    # float(h5f_sub.attrs['parameter/r']),
                    float(h5f_sub.attrs['parameter/omega']),
                    float(h5f_sub.attrs['parameter/psi']),
                    float(h5f_sub.attrs['parameter/f0_inc']),
                    float(h5f_sub.attrs['parameter/f1_rot']),
                    h5f_sub['analysis/rofl/direction'][...].ravel(),
                    h5f_sub['analysis/rofl/inclination'][...].ravel(),
                    h5f_sub['analysis/rofl/t_rel'][...].ravel(),
                    h5f_sub['analysis/epa/0/transmittance'][...].ravel(),
                    h5f_sub['analysis/epa/0/direction'][...].ravel(),
                    h5f_sub['analysis/epa/0/retardation'][...].ravel()
                ]],
                columns=[
                    "microscope", "model", "r", "omega", "psi", "f0_inc",
                    "f1_rot", "rofl_dir", "rofl_inc", "rofl_trel", "epa_trans",
                    "epa_dir", "epa_ret"
                ])
    return df


if __name__ == "__main__":
    os.makedirs(os.path.join(args.input, "analysis"), exist_ok=True)
    file_list = glob.glob(os.path.join(args.input, "*.h5"))

    files = glob.glob(os.path.join(args.input, "*.h5"))
    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm(pool.imap_unordered(run, files),
                            total=len(files),
                            smoothing=0.1)
        ]

    df = [item for sublist in df for item in sublist]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(
        os.path.join(os.path.join(args.input, "analysis"),
                     f"cube_2pop_simulation.pkl"))
