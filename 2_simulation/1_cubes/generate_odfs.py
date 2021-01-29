#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import itertools
import argparse
import h5py
import os
import glob

import odf
import models

import tqdm

import pretty_errors

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

# for microscope, species, model in list(
#                 itertools.product(["PM", "LAP"], ["Roden", "Vervet", "Human"],
#                                   ["r", "p"])):


def run(file):
    df = []

    with h5py.File(file, 'r') as h5f:
        for microscope, species, model in list(
                itertools.product(["PM"], ["Vervet"], ["r"])):
            h5f_sub = h5f[f"/{microscope}/{species}/{model}/"]

            # ROFL
            rofl_direction = h5f_sub['analysis/rofl/direction'][...]
            rofl_inclination = h5f_sub['analysis/rofl/inclination'][...]

            path, name = os.path.split(file)
            odf.table(
                os.path.join(
                    path, "odfs",
                    name[:-3] + f"_{microscope}_{species}_{model}.rofl.dat"),
                rofl_direction.ravel(), rofl_inclination.ravel(), 6, 21, 42)

            # GT
            f0_inc = h5f_sub.attrs["parameter/f0_inc"]
            f1_rot = h5f_sub.attrs["parameter/f1_rot"]
            m_phi, m_theta = models.ori_from_file(
                h5f_sub.attrs['parameter/fiber_path'], f0_inc, f1_rot, 60)

            path, name = os.path.split(file)
            odf.table(
                os.path.join(
                    path, "odfs",
                    name[:-3] + f"_{microscope}_{species}_{model}.gt.dat"),
                m_phi.ravel(), m_theta.ravel(), 6, 21, 42)


if __name__ == "__main__":
    os.makedirs(os.path.join(args.input, "odfs"), exist_ok=True)
    files = glob.glob(os.path.join(args.input, "*.h5"))
    files = list(filter(lambda x: "rot_0.00" in x, files))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(run, files),
                                 total=len(files),
                                 smoothing=0.1)
        ]
