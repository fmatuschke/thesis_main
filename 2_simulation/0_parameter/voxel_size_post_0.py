#! /usr/bin/env python3

import numpy as np
import pandas as pd
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import os

import tqdm
import helper.circular
import fastpli.analysis
import fastpli.tools
import fastpli.io

import models

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
parser.add_argument("-p",
                    "--num_proc",
                    required=True,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()


def run_orientation(file):
    with h5py.File(file, 'r') as h5f:
        omega = h5f['/'].attrs['omega']
        psi = h5f['/'].attrs['psi']
        radius = h5f['/'].attrs['radius']
        v0 = h5f['/'].attrs['v0']
        f0_inc = h5f['/'].attrs['f0_inc']
        f1_rot = h5f['/'].attrs['f1_rot']
        pixel_size = h5f['/'].attrs['pixel_size']
        with h5py.File(str(h5f['fiber_bundles'][...]), 'r') as h5f_:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
            if h5f_['/'].attrs["psi"] != psi or omega == h5f_['/'].attrs[
                    "omega"] != omega:
                raise ValueError("FAIL")

    fiber_bundles = models.rotate(fiber_bundles, f0_inc, f1_rot)
    fiber_bundles = fastpli.objects.fiber_bundles.Cut(
        fiber_bundles, [[-pixel_size / 2, -pixel_size / 2, -30],
                        [pixel_size / 2, pixel_size / 2, 30]])
    phi, theta = fastpli.analysis.orientation.fiber_bundles(fiber_bundles)

    return pd.DataFrame(
        [[radius, v0, omega, psi, f0_inc, f1_rot, pixel_size, phi, theta]],
        columns=[
            "radius", "v0", "omega", "psi", "f0_inc", "f1_rot", "pixel_size",
            "f_phi", "f_theta"
        ])


def run(file):
    df = []

    with h5py.File(file, 'r') as h5f:
        omega = h5f['/'].attrs['omega']
        psi = h5f['/'].attrs['psi']
        radius = h5f['/'].attrs['radius']
        v0 = h5f['/'].attrs['v0']
        f0_inc = h5f['/'].attrs['f0_inc']
        f1_rot = h5f['/'].attrs['f1_rot']
        pixel_size = h5f['/'].attrs['pixel_size']

        for voxel_size in h5f['simpli']:
            for model in h5f[f'simpli/{voxel_size}']:
                for n in h5f[f'simpli/{voxel_size}/{model}']:
                    h5f_sub = h5f[f'simpli/{voxel_size}/{model}/{n}']
                    for m in h5f_sub[f'simulation/optic/0/']:
                        if h5f_sub[
                                f'analysis/epa/0/transmittance/{m}'].size != 1:
                            raise ValueError("FOOOO")
                        df.append(
                            pd.DataFrame(
                                [[
                                    float(voxel_size),
                                    radius,
                                    v0,
                                    model,
                                    omega,
                                    psi,
                                    f0_inc,
                                    f1_rot,
                                    pixel_size,
                                    int(n),
                                    int(m),
                                    # h5f_sub['simulation/data/0'][...],
                                    h5f_sub[f'simulation/optic/0/{m}'][...
                                                                      ].ravel(),
                                    h5f_sub[f'analysis/epa/0/transmittance/{m}']
                                    [...].ravel()[0],
                                    h5f_sub[f'analysis/epa/0/direction/{m}'][
                                        ...].ravel()[0],
                                    h5f_sub[f'analysis/epa/0/retardation/{m}'][
                                        ...].ravel()[0],
                                ]],
                                columns=[
                                    "voxel_size",
                                    "radius",
                                    "v0",
                                    "model",
                                    "omega",
                                    "psi",
                                    "f0_inc",
                                    "f1_rot",
                                    "pixel_size",
                                    "n",
                                    "m",
                                    #  "data",
                                    "optic",
                                    "epa_trans",
                                    "epa_dir",
                                    "epa_ret",
                                ]))

    return pd.concat(df, ignore_index=True)


files = glob.glob(os.path.join(args.input, "*.h5"))

with mp.Pool(processes=args.num_proc) as pool:
    df = [
        sub for sub in tqdm.tqdm(pool.imap_unordered(run_orientation, files),
                                 total=len(files))
    ]
df = pd.concat(df, ignore_index=True)
df.to_pickle(os.path.join(args.input, "voxel_size_orientations.pkl"))

with mp.Pool(processes=args.num_proc) as pool:
    df = [
        sub
        for sub in tqdm.tqdm(pool.imap_unordered(run, files), total=len(files))
    ]
df = pd.concat(df, ignore_index=True)
df.to_pickle(os.path.join(args.input, "voxel_size_post_0.pkl"))
