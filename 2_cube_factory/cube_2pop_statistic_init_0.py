''' find best arbeitspunkt for solver
'''

import numpy as np
import itertools
import h5py
import glob
import gc
import os

import pandas as pd
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD
import random
random.seed(42)

import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

files = glob.glob("output/cube_stat/*.h5")

df = []
for file in files:

    file_name = file[:-3]

    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])
    r = float(file.split("_r_")[1].split("_")[0])
    v0 = float(file.split("_v0_")[1].split("_")[0])
    fl = float(file.split("_fl_")[1].split("_")[0])
    fr = float(file.split("_fr_")[1].split("_")[0])
    state = file.split(".")[-2].split(".")[-1]

    print(omega, psi, r, v0, fl, fr, state)

    # f"output/cube_stat/cube_2pop_statistic_psi_{psi:.2f}_omega_" \
    #         f"{omega:.2f}_r_{omega:.2f}_v0_{omega:.2f}_fl_{omega:.1f}_fr_{omega:.1f}_.{state}"

    if os.path.isfile(file_name + '.pkl'):
        continue

    fbs = fastpli.io.fiber_bundles.load(file_name + '.h5')
    with h5py.File(file_name + ".h5", "r") as h5f:
        meta = {
            "obj_mean_length": h5f['/'].attrs['obj_mean_length'],
            "obj_min_radius": h5f['/'].attrs['obj_min_radius']
        }
        if state != "init":
            meta = {
                **meta,
                **{
                    "overlap": h5f['/'].attrs['overlap'],
                    "num_col_obj": h5f['/'].attrs['num_col_obj'],
                    "num_obj": h5f['/'].attrs['num_obj'],
                    "num_steps": h5f['/'].attrs['num_steps'],
                }
            }

    fbs_ = fbs.copy()
    fbs_ = fastpli.objects.fiber_bundles.Cut(fbs_, [[-30] * 3, [30] * 3])
    phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs_)

    df.append(
        pd.DataFrame([[
            omega, psi, r, v0, fl, fr, state, meta,
            phi.astype(np.float32).ravel(),
            theta.astype(np.float32).ravel()
        ]],
                     columns=[
                         "omega", "psi", "r", "v0", "fl", "fr", "state", "meta",
                         "phi", "theta"
                     ]))

df.to_pickle(file_name + '.pkl')
