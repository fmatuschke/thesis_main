''' extracts the ground truth distribution from the models with respect to
    simulated rotations
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

# get rotation values from simulation data
df = pd.read_pickle("output/analysis/cube_2pop_simulation_LAP_model_p_.pkl")
f0_incs = df.f0_inc.unique()

parameters = list(itertools.product(df.omega.unique(), df.psi.unique()))
random.shuffle(parameters)
parameters = parameters[comm.Get_rank()::comm.Get_size()]

with tqdm(total=len(parameters) * len(f0_incs)) as pbar:
    for omega, psi in parameters:
        if os.path.isfile(
                f"output/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.pkl"
        ):
            continue

        fbs = fastpli.io.fiber_bundles.load(
            f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.h5"
        )

        df_sub = df[(df.omega == omega) & (df.psi == psi)]
        df_ = []
        for f0_inc in f0_incs:
            for f1_rot in df_sub[df_sub.f0_inc == f0_inc].f1_rot.unique():
                rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
                rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
                rot = np.dot(rot_inc, rot_phi)
                fbs_ = fastpli.objects.fiber_bundles.Rotate(fbs, rot)
                fbs_ = fastpli.objects.fiber_bundles.Cut(
                    fbs_, [[-30] * 3, [30] * 3])
                phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs_)

                df_.append(
                    pd.DataFrame([[
                        omega, psi, f0_inc, f1_rot,
                        phi.astype(np.float32).ravel(),
                        theta.astype(np.float32).ravel()
                    ]],
                                 columns=[
                                     "omega", "psi", "f0_inc", "f1_rot", "phi",
                                     "theta"
                                 ]))
            pbar.update()

        df_ = pd.concat(df_, ignore_index=True)
        df_.to_pickle(
            f"output/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.pkl"
        )
        gc.collect()
