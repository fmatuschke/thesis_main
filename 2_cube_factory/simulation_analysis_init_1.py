''' extracts the ground truth distribution from the models with respect to
    simulated rotations
'''

import numpy as np
import h5py
import os
import glob

import pandas as pd
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD

import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

# get rotation values from simulation data
df = pd.read_pickle("output/analysis/cube_2pop_simulation_LAP_model_p_.pkl")

parameters = []
for i, psi in enumerate(df.psi.unique()):
    for j, omega in enumerate(df[df.psi == psi].omega.unique()):
        parameters.append((omega, psi))

# ground truth
for omega, psi in tqdm(parameters[comm.Get_rank()::comm.Get_size()]):
    if os.path.isfile(
            f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.pkl"
    ):
        continue

    fbs = fastpli.io.fiber_bundles.load(
        f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.h5")

    df_sub = df[(df.omega == omega) & (df.psi == psi)]
    df_ = []
    for f0_inc in tqdm(df_sub.f0_inc.unique()):
        for f1_rot in tqdm(df_sub[df_sub.f0_inc == f0_inc].f1_rot.unique(),
                           leave=False):
            rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
            rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
            rot = np.dot(rot_inc, rot_phi)
            fbs = fastpli.objects.fiber_bundles.Rotate(fbs, rot)
            fbs = fastpli.objects.fiber_bundles.Cut(fbs, [[-30] * 3, [30] * 3])
            phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs)

            df_.append(
                pd.DataFrame({
                    "omega": omega,
                    "psi": psi,
                    "f0_inc": f0_inc,
                    "f1_rot": f1_rot,
                    "phi": phi.astype(np.float32),
                    "theta": theta.astype(np.float32)
                }))

    df_ = pd.concat(df_, ignore_index=True)
    df_.to_pickle(
        f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.pkl")
