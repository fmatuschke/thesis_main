''' extracts the ground truth distribution from the models with respect to
    simulated rotations
'''

# TODO: volume = 60

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

parameters = list(
    itertools.product(["init", "solved"], df.omega.unique(), df.psi.unique()))
# random.shuffle(parameters)
parameters = parameters[comm.Get_rank()::comm.Get_size()]

with tqdm(total=len(parameters) * len(f0_incs)) as pbar:
    for state, omega, psi in parameters:
        if os.path.isfile(
                f"output/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.{state}.pkl"
        ):
            pbar.update(len(f0_incs))
            continue
        if not os.path.isfile(
                f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.{state}.h5"
        ):
            pbar.update(len(f0_incs))
            continue

        fbs = fastpli.io.fiber_bundles.load(
            f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.{state}.h5"
        )

        with h5py.File(
                f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.{state}.h5",
                "r") as h5f:

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

        df_sub = df[(df.omega == omega) & (df.psi == psi)]
        df_ = []
        if state == "init":
            f0_inc = 0
            f1_rot = 0

            fbs_ = fbs.copy()
            for v in [120, 60]:  # ascending order!
                fbs_ = fastpli.objects.fiber_bundles.Cut(
                    fbs_, [[-v / 2] * 3, [v / 2] * 3])
                phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs_)

                df_.append(
                    pd.DataFrame([[
                        f0_inc, f1_rot, v, meta,
                        phi.astype(np.float32).ravel(),
                        theta.astype(np.float32).ravel()
                    ]],
                                 columns=[
                                     "f0_inc", "f1_rot", "v", "meta", "phi",
                                     "theta"
                                 ]))
            pbar.update(len(f0_incs))
        else:
            for f0_inc in f0_incs:
                for f1_rot in df_sub[df_sub.f0_inc == f0_inc].f1_rot.unique():
                    rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
                    rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
                    rot = np.dot(rot_inc, rot_phi)
                    fbs_ = fastpli.objects.fiber_bundles.Rotate(fbs, rot)

                    for v in [120, 60]:  # ascending order!
                        fbs_ = fastpli.objects.fiber_bundles.Cut(
                            fbs_, [[-v / 2] * 3, [v / 2] * 3])
                        phi, theta = fastpli.analysis.orientation.fiber_bundles(
                            fbs_)

                        df_.append(
                            pd.DataFrame([[
                                f0_inc, f1_rot, v, meta,
                                phi.astype(np.float32).ravel(),
                                theta.astype(np.float32).ravel()
                            ]],
                                         columns=[
                                             "f0_inc", "f1_rot", "v", "meta",
                                             "phi", "theta"
                                         ]))
                pbar.update()

        df_ = pd.concat(df_, ignore_index=True)
        df_.to_pickle(
            f"output/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.{state}.pkl"
        )
        gc.collect()
