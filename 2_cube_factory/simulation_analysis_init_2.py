import numpy as np
import itertools
import os

import pandas as pd
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD

import helper.spherical_harmonics
import helper.schilling

sim_path = "output/simulation/*.h5"
out_file = "output/analysis/"

for microscope, model in list(itertools.product(
    ["PM", "LAP"], ["r", "p"]))[comm.Get_rank()::comm.Get_size()]:

    if os.path.isfile(
            os.path.join(
                out_file,
                f"cube_2pop_simulation_{microscope}_model_{model}_schilling.pkl"
            )):
        continue

    df = pd.read_pickle(
        os.path.join(out_file,
                     f"cube_2pop_simulation_{microscope}_model_{model}_.pkl"))

    df_acc = pd.DataFrame()
    for f0_inc in tqdm(df.f0_inc.unique()):
        for f1_rot in tqdm(df[df.f0_inc == f0_inc].f1_rot.unique(),
                           leave=False):
            hist_bin = lambda n_phi: np.linspace(
                0, np.pi, n_phi + 1, endpoint=True)

            rofl_dir = df[(df.f0_inc == f0_inc) &
                          (df.f1_rot == f1_rot)].explode("rofl_dir")
            rofl_inc = df[(df.f0_inc == f0_inc) &
                          (df.f1_rot == f1_rot)].explode("rofl_inc")

            for psi in rofl_dir.psi.unique():
                for omega in rofl_dir[rofl_dir.psi == psi].omega.unique():
                    # rofl
                    phi = rofl_dir[(rofl_dir.omega == omega) &
                                   (rofl_dir.psi == psi)].rofl_dir.to_numpy(
                                       dtype=float)
                    theta = np.pi / 2 - rofl_inc[
                        (rofl_inc.omega == omega) &
                        (rofl_inc.psi == psi)].rofl_inc.to_numpy(dtype=float)
                    sh0 = helper.spherical_harmonics.real_spherical_harmonics(
                        phi, theta, 6)

                    # ground truth
                    df_o = pd.read_pickle(
                        f"../data/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.pkl"
                    )
                    phi = df_o[(df_o.f0_inc == f0_inc) & (
                        df_o.f1_rot == f1_rot)].phi.explode().to_numpy(float)
                    theta = df_o[(df_o.f0_inc == f0_inc) & (
                        df_o.f1_rot == f1_rot)].theta.explode().to_numpy(float)
                    sh1 = helper.spherical_harmonics.real_spherical_harmonics(
                        phi, theta, 6)

                    acc = helper.schilling.angular_correlation_coefficient(
                        sh0, sh1)

                    df_acc = df_acc.append(
                        {
                            'f0_inc': f0_inc,
                            'f1_rot': f1_rot,
                            'omega': omega,
                            'psi': psi,
                            'acc': acc
                        },
                        ignore_index=True)
    df_acc.to_pickle(
        os.path.join(
            out_file,
            f"cube_2pop_simulation_{microscope}_model_{model}_schilling.pkl"))
