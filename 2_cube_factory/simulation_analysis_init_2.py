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

    df_acc = []
    with tqdm(total=len(df[['psi', 'omega', 'f0_inc', 'f1_rot'
                           ]].drop_duplicates().index)) as pbar:
        for psi in df.psi.unique():
            for omega in df[df.psi == psi].omega.unique():
                df_sub = df[(df.psi == psi) & (df.omega == omega)]
                df_org = pd.read_pickle(
                    f"output/models/cube_2pop_psi_{psi:.1f}_omega_{omega:.1f}_.solved.pkl"
                )
                for f0_inc in df_sub.f0_inc.unique():
                    for f1_rot in df_sub[df_sub.f0_inc ==
                                         f0_inc].f1_rot.unique():

                        # rofl
                        sub = (df_sub.f0_inc == f0_inc) & (df_sub.f1_rot
                                                           == f1_rot)
                        phi = df_sub[sub].explode("rofl_dir").rofl_dir.to_numpy(
                            dtype=float)
                        theta = np.pi / 2 - df_sub[sub].explode(
                            "rofl_inc").rofl_inc.to_numpy(dtype=float)
                        sh0 = helper.spherical_harmonics.real_spherical_harmonics(
                            phi, theta, 6)

                        # ground truth
                        sub = (df_org.f0_inc == f0_inc) & (df_org.f1_rot
                                                           == f1_rot)
                        phi = df_org[sub].phi.explode().to_numpy(float)
                        theta = df_org[sub].theta.explode().to_numpy(float)
                        sh1 = helper.spherical_harmonics.real_spherical_harmonics(
                            phi, theta, 6)

                        # ACC
                        acc = helper.schilling.angular_correlation_coefficient(
                            sh0, sh1)

                        df_acc.append(
                            pd.DataFrame(
                                {
                                    'f0_inc': f0_inc,
                                    'f1_rot': f1_rot,
                                    'omega': omega,
                                    'psi': psi,
                                    'acc': acc
                                },
                                index=[0]))

                        pbar.update()

    df_acc = pd.concat(df_acc, ignore_index=True)
    df_acc.to_pickle(
        os.path.join(
            out_file,
            f"cube_2pop_simulation_{microscope}_model_{model}_schilling.pkl"))
