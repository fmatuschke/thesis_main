''' extracts the important information from the simulation to a panda file to
    allow further analysis
'''

import numpy as np
import itertools
import warnings
import h5py
import os
import glob

import pandas as pd
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD

sim_path = "output/simulation/*.h5"
out_file = "output/analysis/"

# simulation output

os.makedirs("output/analysis", exist_ok=True)
file_list = sorted(glob.glob(sim_path))

for microscope, model in list(itertools.product(
    ["PM", "LAP"], ["r", "p"]))[comm.Get_rank()::comm.Get_size()]:
    if os.path.isfile(
            os.path.join(
                out_file,
                f"cube_2pop_simulation_{microscope}_model_{model}_.pkl")):
        continue

    df = [None] * len(file_list)
    for f, file in enumerate(tqdm(file_list)):
        with h5py.File(file, 'r') as h5f:
            h5f_sub = h5f[microscope + '/' + model]

            df[f] = pd.DataFrame([[
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
                                     "omega", "psi", "f0_inc", "f1_rot",
                                     "rofl_dir", "rofl_inc", "rofl_trel",
                                     "epa_trans", "epa_dir", "epa_ret"
                                 ])

    df = pd.concat(df, ignore_index=True)

    df.to_pickle(
        os.path.join(out_file,
                     f"cube_2pop_simulation_{microscope}_model_{model}_.pkl"))
