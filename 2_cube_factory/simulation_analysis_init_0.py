''' extracts the important information from the simulation to a panda file to
    allow further analysis
'''

import numpy as np
import itertools
import h5py
import os
import glob

import pandas as pd
from tqdm import tqdm

sim_path = "output/simulation/*.h5"
out_file = "output/analysis/"

# simulation output

os.makedirs("output/analysis", exist_ok=True)
file_list = sorted(glob.glob(sim_path))

for microscope, model in tqdm(list(itertools.product(["PM", "LAP"],
                                                     ["r", "p"]))):
    if not os.path.isfile(
            os.path.join(
                out_file,
                f"cube_2pop_simulation_{microscope}_model_{model}_.pkl")):
        df = [None] * len(file_list)
        for f, file in enumerate(tqdm(file_list)):
            with h5py.File(file, 'r') as h5f:
                h5f_sub = h5f[microscope + '/' + model]
                df[f] = pd.DataFrame({
                    "f0_inc":
                        h5f_sub.attrs['parameter/f0_inc'],
                    "f1_rot":
                        h5f_sub.attrs['parameter/f1_rot'],
                    "omega":
                        h5f_sub.attrs['parameter/omega'],
                    "psi":
                        h5f_sub.attrs['parameter/psi'],
                    "rofl_dir":
                        h5f_sub['analysis/rofl/direction'][...].ravel(),
                    "rofl_inc":
                        h5f_sub['analysis/rofl/inclination'][...].ravel(),
                    "rofl_trel":
                        h5f_sub['analysis/rofl/t_rel'][...].ravel(),
                    "epa_trans":
                        h5f_sub['analysis/epa/0/transmittance'][...].ravel(),
                    "epa_dir":
                        h5f_sub['analysis/epa/0/direction'][...].ravel(),
                    "epa_ret":
                        h5f_sub['analysis/epa/0/retardation'][...].ravel(),
                })

        df = pd.concat(df, ignore_index=True)
        df.to_pickle(
            os.path.join(
                out_file,
                f"cube_2pop_simulation_{microscope}_model_{model}_.pkl"))
