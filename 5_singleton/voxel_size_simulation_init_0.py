
import numpy as np
import pandas as pd
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import os

from tqdm import tqdm
import helper.circular

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()


def run(file):

    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])
    f0_inc = float(file.split("_f0_inc_")[1].split("_")[0])
    f1_rot = float(file.split("_f1_rot_")[1].split("_")[0])
    radius = float(file.split("_r_")[1].split("_")[0])
    pixel_size = float(file.split("_p0_")[1].split("_")[0])

    df = []
    with h5py.File(file, 'r') as h5f:
        for voxel_size in h5f['simpli']:
            for model in h5f[f'simpli/{voxel_size}']:
                h5f_sub = h5f[f'simpli/{voxel_size}/{model}']

                df.append(pd.DataFrame([[omega, psi, f0_inc, f1_rot,    radius, pixel_size,
                                        h5f_sub['simulation/data/0'][...].ravel(),
                                        h5f_sub['simulation/optic/0'][...].ravel(),
                                        h5f_sub['analysis/epa/0/transmittance'][...].ravel(),
                                        np.deg2rad(h5f_sub['analysis/epa/0/direction'][...].ravel()),
                                        h5f_sub['analysis/epa/0/retardation'][...].ravel()]],
                                        columns=[
                                        "omega", "psi", "f0_inc", "f1_rot",
                                        "radius", "pixel_size", "data","optic",
                                        "epa_trans", "epa_dir", "epa_ret"
                                    ]))

    return df


files = glob.glob(os.path.join(args.input, "*.h5"))

run.num_p = args.num_proc
with mp.Pool(processes=run.num_p) as pool:
    df = [item for sub in tqdm(pool.imap_unordered(run, files), total=len(files)) for item in sub ]
    df = pd.concat(df, ignore_index=True)

df.to_pickle(os.path.join(args.input, "voxel_size_simulation.pkl"))
