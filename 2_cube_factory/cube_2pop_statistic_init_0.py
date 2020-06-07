''' find best arbeitspunkt for solver
'''

import numpy as np
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import os

import pandas as pd
from tqdm import tqdm

import fastpli.io
import fastpli.objects
import fastpli.analysis
import fastpli.simulation

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
    r = float(file.split("_r_")[1].split("_")[0])
    v0 = float(file.split("_v0_")[1].split("_")[0])
    fl = float(file.split("_fl_")[1].split("_")[0])
    fr = float(file.split("_fr_")[1].split("_")[0])
    state = file.split(".")[-2].split(".")[-1]

    fbs = fastpli.io.fiber_bundles.load(file)

    if state != "init":
        # Setup Simpli
        simpli = fastpli.simulation.Simpli()
        warnings.filterwarnings("ignore", message="objects overlap")
        # simpli.omp_num_threads = 0
        simpli.voxel_size = 0.1
        simpli.set_voi([-30] * 3, [30] * 3)
        simpli.fiber_bundles = fbs
        simpli.fiber_bundles_properties = [[(1.0, 0, 0, 'p')]] * len(fbs)

        if not run.flag:
            print(f"Single Memory: {simpli.memory_usage():.0f} MB")
            print(f"Total Memory: {simpli.memory_usage() * run.num_p:.0f} MB")
            run.flag = True

        tissue, _, _ = simpli.generate_tissue(only_tissue=True)
        unique_elements, count_elements = np.unique(tissue, return_counts=True)
    else:
        unique_elements, count_elements = 0, 0

    with h5py.File(file, "r") as h5f:

        obj_mean_length = h5f['/'].attrs['obj_mean_length']
        obj_min_radius = h5f['/'].attrs['obj_min_radius']
        overlap = np.nan
        num_col_obj = np.nan
        num_obj = np.nan
        num_steps = np.nan
        time = np.nan
        if state != "init":
            overlap = h5f['/'].attrs['overlap']
            num_col_obj = h5f['/'].attrs['num_col_obj']
            num_obj = h5f['/'].attrs['num_obj']
            num_steps = h5f['/'].attrs['num_steps']
            time = h5f['/'].attrs['time']

    fbs_ = fbs.copy()
    fbs_ = fastpli.objects.fiber_bundles.Cut(fbs_, [[-30] * 3, [30] * 3])
    phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs_)

    df = pd.DataFrame(columns=["phi", "theta"], dtype='object')

    return pd.DataFrame([[
        omega, psi, r, v0, fl, fr, state, overlap, num_col_obj, num_obj,
        num_steps, time, unique_elements, count_elements,
        phi.astype(np.float32).ravel(),
        theta.astype(np.float32).ravel()
    ]],
                        columns=[
                            "omega",
                            "psi",
                            "r",
                            "v0",
                            "fl",
                            "fr",
                            "state",
                            "overlap",
                            "num_col_obj",
                            "num_obj",
                            "num_steps",
                            "time",
                            "unique_elements",
                            "count_elements",
                            "phi",
                            "theta",
                        ])


files = glob.glob(os.path.join(args.input, "*.h5"))

run.flag = False
run.num_p = args.num_proc
with mp.Pool(processes=run.num_p) as pool:
    df = [d for d in tqdm(pool.imap_unordered(run, files), total=len(files))]
    df = pd.concat(df, ignore_index=True)

df.to_pickle(os.path.join(args.input, "cube_stat.pkl"))
