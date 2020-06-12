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
import fastpli.analysis
import fastpli.io

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
        fbs = fastpli.io.fiber_bundles.load_h5(h5f["solver/"])
        fbs = fastpli.objects.fiber_bundles.Cut(
            fbs, [[-pixel_size / 2, -pixel_size / 2, -30],
                  [pixel_size / 2, pixel_size / 2, 30]])
        phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs)

        dset = h5f['solver']
        psi = dset.attrs['psi']
        omega = dset.attrs['omega']
        num_obj = dset.attrs['num_obj']
        num_steps = dset.attrs['num_steps']
        obj_mean_length = dset.attrs['obj_mean_length']
        obj_min_radius = dset.attrs['obj_min_radius']
        time = dset.attrs['time']

        for voxel_size in h5f['simpli']:
            for model in h5f[f'simpli/{voxel_size}']:
                h5f_sub = h5f[f'simpli/{voxel_size}/{model}']

                df.append(
                    pd.DataFrame([[
                        float(voxel_size), model, omega, psi, f0_inc, f1_rot,
                        radius, pixel_size,
                        h5f_sub['simulation/data/0'][...].ravel(),
                        h5f_sub['simulation/optic/0'][...].ravel(),
                        h5f_sub['analysis/epa/0/transmittance'][...].ravel(),
                        h5f_sub['analysis/epa/0/direction'][...].ravel(),
                        h5f_sub['analysis/epa/0/retardation'][...].ravel(), phi,
                        theta, num_obj, num_steps, obj_mean_length,
                        obj_min_radius, time
                    ]],
                                 columns=[
                                     "voxel_size", "model", "omega", "psi",
                                     "f0_inc", "f1_rot", "radius", "pixel_size",
                                     "data", "optic", "epa_trans", "epa_dir",
                                     "epa_ret", "f_phi", "f_theta",
                                     "solver.num_obj", "solver.num_steps",
                                     "solver.obj_mean_length",
                                     "solver.obj_min_radius", "solver.time"
                                 ]))

    return df


files = glob.glob(os.path.join(args.input, "*.h5"))

run.num_p = args.num_proc
with mp.Pool(processes=run.num_p) as pool:
    df = [
        item for sub in tqdm(pool.imap_unordered(run, files), total=len(files))
        for item in sub
    ]
    df = pd.concat(df, ignore_index=True)

df.to_pickle(os.path.join(args.input, "voxel_size_simulation.pkl"))
