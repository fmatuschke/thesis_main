''' extracts the ground truth distribution from the models with respect to
    simulated rotations
'''

import numpy as np
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import gc
import os

import pandas as pd
from tqdm import tqdm

import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

import helper.file
from simulation import f0_incs
from simulation import omega_rotations

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    nargs='+',
                    required=True,
                    help="input files.")
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output Path.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)


def run(file):
    output_file = os.path.join(
        args.output,
        os.path.splitext(os.path.basename(file))[0] + ".pkl")

    if os.path.isfile(output_file):
        warnings.warn("File already exsts")
        return

    fbs = fastpli.io.fiber_bundles.load(file)
    with h5py.File(file, "r") as h5f:
        omega = h5f['/'].attrs['omega']
        psi = h5f['/'].attrs['psi']

        if h5f['/'].attrs['num_steps'] == 0:
            state = "init"
        elif h5f['/'].attrs['overlap'] == 0:
            state = "solved"
        else:
            state = "solved"
            warnings.warn("not solved")

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

    df = []
    if state == "init":
        f0_inc = 0
        f1_rot = 0

        fbs_ = fbs.copy()
        for v in [120, 60]:  # ascending order!
            fbs_ = fastpli.objects.fiber_bundles.Cut(
                fbs_, [[-v / 2] * 3, [v / 2] * 3])
            phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs_)

            df.append(
                pd.DataFrame(
                    [[
                        f0_inc, f1_rot, v, meta,
                        phi.astype(np.float32).ravel(),
                        theta.astype(np.float32).ravel()
                    ]],
                    columns=["f0_inc", "f1_rot", "v", "meta", "phi", "theta"]))
    else:
        for f0_inc in f0_incs():
            for f1_rot in omega_rotations(omega):
                rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
                rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
                rot = np.dot(rot_inc, rot_phi)
                fbs_ = fastpli.objects.fiber_bundles.Rotate(fbs, rot)

                for v in [120, 60]:  # ascending order!
                    fbs_ = fastpli.objects.fiber_bundles.Cut(
                        fbs_, [[-v / 2] * 3, [v / 2] * 3])
                    phi, theta = fastpli.analysis.orientation.fiber_bundles(
                        fbs_)

                    df.append(
                        pd.DataFrame([[
                            f0_inc, f1_rot, v, meta,
                            phi.astype(np.float32).ravel(),
                            theta.astype(np.float32).ravel()
                        ]],
                                     columns=[
                                         "f0_inc", "f1_rot", "v", "meta", "phi",
                                         "theta"
                                     ]))

    df = pd.concat(df, ignore_index=True)
    df.to_pickle(output_file)


if __name__ == "__main__":

    with mp.Pool(processes=args.num_proc) as pool:
        [
            d for d in tqdm(pool.imap_unordered(run, args.input),
                            total=len(args.input),
                            smoothing=0)
        ]
