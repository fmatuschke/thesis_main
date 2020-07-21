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
import fastpli.tools
import fastpli.io

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
parser.add_argument("-p",
                    "--num_proc",
                    required=True,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()


def run(file):
    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])

    df = []
    with h5py.File(file, 'r') as h5f:
        # FIXME
        # omega = h5f['/'].attrs['omega']
        # psi = h5f['/'].attrs['psi']
        # radius = h5f['/'].attrs['radius']
        f0_inc = h5f['/'].attrs['f0_inc']
        f1_rot = h5f['/'].attrs['f1_rot']
        pixel_size = h5f['/'].attrs['pixel_size']
        with h5py.File(str(h5f['fiber_bundles'][...]), 'r') as h5f_:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
            # psi = h5f['/'].attrs["psi"] # FIXME
            # omega = h5f['/'].attrs["omega"]

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)
        fiber_bundles = fastpli.objects.fiber_bundles.Rotate(fiber_bundles, rot)

        # fbs = fastpli.io.fiber_bundles.load_h5(h5f["solver/"])
        fiber_bundles = fastpli.objects.fiber_bundles.Cut(
            fiber_bundles, [[-pixel_size / 2, -pixel_size / 2, -30],
                            [pixel_size / 2, pixel_size / 2, 30]])
        phi, theta = fastpli.analysis.orientation.fiber_bundles(fiber_bundles)

        for voxel_size in h5f['simpli']:
            for model in h5f[f'simpli/{voxel_size}']:
                for n in h5f[f'simpli/{voxel_size}/{model}']:
                    h5f_sub = h5f[f'simpli/{voxel_size}/{model}/{n}']
                    for m in h5f_sub[f'simulation/optic/0/']:
                        df.append(
                            pd.DataFrame(
                                [[
                                    float(voxel_size),
                                    model,
                                    omega,
                                    psi,
                                    f0_inc,
                                    f1_rot,
                                    pixel_size,
                                    int(n),
                                    int(m),
                                    # h5f_sub['simulation/data/0'][...],
                                    h5f_sub[f'simulation/optic/0/{m}'][...
                                                                      ].ravel(),
                                    h5f_sub[f'analysis/epa/0/transmittance/{m}']
                                    [...].ravel(),
                                    h5f_sub[f'analysis/epa/0/direction/{m}'][
                                        ...].ravel(),
                                    h5f_sub[f'analysis/epa/0/retardation/{m}'][
                                        ...].ravel(),
                                    phi,
                                    theta
                                ]],
                                columns=[
                                    "voxel_size",
                                    "model",
                                    "omega",
                                    "psi",
                                    "f0_inc",
                                    "f1_rot",
                                    "pixel_size",
                                    "n",
                                    "m",
                                    #  "data",
                                    "optic",
                                    "epa_trans",
                                    "epa_dir",
                                    "epa_ret",
                                    "f_phi",
                                    "f_theta"
                                ]))

    return df


files = glob.glob(os.path.join(args.input, "*.h5"))

with mp.Pool(processes=args.num_proc) as pool:
    df = [
        item for sub in tqdm(pool.imap_unordered(run, files), total=len(files))
        for item in sub
    ]
df = pd.concat(df, ignore_index=True)
df.to_pickle(os.path.join(args.input, "voxel_size_simulation.pkl"))
