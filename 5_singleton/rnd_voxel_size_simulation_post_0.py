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
    df = []
    with h5py.File(file, 'r') as h5f:
        omega = h5f['/'].attrs['omega']
        psi = h5f['/'].attrs['psi']
        # radius = h5f['/'].attrs['radius']
        f0_inc = h5f['/'].attrs['f0_inc']
        f1_rot = h5f['/'].attrs['f1_rot']
        # pixel_size = h5f['/'].attrs['pixel_size']
        with h5py.File(str(h5f['fiber_bundles'][...]), 'r') as h5f_:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
            if h5f_['/'].attrs["psi"] != psi or omega == h5f_['/'].attrs[
                    "omega"] != omega:
                raise ValueError("FAIL")

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)
        fiber_bundles = fastpli.objects.fiber_bundles.Rotate(fiber_bundles, rot)

        for n in h5f[f'simpli/1.25/r']:
            voi_min = h5f[f'simpli/1.25/r/{n}/norm'].attrs['voi_min']
            voi_max = h5f[f'simpli/1.25/r/{n}/norm/'].attrs['voi_max']
            fiber_bundles_ = fastpli.objects.fiber_bundles.Cut(
                fiber_bundles, [voi_min, voi_max])
            phi, theta = fastpli.analysis.orientation.fiber_bundles(
                fiber_bundles_)
            for voxel_size in h5f['simpli']:
                for model in h5f[f'simpli/{voxel_size}']:
                    h5f_sub_norm = h5f[f'simpli/{voxel_size}/{model}/{n}/norm']
                    h5f_sub_ref = h5f[f'simpli/{voxel_size}/{model}/{n}/ref']

                    for m in h5f_sub_norm[f'simulation/optic/0']:

                        df.append(
                            pd.DataFrame([[
                                float(voxel_size), model, omega, psi, f0_inc,
                                f1_rot,
                                int(n),
                                int(m), h5f_sub_norm[f'simulation/optic/0/{m}'][
                                    ...].ravel(), h5f_sub_norm[
                                        f'analysis/epa/0/transmittance/{m}'][
                                            ...].ravel(),
                                h5f_sub_norm[f'analysis/epa/0/direction/{m}'][
                                    ...].ravel(),
                                h5f_sub_norm[f'analysis/epa/0/retardation/{m}'][
                                    ...].ravel(),
                                h5f_sub_ref[f'simulation/optic/0/{m}'][
                                    ...].ravel(),
                                h5f_sub_ref[f'analysis/epa/0/transmittance/{m}']
                                [...].ravel(),
                                h5f_sub_ref[f'analysis/epa/0/direction/{m}'][
                                    ...].ravel(),
                                h5f_sub_ref[f'analysis/epa/0/retardation/{m}'][
                                    ...].ravel(), phi, theta
                            ]],
                                         columns=[
                                             "voxel_size", "model", "omega",
                                             "psi", "f0_inc", "f1_rot", "n",
                                             "m", "optic", "epa_trans",
                                             "epa_dir", "epa_ret", "ref_optic",
                                             "ref_epa_trans", "ref_epa_dir",
                                             "ref_epa_ret", "f_phi", "f_theta"
                                         ]))

    return pd.concat(df, ignore_index=True)


files = glob.glob(os.path.join(args.input, "*.h5"))

with mp.Pool(processes=args.num_proc) as pool:
    df = [
        sub for sub in tqdm(pool.imap_unordered(run, files), total=len(files))
        # for item in sub
    ]
df = pd.concat(df, ignore_index=True)
df.to_pickle(os.path.join(args.input, "rnd_voxel_size_simulation.pkl"))
