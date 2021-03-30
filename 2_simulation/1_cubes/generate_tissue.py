import numpy as np
import h5py
import argparse
import glob

import os
import warnings
import tqdm

import fastpli.simulation
import fastpli.analysis
import fastpli.tools
import fastpli.io

import models
import helper.file

import pretty_errors

from mpi4py import MPI

comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="input string.")
args = parser.parse_args()
os.makedirs(os.path.join(args.input, "tissue"), exist_ok=True)

if __name__ == "__main__":

    # PARAMETER
    PIXEL_PM = 1.25
    PIXEL_LAP = 20
    LENGTH = 60
    THICKNESS = 60

    file_list = glob.glob(os.path.join(args.input, '*.h5'))

    for file in tqdm.tqdm(file_list[comm.Get_rank()::comm.Get_size()]):

        file_path, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name = os.path.join(file_path, "tissue", file_name)

        with h5py.File(file, 'r') as h5f:
            dset = h5f['PM/Vervet/p']
            fiber_bundles = fastpli.io.fiber_bundles.load(
                dset.attrs['parameter/fiber_path'])
            psi = dset.attrs["parameter/psi"]
            omega = dset.attrs["parameter/omega"]
            radius = dset.attrs["parameter/radius"]
            f0_inc = dset.attrs["parameter/f0_inc"]
            f1_rot = dset.attrs["parameter/f1_rot"]
        voxel_size = helper.file.value(file, "vs")

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)

        with h5py.File(file_name + '.tissue.h5', 'w') as h5f:
            with open(os.path.abspath(__file__), 'r') as script:
                h5f.attrs['script'] = script.read()
                h5f.attrs['input_file'] = file

            for m, (dn, model) in enumerate([(-0.008 / 2, 'p')]):
                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                warnings.filterwarnings("ignore", message="objects overlap")
                simpli.omp_num_threads = 1
                simpli.voxel_size = voxel_size

                simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                               0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
                simpli.tilts = np.deg2rad(np.array([(0, 0)]))

                simpli.fiber_bundles = fiber_bundles.rotate(rot)
                simpli.fiber_bundles.layers = [[(0.75, 0, 0, 'b'),
                                                (1.0, dn, 0, model)]
                                              ] * len(fiber_bundles)

                tissue, _, _ = simpli.generate_tissue(only_tissue=True)
                dset = h5f.create_dataset('tissue',
                                          tissue.shape,
                                          dtype=np.uint8,
                                          compression='gzip')

                dset[:] = tissue

                # tissue = tissue.astype(np.float32)
                # tmp = tissue.ravel()
                # tmp[tmp > 0] = ((tmp[tmp > 0] - 1) // 2) + 1
                # tmp[tmp == 0] = np.nan

                # with warnings.catch_warnings():
                #     warnings.filterwarnings(
                #         "ignore", message='RuntimeWarning: Mean of empty slice')
                #     h5f[f"/tissue_bin_median"] = np.nanmedian(tissue, axis=-1)
                #     h5f[f"/tissue_bin_mean"] = np.nanmean(tissue, axis=-1)
