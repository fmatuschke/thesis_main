import numpy as np
import h5py
import argparse

import os
import warnings
import tqdm

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

import fibers
import helper.file

from mpi4py import MPI
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path.")

parser.add_argument("-i",
                    "--input",
                    nargs='+',
                    required=True,
                    help="input string.")

parser.add_argument("-v",
                    "--voxel_size",
                    type=float,
                    required=True,
                    help="voxel_size in um.")

parser.add_argument(
    "--n_inc",  #4
    type=int,
    required=True,
    help="number of fiber inclinations")

parser.add_argument(
    "--d_rot",  #15
    type=int,
    required=True,
    help="number of fiber inclinations")

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

if __name__ == "__main__":

    # PARAMETER
    PIXEL_PM = 1.25
    PIXEL_LAP = 20
    LENGTH = 60
    THICKNESS = 60

    file_list = args.input

    # simulation loop
    parameter = []
    fiber_inc = [
        (f, i) for f in file_list for i in fibers.inclinations(args.n_inc)
    ]
    for file, f0_inc in fiber_inc:
        omega = helper.file.value(file, "omega")

        for f1_rot in fibers.omega_rotations(omega, args.d_rot):
            parameter.append((file, f0_inc, f1_rot))

    for file, f0_inc, f1_rot in tqdm.tqdm(
            parameter[comm.Get_rank()::comm.Get_size()]):

        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += f'_vs_{args.voxel_size:.4f}'
        file_name += f'_inc_{f0_inc:.2f}'
        file_name += f'_rot_{f1_rot:.2f}'
        file_name = os.path.join(args.output, file_name)

        if not os.path.isfile(file_name + '.h5'):
            print(f"file: {file_name}.h5 does not exist")
            exit(1)

        # if os.path.isfile(file_name + '.tissue.h5'):
        #     print(f"file: {file_name}.tissue.h5 already exists")
        #     continue

        with h5py.File(file, 'r') as h5f:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f)
            psi = h5f['/'].attrs["psi"]
            omega = h5f['/'].attrs["omega"]
            radius = h5f['/'].attrs["radius"]
            v0 = h5f['/'].attrs["v0"]

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)

        with h5py.File(file_name + '.tissue.h5', 'w') as h5f:
            with open(os.path.abspath(__file__), 'r') as script:
                h5f.attrs['script'] = script.read()
                h5f.attrs['input_file'] = file

            # for m, (dn, model) in enumerate([(-0.008 / 2, 'p'), (0.008, 'r')]):
            for m, (dn, model) in enumerate([(-0.008 / 2, 'p')]):
                mu = 0

                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                warnings.filterwarnings("ignore", message="objects overlap")
                simpli.omp_num_threads = 1
                simpli.voxel_size = args.voxel_size
                simpli.filter_rotations = np.linspace(0, np.pi, 9, False)
                simpli.interpolate = "Slerp"
                simpli.wavelength = 525  # in nm
                simpli.optical_sigma = 0.75  # in pixel size
                simpli.verbose = 0

                simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                               0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
                simpli.tilts = np.deg2rad(np.array([(0, 0)]))

                simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                    fiber_bundles, rot)
                simpli.fiber_bundles_properties = [[(0.75, 0, mu, 'b'),
                                                    (1.0, dn, mu, model)]
                                                  ] * len(fiber_bundles)

                tissue, _, _ = simpli.generate_tissue(only_tissue=True)
                dset = h5f.create_dataset('tissue',
                                          tissue.shape,
                                          dtype=np.uint16,
                                          compression='gzip',
                                          compression_opts=4)

                dset[:] = tissue
                # h5f[f"/tissue_max_proj"] = np.max(tissue,
                #                                   axis=-1).astype(np.uint16)
                # h5f[f"/tissue_mean_proj"] = np.mean(tissue,
                #                                     axis=-1).astype(np.float32)
                # h5f[f"/tissue_sum_proj"] = np.sum(tissue,
                #                                   axis=-1).astype(np.uint16)

                tissue = tissue.astype(np.float32)
                tmp = tissue.ravel()
                tmp[tmp > 0] = ((tmp[tmp > 0] - 1) // 2) + 1
                tmp[tmp == 0] = np.nan

                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", message='RuntimeWarning: Mean of empty slice')
                    h5f[f"/tissue_bin_median"] = np.nanmedian(tissue, axis=-1)
                    h5f[f"/tissue_bin_mean"] = np.nanmean(tissue, axis=-1)

                # dset = h5f.create_dataset('tissue_bin',
                #                           tissue.shape,
                #                           dtype=np.uint16,
                #                           compression='gzip',
                #                           compression_opts=4)

                # dset[:] = tissue

                # h5f[f"/tissue_bin_max_proj"] = np.max(tissue,
                #                                       axis=-1).astype(np.uint16)
                # h5f[f"/tissue_bin_mean_proj"] = np.mean(tissue, axis=-1).astype(
                #     np.float32)
                # h5f[f"/tissue_bin_sum_proj"] = np.sum(tissue,
                #                                       axis=-1).astype(np.uint16)
