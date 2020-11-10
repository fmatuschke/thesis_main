import numpy as np
import h5py
import argparse
import logging
import subprocess
import glob
import sys
import os

import warnings

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

import tqdm

# from simulation_repeat import run_simulation_pipeline_n
import helper.mpi
import fibers

from mpi4py import MPI
comm = MPI.COMM_WORLD
import multiprocessing as mp

# reproducability
np.random.seed(42)

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

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

# logger
logger = logging.getLogger("rank[%i]" % comm.rank)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    os.path.join(
        args.output,
        f'simulation_{args.voxel_size}_{comm.Get_size()}_{comm.Get_rank()}.log')
)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("args: " + " ".join(sys.argv[1:]))
logger.info(
    f"git: {subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()}")
logger.info("script:\n" + open(os.path.abspath(__file__), 'r').read())

if __name__ == "__main__":

    # PARAMETER
    PIXEL_PM = 1.25
    PIXEL_LAP = 20
    LENGTH = 60
    THICKNESS = 60

    file_list = args.input
    # print Memory
    simpli = fastpli.simulation.Simpli()
    simpli.voxel_size = args.voxel_size  # in mu meter
    simpli.pixel_size = PIXEL_LAP  # in mu meter
    simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                   0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
    simpli.tilts = np.deg2rad(np.array([(5.5, 0)]))
    simpli.add_crop_tilt_halo()

    if comm.Get_rank() == 0:
        print(f"Single Memory: {simpli.memory_usage():.0f} MB")
        print(f"Total Memory: {simpli.memory_usage()* comm.Get_size():.0f} MB")
    logger.info(f"Single Memory: {simpli.memory_usage():.0f} MB")
    logger.info(
        f"Total Memory: {simpli.memory_usage()* comm.Get_size():.0f} MB")
    del simpli

    # simulation loop
    parameter = []
    fiber_inc = [(f, i) for f in file_list for i in fibers.inclinations(5)]
    for file, f0_inc in fiber_inc:
        # logger.info(f"input file: {file}")

        with h5py.File(file, 'r') as h5f:
            omega = h5f['/'].attrs["omega"]

        for f1_rot in fibers.omega_rotations(omega, 22.5):
            parameter.append((file, f0_inc, f1_rot))

    for file, f0_inc, f1_rot in tqdm.tqdm(
            parameter[comm.Get_rank()::comm.Get_size()]):
        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += f'_vs_{args.voxel_size:.4f}'
        file_name += f'_inc_{f0_inc:.2f}'
        file_name += f'_rot_{f1_rot:.2f}'
        file_name = os.path.join(args.output, file_name)
        logger.info(f"input file: {file}")
        logger.info(f"output file: {file_name}")

        with h5py.File(file, 'r') as h5f:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f)
            psi = h5f['/'].attrs["psi"]
            omega = h5f['/'].attrs["omega"]
            # radius = h5f['/'].attrs["r"]  # FIXME: change
            # v0 = h5f['/'].attrs["v0"]  # FIXME: change

        logger.info(f"omega: {omega}")
        logger.info(f"psi: {psi}")
        logger.info(f"inclination : {f0_inc}")
        logger.info(f"rotation : {f1_rot}")

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)

        with h5py.File(file_name + '.h5', 'w-') as h5f:
            with open(os.path.abspath(__file__), 'r') as script:
                h5f.attrs['script'] = script.read()
                h5f.attrs['input_file'] = file

            for m, (dn, model) in enumerate([(-0.008 / 2, 'p'), (0.008, 'r')]):
                for name, gain, intensity, res, tilt_angle, sigma in [
                    ('LAP', 3, 35000, PIXEL_LAP, 5.5, 0.75),
                    ('PM', 0.117, 16000, PIXEL_PM, 3.9, 0.75)
                ]:
                    dset = h5f.create_group(name + '/' + model)

                    # Setup Simpli
                    simpli = fastpli.simulation.Simpli()
                    warnings.filterwarnings("ignore", message="objects overlap")
                    simpli.omp_num_threads = 1
                    simpli.voxel_size = args.voxel_size
                    simpli.pixel_size = res
                    simpli.filter_rotations = np.linspace(0, np.pi, 9, False)
                    simpli.interpolate = "Slerp"
                    simpli.wavelength = 525  # in nm
                    simpli.optical_sigma = 0.75  # in pixel size
                    simpli.verbose = 0

                    simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                                   0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
                    simpli.tilts = np.deg2rad(
                        np.array([(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                                  (tilt_angle, 180), (tilt_angle, 270)]))
                    simpli.add_crop_tilt_halo()

                    simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                        fiber_bundles, rot)
                    simpli.fiber_bundles_properties = [[(0.75, 0, 10, 'b'),
                                                        (1.0, dn, 10, model)]
                                                      ] * len(fiber_bundles)

                    logger.info(f"tissue_pipeline: model:{model}")

                    save = ['optic', 'epa', 'rofl']
                    # save += ['tissue'] if m == 0 and name == 'LAP' else []
                    label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                        h5f=dset, save=save)

                    unique_elements, counts_elements = np.unique(
                        label_field, return_counts=True)
                    dset.attrs['label_field_stats'] = np.asarray(
                        (unique_elements, counts_elements))

                    # Simulate PLI Measurement
                    logger.info(f"simulation_pipeline: model:{model}")

                    simpli.light_intensity = intensity  # a.u.
                    simpli.noise_model = lambda x: np.round(
                        np.random.normal(x, np.sqrt(gain * x))).astype(np.uint16
                                                                      )

                    simpli.save_parameter_h5(h5f=dset)

                    simpli.run_simulation_pipeline(label_field,
                                                   vector_field,
                                                   tissue_properties,
                                                   h5f=dset,
                                                   save=save,
                                                   crop_tilt=True)

                    dset.attrs['parameter/psi'] = psi
                    dset.attrs['parameter/omega'] = omega
                    dset.attrs['parameter/fiber_path'] = file
                    dset.attrs['parameter/v'] = LENGTH
                    dset.attrs['parameter/f0_inc'] = f0_inc
                    dset.attrs['parameter/f1_rot'] = f1_rot
                    dset.attrs[
                        'parameter/crop_tilt_voxel'] = simpli.crop_tilt_voxel()

                    del label_field
                    del vector_field
                    del simpli
