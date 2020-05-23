import numpy as np
import copy
import h5py
import argparse
import logging
import glob
import sys
import os

import warnings

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

from tqdm import tqdm

from simulate_n import run_simulation_pipeline_n
from MPIFileHandler import MPIFileHandler

from mpi4py import MPI
comm = MPI.COMM_WORLD
import multiprocessing as mp
NUM_THREADS = 2
mp_pool = mp.Pool(NUM_THREADS)

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path.")

parser.add_argument("-i",
                    "--input",
                    nargs='*',
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
log_file = os.path.join(args.output, "simulation.log")
mh = MPIFileHandler(log_file, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
mh.setFormatter(formatter)
logger.addHandler(mh)
logger.info("args: " + " ".join(sys.argv[1:]))

# PARAMETER
PIXEL_PM = 1.25
PIXEL_LAP = 20
LENGTH = 60
THICKNESS = 60
FIBER_INCLINATION = np.linspace(0, 90, 10, True)

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
logger.info(f"Total Memory: {simpli.memory_usage()* comm.Get_size():.0f} MB")
del simpli

# simulation loop
FILE_AND_INC = [(f, i) for f in file_list for i in FIBER_INCLINATION]
for file, f0_inc in tqdm(FILE_AND_INC[comm.Get_rank()::comm.Get_size()]):
    logger.info(f"input file: {file}")

    # Loading fiber models and prepair rotations
    with h5py.File(file, 'r') as h5f:
        fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f)
        psi = h5f['/'].attrs["psi"]
        omega = h5f['/'].attrs["omega"]

    n_rot = int(
        np.round(
            np.sqrt((1 - np.cos(2 * np.deg2rad(omega))) /
                    (1 - np.cos(np.deg2rad(10))))))

    if n_rot == 0:
        fiber_phi_rotations = [0]
    else:
        n_rot = max(n_rot, 3)
        fiber_phi_rotations = np.linspace(0, 90, n_rot, True)

    logger.info(f"omega: {omega}, n_rot: {n_rot}")

    # for f0_inc in FIBER_INCLINATION:
    for f1_rot in fiber_phi_rotations:
        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += '_inc_{:.2f}'.format(f0_inc)
        file_name += '_rot_{:.2f}'.format(f1_rot)
        file_name = os.path.join(args.output, file_name)

        logger.info(f"inclination : {f0_inc}")
        logger.info(f"rotation : {f1_rot}")
        logger.info(f"output file: {file_name}")

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)

        with h5py.File(file_name + '.h5', 'w-') as h5f:
            with open(os.path.abspath(__file__), 'r') as script:
                h5f.attrs['script'] = script.read()
                h5f.attrs['input_file'] = file

            for m, (dn, model) in enumerate([(-0.001, 'p'), (0.002, 'r')]):
                for name, gain, intensity, res, tilt_angle, sigma in [
                    ('LAP', 3, 26000, PIXEL_LAP, 5.5, 0.71),
                    ('PM', 1.5, 50000, PIXEL_PM, 3.9, 0.71)
                ]:
                    dset = h5f.create_group(name + '/' + model)

                    # Setup Simpli
                    simpli = fastpli.simulation.Simpli()
                    simpli.omp_num_threads = NUM_THREADS
                    simpli.voxel_size = args.voxel_size
                    simpli.pixel_size = res
                    simpli.filter_rotations = np.deg2rad(
                        [0, 30, 60, 90, 120, 150])
                    simpli.interpolate = True
                    simpli.wavelength = 525  # in nm
                    simpli.optical_sigma = 0.71  # in pixel size
                    simpli.verbose = 0

                    simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                                   0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
                    simpli.tilts = np.deg2rad(
                        np.array([(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                                  (tilt_angle, 180), (tilt_angle, 270)]))
                    simpli.add_crop_tilt_halo()

                    simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                        copy.deepcopy(fiber_bundles), rot)
                    simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                        (1.0, dn, 1, model)]
                                                      ] * len(fiber_bundles)

                    logger.info(f"tissue_pipeline: model:{model}")

                    save = ['optic', 'epa', 'mask', 'rofl']
#                     save += ['tissue'] if m == 0 and name == 'LAP' else []
                    label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                        h5f=dset, save=save)

                    # Simulate PLI Measurement
                    logger.info(f"simulation_pipeline: model:{model}")
                    # FIXME: LAP sigma ist bei einem pixel sinnfrei

                    simpli.light_intensity = intensity  # a.u.
                    simpli.sensor_gain = gain

                    simpli.save_parameter_h5(h5f=dset)

                    if name == 'LAP':
                        run_simulation_pipeline_n(simpli,
                                                  label_field,
                                                  vector_field,
                                                  tissue_properties,
                                                  256,
                                                  h5f=dset,
                                                  crop_tilt=True,
                                                  mp_pool=mp_pool)
                    else:
                        simpli.run_simulation_pipeline(label_field,
                                                       vector_field,
                                                       tissue_properties,
                                                       h5f=dset,
                                                       crop_tilt=True,
                                                       mp_pool=mp_pool)

                    dset.attrs['parameter/f0_inc'] = f0_inc
                    dset.attrs['parameter/psi'] = psi
                    dset.attrs['parameter/omega'] = omega
                    dset.attrs['parameter/f1_rot'] = f1_rot
                    dset.attrs[
                        'parameter/crop_tilt_voxel'] = simpli.crop_tilt_voxel()

                    del label_field
                    del vector_field
                    del simpli