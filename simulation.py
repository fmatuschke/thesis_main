#%% import
import numpy as np
import copy
import h5py
import argparse
import logging
import glob
import sys
import os

import fastpli.simulation
import fastpli.objects
import fastpli.tools

from tqdm import tqdm

from MPIFileHandler import MPIFileHandler
from mpi4py import MPI
comm = MPI.COMM_WORLD
import multiprocessing as mp
NUM_THREADS = 4
mp_pool = mp.Pool(NUM_THREADS)

# reproducability
np.random.seed(42)

#%% path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input regex string.")

args = parser.parse_args()
os.makedirs(os.path.join(args.output, 'simulations'), exist_ok=True)

#%% logger
logger = logging.getLogger("rank[%i]" % comm.rank)
logger.setLevel(logging.DEBUG)

log_file = fastpli.tools.helper.version_file_name(
    os.path.join(args.output, 'simulation')) + ".log"
mh = MPIFileHandler(log_file)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
mh.setFormatter(formatter)
logger.addHandler(mh)

#%% PARAMETER
VOXEL_SIZE = 0.1
LENGTH = 65
THICKNESS = 60

file_list = sorted(glob.glob(os.path.join(args.output, 'models', args.input)))
if not file_list:
    logger.error("no files detected")
    comm.Abort()
    sys.exit()

# print Memory
simpli = fastpli.simulation.Simpli()
simpli.voxel_size = VOXEL_SIZE  # in mu meter
simpli.set_voi(-0.5 * np.array([65, 65, 60]),
               0.5 * np.array([65, 65, 60]))  # in mu meter

logger.info(f"Single Memory: {round(simpli.memory_usage())} MB")
logger.info(f"Total Memory: {round(simpli.memory_usage()* comm.Get_size())} MB")
del simpli

#%% simulation loop
for file in tqdm(file_list[comm.Get_rank()::comm.Get_size()]):
    logger.info(f"input file: {file}")

    # Loading fiber models and prepair rotations
    fiber_bundles = fastpli.io.fiber.load(file, 'fiber_bundles')

    with h5py.File(file, 'r') as h5f:
        psi = h5f['fiber_bundles'].attrs["psi"]
        dphi = h5f['fiber_bundles'].attrs["dphi"]

    n_rot = int(
        np.round(
            np.sqrt((1 - np.cos(2 * np.deg2rad(dphi))) /
                    (1 - np.cos(np.deg2rad(10))))))

    if n_rot > 0:
        n_rot = max(n_rot, 2)

    fiber_phi_rotations = np.linspace(0, 90, n_rot, True)
    logger.info(f"dphi: {dphi}, n_rot: {n_rot}")

    if n_rot == 0:
        fiber_phi_rotations = [0]

    for f_phi in fiber_phi_rotations:
        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += '_phi_{:.2f}'.format(f_phi)
        file_name = fastpli.tools.helper.version_file_name(
            os.path.join(args.output, 'simulations', file_name))

        logger.info(f"rotation : {f_phi}")
        logger.info(f"output file: {file_name}")

        rot = fastpli.tools.rotation.x(np.deg2rad(f_phi))

        with h5py.File(file_name + '.h5', 'w-') as h5f:
            with open(os.path.abspath(__file__), 'r') as script:
                h5f['parameter/script'] = script.read()
            h5f['parameter/input_file'] = file

            for m, (dn, model) in enumerate([(-0.001, 'p'), (0.002, 'r')]):
                for name, gain, intensity, res, tilt_angle in [
                    ('LAP', 3, 26000, 20, 5.5), ('PM', 1.5, 50000, 1.25, 3.9)
                ]:
                    dset = h5f.create_group(name + '/' + model)

                    # Setup Simpli
                    simpli = fastpli.simulation.Simpli()
                    simpli.omp_num_threads = NUM_THREADS
                    simpli.voxel_size = VOXEL_SIZE
                    simpli.resolution = res
                    simpli.filter_rotations = np.deg2rad(
                        [0, 30, 60, 90, 120, 150])
                    simpli.interpolate = True
                    simpli.wavelength = 525  # in nm
                    simpli.optical_sigma = 0.71  # in pixel size
                    simpli.verbose = 0

                    simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                                   0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
                    simpli.tilts = np.deg2rad(
                        np.array([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180),
                                  (5.5, 270)]))
                    simpli.add_crop_tilt_halo()

                    simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                        copy.deepcopy(fiber_bundles), rot)
                    simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                        (1.0, dn, 1, model)]
                                                      ] * len(fiber_bundles)

                    logger.info(f"tissue_pipeline: model:{model}")
                    label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                        h5f=dset, save=["label_field"])

                    # Simulate PLI Measurement
                    logger.info(f"simulation_pipeline: model:{model}")
                    # TODO: LAP run multiple pixels for statistics

                    simpli.light_intensity = intensity  # a.u.
                    simpli.sensor_gain = gain

                    simpli.save_parameter(h5f=dset)
                    simpli.run_simulation_pipeline(label_field,
                                                   vector_field,
                                                   tissue_properties,
                                                   h5f=dset,
                                                   crop_tilt=True,
                                                   mp_pool=mp_pool)

                    dset['parameter/model/psi'] = psi
                    dset['parameter/model/dphi'] = dphi
                    dset['parameter/model/phi'] = f_phi

                    del label_field
                    del vector_field
                    del simpli
