import numpy as np
import copy
import h5py
import os
import sys
import glob
import gc

import fastpli.simulation
import fastpli.objects
import fastpli.tools

from mpi4py import MPI
from tqdm import tqdm

# reproducability
np.random.seed(42)

# PARAMETER
NUM_THREADS = 4
VOXEL_SIZE = 0.1
LENGTH = 65
THICKNESS = 60

import multiprocessing as mp
mp_pool = mp.Pool(NUM_THREADS)

comm = MPI.COMM_WORLD
FILE_NAME = os.path.abspath(__file__)
# FILE_PATH = os.path.dirname(FILE_NAME) + '/output
FILE_BASE = os.path.basename(FILE_NAME)

if len(sys.argv) > 1:
    FILE_PATH = sys.argv[1]
else:
    raise ValueError("need file output path")

os.makedirs(os.path.join(FILE_PATH, 'simulations'), exist_ok=True)
file_list = sorted(glob.glob(os.path.join(FILE_PATH, 'models', '*.solved*.h5')))

# print Memory
simpli = fastpli.simulation.Simpli()
simpli.voxel_size = VOXEL_SIZE  # in mu meter
simpli.set_voi(-0.5 * np.array([65, 65, 60]),
               0.5 * np.array([65, 65, 60]))  # in mu meter
print('Single Memory: ' + str(round(simpli.memory_usage())) + ' MB')
print('Total Memory: ' + str(round(simpli.memory_usage() * comm.Get_size())) +
      ' MB')
del simpli

# begin loop
for file in tqdm(file_list[comm.Get_rank()::comm.Get_size()]):

    tqdm.write(file)

    # Loading fiber models and prepair rotations
    # tqdm.write('loading models')
    fiber_bundles = fastpli.io.fiber.load(file, 'fiber_bundles')

    with h5py.File(file, 'r')['fiber_bundles'] as h5f:
        psi = h5f.attrs["psi"]
        dphi = h5f.attrs["dphi"]

    # dphi = float(file.split("dphi_")[-1].split("_psi")[0])
    n_rot = int(
        np.round(
            np.sqrt((1 - np.cos(2 * np.deg2rad(dphi))) /
                    (1 - np.cos(np.deg2rad(10))))))

    if n_rot > 0:
        n_rot = max(n_rot, 2)

    FIBER_ROTATIONS_PHI = np.linspace(0, 90, n_rot, True)
    tqdm.write("{}: {}".format(str(dphi), str(n_rot)))

    if n_rot == 0:
        FIBER_ROTATIONS_PHI = [0]

    for f_phi in FIBER_ROTATIONS_PHI:
        # tqdm.write("rotation: " + str(f_phi))

        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += '_phi_{:.2f}'.format(f_phi)

        file_name = fastpli.tools.helper.version_file_name(
            os.path.join(FILE_PATH, 'simulations', file_name))

        tqdm.write(file_name)

        # Setup Simpli
        simpli = fastpli.simulation.Simpli()
        simpli.omp_num_threads = NUM_THREADS
        simpli.voxel_size = VOXEL_SIZE  # in mu meter
        # simpli.set_voi(
        #     -0.5 * np.array([65 + DELTA_V, 65 + DELTA_V, 60]),
        #     0.5 * np.array([65 + DELTA_V, 65 + DELTA_V, 60]))  # in mu meter
        simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
        simpli.interpolate = True
        simpli.wavelength = 525  # in nm
        simpli.tilts = np.deg2rad(
            np.array([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180), (5.5, 270)]))
        simpli.optical_sigma = 0.71  # in voxel size
        simpli.verbose = 0

        # tqdm.write('rotating models')
        rot = fastpli.tools.rotation.x(np.deg2rad(f_phi))
        simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
            copy.deepcopy(fiber_bundles), rot)

        with h5py.File(file_name + '.h5', 'w-') as h5f:
            with open(os.path.abspath(__file__), 'r') as script:
                h5f['parameter/script'] = script.read()

            save = ["label_field"]
            for m, (dn, model) in enumerate([(-0.001, 'p'), (0.002, 'r')]):
                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 1, model)]
                                                  ] * len(fiber_bundles)

                label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                    h5f=h5f, save=save)
                save = None

                # Simulate PLI Measurement
                for name, gain, intensity, res, tilt_angle in [
                    ('LAP', 3, 26000, 20, 5.5), ('PM', 1.5, 50000, 1.25, 3.9)
                ]:
                    simpli.light_intensity = intensity  # a.u.
                    simpli.resolution = res  # in mu meter
                    simpli.sensor_gain = gain
                    simpli.set_voi(
                        -0.5 * np.array([LENGTH, LENGTH, THICKNESS]), 0.5 *
                        np.array([LENGTH, LENGTH, THICKNESS]))  # in mu meter
                    simpli.add_crop_tilt_halo()

                    dset = h5f.create_group(name + '/' + model)
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

        gc.collect()

MPI.Finalize()
