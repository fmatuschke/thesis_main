import numpy as np
import os
import sys
import glob
import argparse
import logging
import h5py

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

from tqdm import tqdm
from mpi4py import MPI
from MPIFileHandler import MPIFileHandler
comm = MPI.COMM_WORLD

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

# Parameter
OMEGAS = [0, 30, 60, 90]
# VOXEL_SIZES = [0.025, 0.05, 0.1, 0.25, 0.75, 1.25]
# LENGTH = VOXEL_SIZES[-1]*10

# VOXEL_SIZES = [0.1, 0.25, 0.75, 1.25, 10, 20, 40, 60]
# LENGTH = VOXEL_SIZES[-1]

# RESOLUTIONS = [1.25, 2.5]

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, nargs='+')
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('--voxel-size', type=float, required=True, nargs='+')
parser.add_argument('--length', type=float, required=True)
parser.add_argument('-t', '--num-threads-per-process', type=int, required=True)
parser.add_argument('--tilting', action='store_true')
args = parser.parse_args()

# logger
logger = logging.getLogger("rank[%i]" % comm.rank)
logger.setLevel(logging.DEBUG)
log_file = os.path.join(args.output, FILE_NAME) + ".log"
os.makedirs(args.output, exist_ok=True)
mh = MPIFileHandler(log_file, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
mh.setFormatter(formatter)
logger.addHandler(mh)
logger.info("args: " + " ".join(sys.argv[1:]))


def simulation(input,
               output,
               voxel_sizes,
               length,
               thickness,
               rot_f0=0,
               rot_f1=0,
               tilting=False,
               num_threads=1):

    logger.info(f'simulation()')

    if os.path.isfile(output):
        print("output already exists:", output)
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, 'w-') as h5f:
        logger.info(f"output:{output}")

        fiber_bundles = fastpli.io.fiber_bundles.load(input, '/')

        if rot_f0 or rot_f1:
            rot_inc = fastpli.tools.rotation.y(-np.deg2rad(rot_f0))
            rot_phi = fastpli.tools.rotation.x(np.deg2rad(rot_f1))
            fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                fiber_bundles, np.dot(rot_inc, rot_phi))

        h5f['/'].attrs['input'] = input
        h5f['/'].attrs['output'] = output

        with h5py.File(input, 'r') as h5f_in:
            omega = h5f_in['/'].attrs['omega']
            psi = h5f_in['/'].attrs['psi']

        h5f['/'].attrs['omega'] = omega
        h5f['/'].attrs['psi'] = psi
        h5f['/'].attrs['voxel_sizes'] = voxel_sizes
        h5f['/'].attrs['length'] = length
        h5f['/'].attrs['thickness'] = thickness
        h5f['/'].attrs['rot_f0'] = rot_f0
        h5f['/'].attrs['rot_f1'] = rot_f1
        h5f['/'].attrs['tilting'] = tilting
        h5f['/'].attrs['num_threads'] = num_threads

        logger.info(f'omega: {omega}')
        logger.info(f'psi: {psi}')
        logger.info(f'voxel_sizes: {voxel_sizes}')
        logger.info(f'length: {length}')
        logger.info(f'thickness: {thickness}')
        logger.info(f'rot_f0: {rot_f0}')
        logger.info(f'rot_f1: {rot_f1}')
        logger.info(f'tilting: {tilting}')
        logger.info(f'num_threads: {num_threads}')

        with open(os.path.abspath(__file__), 'r') as script:
            h5f.attrs['script'] = script.read()

        for voxel_size in voxel_sizes:
            logger.info(f'voxel_size: {voxel_size}')
            for dn, model in [(-0.001, 'p'), (0.002, 'r')]:
                dset = h5f.create_group(str(voxel_size) + '/' + model)
                logger.info(f'dn: {dn}, model: {model}')

                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                simpli.omp_num_threads = num_threads
                simpli.voxel_size = voxel_size
                simpli.resolution = voxel_size
                simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
                simpli.interpolate = True
                simpli.untilt_sensor_view = True
                simpli.wavelength = 525  # in nm

                simpli.set_voi(-0.5 * np.array([length, length, thickness]),
                               0.5 * np.array([length, length, thickness]))

                if tilting:
                    simpli.tilts = np.deg2rad(
                        np.array([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180),
                                  (5.5, 270)]))
                else:
                    simpli.tilts = np.deg2rad(np.array([(0, 0)]))
                simpli.add_crop_tilt_halo()

                print(f'memory: {round(simpli.memory_usage(), 2)} MB')
                return

                logger.info(f'memory: {round(simpli.memory_usage(), 2)} MB')
                if simpli.memory_usage() > 24 * 1e3:
                    print(str(round(simpli.memory_usage(), 2)) + 'MB')

                simpli.fiber_bundles = fiber_bundles
                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 0, model)]
                                                  ] * len(fiber_bundles)
                label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                    h5f=dset, save=["label_field"])

                # Simulate PLI Measurement
                simpli.light_intensity = 1  # a.u.
                simpli.save_parameter_h5(h5f=dset)
                for t, tilt in enumerate(simpli._tilts):
                    theta, phi = tilt[0], tilt[1]
                    logger.info(f'theta: {theta}, phi: {phi}')
                    images = simpli.run_simulation(label_field, vector_field,
                                                   tissue_properties, theta,
                                                   phi)

                    images = simpli.rm_crop_tilt_halo(images)

                    dset['simulation/data/' + str(t)] = images
                    dset['simulation/data/' + str(t)].attrs['theta'] = theta
                    dset['simulation/data/' + str(t)].attrs['phi'] = phi

                del label_field
                del vector_field
                del simpli

    logger.info(f'simulation - finished')


def _rot_pop2_factory(omega, delta_omega):
    import fastpli.tools
    import fastpli.objects

    inc_list = np.arange(0, 90 + 1e-9, delta_omega)

    n_rot = int(
        round(2 * np.pi / np.deg2rad(delta_omega) * np.sin(np.deg2rad(omega))))

    if n_rot == 0:
        fiber_phi_rotations = [0]
    else:
        n_rot = max(n_rot, 8)
        fiber_phi_rotations = np.linspace(0, 360, n_rot, False)

    for f0_inc in inc_list:
        for f1_rot in fiber_phi_rotations:
            yield f0_inc, f1_rot


if __name__ == '__main__':
    # Simulation
    files = args.input
    parameter = []

    for file in files:
        if os.path.isdir(file):
            raise IOError("path is a directory")
        omega = float(file.split("_omega_")[-1].split("_")[0])
        if not omega in OMEGAS:
            continue
        for f0, f1 in _rot_pop2_factory(omega, OMEGAS[1] - OMEGAS[0]):
            parameter.append((file, f0, f1))

    logger.info(f"len files: {len(files)}")
    logger.info(f"num parameters: {len(parameter)}")

    for file, f0, f1 in parameter[comm.Get_rank()::comm.Get_size()]:
        file_name = os.path.basename(file)
        file_name = file_name.rpartition(".solved")[0]
        output = os.path.join(
            args.output,
            f"{file_name}_vref_{args.voxel_size[0]}_length_{args.length}_f0_{round(f0,1)}_f1_{round(f1,1)}_.simulation.h5"
        )

        if os.path.isfile(output):
            logger.info(f"{output} already exists")

        simulation(input=file,
                   output=output,
                   voxel_sizes=args.voxel_size,
                   length=args.length,
                   thickness=60,
                   rot_f0=f0,
                   rot_f1=f1,
                   tilting=args.tilting,
                   num_threads=args.num_threads_per_process)

    logger.info("finished")
