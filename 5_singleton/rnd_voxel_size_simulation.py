import numpy as np
import multiprocessing as mp
import subprocess
import itertools
import argparse
import logging
import datetime
import warnings
import time
import h5py
import sys
import os

from tqdm import tqdm

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.tools
import fastpli.io

import helper.mplog

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    nargs='+',
                    required=True,
                    help="input string.")

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

# parser.add_argument("-n",
#                     "--repeat",
#                     type=int,
#                     required=True,
#                     help="repeat n times")

parser.add_argument("-m",
                    "--repeat_noise",
                    type=int,
                    required=True,
                    help="repeat noise m times")

parser.add_argument("-p",
                    "--num_proc",
                    type=int,
                    required=True,
                    help="Number of parallel processes.")

parser.add_argument("-t",
                    "--num_threads",
                    type=int,
                    required=True,
                    help="Number of threads per process.")

args = parser.parse_args()
output_name = os.path.join(args.output, FILE_NAME)
os.makedirs(args.output, exist_ok=True)

# logger
FORMAT = '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    filename=output_name,
    # + f'.{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
    filemode='a')
logger = logging.getLogger()
helper.mplog.install_mp_handler(logger)

# VOXEL_SIZES = [0.01, 0.025, 0.05, 0.125, 0.25, 0.625, 1.25]
VOXEL_SIZES = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.125, 0.25, 0.625, 1.25
]
# PIXEL_SIZE = 1.25  # PM
THICKNESS = 60


def get_file_pref(parameter):
    file, f0_inc, f1_rot = parameter

    base = os.path.basename(file)
    pre = base.split("_psi_")[0]
    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])

    return output_name + f"_{pre}_psi_{psi:.2f}_omega_{omega:.2f}_" \
        f"f0_inc_{f0_inc:.2f}_" \
        f"f1_rot_{f1_rot:.2f}_" \
        f"vs_0_{VOXEL_SIZES[0]:.2f}_"


def run(parameter):
    file, f0_inc, f1_rot = parameter

    # generate model
    file_pref = get_file_pref(parameter)
    logger.info(f"file_pref: {file_pref}")

    with h5py.File(file_pref + '.h5', 'w') as h5f:

        h5f.attrs['script'] = open(os.path.abspath(__file__), 'r').read()

        with h5py.File(file, 'r') as h5f_:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
            psi = h5f_.attrs["psi"]
            omega = h5f_.attrs["omega"]

        h5f.attrs['psi'] = psi
        h5f.attrs['omega'] = omega
        h5f.attrs['f0_inc'] = f0_inc
        h5f.attrs['f1_rot'] = f1_rot
        # h5f['/'].attrs['pixel_size'] = PIXEL_SIZE
        h5f['fiber_bundles'] = file

        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)
        fiber_bundles = fastpli.objects.fiber_bundles.Rotate(fiber_bundles, rot)

        # for n in range(args.repeat):
        Nx = 60 / VOXEL_SIZES[-1]
        Ny = 60 / VOXEL_SIZES[-1]
        print("test")
        for n, (nx, ny) in enumerate(
                itertools.product(range(-int(Nx // 2), int(Nx // 2), 10),
                                  range(-int(Ny // 2), int(Ny // 2), 10))):
            logger.info(f"n_repeat: {n}")
            # Setup Simpli
            logger.info(f"prepair simulation")
            simpli = fastpli.simulation.Simpli()
            simpli.omp_num_threads = args.num_threads
            # simpli.sensor_gain = 1.5  # pm
            simpli.optical_sigma = 0.71  # in voxel size
            simpli.filter_rotations = np.linspace(0, np.pi, 9, False)
            simpli.interpolate = "Slerp"
            simpli.untilt_sensor_view = True
            simpli.wavelength = 525  # in nm
            simpli.light_intensity = 50000  # a.u.
            simpli.fiber_bundles = fiber_bundles
            simpli.tilts = np.deg2rad(np.array([(0, 0)]))

            dim_origin = np.array([nx, ny, 0]) * VOXEL_SIZES[-1]
            # print(dim_origin)
            # continue
            # dim_origin = simpli.dim_origin + np.random.uniform(
            #     -THICKNESS / 2, THICKNESS / 2, [3])
            logger.info(f"dim_origin: {dim_origin}")

            for voxel_size in VOXEL_SIZES:
                logger.info(f"voxel_size: {voxel_size}")

                for mode in ["ref", "norm"]:
                    # normal

                    if mode == "ref":
                        simpli.pixel_size = voxel_size
                        simpli.voxel_size = 1 / 2 * voxel_size
                    elif mode == "norm":
                        simpli.pixel_size = voxel_size
                        simpli.voxel_size = voxel_size
                    else:
                        raise ValueError("FOO")
                    simpli.set_voi(
                        -0.5 * np.array(
                            [simpli.pixel_size, simpli.pixel_size, THICKNESS]),
                        0.5 * np.array(
                            [simpli.pixel_size, simpli.pixel_size, THICKNESS]))
                    simpli.dim_origin = dim_origin

                    logger.info("memory: " +
                                str(round(simpli.memory_usage(), 2)) + 'MB')
                    if simpli.memory_usage() * args.num_proc > 100000:
                        print(str(round(simpli.memory_usage(), 2)) + 'MB')
                        sys.exit(1)

                    for dn, model in [(-0.001, 'p'), (0.002, 'r')]:
                        logger.info(f"model: {model}")
                        dset = h5f.create_group(
                            f'simpli/{voxel_size}/{model}/{n}/{mode}')
                        dset.attrs['voxel_size'] = simpli.voxel_size
                        dset.attrs['pixel_size'] = simpli.pixel_size
                        dset.attrs['model'] = model
                        dset.attrs['mode'] = mode
                        dset.attrs['n'] = n
                        dset.attrs['dim_origin'] = simpli.dim_origin
                        dset.attrs['voi_min'] = simpli.get_voi()[0]
                        dset.attrs['voi_max'] = simpli.get_voi()[1]

                        simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                            (1.0, dn, 0, model)]
                                                          ] * len(fiber_bundles)

                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore",
                                                    message="objects overlap")
                            label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                            )

                        unique_elements, counts_elements = np.unique(
                            label_field, return_counts=True)
                        dset.attrs['label_field_stats'] = np.asarray(
                            (unique_elements, counts_elements))

                        # Simulate PLI Measurement
                        # simpli.save_parameter_h5(h5f=dset)
                        for t, tilt in enumerate(simpli._tilts):
                            theta, phi = tilt[0], tilt[1]
                            logger.info(f"simulation: {theta}, {phi}")
                            images = simpli.run_simulation(
                                label_field, vector_field, tissue_properties,
                                theta, phi)

                            dset['simulation/data/' + str(t)] = images
                            dset['simulation/data/' +
                                 str(t)].attrs['theta'] = theta
                            dset['simulation/data/' + str(t)].attrs['phi'] = phi

                            for m in range(args.repeat_noise):
                                logger.info(f"m_repeat_noise: {m}")
                                if m == 0:
                                    simpli.sensor_gain = 0
                                else:
                                    simpli.sensor_gain = 1.5
                                # apply optic to simulation
                                logger.info(f"apply_optic")
                                images_ = simpli.apply_optic(images)
                                dset[f'simulation/optic/{t}/{m}'] = images_
                                # calculate modalities
                                logger.info(f"epa")
                                epa = simpli.apply_epa(images_)
                                dset[
                                    f'analysis/epa/{t}/transmittance/{m}'] = epa[
                                        0]
                                dset[f'analysis/epa/{t}/direction/{m}'] = epa[1]
                                dset[f'analysis/epa/{t}/retardation/{m}'] = epa[
                                    2]

                            dset[f'simulation/optic/{t}'].attrs['theta'] = theta
                            dset[f'simulation/optic/{t}'].attrs['phi'] = phi
                            dset[f'analysis/epa/{t}'].attrs['theta'] = theta
                            dset[f'analysis/epa/{t}'].attrs['phi'] = phi

                        del label_field
                        del vector_field


def check_file(p):
    file_pref = get_file_pref(p)
    return not os.path.isfile(file_pref + '.h5')


def f0_incs(n=10):
    return np.linspace(0, 90, n, True)


def omega_rotations(omega, dphi=np.deg2rad(10)):

    rot = []

    n_rot = int(np.round(np.sqrt((1 - np.cos(2 * omega)) / (1 - np.cos(dphi)))))
    if n_rot == 0:
        rot.append(0)
    else:
        n_rot += (n_rot + 1) % 2
        n_rot = max(n_rot, 3)
        for f_rot in np.linspace(-180, 180, n_rot, True):
            f_rot = np.round(f_rot, 2)
            rot.append(f_rot)

    return rot


if __name__ == "__main__":
    logger.info("args: " + " ".join(sys.argv[1:]))
    logger.info(f"git: {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}")
    logger.info("script:\n" + open(os.path.abspath(__file__), 'r').read())

    file_list = args.input

    parameters = []
    for file, f0_inc in list(itertools.product(file_list, f0_incs(10))):
        omega = float(file.split("_omega_")[1].split("_")[0])

        for f1_rot in omega_rotations(omega):
            parameters.append((file, f0_inc, f1_rot))

    # filter
    logger.info("parameters before filter: " + str(parameters))
    parameters = list(filter(check_file, parameters))
    logger.info("parameters after filter: " + str(parameters))

    # run(parameters[0])
    # run((1.0, 0, 90, 0))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            d for d in tqdm(pool.imap_unordered(run, parameters),
                            total=len(parameters),
                            smoothing=0)
        ]

# sort -t- -k3.1,3.4
