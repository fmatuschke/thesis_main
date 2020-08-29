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

from tqdm import tqdm

# from simulation_repeat import run_simulation_pipeline_n
import helper.mpi
import fibers

from mpi4py import MPI
comm = MPI.COMM_WORLD
import multiprocessing as mp

if __name__ == "__main__":
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
                        nargs='+',
                        required=True,
                        help="input string.")

    parser.add_argument("-v",
                        "--voxel_size",
                        type=float,
                        required=True,
                        help="voxel_size in um.")

    parser.add_argument("-t",
                        "--threads",
                        type=int,
                        required=True,
                        help="number of threads")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    mp_pool = mp.Pool(args.threads)

    # logger
    logger = logging.getLogger("rank[%i]" % comm.rank)
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(args.output, "simulation.log")

    if os.path.isfile(log_file):
        print("Log file already exists")
        sys.exit(1)

    mh = helper.mpi.FileHandler(log_file,
                                mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
    mh.setFormatter(formatter)
    logger.addHandler(mh)
    logger.info("args: " + " ".join(sys.argv[1:]))
    logger.info(
        f"git: {subprocess.check_output(['git' 'rev-parse' 'HEAD']).strip()}")
    logger.info("script:\n" + open(os.path.abspath(__file__), 'r').read())

    # PARAMETER
    PIXEL_PM = 1.25
    PIXEL_LAP = 20
    LENGTH = 60
    THICKNESS = 60
    # FIBER_INCLINATION = np.linspace(0, 90, 10, True)

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
    fiber_inc = [(f, i) for f in file_list for i in fibers.inclinations()]
    for file, f0_inc in fiber_inc:
        logger.info(f"input file: {file}")

        with h5py.File(file, 'r') as h5f:
            omega = h5f['/'].attrs["omega"]

        for f1_rot in fibers.omega_rotations(omega):
            parameter.append((file, f0_inc, f1_rot))

    for file, f0_inc, f1_rot in tqdm(
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

            for m, (dn, model) in enumerate([(-0.001, 'p'), (0.002, 'r')]):
                for name, gain, intensity, res, tilt_angle, sigma in [
                    ('LAP', 3, 26000, PIXEL_LAP, 5.5, 0.71),
                    ('PM', 1.5, 50000, PIXEL_PM, 3.9, 0.71)
                ]:
                    dset = h5f.create_group(name + '/' + model)

                    # Setup Simpli
                    simpli = fastpli.simulation.Simpli()
                    warnings.filterwarnings("ignore", message="objects overlap")
                    simpli.omp_num_threads = args.threads
                    simpli.voxel_size = args.voxel_size
                    simpli.pixel_size = res
                    simpli.filter_rotations = np.linspace(0, np.pi, 9, False)
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
                        fiber_bundles, rot)
                    simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                        (1.0, dn, 1, model)]
                                                      ] * len(fiber_bundles)

                    logger.info(f"tissue_pipeline: model:{model}")

                    save = ['optic', 'epa', 'rofl']
                    #                     save += ['tissue'] if m == 0 and name == 'LAP' else []
                    label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                        h5f=dset, save=save)

                    unique_elements, counts_elements = np.unique(
                        label_field, return_counts=True)
                    dset.attrs['label_field_stats'] = np.asarray(
                        (unique_elements, counts_elements))

                    # Simulate PLI Measurement
                    logger.info(f"simulation_pipeline: model:{model}")
                    # FIXME: LAP sigma ist bei einem pixel sinnfrei -> 20µm

                    simpli.light_intensity = intensity  # a.u.
                    simpli.sensor_gain = gain

                    simpli.save_parameter_h5(h5f=dset)

                    # if name == 'LAP':
                    #     run_simulation_pipeline_n(simpli,
                    #                               label_field,
                    #                               vector_field,
                    #                               tissue_properties,
                    #                               int((PIXEL_LAP / PIXEL_PM)**2),
                    #                               h5f=dset,
                    #                               save=save,
                    #                               crop_tilt=True,
                    #                               mp_pool=mp_pool)
                    # else:
                    simpli.run_simulation_pipeline(label_field,
                                                   vector_field,
                                                   tissue_properties,
                                                   h5f=dset,
                                                   save=save,
                                                   crop_tilt=True,
                                                   mp_pool=mp_pool)

                    dset.attrs['parameter/psi'] = psi
                    dset.attrs['parameter/omega'] = omega
                    dset.attrs['parameter/v'] = LENGTH
                    dset.attrs['parameter/f0_inc'] = f0_inc
                    dset.attrs['parameter/f1_rot'] = f1_rot
                    dset.attrs[
                        'parameter/crop_tilt_voxel'] = simpli.crop_tilt_voxel()

                    del label_field
                    del vector_field
                    del simpli
