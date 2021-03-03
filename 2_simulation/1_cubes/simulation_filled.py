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

import simulation_filled_helper

import tqdm

# from simulation_repeat import run_simulation_pipeline_n
import helper.mpi
import helper.file
import models

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

parser.add_argument("--start", type=int, required=True, help="mpi start.")

parser.add_argument('--Vervet', default=False, action='store_true')

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

# logger
logger = logging.getLogger("rank[%i]" % comm.rank)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    os.path.join(
        args.output,
        f'simulation_{args.voxel_size}_{comm.Get_size()}_{comm.Get_rank()}_{args.start}.log',
    ), 'a')
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
    fiber_inc = [
        (f, i) for f in file_list for i in models.inclinations(args.n_inc)
    ]
    for file, f0_inc in fiber_inc:
        # h5py to slow
        # with h5py.File(file, 'r') as h5f:
        #     omega = h5f['/'].attrs["omega"]
        omega = helper.file.value(file, "omega")

        for f1_rot in models.omega_rotations(omega, args.d_rot):
            parameter.append((file, f0_inc, f1_rot))

    # print(len(parameter))
    # exit(0)
    logger.info(f"len parameter {len(parameter)}")

    for file, f0_inc, f1_rot in tqdm.tqdm(
            parameter[args.start + comm.Get_rank()::comm.Get_size()]):

        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += f'_filled_'
        file_name += f'_vs_{args.voxel_size:.4f}'
        file_name += f'_inc_{f0_inc:.2f}'
        file_name += f'_rot_{f1_rot:.2f}'
        file_name = os.path.join(args.output, file_name)
        logger.info(f"input file: {file}")
        logger.info(f"output file: {file_name}")

        if os.path.isfile(file_name + '.h5'):
            logger.info(f"file exists: {file_name}.h5")
            continue

        with h5py.File(file, 'r') as h5f:
            fiber_bundles_h5 = fastpli.io.fiber_bundles.load_h5(h5f)
            psi = h5f['/'].attrs["psi"]
            omega = h5f['/'].attrs["omega"]
            radius = h5f['/'].attrs["radius"]
            v0 = h5f['/'].attrs["v0"]

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

            dn, model = (-0.008 / 2, 'p') if radius > 0.5 else (0.008, 'r')

            for name, gain, intensity, res, tilt_angle, sigma in [
                ('LAP', 3, 35000, PIXEL_LAP, 5.5, 0.75),
                ('PM', 0.1175, 8000, PIXEL_PM, 3.9, 0.75)
            ]:
                mu = 0

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

                fiber_bundles = simulation_filled_helper.fill_fb(
                    fiber_bundles_h5, radius, 0.5)
                simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                    fiber_bundles, rot)
                simpli.fiber_bundles_properties = [[(0.75, 0, mu, 'b'),
                                                    (1.0, dn, mu, model)]
                                                  ] * len(fiber_bundles)

                logger.info(f"tissue_pipeline: model:{model}")

                tissue, optical_axis, tissue_properties = simpli.run_tissue_pipeline(
                )
                tissue_thickness = np.sum(tissue > 0, -1)

                # Simulate PLI Measurement
                logger.info(f"simulation_pipeline: model:{model}")

                simpli.light_intensity = intensity  # a.u.
                simpli.noise_model = lambda x: np.round(
                    np.random.normal(x, np.sqrt(gain * x))).astype(np.uint16)

                dset = h5f.create_group(f'{name}/{model}')
                simpli.save_parameter_h5(h5f=dset)
                if 'tissue_stats' not in dset:
                    unique_elements, counts_elements = np.unique(
                        tissue, return_counts=True)
                    dset.attrs['tissue_stats'] = np.asarray(
                        (unique_elements, counts_elements))

                images_stack = [None] * 5
                # print('Run Simulation:')
                for t, (theta, phi) in enumerate(simpli.tilts):
                    # print(round(np.rad2deg(theta), 1),
                    #       round(np.rad2deg(phi), 1))
                    images_stack[t] = simpli.run_simulation(
                        tissue, optical_axis, tissue_properties, theta, phi)

                species_list = [('Roden', 8), ('Vervet', 30), ('Human', 65)]
                if args.Vervet:
                    species_list = [('Vervet', 30)]

                for species, mu in species_list:
                    dset = h5f.create_group(f'{name}/{species}/{model}')

                    tilting_stack = [None] * 5
                    for t, (theta, phi) in enumerate(simpli.tilts):
                        # absorption
                        images = np.multiply(
                            images_stack[t],
                            np.exp(-mu * tissue_thickness * 1e-3 *
                                   simpli.voxel_size)[:, :, None])

                        images = simpli.rm_crop_tilt_halo(images)

                        # apply optic to simulation
                        resample, images = simpli.apply_optic(images)
                        dset['simulation/optic/' + str(t)] = images
                        dset['simulation/resample/' + str(t)] = resample

                        # calculate modalities
                        epa = simpli.apply_epa(images)
                        dset['analysis/epa/' + str(t) +
                             '/transmittance'] = epa[0]
                        dset['analysis/epa/' + str(t) + '/direction'] = epa[1]
                        dset['analysis/epa/' + str(t) + '/retardation'] = epa[2]

                        tilting_stack[t] = images

                    mask = None  # keep analysing all pixels

                    # print('Run ROFL analysis:')
                    rofl_direction, rofl_incl, rofl_t_rel, param = simpli.apply_rofl(
                        tilting_stack, mask=mask)

                    dset['analysis/rofl/direction'] = rofl_direction
                    dset['analysis/rofl/inclination'] = rofl_incl
                    dset['analysis/rofl/t_rel'] = rofl_t_rel

                    dset['analysis/rofl/direction_conf'] = param[0]
                    dset['analysis/rofl/inclination_conf'] = param[1]
                    dset['analysis/rofl/t_rel_conf'] = param[2]
                    dset['analysis/rofl/func'] = param[3]
                    dset['analysis/rofl/n_iter'] = param[4]

                    dset.attrs['parameter/radius'] = radius
                    dset.attrs['parameter/psi'] = psi
                    dset.attrs['parameter/omega'] = omega
                    dset.attrs['parameter/fiber_path'] = file
                    dset.attrs['parameter/volume'] = LENGTH
                    dset.attrs['parameter/f0_inc'] = f0_inc
                    dset.attrs['parameter/f1_rot'] = f1_rot
                    dset.attrs[
                        'parameter/crop_tilt_voxel'] = simpli.crop_tilt_voxel()

                h5f.flush()
                del tissue
                del optical_axis
                del simpli
