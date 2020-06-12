import numpy as np
import multiprocessing as mp
import itertools
import argparse
import logging
import datetime
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
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

parser.add_argument("-p",
                    "--num_proc",
                    type=int,
                    required=True,
                    help="Number of processes.")

args = parser.parse_args()
output_name = os.path.join(args.output, FILE_NAME)
os.makedirs(args.output, exist_ok=True)

# logger
FORMAT = '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    filename=output_name +
    f'.{datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")}.log',
    filemode='w')
logger = logging.getLogger()
helper.mplog.install_mp_handler(logger)

# TODO: add noise and ref voxel size without noise

PIXEL_SIZE = 1.0
THICKNESS = 60
VOXEL_SIZES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
FIBER_RADII = [1.0]


def run(parameter):
    psi, omega, f0_inc, f1_rot = parameter
    radius = FIBER_RADII[0]

    # generate model
    file_pref = output_name + f"_psi_{psi:.2f}_omega_{omega:.2f}_" \
                               f"f0_inc_{f0_inc:.2f}_" \
                               f"f1_rot_{f1_rot:.2f}_" \
                               f"r_{radius:.2f}_p0_{PIXEL_SIZE:.2f}_"
    logger.info(f"file_pref: {file_pref}")

    # setup solver
    logger.info(f"prepair generation")

    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = radius * 2
    solver.obj_min_radius = radius * 2
    solver.omp_num_threads = 1

    seeds = fastpli.model.sandbox.seeds.triangular_grid(THICKNESS * 2,
                                                        THICKNESS * 2,
                                                        1 * radius,
                                                        center=True)

    # pick random seeds for fiber population distribution
    seeds_0 = seeds[np.random.rand(seeds.shape[0]) < psi, :]
    seeds_1 = seeds[np.random.rand(seeds.shape[0]) < (1 - psi), :]
    rnd_radii_0 = radius * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = radius * np.random.lognormal(0, 0.1, seeds_1.shape[0])

    vec = np.array([1, 0, 0])
    rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
    rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
    rot = np.dot(rot_inc, rot_phi)
    vec = np.dot(rot, vec)

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 *
            np.array([PIXEL_SIZE * 3, PIXEL_SIZE * 3, THICKNESS * 1.25]),
            q=0.5 *
            np.array([PIXEL_SIZE * 3, PIXEL_SIZE * 3, THICKNESS * 1.25]),
            phi=np.deg2rad(0),
            theta=np.pi / 2 - np.deg2rad(f0_inc),
            seeds=seeds_0,
            radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 *
            np.array([PIXEL_SIZE * 3, PIXEL_SIZE * 3, THICKNESS * 1.25]),
            q=0.5 *
            np.array([PIXEL_SIZE * 3, PIXEL_SIZE * 3, THICKNESS * 1.25]),
            phi=np.arctan2(vec[1], vec[0]),
            theta=np.arccos(vec[2]),
            seeds=seeds_1 + radius,
            radii=rnd_radii_1)
    ]

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)
    logger.info(f"init: {solver.num_obj}/{solver.num_col_obj}")

    # add rnd displacement
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.05 * radius, (f.shape[0], 3))
            f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)
    logger.info(f"rnd displacement: {solver.num_obj}/{solver.num_col_obj}")

    # Run Solver
    logger.info(f"run solver")
    start_time = time.time()

    for i in range(1000):
        if solver.step():
            break
        if i % 50 == 0:
            overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0
            logger.info(
                f"step: {i}, {solver.num_obj}/{solver.num_col_obj} {round(overlap * 100)}%"
            )
            solver.fiber_bundles = fastpli.objects.fiber_bundles.Cut(
                solver.fiber_bundles, [
                    -0.5 * np.array(
                        [PIXEL_SIZE * 3, PIXEL_SIZE * 3, THICKNESS * 1.25]),
                    0.5 *
                    np.array([PIXEL_SIZE * 3, PIXEL_SIZE * 3, THICKNESS * 1.25])
                ])

        # solver.draw_scene()

    end_time = time.time()
    fiber_bundles = solver.fiber_bundles

    logger.info(f"steps: {i}")
    logger.info(f"time: {end_time - start_time}")
    logger.info(f"saveing solved")
    with h5py.File(file_pref + '.solved.h5', 'w') as h5f:
        dset = h5f.create_group(f'solver/')
        solver.save_h5(dset, script=open(os.path.abspath(__file__), 'r').read())
        dset.attrs['psi'] = psi
        dset.attrs['omega'] = omega
        dset.attrs['num_obj'] = solver.num_obj
        dset.attrs['num_steps'] = solver.num_steps
        dset.attrs['obj_mean_length'] = solver.obj_mean_length
        dset.attrs['obj_min_radius'] = solver.obj_min_radius
        dset.attrs['time'] = end_time - start_time

        logger.info(f"generation done")

        # Setup Simpli
        logger.info(f"prepair simulation")
        simpli = fastpli.simulation.Simpli()
        simpli.omp_num_threads = 1
        simpli.pixel_size = PIXEL_SIZE
        simpli.sensor_gain = 0
        simpli.optical_sigma = 0  # in voxel size
        simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
        simpli.interpolate = True
        simpli.untilt_sensor_view = True
        simpli.wavelength = 525  # in nm

        simpli.tilts = np.deg2rad(np.array([(0, 0)]))

        simpli.fiber_bundles = fiber_bundles

        for voxel_size in VOXEL_SIZES:
            simpli.voxel_size = voxel_size
            simpli.set_voi(-0.5 * np.array([PIXEL_SIZE, PIXEL_SIZE, THICKNESS]),
                           0.5 * np.array([PIXEL_SIZE, PIXEL_SIZE, THICKNESS]))

            if simpli.memory_usage() * args.num_proc > 200000:
                print(str(round(simpli.memory_usage(), 2)) + 'MB')
                return

            for dn, model in [(-0.001, 'p'), (0.002, 'r')]:
                dset = h5f.create_group(f'simpli/{voxel_size}/{model}')

                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 0, model)]
                                                  ] * len(fiber_bundles)

                label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                )

                # Simulate PLI Measurement
                simpli.light_intensity = 1  # a.u.
                simpli.save_parameter_h5(h5f=dset)
                for t, tilt in enumerate(simpli._tilts):
                    theta, phi = tilt[0], tilt[1]
                    images = simpli.run_simulation(label_field, vector_field,
                                                   tissue_properties, theta,
                                                   phi)

                    dset['simulation/data/' + str(t)] = images
                    dset['simulation/data/' + str(t)].attrs['theta'] = theta
                    dset['simulation/data/' + str(t)].attrs['phi'] = phi

                    # apply optic to simulation
                    images = simpli.apply_optic(images)
                    dset['simulation/optic/' + str(t)] = images

                    # calculate modalities
                    epa = simpli.apply_epa(images)
                    dset['analysis/epa/' + str(t) + '/transmittance'] = epa[0]
                    dset['analysis/epa/' + str(t) + '/direction'] = epa[1]
                    dset['analysis/epa/' + str(t) + '/retardation'] = epa[2]

                del label_field
                del vector_field


if __name__ == "__main__":
    logger.info("args: " + " ".join(sys.argv[1:]))

    # VOXEL_SIZES = [
    #     0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12,
    #     0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
    # ]

    PSI = [0.25, 0.5]  # fiber fraction: PSI * f0 + (1-PSI) * f1
    OMEGA = [45, 90]  # angle of opening (f0, f1)
    PSI_OMEGA = list(itertools.product(PSI, OMEGA))
    PSI_OMEGA.append((1.0, 0))

    FIBER_INCLINATION = np.linspace(0, 90, 10, True)
    PSI_OMEGA_F0_F1 = []
    for (psi,
         omega), f0_inc in list(itertools.product(PSI_OMEGA,
                                                  FIBER_INCLINATION)):
        n_rot = int(
            np.round(
                np.sqrt((1 - np.cos(2 * np.deg2rad(omega))) /
                        (1 - np.cos(np.deg2rad(10))))))

        if n_rot == 0:
            PSI_OMEGA_F0_F1.append((psi, omega, f0_inc, 0))
        else:
            n_rot += (n_rot + 1) % 2
            n_rot = max(n_rot, 3)
            for f1_rot in np.linspace(-180, 180, n_rot, True):
                f1_rot = np.round(f1_rot, 2)
                PSI_OMEGA_F0_F1.append((psi, omega, f0_inc, f1_rot))

    # parameters = list(
    #     itertools.product(PSI_OMEGA_F0_F1, (VOXEL_SIZES), FIBER_RADII))

    # print(parameters[1])

    # run(PSI_OMEGA_F0_F1[0])

    with mp.Pool(processes=args.num_proc) as pool:
        [
            d for d in tqdm(pool.imap_unordered(run, PSI_OMEGA_F0_F1),
                            total=len(PSI_OMEGA_F0_F1))
        ]
