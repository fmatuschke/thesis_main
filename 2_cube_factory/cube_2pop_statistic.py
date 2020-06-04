import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.tools
import fastpli.io

import numpy as np
import itertools
import argparse
import logging
import time
import h5py
import sys
import os

from tqdm import tqdm
import helper.mpi

from mpi4py import MPI
comm = MPI.COMM_WORLD
NUM_THREADS = 1

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

parser.add_argument("-n",
                    "--max_steps",
                    type=int,
                    required=True,
                    help="Number of max_steps.")

args = parser.parse_args()
output_name = os.path.join(args.output, FILE_NAME)
os.makedirs(args.output, exist_ok=True)

# logger
logger = logging.getLogger("rank[%i]" % comm.rank)
logger.setLevel(logging.DEBUG)
log_file = output_name + ".log"
mh = helper.mpi.FileHandler(log_file, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
mh.setFormatter(formatter)
logger.addHandler(mh)
logger.info("args: " + " ".join(sys.argv[1:]))

# Fiber Model
SIZE = 90  # to create a 60 micro meter cube
FIBER_RADII = [1.0]
OBJ_MEAN_LENGTH_F = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
OBJ_MIN_RADIUS_F = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
PSI = [0.25, 0.5, 0.75]  # fiber fraction: PSI * f0 + (1-PSI) * f1
OMEGA = [0, 30, 60, 90]  # angle of opening (f0, f1)
PSI, OMEGA = np.meshgrid(PSI, OMEGA)
PARAMETER = list(zip(PSI.flatten(), OMEGA.flatten()))
PARAMETER.append((0.0, 0.0))
PARAMETER.append((1.0, 0.0))

parameters = list(
    itertools.product(PARAMETER, FIBER_RADII, OBJ_MEAN_LENGTH_F,
                      OBJ_MIN_RADIUS_F))
parameters = parameters[comm.Get_rank()::comm.Get_size()]

for (psi, omega), radius, mean_length_f, min_radius_f in tqdm(parameters):

    logger.info(
        f"psi:{psi}, omega:{omega}, radius:{radius}," \
            " mean_length_f:{mean_length_f}, min_radius_f:{min_radius_f}"
    )

    # setup solver
    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = radius * mean_length_f
    solver.obj_min_radius = radius * min_radius_f
    solver.omp_num_threads = NUM_THREADS

    file_pref = output_name + f"_psi_{psi:.2f}_omega_{omega:.2f}_r_" \
                               f"{radius:.2f}_v0_{SIZE:.0f}_fl_{mean_length_f:.2f}_" \
                               f"fr_{min_radius_f:.2f}_"
    logger.info(f"file_pref: {file_pref}")

    seeds = fastpli.model.sandbox.seeds.triangular_grid(SIZE * 2,
                                                        SIZE * 2,
                                                        2 * radius,
                                                        center=True)

    # pick random seeds for fiber population distribution
    seeds_0 = seeds[np.random.rand(seeds.shape[0]) < psi, :]
    seeds_1 = seeds[np.random.rand(seeds.shape[0]) < (1 - psi), :]
    rnd_radii_0 = radius * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = radius * np.random.lognormal(0, 0.1, seeds_1.shape[0])

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(p=-0.5 *
                                           np.array([SIZE, SIZE, SIZE]),
                                           q=0.5 * np.array([SIZE, SIZE, SIZE]),
                                           phi=np.deg2rad(0),
                                           theta=np.deg2rad(90),
                                           seeds=seeds_0,
                                           radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(p=-0.5 *
                                           np.array([SIZE, SIZE, SIZE]),
                                           q=0.5 * np.array([SIZE, SIZE, SIZE]),
                                           phi=np.deg2rad(omega),
                                           theta=np.deg2rad(90),
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

    # if comm.Get_size() == 1:
    #     solver.draw_scene()

    # Save Data
    logger.debug(f"save init")
    with h5py.File(file_pref + '.init.h5', 'w-') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['psi'] = psi
        h5f['/'].attrs['omega'] = omega
        # print("OTHER META DATA?")
        # h5f['/'].attrs['overlap'] = overlap
        # h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        # h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_steps'] = 0
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius

    # fastpli.io.fiber_bundles.save(file_pref + '.init.dat', solver.fiber_bundles)

    # Run Solver
    logger.info(f"run solver")
    start_time = time.time()
    for i in tqdm(range(1, args.max_steps)):
        if solver.step():
            break

        overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0
        if i % 50 == 0:
            logger.info(
                f"step: {i}, {solver.num_obj}/{solver.num_col_obj} {round(overlap * 100)}%"
            )
            solver.fiber_bundles = fastpli.objects.fiber_bundles.Cut(
                solver.fiber_bundles,
                [-0.5 * np.array([SIZE] * 3), 0.5 * np.array([SIZE] * 3)])

        # if i > args.max_steps / 2 and overlap <= 0.001:
        #     break

    end_time = time.time()
    overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0

    logger.info(f"time: {end_time - start_time}")
    logger.info(
        f"solved: {i}, {solver.num_obj}/{solver.num_col_obj}, {overlap}")
    logger.debug(f"save solved")
    with h5py.File(file_pref + '.solved.h5', 'w-') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['psi'] = psi
        h5f['/'].attrs['omega'] = omega
        h5f['/'].attrs['overlap'] = overlap
        h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_steps'] = solver.num_steps
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
        h5f['/'].attrs['time'] = end_time - start_time

    # fastpli.io.fiber_bundles.save(file_pref + '.solved.dat',
    #                               solver.fiber_bundles)
