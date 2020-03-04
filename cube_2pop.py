import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.tools
import fastpli.io

import numpy as np
import argparse
import logging
import h5py
import sys
import os

from tqdm import tqdm

from MPIFileHandler import MPIFileHandler
from mpi4py import MPI
comm = MPI.COMM_WORLD
NUM_THREADS = 4

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

parser.add_argument("-r",
                    "--fiber_radius",
                    default=1,
                    help="mean value of fiber radius")

args = parser.parse_args()
output_name = os.path.join(args.output, FILE_NAME)
os.makedirs(args.output, exist_ok=True)

# logger
logger = logging.getLogger("rank[%i]" % comm.rank)
logger.setLevel(logging.DEBUG)
log_file = output_name + ".log"
mh = MPIFileHandler(log_file, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
mh.setFormatter(formatter)
logger.addHandler(mh)
logger.info("args: " + " ".join(sys.argv[1:]))

# Fiber Model
LENGTH = 210  # to rotate a 120 um cube inside
RADIUS_LOGMEAN = args.fiber_radius
PSI = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
       0.9]  # fiber fraction: PSI * f0 + (1-PSI) * f1
OMEGA = np.linspace(0, 90, 10, True)  # angle of opening (f0, f1)
PSI, OMEGA = np.meshgrid(PSI, OMEGA)
PARAMETER = list(zip(PSI.flatten(), OMEGA.flatten()))
PARAMETER.append((0.0, 0.0))
PARAMETER.append((1.0, 0.0))

# solve
for psi, omega in tqdm(PARAMETER[comm.Get_rank()::comm.Get_size()]):
    logger.info(f"psi:{psi}, omega:{omega}")

    # setup solver
    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = RADIUS_LOGMEAN * 2
    solver.obj_min_radius = RADIUS_LOGMEAN * 5
    solver.omp_num_threads = NUM_THREADS

    file_pref = output_name + f"_psi_{round(psi, 2)}_omega_{round(omega, 2)}_"
    logger.info(f"file_pref: {file_pref}")

    seeds = fastpli.model.sandbox.seeds.triangular_grid(LENGTH * 2,
                                                        LENGTH * 2,
                                                        2 * RADIUS_LOGMEAN,
                                                        center=True)

    # pick random seeds for fiber population distribution
    seeds_0 = seeds[np.random.rand(seeds.shape[0]) < psi, :]
    seeds_1 = seeds[np.random.rand(seeds.shape[0]) < (1 - psi), :]
    rnd_radii_0 = RADIUS_LOGMEAN * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = RADIUS_LOGMEAN * np.random.lognormal(0, 0.1, seeds_1.shape[0])

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            q=0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            phi=np.deg2rad(0),
            theta=np.deg2rad(90),
            seeds=seeds_0,
            radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            q=0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            phi=np.deg2rad(omega),
            theta=np.deg2rad(90),
            seeds=seeds_1 + RADIUS_LOGMEAN,
            radii=rnd_radii_1)
    ]

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)
    logger.info(f"init: {solver.num_obj}/{solver.num_col_obj}")

    # add rnd displacement
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.05 * RADIUS_LOGMEAN,
                                          (f.shape[0], 3))
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
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius

    # fastpli.io.fiber_bundles.save(file_pref + '.init.dat', solver.fiber_bundles)

    # Run Solver
    logger.info(f"run solver")
    for i in tqdm(range(args.max_steps)):
        if solver.step():
            break

        overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0
        if i % 50 == 0:
            logger.info(
                f"step: {i}, {solver.num_obj}/{solver.num_col_obj} {round(overlap * 100)}%"
            )

        if overlap <= 0.01:
            break

    overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0

    logger.info(f"solved: {i}, {solver.num_obj}/{solver.num_col_obj}")
    logger.debug(f"save solved")
    with h5py.File(file_pref + '.solved.h5', 'w-') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['psi'] = psi
        h5f['/'].attrs['omega'] = omega
        h5f['/'].attrs['overlap'] = overlap
        h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_steps'] = solver.step_num
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius

    # fastpli.io.fiber_bundles.save(file_pref + '.solved.dat',
    #                               solver.fiber_bundles)
