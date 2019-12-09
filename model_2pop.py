import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.io

import numpy as np
import h5py
import os

from tqdm import tqdm
from mpi4py import MPI

import fastpli_helper as helper

# reproducability
np.random.seed(42)

# GLOBAL PARAMETERS
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

MODEL_NAME = "cube_2pop"
OUTPUT_PATH = os.path.join(FILE_PATH, 'output', 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# FIBERS
DPHI = np.linspace(0, 90, 10, True)
RADIUS_LOGMEAN = 1
LENGTH = 65 * np.sqrt(3)
PARAMETERS = DPHI

# setup solver
solver = fastpli.model.solver.Solver()
solver.obj_mean_length = RADIUS_LOGMEAN * 2
solver.obj_min_radius = RADIUS_LOGMEAN * 5
solver.omp_num_threads = 8

# mpi
index = 0
comm = MPI.COMM_WORLD

radius_old = None
for rank, dphi in enumerate(PARAMETERS[comm.Get_rank()::comm.Get_size()]):
    print("rank:" + str(comm.Get_rank()), "parameter:", PARAMETERS[rank])

    file_pref = helper.version_file_name(
        os.path.join(OUTPUT_PATH, MODEL_NAME + '_dphi_' + str(round(dphi, 1))))
    print("rank:" + file_pref)

    seeds = fastpli.model.sandbox.seeds.triangular_grid(LENGTH,
                                                        LENGTH,
                                                        2 * RADIUS_LOGMEAN,
                                                        center=True)
    rnd_radii = RADIUS_LOGMEAN * np.random.lognormal(0, 0.1, seeds.shape[0])

    print(np.mean(rnd_radii), np.std(rnd_radii))

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            q=0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            phi=np.deg2rad(0),
            theta=np.deg2rad(dphi),
            seeds=seeds,
            radii=rnd_radii)
    ]

    solver.fiber_bundles = fiber_bundles
    solver.boundry_checking(100)
    fiber_bundles = solver.fiber_bundles

    print("rank:" + str(comm.Get_rank()), "init:", solver.num_obj,
          solver.num_col_obj)

    # check boundry to split fiber into segments
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.25 * RADIUS_LOGMEAN,
                                          (f.shape[0], 3))
            f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])
    solver.fiber_bundles = fiber_bundles

    solver.boundry_checking(100)
    fiber_bundles = solver.fiber_bundles

    print("rank:" + str(comm.Get_rank()), "init:", solver.num_obj,
          solver.num_col_obj)

    # Save Data
    helper.save_h5_fibers(file_pref + '.init.h5', solver.fiber_bundles,
                          __file__)
    fastpli.io.fiber.save(file_pref + '.init.dat', solver.fiber_bundles)

    # Run Solver
    for i in tqdm(range(10000)):
        if solver.step():
            break

        if i % 50 == 0:
            tqdm.write("rank: {} step: {} {} {} {}%".format(
                comm.Get_rank(), i, solver.num_obj, solver.num_col_obj,
                round(solver.overlap / solver.num_col_obj * 100)))

    print("rank: {} step: {} {}".format(comm.Get_rank(), i, solver.num_obj,
                                        solver.num_col_obj))

    helper.save_h5_fibers(file_pref + '.solved.h5', solver.fiber_bundles,
                          __file__, solver.as_dict(), i, solver.num_col_obj,
                          solver.overlap)
    fastpli.io.fiber.save(file_pref + '.solved.dat', solver.fiber_bundles)
