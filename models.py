import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.io

import numpy as np
import h5py
import os

from tqdm import tqdm
from mpi4py import MPI
from numba import njit

# reproducability
np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

# PARAMETERS
RND_LIST = np.array([0, 0.05, 0.10, 0.20]) * 1
RADIUS_LIST = np.array([0.5, 1.0, 1.5]) * 1
PARAMETERS = []
for radius in RADIUS_LIST:
    for rnd in RND_LIST * radius:
        PARAMETERS.append((radius, rnd))

# SIZES
ORG = 64
LAP = 64
PM = 1.28
NUM_LAP_PIXEL = 5
LENGTH = NUM_LAP_PIXEL * LAP * np.sqrt(3)

os.makedirs(os.path.join(FILE_PATH, 'output', 'models'), exist_ok=True)

### create fiber_bundle(s) ###
solver = fastpli.model.solver.Solver()
solver.drag = 0
solver.omp_num_threads = 8

# mpi
index = 0
comm = MPI.COMM_WORLD

radius_old = None
for radius, rnd in PARAMETERS[comm.Get_rank()::comm.Get_size()]:
    print("rank:" + str(comm.Get_rank()), "parameter:", round(radius, 2),
          round(rnd, 2))

    solver.obj_mean_length = 4 * radius
    solver.obj_min_radius = 8 * radius

    solver.fiber_bundles = [
        fastpli.model.sandbox.shape.box(
            -0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            0.5 * np.array([LENGTH, LENGTH, LENGTH]), np.deg2rad(0),
            np.deg2rad(90), 2 * radius + 1e-9, radius, 2)
    ]

    print("rank:" + str(comm.Get_rank()), "init:", solver.num_obj,
          solver.num_col_obj)

    i = 0
    if rnd != 0:
        solver.boundry_checking(100)

        # check boundry to split fiber into segments
        fiber_bundle = solver.fiber_bundles[0]
        for f, fiber in enumerate(fiber_bundle):
            fiber_bundle[f][:, :3] += np.random.normal(0, rnd,
                                                       [fiber.shape[0], 3])
        solver.fiber_bundles = [fiber_bundle]
        del fiber_bundle

        ### run solver ###
        # solver.draw_scene()
        for i in range(100):
            print("rank:" + str(comm.Get_rank()), "step:", i, solver.num_obj,
                  solver.num_col_obj)
            # solver.draw_scene()
            if solver.step():
                break

    ### save data ###
    file_name = 'cube_hom_radius_' + str(round(radius, 2)) + '_rnd_' + str(
        round(rnd, 2))
    file_name = os.path.join(FILE_PATH, 'output', 'models', file_name + '.h5')

    fastpli.io.fiber.save(file_name,
                          solver.fiber_bundles,
                          'fiber_bundles',
                          mode='w-')

    with h5py.File(file_name, 'a') as h5f:
        with open(os.path.abspath(__file__), 'r') as f:
            h5f['script'] = f.read()

        h5f['fiber_bundles'].attrs['solver'] = str(solver.as_dict())
        h5f['fiber_bundles'].attrs['solver.steps'] = i
        h5f['fiber_bundles'].attrs['solver.num_col_obj'] = solver.num_col_obj
