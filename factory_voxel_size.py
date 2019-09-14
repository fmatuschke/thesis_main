import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.io

import numpy as np
import h5py
from tqdm import tqdm
import os, sys

from mpi4py import MPI

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

FOLDER_NAME = 'voxel_size'

RND_LIST = np.array([0, 0.1, 0.25, 0.5, 0.75, 1.0])
RADIUS_LIST = np.array([0.5, 0.75, 1.0, 1.25, 1.5])

# relative pixel size
ORG = 64
LAP = 64
PM = 1.28
SCALE = LAP / ORG

# THICKNESS = 60.0
NUM_LAP_PIXEL = 5
LENGTH = NUM_LAP_PIXEL * LAP * np.sqrt(3)

print()
print("NUM_LAP_PIXEL", NUM_LAP_PIXEL)
print("LENGTH", LENGTH)
print()
print(RND_LIST)
print(RADIUS_LIST)
print()

np.random.seed(42)

### create fiber_bundle(s) ###
solver = fastpli.model.solver.Solver()
solver.drag = 0

solver.omp_num_threads = 4

# mpi
index = 0
comm = MPI.COMM_WORLD

for radius in tqdm(RADIUS_LIST):
    solver.obj_mean_length = 2 * radius
    solver.obj_min_radius = 4 * radius

    for rnd in tqdm(RND_LIST * radius):

        index += 1
        if index % comm.Get_size() != comm.Get_rank():
            continue

        fiber_bundle = fastpli.model.sandbox.shape.box(
            -0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            0.5 * np.array([LENGTH, LENGTH, LENGTH]), np.deg2rad(0),
            np.deg2rad(90), 1.95 * radius + rnd, radius, 2)

        ### setup solver ###
        if rnd == 0:
            solver.fiber_bundles = [fiber_bundle]
        elif rnd > 0:
            for f, fiber in enumerate(fiber_bundle):
                fiber_bundle[f][:, :3] += np.random.normal(
                    0, rnd, [fiber.shape[0], 3])
            solver.fiber_bundles = [fiber_bundle]
            solver.boundry_checking(100)
        else:
            raise ValueError("FOOBAR")

        ### run solver ###
        # solver.draw_scene()
        for i in range(100):
            if solver.step():
                break

        # print("step:", i, solver.num_obj, solver.num_col_obj)
        # solver.draw_scene()

        # save
        os.makedirs(os.path.join(FILE_PATH, 'output', FOLDER_NAME, 'models'),
                    exist_ok=True)

        file_name = 'radius_' + str(round(radius, 1)) + '_rnd_' + str(
            round(rnd, 1))
        file_name = os.path.join(FILE_PATH, 'output', FOLDER_NAME, 'models',
                                 file_name + '.h5')

        fastpli.io.fiber.save(file_name,
                              solver.fiber_bundles,
                              'fiber_bundles',
                              mode='w')

        with h5py.File(file_name, 'a') as h5f:
            with open(os.path.abspath(__file__), 'r') as f:
                h5f['script'] = f.read()

            h5f['fiber_bundles'].attrs['solver'] = str(solver.as_dict())
            h5f['fiber_bundles'].attrs['solver.steps'] = i
            h5f['fiber_bundles'].attrs[
                'solver.num_col_obj'] = solver.num_col_obj
