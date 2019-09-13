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

PHI_LIST = np.array([0, 45])
THETA_LIST = np.array([0, 30, 60, 70, 80, 85, 90])
RND_LIST = np.array([0, 0.2, 0.5])

# relative pixel size
ORG = 64
LAP = 64
PM = 1.33
SCALE = LAP / ORG

THICKNESS = 60.0 * SCALE
NUM_LAP_PIXEL = 5
LENGTH = NUM_LAP_PIXEL * LAP
FIBER_RADIUS = 0.5 * SCALE

print()
print("THICKNESS", THICKNESS)
print("NUM_LAP_PIXEL", NUM_LAP_PIXEL)
print("LENGTH", LENGTH)
print("FIBER_RADIUS", FIBER_RADIUS)
print()
print(PHI_LIST)
print(THETA_LIST)
print(RND_LIST)
print()

np.random.seed(42)

### create fiber_bundle(s) ###
solver = fastpli.model.solver.Solver()
solver.drag = 0
solver.obj_mean_length = 2 * FIBER_RADIUS
solver.obj_min_radius = 4 * FIBER_RADIUS
solver.omp_num_threads = 8

# mpi
index = 0
comm = MPI.COMM_WORLD

for phi in tqdm(PHI_LIST):
    for theta in tqdm(THETA_LIST):
        index += 1
        if index % comm.Get_size() != comm.Get_rank():
            continue

        for rnd in tqdm(RND_LIST * FIBER_RADIUS):

            # print(phi, theta, round(rnd, 2))

            fiber_bundle = fastpli.model.sandbox.shape.box(
                [0, 0, 0], [LENGTH, LENGTH, THICKNESS], np.deg2rad(phi),
                np.deg2rad(theta), 2 * FIBER_RADIUS + 1e-9 + rnd, FIBER_RADIUS,
                2)

            ### setup solver ###
            solver.fiber_bundles = [fiber_bundle]
            if rnd > 0:
                solver.boundry_checking(100)
                fiber_bundle = solver.fiber_bundles[0]
                for f, fiber in enumerate(fiber_bundle):
                    fiber_bundle[f][:, :3] += np.random.normal(
                        0, rnd, [fiber.shape[0], 3])
                solver.fiber_bundles = [fiber_bundle]

            ### run solver ###
            # solver.draw_scene()
            for i in range(1000):
                if solver.step():
                    break

                # if i % 10 == 0:
                #     print("step:", i, solver.num_obj, solver.num_col_obj)
                #     solver.draw_scene()

            # print("step:", i, solver.num_obj, solver.num_col_obj)

            # save
            file_name = 'phi_' + str(round(phi, 0)) + '_theta_' + str(
                round(theta, 0)) + '_rnd_' + str(round(rnd, 1))
            file_name = os.path.join(FILE_PATH,
                                     'output/models/model_' + file_name + '.h5')

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
