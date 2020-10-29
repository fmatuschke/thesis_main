# /data/PLI-Group/Marius/Master/Nicole/tractogramsMSA/tractogram_msa50.h5

import multiprocessing as mp
import numpy as np
import h5py
import tqdm
import os

import fastpli.model.solver
import fastpli.objects
import fastpli.io

from load_fbs_plivis import read

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_OUT = os.path.join(FILE_PATH, f'fastpli.example.{FILE_BASE}')

fbs = read(
    "/data/PLI-Group/Marius/Master/Nicole/tractogramsMSA/tractogram_msa50.h5")

solver = fastpli.model.solver.Solver()
fbs = fastpli.objects.fiber_bundles.Rescale(fbs, 60, 'points')
fbs = fastpli.objects.fiber_bundles.Rescale(fbs, 30, 'radii')
solver.fiber_bundles = fbs
solver.obj_min_radius = 8 * 60
solver.obj_mean_length = 8 * 60
solver.omp_num_threads = 4

# run solver
solver.apply_boundary_conditions(100)
# solver.toggle_axis()
# solver.draw_scene()

for i in tqdm.tqdm(range(1000)):
    solved = solver.step()

    if i % 5 == 0:
        tqdm.tqdm.write(
            f'step: {i}, {solver.num_obj}/{solver.num_col_obj} {solver.overlap/solver.num_obj}'
        )
        # solver.draw_scene()
        # solver.save_ppm(f'solver_{i:03}.ppm')  # save a ppm image

    if i % 100 == 0:
        fastpli.io.fiber_bundles.save(
            f'output/model_{solver.obj_min_radius}_{solver.obj_mean_length}.tmp.dat',
            solver.fiber_bundles,
            mode='w')

    if solved:
        print(f'solved: {i}, {solver.num_obj}/{solver.num_col_obj}')
        break

fastpli.io.fiber_bundles.save(
    f'output/model_{solver.obj_min_radius}_{solver.obj_mean_length}.dat',
    solver.fiber_bundles)

print('Done')
