import yep

import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.io

import numpy as np
import os, sys

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_OUTPUT = os.path.join(FILE_PATH, 'output')
os.makedirs(FILE_OUTPUT, exist_ok=True)

# yep.start(
#     os.path.join(FILE_OUTPUT,
#                  FILE_BASE + '.' + fastpli.__git__hash__ + '.prof'))

np.random.seed(42)

### create fiber_bundle(s) ###
population = fastpli.model.sandbox.seeds.triangular_grid(60, 60, 1, True)
fiber_bundle = fastpli.model.sandbox.build.cuboid(
    [0, 0, 0], [60, 60, 60], 0, np.pi / 2, population,
    np.random.uniform(0.5, 1.5, population.shape[0]))

### setup solver ###
solver = fastpli.model.solver.Solver()
solver.fiber_bundles = [fiber_bundle]
solver.drag = 0
solver.obj_min_radius = 1
solver.obj_mean_length = 2
solver.omp_num_threads = 1

### run solver ###
yep.start(
    os.path.join(FILE_OUTPUT,
                 FILE_BASE + '.' + fastpli.__git__hash__ + '.prof'))

solved = solver.step()

yep.stop()

os.system("google-pprof --callgrind " + sys.executable + " " + FILE_OUTPUT +
          '/' + FILE_BASE + '.' + fastpli.__git__hash__ + ".prof > " +
          FILE_OUTPUT + '/' + FILE_BASE + '.' + fastpli.__git__hash__ +
          ".callgrind")
