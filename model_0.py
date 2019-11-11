import fastpli.model.sandbox
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.model.solver

import numpy as np
import h5py
from tqdm import tqdm, trange
import os


def save_fibers(file_name, fiber_bundles, solver_dict=None, i=0, n=0):
    fastpli.io.fiber.save(file_name, fiber_bundles)
    with h5py.File(file_name, 'r+') as h5f:
        h5f['version'] = fastpli.__version__
        with open(os.path.abspath(__file__), 'r') as f:
            h5f['/'].attrs['script'] = f.read()
        if solver_dict:
            h5f['/'].attrs['solver'] = str(solver_dict)
            h5f['/'].attrs['solver.steps'] = i
            h5f['/'].attrs['solver.num_col_obj'] = n


# reproducability
np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

### 3 crossing fiber bundles ###
RADIUS_OUT = 640
RADIUS_IN = RADIUS_OUT * 0.6
FIBER_RADIUS = 1.0
FIBER_SPACING = FIBER_RADIUS * 3.0 + 1e-9
# FIBER_STEPS = 25
MODEL_NAME = "y_shape_hom"
OUTPUT_PATH = os.path.join(FILE_PATH, 'output', 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)


def next_file_name():
    import glob

    name = os.path.join(OUTPUT_PATH, MODEL_NAME)
    files = glob.glob(os.path.join(OUTPUT_PATH, MODEL_NAME + '*'))

    def in_list(i, file):
        for f in files:
            if name + ".%s" % i in f:
                return True
        return False

    i = 0
    while in_list(i, files):
        i += 1

    return name + ".%s" % i


p0 = np.array((0, 0, 0))
p1 = np.array((0, 0, 60))
p_shift = np.array([(RADIUS_OUT + RADIUS_IN) * 0.5 / np.cos(np.deg2rad(30)), 0,
                    0])

fiber_bundles = []

# fiber seeds
print("seeding")
seeds = fastpli.model.sandbox.seeds.triangular_circle(RADIUS_OUT, FIBER_SPACING)

# first
print("fiber_bundle 0")
dp = np.dot(fastpli.tools.rotation.z(np.deg2rad(0)), p_shift)
data = fastpli.model.sandbox.build.cylinder(p0 + dp + 0 / 3 * FIBER_RADIUS,
                                            p1 + dp + 0 / 3 * FIBER_RADIUS,
                                            RADIUS_IN, RADIUS_OUT, seeds,
                                            FIBER_RADIUS, np.deg2rad(180 - 30),
                                            np.deg2rad(180 + 30), 'c')

fiber_bundles.append(data)

# second
print("fiber_bundle 1")
dp = np.dot(fastpli.tools.rotation.z(np.deg2rad(120)), p_shift)
data = fastpli.model.sandbox.build.cylinder(p0 + dp + 1 / 3 * FIBER_RADIUS,
                                            p1 + dp + 1 / 3 * FIBER_RADIUS,
                                            RADIUS_IN, RADIUS_OUT, seeds,
                                            FIBER_RADIUS,
                                            np.deg2rad(180 - 30 + 120),
                                            np.deg2rad(180 + 30 + 120), 'c')
fiber_bundles.append(data)

# # third
print("fiber_bundle 2")
dp = np.dot(fastpli.tools.rotation.z(np.deg2rad(240)), p_shift)
data = fastpli.model.sandbox.build.cylinder(p0 + dp + 1 / 3 * FIBER_RADIUS,
                                            p1 + dp + 1 / 3 * FIBER_RADIUS,
                                            RADIUS_IN, RADIUS_OUT, seeds,
                                            FIBER_RADIUS,
                                            np.deg2rad(180 - 30 + 240),
                                            np.deg2rad(180 + 30 + 240), 'c')
fiber_bundles.append(data)

# add displacement
print("displacement")
for fb in fiber_bundles:
    for f in fb:
        f[:, :-1] += np.random.uniform(-FIBER_RADIUS * 0.25,
                                       FIBER_RADIUS * 0.25, (f.shape[0], 3))

file_pref = next_file_name()
print(file_pref)
save_fibers(file_pref + '.init.h5', fiber_bundles)

solver = fastpli.model.solver.Solver()
solver.omp_num_threads = 8
solver.fiber_bundles = fiber_bundles
solver.obj_mean_length = FIBER_RADIUS * 4
solver.obj_min_radius = FIBER_RADIUS * 8

solver.boundry_checking(100)

solver.draw_scene()
for i in trange(10000):
    solved = solver.step()
    if (i % 25) == 0:
        solver.draw_scene()
        tqdm.write("step %i, %i, %i" % (i, solver.num_obj, solver.num_col_obj))

    if solved:
        break

# solver.draw_scene()
print("step:", i, solver.num_obj, solver.num_col_obj)

save_fibers(file_pref + '.solved.h5', solver.fiber_bundles, solver.as_dict(), i,
            solver.num_col_obj)
