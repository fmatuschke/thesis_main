import fastpli.model.sandbox
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.model.solver

import time
import numpy as np
import h5py
from tqdm import tqdm, trange
import os


def get_pip_freeze():
    try:
        from pip._internal.operations import freeze
    except ImportError:
        from pip.operations import freeze
    return "\n".join(freeze.freeze())


def save_fibers(file_name, fiber_bundles, solver_dict=None, i=0, n=0):
    fastpli.io.fiber.save(file_name, fiber_bundles, '/fiber_bundles', 'w-')
    with h5py.File(file_name, 'r+') as h5f:
        h5f['/fiber_bundles'].attrs['version'] = fastpli.__version__
        h5f['/fiber_bundles'].attrs['pip_freeze'] = get_pip_freeze()
        with open(os.path.abspath(__file__), 'r') as f:
            h5f['/fiber_bundles'].attrs['script'] = f.read()
        if solver_dict:
            h5f['/fiber_bundles'].attrs['solver'] = str(solver_dict)
            h5f['/fiber_bundles'].attrs['solver.steps'] = i
            h5f['/fiber_bundles'].attrs['solver.num_col_obj'] = n


# reproducability
np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

MODEL_NAME = "y_shape_fb"
OUTPUT_PATH = os.path.join(FILE_PATH, 'output', 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)

### 3 crossing fiber bundles ###
RADIUS_OUT = 640
RADIUS_IN = RADIUS_OUT * 0.6
F_RADIUS = 1.0
FB_RADIUS = 10.0
FB_SPACING = FB_RADIUS * 2
FB_STEPS = 25
p0 = np.array((0, 0, 0))
p1 = np.array((0, 0, 60))
p_shift = np.array([(RADIUS_OUT + RADIUS_IN) * 0.5 / np.cos(np.deg2rad(30)), 0,
                    0])
V_FRACTION = 0.5

solver = fastpli.model.solver.Solver()
solver.omp_num_threads = 8


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


# fiber seeds
print("seeding")
seeds = fastpli.model.sandbox.seeds.triangular_circle(RADIUS_OUT, FB_SPACING)
fiber_bundles = []

# first
print("fiber_bundle 0")
dp = np.dot(fastpli.tools.rotation.z(np.deg2rad(0)), p_shift)
data = fastpli.model.sandbox.build.cylinder(p0 + dp + 0 / 3 * FB_RADIUS,
                                            p1 + dp + 0 / 3 * FB_RADIUS,
                                            RADIUS_IN,
                                            RADIUS_OUT,
                                            seeds,
                                            FB_RADIUS,
                                            np.deg2rad(180 - 30),
                                            np.deg2rad(180 + 30),
                                            'c',
                                            steps=FB_STEPS)

fiber_bundles.append(data)

# second
print("fiber_bundle 1")
dp = np.dot(fastpli.tools.rotation.z(np.deg2rad(120)), p_shift)
data = fastpli.model.sandbox.build.cylinder(p0 + dp + 1 / 3 * FB_RADIUS,
                                            p1 + dp + 1 / 3 * FB_RADIUS,
                                            RADIUS_IN,
                                            RADIUS_OUT,
                                            seeds,
                                            FB_RADIUS,
                                            np.deg2rad(180 - 30 + 120),
                                            np.deg2rad(180 + 30 + 120),
                                            'c',
                                            steps=FB_STEPS)
fiber_bundles.append(data)

# third
print("fiber_bundle 2")
dp = np.dot(fastpli.tools.rotation.z(np.deg2rad(240)), p_shift)
data = fastpli.model.sandbox.build.cylinder(p0 + dp + 1 / 3 * FB_RADIUS,
                                            p1 + dp + 1 / 3 * FB_RADIUS,
                                            RADIUS_IN,
                                            RADIUS_OUT,
                                            seeds,
                                            FB_RADIUS,
                                            np.deg2rad(180 - 30 + 240),
                                            np.deg2rad(180 + 30 + 240),
                                            'c',
                                            steps=FB_STEPS)
fiber_bundles.append(data)

solver.fiber_bundles = fiber_bundles
solver.obj_mean_length = FB_RADIUS * 4
solver.obj_min_radius = FB_RADIUS * 8
solver.boundry_checking(10)
fiber_bundles = solver.fiber_bundles
solver.draw_scene()
# input()

# add displacement
print("displacement")
for fb in fiber_bundles:
    for f in fb:
        f[:, :-1] += np.random.normal(0, 0.25 * FB_RADIUS, (f.shape[0], 3))

solver.fiber_bundles = fiber_bundles
solver.obj_mean_length = FB_RADIUS * 2
solver.obj_min_radius = FB_RADIUS * 5
solver.boundry_checking(10)
fiber_bundles = solver.fiber_bundles
solver.draw_scene()
# input()

# solve fiber bundles
print("FB solving")
for i in trange(25):
    solved = solver.step()
    if (i % 25) == 0:
        solver.draw_scene()
        tqdm.write("fb step %i, %i, %i" %
                   (i, solver.num_obj, solver.num_col_obj))

    if solved:
        break

# add rnd fibers insider fiber_bundle
new_fbs = []
num_fibers = 0
for fb in fiber_bundles:
    for f in fb:
        rnd_num = np.random.poisson(FB_RADIUS**2 / F_RADIUS**2 * V_FRACTION)

        # unform rnd seed points inside circle
        rnd_phi = np.random.uniform(0, 2 * np.pi, rnd_num)
        rnd_r = FB_RADIUS * np.sqrt(np.random.uniform(0, 1, rnd_num))
        rnd_seeds = np.transpose(
            np.array([rnd_r * np.cos(rnd_phi), rnd_r * np.sin(rnd_phi)]))

        rnd_radii = F_RADIUS * np.random.lognormal(0, 0.1, rnd_num)

        new_fbs.append(
            fastpli.model.sandbox.build.bundle(f[:, :-1], rnd_seeds, rnd_radii))

        num_fibers += len(new_fbs[-1])

print("Num Fibers: ", num_fibers)

fiber_bundles = new_fbs
solver.fiber_bundles = fiber_bundles
solver.obj_mean_length = F_RADIUS * 2
solver.obj_min_radius = F_RADIUS * 5
solver.boundry_checking(100)
fiber_bundles = solver.fiber_bundles
solver.draw_scene()
# input()

# add displacement
print("final rnd displacement and radii")
for fb in fiber_bundles:
    for f in fb:
        f[:, :-1] += np.random.normal(0, 0.25 * F_RADIUS, (f.shape[0], 3))
        f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

solver.fiber_bundles = fiber_bundles
solver.boundry_checking(100)
fiber_bundles = solver.fiber_bundles
solver.draw_scene()
# input()

file_pref = next_file_name()
print(file_pref)
save_fibers(file_pref + '.init.h5', solver.fiber_bundles)
fastpli.io.fiber.save(file_pref + '.init.dat', solver.fiber_bundles)

print("objs:", solver.num_obj)
for i in trange(10000):
    solved = solver.step()
    if (i % 25) == 0:
        solver.draw_scene()
        tqdm.write("f step %i, %i, %i" %
                   (i, solver.num_obj, solver.num_col_obj))
    if (i % 1000) == 0:
        save_fibers(file_pref + '.step.' + str(i) + '.h5', solver.fiber_bundles,
                    solver.as_dict(), i, solver.num_col_obj)
        fastpli.io.fiber.save(file_pref + '.step.' + str(i) + '.dat',
                              solver.fiber_bundles)

    if solved:
        break

print("step:", i, solver.num_obj, solver.num_col_obj)

save_fibers(file_pref + '.solved.h5', solver.fiber_bundles, solver.as_dict(), i,
            solver.num_col_obj)
fastpli.io.fiber.save(file_pref + '.solved.dat', solver.fiber_bundles)
