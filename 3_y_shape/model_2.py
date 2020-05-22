import fastpli.model.sandbox
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.model.solver
import fastpli_helper as helper

import time
import numpy as np
import h5py
from tqdm import tqdm, trange
import os
import random

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

# add rnd fibers insider fiber_bundle
num_f_in_fb = []

# convert f to fb

# num_fibers = 0
new_fiber_bundles = []
i = 0
for fb in fiber_bundles:
    # num_f_in_fb.append([])
    for f in fb:
        rnd_num = np.random.poisson(FB_RADIUS**2 / F_RADIUS**2 * V_FRACTION) + 1
        num_f_in_fb.append([i, rnd_num])

        rnd_radii = F_RADIUS * np.random.lognormal(0, 0.1)
        f[:, -1] = rnd_radii

        new_fiber_bundles.append([f.copy()])
        i += 1

# fiber_bundles = new_fbs
solver.fiber_bundles = new_fiber_bundles
solver.obj_mean_length = F_RADIUS * 2
solver.obj_min_radius = F_RADIUS * 5
solver.boundry_checking(10)
fiber_bundles = solver.fiber_bundles
solver.draw_scene()
# input()

# add displacement
# print("final rnd displacement and radii")
# for fb in fiber_bundles:
#     for f in fb:
#         f[:, :-1] += np.random.normal(0, 0.25 * F_RADIUS, (f.shape[0], 3))
#         f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

# solver.fiber_bundles = fiber_bundles
# solver.boundry_checking(100)
# fiber_bundles = solver.fiber_bundles
# solver.draw_scene()
# input()

file_pref = helper.version_file_name(os.path.join(OUTPUT_PATH, MODEL_NAME))
print(file_pref)
# helper.save_h5_fibers(file_pref + '.init.h5', solver.fiber_bundles, __file__)
# fastpli.io.fiber.save(file_pref + '.init.dat', solver.fiber_bundles)

# print("objs:", solver.num_obj)

# random.shuffle(num_f_in_fb)
n_f_to_pick = 0
for i in trange(10000):
    solved = solver.step()

    v = solver.overlap / (solver.num_col_obj + 1e-16)
    if (solver.num_col_obj == 0 or v < 0.01) and len(num_f_in_fb) > 0:
        pick_n = max(int(n_f_to_pick / 10), 100)
        fiber_bundles = solver.fiber_bundles
        for _ in range(pick_n):
            if len(num_f_in_fb) == 0:
                break
            index = random.randint(0, len(num_f_in_fb) - 1)
            n, rnd_num = num_f_in_fb[index]
            if rnd_num == 0:
                del num_f_in_fb[index]
                continue
            num_f_in_fb[index][1] -= 1
            f = random.choice(fiber_bundles[n])
            dv = np.random.uniform(-1, 1, 3)
            dv = dv / np.linalg.norm(dv) * np.random.uniform(
                -FB_RADIUS, FB_RADIUS)
            f[:, :-1] += dv
            f[:, :-1] += np.random.uniform(-.1 * F_RADIUS, .1 * F_RADIUS,
                                           (f.shape[0], 3))
            f[:, -1] = F_RADIUS * np.random.lognormal(0, 0.1, f.shape[0])
            fiber_bundles[n].append(f)
            solved = False

        solver.fiber_bundles = fiber_bundles
        # solver.draw_scene()

        n_f_to_pick = 0
        for elm in num_f_in_fb:
            n_f_to_pick += elm[1]
        tqdm.write("rest: {}".format(n_f_to_pick))

    fiber_bundles = solver.fiber_bundles

    for fb_i, fb in enumerate(fiber_bundles):
        if len(fb) == 1:
            continue
        solver.fiber_bundles = [fb]
        solved = solved and not solver.step()
        while (not solver.step() and
               round(solver.overlap / (solver.num_col_obj + 1e-16) * 100) > 1):
            pass
        # solver.draw_scene()
        fiber_bundles[fb_i] = solver.fiber_bundles[0]

    solver.fiber_bundles = fiber_bundles

    # if i % 100 == 0:
    solved = solved and not solver.step()
    solver.draw_scene()

    if (i % 25) == 0:
        tqdm.write("f step {}, {}, {}, {}%".format(
            i, solver.num_obj, solver.num_col_obj,
            round(solver.overlap / (solver.num_col_obj + 1e-16) * 100)))

    if solved:
        break

print("step:", i, solver.num_obj, solver.num_col_obj,
      solver.overlap / solver.num_col_obj * 100)
helper.save_h5_fibers(file_pref + '.solved.h5', solver.fiber_bundles, __file__,
                      solver.as_dict(), i, solver.num_col_obj, solver.overlap)
fastpli.io.fiber.save(file_pref + '.solved.dat', solver.fiber_bundles)
