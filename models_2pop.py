import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.io

import numpy as np
import h5py
import os

from tqdm import tqdm
from mpi4py import MPI


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

MODEL_NAME = "cube_2pop"
OUTPUT_PATH = os.path.join(FILE_PATH, 'output', 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# PARAMETERS
DPHI = np.linspace(0, 90, 5, True)
RADIUS_LOGMEAN = 1
PARAMETERS = DPHI

ORG = 64
LAP = 64
PM = 1.28
NUM_LAP_PIXEL = 1
LENGTH = NUM_LAP_PIXEL * LAP * np.sqrt(3)


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

    # solver.draw_scene()
    # input()

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

    # solver.draw_scene()
    # input()

    # Save Data
    file_pref = next_file_name() + 'dphi_' + str(dphi)
    print(file_pref)
    save_fibers(file_pref + '.init.h5', solver.fiber_bundles)
    fastpli.io.fiber.save(file_pref + '.init.dat', solver.fiber_bundles)

    # Run Solver
    for i in tqdm(range(1000)):
        if i % 25 == 0:
            tqdm.write("rank: {} step: {} {} {}".format(comm.Get_rank(), i,
                                                        solver.num_obj,
                                                        solver.num_col_obj))
        if solver.step():
            break
    print("rank:" + str(comm.Get_rank()), "step:", i, solver.num_obj,
          solver.num_col_obj)

    save_fibers(file_pref + '.solved.h5', solver.fiber_bundles,
                solver.as_dict(), i, solver.num_col_obj)
    fastpli.io.fiber.save(file_pref + '.solved.dat', solver.fiber_bundles)
