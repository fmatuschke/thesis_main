import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.tools
import fastpli.io

import numpy as np
import h5py
import sys
import os

from tqdm import tqdm
from mpi4py import MPI

# reproducability
np.random.seed(42)

# GLOBAL PARAMETERS
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

MODEL_NAME = "cube_2pop"

OUTPUT_PATH = os.path.join(FILE_PATH, 'models')
if len(sys.argv) > 1:
    OUTPUT_PATH = os.path.join(sys.argv[1], 'models')
else:
    raise ValueError("No input")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# FIBERS
LENGTH = 120
RADIUS_LOGMEAN = 1
DPHI = np.linspace(0, 90, 10, True)
PSI = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PDPHI, PPSI = np.meshgrid(DPHI, PSI)
PARAMETER = list(zip(PDPHI.flatten(), PPSI.flatten()))
PARAMETER.append((0.0, 0.0))
PARAMETER.append((0.0, 1.0))

# mpi
comm = MPI.COMM_WORLD

print(PDPHI.size)

for dphi, psi in PARAMETER[comm.Get_rank()::comm.Get_size()]:

    print(dphi, psi)

    # setup solver
    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = RADIUS_LOGMEAN * 2
    solver.obj_min_radius = RADIUS_LOGMEAN * 5
    solver.omp_num_threads = 4

    file_pref = fastpli.tools.helper.version_file_name(
        os.path.join(OUTPUT_PATH, MODEL_NAME + '_dphi_' + str(round(dphi, 1))) +
        '_psi_' + str(round(psi, 1)))
    print("rank {}: {}".format(str(comm.Get_rank()), file_pref))

    seeds = fastpli.model.sandbox.seeds.triangular_grid(LENGTH * 2,
                                                        LENGTH * 2,
                                                        2 * RADIUS_LOGMEAN,
                                                        center=True)

    # print(np.mean(rnd_radii), np.std(rnd_radii))

    # pick random seeds for volume distribution
    seeds_0 = seeds[np.random.rand(seeds.shape[0]) < psi, :]
    seeds_1 = seeds[np.random.rand(seeds.shape[0]) < (1 - psi), :]
    rnd_radii_0 = RADIUS_LOGMEAN * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = RADIUS_LOGMEAN * np.random.lognormal(0, 0.1, seeds_1.shape[0])

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            q=0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            phi=np.deg2rad(0),
            theta=np.deg2rad(90),
            seeds=seeds_0,
            radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            q=0.5 * np.array([LENGTH, LENGTH, LENGTH]),
            phi=np.deg2rad(dphi),
            theta=np.deg2rad(90),
            seeds=seeds_1 + RADIUS_LOGMEAN,
            radii=rnd_radii_1)
    ]

    fiber_bundles = solver.boundry_check(fiber_bundles, 100)

    print("rank:" + str(comm.Get_rank()), "init:", solver.num_obj,
          solver.num_col_obj)

    # check boundry to split fiber into segments
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.05 * RADIUS_LOGMEAN,
                                          (f.shape[0], 3))
            f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

    fiber_bundles = solver.boundry_check(fiber_bundles, 100)

    # if comm.Get_rank() == 0:
    #     solver.draw_scene()

    print("rank:" + str(comm.Get_rank()), "init:", solver.num_obj,
          solver.num_col_obj)

    # Save Data
    fastpli.tools.helper.save_h5_fibers(file_pref + '.init.h5',
                                        solver.fiber_bundles, 'fiber_bundles',
                                        __file__)
    fastpli.io.fiber.save(file_pref + '.init.dat', solver.fiber_bundles)

    # Run Solver
    for i in tqdm(range(10000)):
        if solver.step():
            break

        overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0
        if i % 50 == 0:
            tqdm.write("rank: {} step: {} {} {} {}%".format(
                comm.Get_rank(), i, solver.num_obj, solver.num_col_obj,
                round(overlap * 100)))

        if overlap <= 0.01:
            break

    print("rank: {} step: {} {}".format(comm.Get_rank(), i, solver.num_obj,
                                        solver.num_col_obj))

    fastpli.tools.helper.save_h5_fibers(file_pref + '.solved.h5',
                                        solver.fiber_bundles,
                                        'fiber_bundles', __file__,
                                        solver.as_dict(), i, solver.num_col_obj,
                                        solver.overlap)
    fastpli.io.fiber.save(file_pref + '.solved.dat', solver.fiber_bundles)
