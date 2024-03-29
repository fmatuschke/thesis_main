#!/usr/bin/env python

import argparse
import itertools
import multiprocessing as mp
import os
import subprocess
import time

import fastpli.io
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.tools
import h5py
import numpy as np
import tqdm

# reproducability -> see run()
# np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o',
                    '--output',
                    type=str,
                    required=True,
                    help='Output path of solver.')

parser.add_argument('-p',
                    '--num_proc',
                    type=int,
                    required=True,
                    help='Number of threads.')

args = parser.parse_args()
output_name = os.path.join(args.output, FILE_NAME)
os.makedirs(args.output, exist_ok=True)
subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)


def run(parameters):
    rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(rnd_seed)

    radius, mean_length_f, min_radius_f, (psi, omega), n = parameters

    file_pref = output_name + f'_psi_{psi:.2f}_omega_{omega:.2f}_r_' \
                               f'{radius:.2f}_v0_{SIZE:.0f}_fl_{mean_length_f:.2f}_' \
                               f'fr_{min_radius_f:.2f}_n_{n}_'

    if os.path.isfile(file_pref + '.solved.h5'):
        return

    # logger.info(f'file_pref: {file_pref}')

    # setup solver
    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = radius * mean_length_f
    solver.obj_min_radius = radius * min_radius_f
    solver.omp_num_threads = N_OMP_THREADS

    # logger.info(f'seed')
    seeds = fastpli.model.sandbox.seeds.triangular_grid(SIZE * 2,
                                                        SIZE * 2,
                                                        2 * radius,
                                                        center=True)

    # pick random seeds for fiber population distribution
    seeds_0 = seeds[np.random.rand(seeds.shape[0]) < psi, :]
    seeds_1 = seeds[np.random.rand(seeds.shape[0]) < (1 - psi), :]
    rnd_radii_0 = radius * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = radius * np.random.lognormal(0, 0.1, seeds_1.shape[0])

    # logger.info(f'creating fiber_bundles')
    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(p=-0.5 *
                                           np.array([SIZE, SIZE, SIZE]),
                                           q=0.5 * np.array([SIZE, SIZE, SIZE]),
                                           phi=np.deg2rad(0),
                                           theta=np.deg2rad(90),
                                           seeds=seeds_0,
                                           radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(p=-0.5 *
                                           np.array([SIZE, SIZE, SIZE]),
                                           q=0.5 * np.array([SIZE, SIZE, SIZE]),
                                           phi=np.deg2rad(omega),
                                           theta=np.deg2rad(90),
                                           seeds=seeds_1 + radius,
                                           radii=rnd_radii_1)
    ]

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)
    # logger.info(f'init: {solver.num_obj}/{solver.num_col_obj}')

    # add rnd displacement
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.05 * radius, (f.shape[0], 3))
            f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)
    # logger.info(f'rnd displacement: {solver.num_obj}/{solver.num_col_obj}')

    # if comm.Get_size() == 1:
    #     solver.draw_scene()

    # Save Data
    # logger.debug(f'save init')
    with h5py.File(file_pref + '.init.h5', 'w') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['psi'] = psi
        h5f['/'].attrs['omega'] = omega
        # print('OTHER META DATA?')
        # h5f['/'].attrs['overlap'] = overlap
        # h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        # h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_steps'] = 0
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
        h5f['/'].attrs['rnd_seed'] = rnd_seed

    # Run Solver
    # logger.info(f'run solver')
    start_time = time.time()

    times = []
    steps = []
    overlaps = []
    num_objs = []
    num_col_objs = []
    for i in tqdm.trange(1, N_MAX_STEPS + 1, leave=False, position=1, ncols=80):
        if solver.step():
            break

        if i % 50 == 0:
            # logger.info(
            #     f'step: {i}, {solver.num_obj}/{solver.num_col_obj}, {solver.overlap}'
            # )

            times.append(time.time() - start_time)
            steps.append(i)
            overlaps.append(solver.overlap)
            num_objs.append(solver.num_obj)
            num_col_objs.append(solver.num_col_obj)

            if i % 250 == 0 and (times[-1] - times[-2]) > 300:
                with h5py.File(f'{file_pref}.tmp.h5', 'w') as h5f:
                    solver.save_h5(h5f,
                                   script=open(os.path.abspath(__file__),
                                               'r').read())
                    h5f['/'].attrs['psi'] = psi
                    h5f['/'].attrs['omega'] = omega
                    h5f['/'].attrs['overlap'] = solver.overlap
                    h5f['/'].attrs['step'] = i
                    h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
                    h5f['/'].attrs['num_obj'] = solver.num_obj
                    h5f['/'].attrs['num_steps'] = solver.num_steps
                    h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
                    h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
                    h5f['/'].attrs['time'] = time.time() - start_time

                    h5f['/'].attrs['times'] = np.array(times)
                    h5f['/'].attrs['steps'] = np.array(steps)
                    h5f['/'].attrs['overlaps'] = np.array(overlaps)
                    h5f['/'].attrs['num_objs'] = np.array(num_objs)
                    h5f['/'].attrs['num_col_objs'] = np.array(num_col_objs)
                    h5f['/'].attrs['rnd_seed'] = rnd_seed

            if i != N_MAX_STEPS:
                solver.fiber_bundles = solver.fiber_bundles.cut([
                    -0.5 * np.array([SIZE + 2 * radius] * 3),
                    0.5 * np.array([SIZE + 2 * radius] * 3)
                ])

    end_time = time.time()

    times.append(end_time - start_time)
    steps.append(i)
    overlaps.append(solver.overlap)
    num_objs.append(solver.num_obj)
    num_col_objs.append(solver.num_col_obj)

    # logger.info(f'time: {end_time - start_time}')
    # logger.info(
    # f'solved: {i}, {solver.num_obj}/{solver.num_col_obj}, {solver.overlap}')
    # logger.info(f'saveing solved')
    with h5py.File(file_pref + '.solved.h5', 'w-') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['psi'] = psi
        h5f['/'].attrs['omega'] = omega
        h5f['/'].attrs['overlap'] = solver.overlap
        h5f['/'].attrs['step'] = i
        h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_steps'] = solver.num_steps
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
        h5f['/'].attrs['time'] = end_time - start_time

        h5f['/'].attrs['times'] = times
        h5f['/'].attrs['steps'] = steps
        h5f['/'].attrs['overlaps'] = overlaps
        h5f['/'].attrs['num_objs'] = num_objs
        h5f['/'].attrs['num_col_objs'] = num_col_objs
        h5f['/'].attrs['rnd_seed'] = rnd_seed

    if os.path.isfile(f'{file_pref}.tmp.h5'):
        os.remove(f'{file_pref}.tmp.h5')

    # logger.info(f'done')


def check_file(p):
    radius, mean_length_f, min_radius_f, (psi, omega), n = p
    file_pref = output_name + f'_psi_{psi:.2f}_omega_{omega:.2f}_r_' \
                            f'{radius:.2f}_v0_{SIZE:.0f}_fl_{mean_length_f:.2f}_' \
                            f'fr_{min_radius_f:.2f}_n_{n}_'
    return not os.path.isfile(file_pref + '.solved.h5')


if __name__ == '__main__':
    # logger.info('args: ' + ' '.join(sys.argv[1:]))
    # logger.info(
    # f'git: {subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()}')
    # logger.info('script:\n' + open(os.path.abspath(__file__), 'r').read())

    # Fiber Model
    N_OMP_THREADS = 1
    N_MAX_STEPS = 100000
    N_REPEAT = range(24)
    SIZE = 60  # <- for pre stats ok, no simulation required
    FIBER_RADII = [0.5, 1.0, 2.0, 5.0, 10]
    OBJ_MEAN_LENGTH_F = [1.0, 2.0, 4.0, 8.0]
    OBJ_MIN_RADIUS_F = [1.0, 2.0, 4.0, 8.0]
    PARAMETER = []
    PARAMETER.append((1.0, 0))
    PARAMETER.append((0.5, 90))

    FIBER_RADII.reverse()  # fastest first
    with mp.Pool(args.num_proc) as pool:
        for radii in FIBER_RADII:
            parameters = list(
                itertools.product([radii], OBJ_MEAN_LENGTH_F, OBJ_MIN_RADIUS_F,
                                  PARAMETER, N_REPEAT))

            for _ in tqdm.tqdm(pool.imap_unordered(run, parameters),
                               total=len(parameters),
                               position=0,
                               ncols=80,
                               desc=f'radius: {radii}'):
                pass
