import argparse
import multiprocessing as mp
import os
import subprocess
import time
import typing

import fastpli.io
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.tools
import h5py
import numpy as np
import tqdm


class Parameter(typing.NamedTuple):
    """  """
    radius: float
    volume: float
    psi: float
    omega: float
    max_steps: int
    num_proc: int
    output: str
    n: int


def run(p):
    # sys.stdout = open(f'{p.output}_rank_{MPI.COMM_WORLD.Get_rank()}.log',
    #                   'w+',
    #                   buffering=1)

    rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(rnd_seed)

    # setup solver
    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = p.radius * 2
    solver.obj_min_radius = p.radius * 2
    solver.omp_num_threads = 1

    file_pref = p.output + f'_repn_{p.n}_psi_{p.psi:.2f}_omega_{p.omega:.2f}_r_{p.radius:.2f}_v0_{p.volume:.0f}_'

    # pick random seeds for fiber population distribution
    seeds_0 = np.random.uniform(-p.volume, p.volume,
                                (int(p.psi * (2 * p.volume)**2 /
                                     (np.pi * p.radius**2)), 2))
    seeds_1 = np.random.uniform(-p.volume, p.volume, (int(
        (1 - p.psi) * (2 * p.volume)**2 / (np.pi * p.radius**2)), 2))

    rnd_radii_0 = p.radius * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = p.radius * np.random.lognormal(0, 0.1, seeds_1.shape[0])

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([p.volume + 10 * p.radius] * 3),
            q=0.5 * np.array([p.volume + 10 * p.radius] * 3),
            phi=np.deg2rad(0),
            theta=np.deg2rad(90),
            seeds=seeds_0,
            radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([p.volume + 10 * p.radius] * 3),
            q=0.5 * np.array([p.volume + 10 * p.radius] * 3),
            phi=np.deg2rad(p.omega),
            theta=np.deg2rad(90),
            seeds=seeds_1 + p.radius,
            radii=rnd_radii_1)
    ]

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)

    # add rnd displacement
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.05 * p.radius, (f.shape[0], 3))
            f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

    solver.fiber_bundles = fiber_bundles
    fiber_bundles = solver.apply_boundary_conditions(100)

    # Save Data
    with h5py.File(file_pref + '.init.h5', 'w-') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['rep_n'] = p.n
        h5f['/'].attrs['psi'] = p.psi
        h5f['/'].attrs['omega'] = p.omega
        h5f['/'].attrs['v0'] = p.volume
        h5f['/'].attrs['radius'] = p.radius
        h5f['/'].attrs['step'] = 0
        h5f['/'].attrs['overlap'] = solver.overlap
        h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        h5f['/'].attrs['num_steps'] = 0
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
        h5f['/'].attrs['time'] = 0
        h5f['/'].attrs['rnd_seed'] = rnd_seed

    # Run Solver
    start_time = time.time()
    solver.fiber_bundles = solver.fiber_bundles.cut_sphere(
        0.5 * (p.volume + 10 * p.radius))
    for i in tqdm.trange(1, p.max_steps + 1):
        if solver.step():
            break

        if i % 50 == 0:
            if i % 250 == 0:
                with h5py.File(f'{file_pref}.tmp.h5', 'w') as h5f:
                    solver.save_h5(h5f,
                                   script=open(os.path.abspath(__file__),
                                               'r').read())
                    h5f['/'].attrs['rep_n'] = p.n
                    h5f['/'].attrs['psi'] = p.psi
                    h5f['/'].attrs['omega'] = p.omega
                    h5f['/'].attrs['v0'] = p.volume
                    h5f['/'].attrs['radius'] = p.radius
                    h5f['/'].attrs['step'] = i
                    h5f['/'].attrs['overlap'] = solver.overlap
                    h5f['/'].attrs['num_obj'] = solver.num_obj
                    h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
                    h5f['/'].attrs['num_steps'] = solver.num_steps
                    h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
                    h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
                    h5f['/'].attrs['time'] = time.time() - start_time
                    h5f['/'].attrs['rnd_seed'] = rnd_seed

            if i != p.max_steps:
                solver.fiber_bundles = solver.fiber_bundles.cut_sphere(
                    0.5 * (p.volume + 10 * p.radius))

    with h5py.File(file_pref + '.solved.h5', 'w-') as h5f:
        solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
        h5f['/'].attrs['rep_n'] = p.n
        h5f['/'].attrs['psi'] = p.psi
        h5f['/'].attrs['omega'] = p.omega
        h5f['/'].attrs['v0'] = p.volume
        h5f['/'].attrs['radius'] = p.radius
        h5f['/'].attrs['step'] = i
        h5f['/'].attrs['overlap'] = solver.overlap
        h5f['/'].attrs['num_obj'] = solver.num_obj
        h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
        h5f['/'].attrs['num_steps'] = solver.num_steps
        h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
        h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
        h5f['/'].attrs['time'] = time.time() - start_time
        h5f['/'].attrs['rnd_seed'] = rnd_seed

    if os.path.isfile(f'{file_pref}.tmp.h5'):
        os.remove(f'{file_pref}.tmp.h5')


def main():
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

    parser.add_argument('-n',
                        '--max_steps',
                        type=int,
                        required=True,
                        help='Number of max_steps.')

    parser.add_argument('-r',
                        '--radius',
                        required=True,
                        type=float,
                        help='mean value of fiber radius')

    parser.add_argument('-v',
                        '--volume',
                        required=True,
                        type=int,
                        help='volume size')

    parser.add_argument("-p",
                        "--num_proc",
                        type=int,
                        required=True,
                        help="Number of threads.")
    parser.add_argument("-N",
                        "--num_repo",
                        default=1,
                        type=int,
                        help="Number of processes.")

    args = parser.parse_args()
    output_name = os.path.join(args.output, FILE_NAME)
    os.makedirs(args.output, exist_ok=True)
    subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
    subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)

    psi_omega_list = []
    psi_omega_list.extend([(1.0, 0.0)] * args.num_repo)
    psi_omega_list.extend([(0.5, 90.0)] * args.num_repo)

    parameters = [
        Parameter(radius=args.radius,
                  volume=args.volume,
                  psi=psi,
                  omega=omega,
                  max_steps=args.max_steps,
                  num_proc=args.num_proc,
                  output=output_name,
                  n=i % args.num_repo)
        for i, (psi, omega) in enumerate(psi_omega_list)
    ]

    with mp.Pool(args.num_proc) as pool:
        [_ for _ in pool.imap_unordered(run, parameters)]


if __name__ == '__main__':
    main()
