import argparse
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
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


class Parameter(typing.NamedTuple):
    """  """
    radius: float
    volume: float
    psi_0: float
    psi_1: float
    psi_2: float
    phi_0: float
    phi_1: float
    phi_2: float
    theta_0: float
    theta_1: float
    theta_2: float
    max_steps: int
    num_proc: int
    output: str


def run(p):
    rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(rnd_seed)

    # setup solver
    solver = fastpli.model.solver.Solver()
    solver.obj_mean_length = p.radius * 2
    solver.obj_min_radius = p.radius * 2
    solver.omp_num_threads = p.num_proc

    file_pref = p.output + f'_r_{p.radius:.2f}' + f'_v0_{p.volume:.0f}' + f'_psi0_{p.psi_0:.2f}' + f'_psi1_{p.psi_1:.2f}' + f'_psi2_{p.psi_2:.2f}' + f'_phi0_{p.phi_0:.2f}' + f'_phi1_{p.phi_1:.2f}' + f'_phi2_{p.phi_2:.2f}' + f'_theta0_{p.theta_0:.2f}' + f'_theta1_{p.theta_1:.2f}' + f'_theta2_{p.theta_2:.2f}'

    # pick random seeds for fiber population distribution
    seeds_0 = np.random.uniform(-p.volume, p.volume,
                                (int(p.psi_0 * (2 * p.volume)**2 /
                                     (np.pi * p.radius**2)), 2))
    seeds_1 = np.random.uniform(-p.volume, p.volume,
                                (int(p.psi_1 * (2 * p.volume)**2 /
                                     (np.pi * p.radius**2)), 2))
    seeds_2 = np.random.uniform(-p.volume, p.volume,
                                (int(p.psi_2 * (2 * p.volume)**2 /
                                     (np.pi * p.radius**2)), 2))

    rnd_radii_0 = p.radius * np.random.lognormal(0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = p.radius * np.random.lognormal(0, 0.1, seeds_1.shape[0])
    rnd_radii_2 = p.radius * np.random.lognormal(0, 0.1, seeds_2.shape[0])

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([p.volume + 10 * p.radius] * 3),
            q=0.5 * np.array([p.volume + 10 * p.radius] * 3),
            phi=np.deg2rad(p.phi_0),
            theta=np.deg2rad(p.theta_0),
            seeds=seeds_0,
            radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([p.volume + 10 * p.radius] * 3),
            q=0.5 * np.array([p.volume + 10 * p.radius] * 3),
            phi=np.deg2rad(p.phi_1),
            theta=np.deg2rad(p.theta_1),
            seeds=seeds_1,
            radii=rnd_radii_1),
        fastpli.model.sandbox.build.cuboid(
            p=-0.5 * np.array([p.volume + 10 * p.radius] * 3),
            q=0.5 * np.array([p.volume + 10 * p.radius] * 3),
            phi=np.deg2rad(p.phi_2),
            theta=np.deg2rad(p.theta_2),
            seeds=seeds_2,
            radii=rnd_radii_2)
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
        h5f['/'].attrs['psi_0'] = p.psi_0
        h5f['/'].attrs['psi_1'] = p.psi_1
        h5f['/'].attrs['psi_2'] = p.psi_2
        h5f['/'].attrs['phi_0'] = p.phi_0
        h5f['/'].attrs['phi_1'] = p.phi_1
        h5f['/'].attrs['phi_2'] = p.phi_2
        h5f['/'].attrs['theta_0'] = p.theta_0
        h5f['/'].attrs['theta_1'] = p.theta_1
        h5f['/'].attrs['theta_2'] = p.theta_2
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
    for i in range(1, p.max_steps + 1):
        if solver.step():
            break

        if i % 50 == 0:
            if i % 250 == 0:
                with h5py.File(f'{file_pref}.tmp.h5', 'w') as h5f:
                    solver.save_h5(h5f,
                                   script=open(os.path.abspath(__file__),
                                               'r').read())
                    h5f['/'].attrs['psi_0'] = p.psi_0
                    h5f['/'].attrs['psi_1'] = p.psi_1
                    h5f['/'].attrs['psi_2'] = p.psi_2
                    h5f['/'].attrs['phi_0'] = p.phi_0
                    h5f['/'].attrs['phi_1'] = p.phi_1
                    h5f['/'].attrs['phi_2'] = p.phi_2
                    h5f['/'].attrs['theta_0'] = p.theta_0
                    h5f['/'].attrs['theta_1'] = p.theta_1
                    h5f['/'].attrs['theta_2'] = p.theta_2
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
        h5f['/'].attrs['psi_0'] = p.psi_0
        h5f['/'].attrs['psi_1'] = p.psi_1
        h5f['/'].attrs['psi_2'] = p.psi_2
        h5f['/'].attrs['phi_0'] = p.phi_0
        h5f['/'].attrs['phi_1'] = p.phi_1
        h5f['/'].attrs['phi_2'] = p.phi_2
        h5f['/'].attrs['theta_0'] = p.theta_0
        h5f['/'].attrs['theta_1'] = p.theta_1
        h5f['/'].attrs['theta_2'] = p.theta_2
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

    print(f'rank {MPI.COMM_WORLD.Get_rank()}: finished with {p}')


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
                        default=1,
                        type=float,
                        help='mean value of fiber radius')

    parser.add_argument('-v',
                        '--volume',
                        default=1,
                        type=int,
                        help='volume size')

    parser.add_argument("-p",
                        "--num_proc",
                        type=int,
                        required=True,
                        help="Number of threads.")

    args = parser.parse_args()
    output_name = os.path.join(args.output, FILE_NAME)
    os.makedirs(args.output, exist_ok=True)
    subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
    subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)

    parameters = [
        Parameter(radius=args.radius,
                  volume=args.volume,
                  psi_0=0.333,
                  psi_1=0.333,
                  psi_2=0.333,
                  phi_0=0,
                  phi_1=90,
                  phi_2=0,
                  theta_0=90,
                  theta_1=90,
                  theta_2=0,
                  max_steps=args.max_steps,
                  num_proc=args.num_proc,
                  output=output_name),
        Parameter(radius=args.radius,
                  volume=args.volume,
                  psi_0=0.333,
                  psi_1=0.333,
                  psi_2=0.333,
                  phi_0=0,
                  phi_1=60,
                  phi_2=0,
                  theta_0=90,
                  theta_1=90,
                  theta_2=30,
                  max_steps=args.max_steps,
                  num_proc=args.num_proc,
                  output=output_name)
    ]

    [run(p) for p in parameters]  # for testing

    # with MPIPoolExecutor() as executor:
    #     executor.map(run, parameters)


if __name__ == '__main__':
    main()
