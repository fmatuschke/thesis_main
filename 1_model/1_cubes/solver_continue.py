import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.tools
import fastpli.io

import numpy as np
import subprocess
import itertools
import argparse
import datetime
import warnings
import logging
import time
import h5py
import sys
import os

import tqdm
import helper.mpi
import helper.file

from mpi4py import MPI
comm = MPI.COMM_WORLD

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    nargs='+',
                    required=True,
                    help="input string.")

parser.add_argument("-n",
                    "--max_steps",
                    type=int,
                    required=True,
                    help="Number of max_steps.")

parser.add_argument("--start",
                    type=int,
                    required=True,
                    help="start value of mpi process")

parser.add_argument("-p",
                    "--num_proc",
                    type=int,
                    required=True,
                    help="Number of threads.")

parser.add_argument("-t",
                    "--time",
                    type=float,
                    required=True,
                    help="allocation time in hours.")

args = parser.parse_args()
# os.makedirs(args.output, exist_ok=True)

# setup solver
solver = fastpli.model.solver.Solver()
solver.omp_num_threads = args.num_proc

# file = args.input[args.start + comm.Get_rank()]
for file in args.input[args.start + comm.Get_rank()::comm.Get_size()]:
    file_pref = file.split('.tmp.h5')[0]

    if os.path.isfile(file_pref + 'solved.h5'):
        print("already finished!")
    else:
        with h5py.File(file, 'r') as h5f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver.load_h5(h5f)
                i_start = h5f['/'].attrs['step'] + 1
                dt = h5f['/'].attrs['time']

                # SIZE = h5f['/'].attrs['v0']
                # RADIUS_LOGMEAN = h5f['/'].attrs['radius']

                SIZE = helper.file.value(file, 'v0')
                RADIUS_LOGMEAN = helper.file.value(file, 'r')

                psi = h5f['/'].attrs['psi']
                omega = h5f['/'].attrs['omega']

        # Run Solver
        start_time = time.time()
        for i in tqdm.trange(i_start, i_start + args.max_steps):
            if solver.step():
                break

            overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0
            if i % 50 == 0:

                print(solver.overlap)
                print(solver.num_obj)
                print(solver.num_col_obj)
                print(solver.num_steps)
                print(solver.obj_mean_length)
                print(solver.obj_min_radius)

                if i % 100 == 0:
                    if (time.time() - start_time) > 0.9 * args.time * 60 * 60:
                        with h5py.File(f'{file_pref}.tmp_.h5', 'w') as h5f:
                            solver.save_h5(h5f,
                                           script=open(
                                               os.path.abspath(__file__),
                                               'r').read())
                            h5f['/'].attrs['psi'] = psi
                            h5f['/'].attrs['omega'] = omega
                            h5f['/'].attrs['step'] = i
                            h5f['/'].attrs['overlap'] = solver.overlap
                            h5f['/'].attrs['num_obj'] = solver.num_obj
                            h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
                            h5f['/'].attrs['num_steps'] = solver.num_steps
                            h5f['/'].attrs[
                                'obj_mean_length'] = solver.obj_mean_length
                            h5f['/'].attrs[
                                'obj_min_radius'] = solver.obj_min_radius
                            h5f['/'].attrs['time'] = time.time() - start_time

                if i != args.max_steps:
                    solver.fiber_bundles = fastpli.objects.fiber_bundles.CutSphere(
                        solver.fiber_bundles,
                        0.5 * (SIZE + 10 * RADIUS_LOGMEAN))

        overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0

        with h5py.File(file_pref + '.solved.h5', 'w-') as h5f:
            solver.save_h5(h5f,
                           script=open(os.path.abspath(__file__), 'r').read())
            h5f['/'].attrs['psi'] = psi
            h5f['/'].attrs['omega'] = omega
            h5f['/'].attrs['v0'] = SIZE
            h5f['/'].attrs['radius'] = RADIUS_LOGMEAN
            h5f['/'].attrs['step'] = i
            h5f['/'].attrs['overlap'] = solver.overlap
            h5f['/'].attrs['num_obj'] = solver.num_obj
            h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
            h5f['/'].attrs['num_steps'] = solver.num_steps
            h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
            h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
            h5f['/'].attrs['time'] = time.time() - start_time + dt

        # if os.path.isfile(f'{file_pref}.tmp.h5'):
        #     os.remove(f'{file_pref}.tmp.h5')
