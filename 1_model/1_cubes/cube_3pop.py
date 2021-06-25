import argparse
import os
import time
import ast

import fastpli.io
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.tools
import h5py
import numpy as np
from tqdm import tqdm

rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
np.random.seed(rnd_seed)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Output path of solver.")

parser.add_argument("-n",
                    "--max_steps",
                    type=int,
                    required=True,
                    help="Number of max_steps.")

args = parser.parse_args()
file_name = args.input

solver = fastpli.model.solver.Solver()

with h5py.File(file_name, 'r') as h5f:
    solver.set_dict(ast.literal_eval(h5f.attrs['fastpli/solver']))
    psi_0 = h5f['/'].attrs['psi_0']
    psi_1 = h5f['/'].attrs['psi_1']
    psi_2 = h5f['/'].attrs['psi_2']
    phi_0 = h5f['/'].attrs['phi_0']
    phi_1 = h5f['/'].attrs['phi_1']
    phi_2 = h5f['/'].attrs['phi_2']
    theta_0 = h5f['/'].attrs['theta_0']
    theta_1 = h5f['/'].attrs['theta_1']
    theta_2 = h5f['/'].attrs['theta_2']

    volume = h5f['/'].attrs['v0']
    radius = h5f['/'].attrs['radius']
    rnd_seed = h5f['/'].attrs['rnd_seed']

# Run Solver
solver.fiber_bundles = solver.fiber_bundles.cut_sphere(0.5 *
                                                       (volume + 10 * radius))
for i in tqdm(range(1, args.max_steps + 1), leave=False):
    if solver.step():
        break

    overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0
    if i % 50 == 0:
        # logger.info(
        #     f"step: {i}, {solver.num_obj}/{solver.num_col_obj} {round(overlap * 100)}%"
        # )

        if i % 250 == 0:
            with h5py.File(f'{file_name}.tmp.h5', 'w') as h5f:
                solver.save_h5(h5f,
                               script=open(os.path.abspath(__file__),
                                           'r').read())
                h5f['/'].attrs['psi_0'] = psi_0
                h5f['/'].attrs['psi_1'] = psi_1
                h5f['/'].attrs['psi_2'] = psi_2
                h5f['/'].attrs['phi_0'] = phi_0
                h5f['/'].attrs['phi_1'] = phi_1
                h5f['/'].attrs['phi_2'] = phi_2
                h5f['/'].attrs['theta_0'] = theta_0
                h5f['/'].attrs['theta_1'] = theta_1
                h5f['/'].attrs['theta_2'] = theta_2
                h5f['/'].attrs['v0'] = volume
                h5f['/'].attrs['radius'] = radius
                h5f['/'].attrs['step'] = i
                h5f['/'].attrs['overlap'] = solver.overlap
                h5f['/'].attrs['num_obj'] = solver.num_obj
                h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
                h5f['/'].attrs['num_steps'] = solver.num_steps
                h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
                h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
                h5f['/'].attrs['rnd_seed'] = rnd_seed

        if i != args.max_steps:
            solver.fiber_bundles = solver.fiber_bundles.cut_sphere(
                0.5 * (volume + 10 * radius))

overlap = solver.overlap / solver.num_col_obj if solver.num_col_obj else 0

with h5py.File(file_name + '.solved.h5', 'w-') as h5f:
    solver.save_h5(h5f, script=open(os.path.abspath(__file__), 'r').read())
    h5f['/'].attrs['psi_0'] = psi_0
    h5f['/'].attrs['psi_1'] = psi_1
    h5f['/'].attrs['psi_2'] = psi_2
    h5f['/'].attrs['phi_0'] = phi_0
    h5f['/'].attrs['phi_1'] = phi_1
    h5f['/'].attrs['phi_2'] = phi_2
    h5f['/'].attrs['theta_0'] = theta_0
    h5f['/'].attrs['theta_1'] = theta_1
    h5f['/'].attrs['theta_2'] = theta_2
    h5f['/'].attrs['v0'] = volume
    h5f['/'].attrs['radius'] = radius
    h5f['/'].attrs['step'] = i
    h5f['/'].attrs['overlap'] = solver.overlap
    h5f['/'].attrs['num_obj'] = solver.num_obj
    h5f['/'].attrs['num_col_obj'] = solver.num_col_obj
    h5f['/'].attrs['num_steps'] = solver.num_steps
    h5f['/'].attrs['obj_mean_length'] = solver.obj_mean_length
    h5f['/'].attrs['obj_min_radius'] = solver.obj_min_radius
    h5f['/'].attrs['rnd_seed'] = rnd_seed

if os.path.isfile(f'{file_name}.tmp.h5'):
    os.remove(f'{file_name}.tmp.h5')
