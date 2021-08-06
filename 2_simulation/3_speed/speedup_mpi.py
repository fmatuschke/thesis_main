"""
for n in {1,2,3,4,5,6,7,8,16,24,32,40,48}; do; mpiexec -n $n python3 -m mpi4py speedup_mpi.py -n 10; done
"""

import argparse
import os
import time

import fastpli.io
import fastpli.simulation
import numpy as np
import pandas as pd
import tqdm
from mpi4py import MPI

import parameter

if MPI.COMM_WORLD.Get_rank() == 0:
    print('MPI size:', MPI.COMM_WORLD.Get_size())

parser = argparse.ArgumentParser()
parser.add_argument("-n",
                    "--n_repeat",
                    required=True,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

CONFIG = parameter.get_tupleware()
SETUP = CONFIG.simulation.setup.pm

# Setup Simpli for Tissue Generation
simpli = fastpli.simulation.Simpli(MPI.COMM_WORLD)
simpli.omp_num_threads = 1
simpli.voxel_size = CONFIG.simulation.voxel_size
simpli.pixel_size = SETUP.pixel_size
simpli.filter_rotations = np.linspace(0, np.pi,
                                      CONFIG.simulation.num_filter_rot, False)
simpli.interpolate = "Slerp"
simpli.wavelength = CONFIG.simulation.wavelength  # in nm
simpli.optical_sigma = CONFIG.simulation.optical_sigma  # in pixel size
simpli.verbose = 0
simpli.set_voi(CONFIG.simulation.voi[0], CONFIG.simulation.voi[1])
tilt_angle = SETUP.tilt_angle
simpli.tilts = np.deg2rad(
    np.array([(0, 0), (tilt_angle, 0), (tilt_angle, 90), (tilt_angle, 180),
              (tilt_angle, 270)]))
simpli.add_crop_tilt_halo()

simpli.fiber_bundles = fastpli.io.fiber_bundles.load(
    os.path.join(THESIS, '1_model/1_cubes/output/cube_2pop_135_rc1',
                 'cube_2pop_psi_0.50_omega_90.00_r_0.50_v0_135_.solved.h5'))

# define layers (e.g. axon, myelin) inside fibers of each fiber_bundle
LAYERS = CONFIG.models.layers
layers = [(LAYERS.b.radius, LAYERS.b.dn, 0, LAYERS.b.model)]
layers.append((LAYERS.r.radius, LAYERS.r.dn, 0, LAYERS.r.model))
simpli.fiber_bundles.layers = [layers] * len(simpli.fiber_bundles)

if MPI.COMM_WORLD.Get_size() == 1:
    print('VOI:', simpli.get_voi())
    print('Memory:', str(round(simpli.memory_usage('MB'), 2)) + ' MB')

# Generate Tissue
# print('Run Generation:')

t_generator = []
for n in tqdm.trange(args.n_repeat, disable=MPI.COMM_WORLD.Get_rank() > 0):
    MPI.COMM_WORLD.Barrier()
    t0 = time.time()
    tissue, optical_axis, tissue_properties = simpli.generate_tissue()
    MPI.COMM_WORLD.Barrier()
    t1 = time.time()
    t_generator.append(t1 - t0)

if MPI.COMM_WORLD.Get_rank() == 0:
    fname = f"output/generation_mpi_v_{simpli.voxel_size}.pkl"
    if MPI.COMM_WORLD.Get_size() == 1:
        if os.path.exists(fname):
            os.remove(fname)
        df = pd.DataFrame()
    else:
        df = pd.read_pickle(fname)
    df[f'p{MPI.COMM_WORLD.Get_size()}'] = t_generator
    df.to_pickle(fname)

    df_ = df.copy()
    for c in df_:
        df_[c] = np.divide(np.mean(df_['p1']), df_[c])
    df_.to_csv(fname[:-4] + '.csv', index=False)

# #########################
simpli.light_intensity = SETUP.light_intensity  # a.u.
tissue_properties[1:, 1] = CONFIG.species.vervet.mu

t_generator = np.empty((args.n_repeat, len(simpli.tilts)))
for n in tqdm.trange(args.n_repeat, disable=MPI.COMM_WORLD.Get_rank() > 0):
    tt = []
    for t, (theta, phi) in enumerate(simpli.tilts):
        MPI.COMM_WORLD.Barrier()
        t0 = time.time()
        images = simpli.run_simulation(tissue, optical_axis, tissue_properties,
                                       theta, phi)
        MPI.COMM_WORLD.Barrier()
        t1 = time.time()
        t_generator[n, t] = t1 - t0

if MPI.COMM_WORLD.Get_rank() == 0:
    fname = f"output/simulation_mpi_v_{simpli.voxel_size}.pkl"
    if MPI.COMM_WORLD.Get_size() == 1:
        if os.path.exists(fname):
            os.remove(fname)
        df = pd.DataFrame()
    else:
        df = pd.read_pickle(fname)
    for n, tt in enumerate(t_generator.T):
        df[f'p{MPI.COMM_WORLD.Get_size()}_t{n}'] = tt
    df.to_pickle(fname)

    df_ = df.copy()
    for c in df_:
        n = int(c.split('_t')[-1])
        df_[c] = df_[f'p1_t{n}'].mean() / df_[c]
    df_.to_csv(fname[:-4] + '.csv', index=False)

#%%
import pandas as pd
import numpy

fname = f"output/generation_mpi_v_1.0.pkl"
df = pd.read_pickle(fname)
df_ = df.copy()
for c in df_:
    df_[c] = df[f'p1'].mean() / df[c]
df_.to_csv(fname[:-4] + '.csv', index=False)

fname = f"output/simulation_mpi_v_1.0.pkl"
df = pd.read_pickle(fname)
df_ = df.copy()
for c in df_:
    n = int(c.split('_t')[-1])
    df_[c] = df[f'p1_t{n}'].mean() / df[c]
df_.to_csv(fname[:-4] + '.csv', index=False)

# %%
