"""
MPI executable example.

NOTE: if h5py is used, it has to be compiled in parallel mode. See
https://docs.h5py.org/en/stable/mpi.html for more information

Ecexution:
mpirun -n 2 python3 -m mpi4py examples/simpli_mpi.py
"""

import argparse
import os
import time

import fastpli.io
import fastpli.simulation
import numpy as np
import pandas as pd
import tqdm

import parameter

# from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("-p",
                    "--num_proc",
                    required=True,
                    type=int,
                    help="Number of processes.")
parser.add_argument("-n",
                    "--n_repeat",
                    required=True,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()

print(args.num_proc)

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

CONFIG = parameter.get_tupleware()
SETUP = CONFIG.simulation.setup.pm

# Setup Simpli for Tissue Generation
simpli = fastpli.simulation.Simpli()
simpli.omp_num_threads = args.num_proc
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

if args.num_proc == 1:
    print('VOI:', simpli.get_voi())
    print('Memory:', str(round(simpli.memory_usage('MB'), 2)) + ' MB')

t_generator = []
for n in tqdm.trange(args.n_repeat):
    t0 = time.time()
    tissue, optical_axis, tissue_properties = simpli.generate_tissue()
    t1 = time.time()
    t_generator.append(t1 - t0)

fname = f"output/generation_mp_v_{simpli.voxel_size}.pkl"
if args.num_proc == 1:
    if os.path.exists(fname):
        os.remove(fname)
    df = pd.DataFrame()
else:
    if os.path.exists(fname):
        df = pd.read_pickle(fname)
    else:
        df = pd.DataFrame()
df[f'p{args.num_proc}'] = t_generator
df.to_pickle(fname)

df_ = df.copy()
for c in df_:
    df_[c] = df[f'p{1}'].mean() / df[c]
df_.to_csv(fname[:-4] + '.csv', index=False)

# ###################
simpli.light_intensity = SETUP.light_intensity  # a.u.
tissue_properties[1:, 1] = CONFIG.species.vervet.mu

t_generator = np.empty((args.n_repeat, len(simpli.tilts)))
for n in tqdm.trange(args.n_repeat):
    tt = []
    for t, (theta, phi) in enumerate(simpli.tilts):
        t0 = time.time()
        images = simpli.run_simulation(tissue, optical_axis, tissue_properties,
                                       theta, phi)
        t1 = time.time()
        t_generator[n, t] = t1 - t0

fname = f"output/simulation_mp_v_{simpli.voxel_size}.pkl"
if args.num_proc == 1:
    if os.path.exists(fname):
        os.remove(fname)
    df = pd.DataFrame()
else:
    if os.path.exists(fname):
        df = pd.read_pickle(fname)
    else:
        df = pd.DataFrame()
for n, tt in enumerate(t_generator.T):
    df[f'p{args.num_proc}_t{n}'] = tt
df.to_pickle(fname)

df_ = df.copy()
for c in df_:
    n = int(c.split('_t')[-1])
    df_[c] = df[f'p1_t{n}'].mean() / df[c]
df_.to_csv(fname[:-4] + '.csv', index=False)
