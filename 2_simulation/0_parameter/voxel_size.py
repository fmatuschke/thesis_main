import argparse
import datetime
import glob
import itertools
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import warnings

import fastpli.analysis
import fastpli.io
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.simulation
import fastpli.tools
import h5py
import helper.file
import numpy as np
import parameter
import tqdm
from mpi4py import MPI

import models

# reproducability -> see run()
# np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
CONFIG = parameter.get_tupleware()
# arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    # nargs='+',
    required=True,
    help="input string.")

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

parser.add_argument("-n",
                    "--repeat",
                    type=int,
                    required=True,
                    help="repeat n times")

parser.add_argument("-m",
                    "--repeat_noise",
                    type=int,
                    required=True,
                    help="repeat noise m times")

parser.add_argument("-p",
                    "--num_proc",
                    type=int,
                    required=True,
                    help="Number of parallel processes.")

parser.add_argument("-t",
                    "--num_threads",
                    type=int,
                    required=True,
                    help="Number of threads per process.")

args = parser.parse_args()
output_name = os.path.join(args.output, FILE_NAME)
os.makedirs(args.output, exist_ok=False)
subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)

VOXEL_SIZES = [0.01, 0.025, 0.05, 0.1, 0.26, 0.65, 1.3]
SETUP = CONFIG.simulation.setup.pm


def get_file_pref(parameter):
    file, f0_inc, f1_rot = parameter

    base = os.path.basename(file)
    pre = base.split("_psi_")[0]
    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])
    r = float(file.split("_r_")[1].split("_")[0])

    return output_name + f"_{pre}_psi_{psi:.2f}_omega_{omega:.2f}_" \
        f"r_{r:.2f}_" \
        f"f0_inc_{f0_inc:.2f}_" \
        f"f1_rot_{f1_rot:.2f}_" \
        f"p0_{SETUP.pixel_size:.2f}_"


def run(parameter):
    file, f0_inc, f1_rot = parameter

    file_pref = get_file_pref(parameter)
    rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(rnd_seed)

    # generate model
    with h5py.File(file_pref + '.h5', 'w-') as h5f:

        h5f.attrs['script'] = open(os.path.abspath(__file__), 'r').read()

        with h5py.File(file, 'r') as h5f_:
            fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
            psi = h5f_['/'].attrs["psi"]
            omega = h5f_['/'].attrs["omega"]
            radius = h5f_['/'].attrs["radius"]  # FIXME
            v0 = h5f_['/'].attrs["v0"]  # FIXME
            # radius = float(file.split("_r_")[1].split("_")[0])
            # v0 = float(file.split("_v0_")[1].split("_")[0])

        h5f.attrs['psi'] = psi
        h5f.attrs['omega'] = omega
        h5f.attrs['radius'] = radius
        h5f.attrs['v0'] = v0
        h5f.attrs['f0_inc'] = f0_inc
        h5f.attrs['f1_rot'] = f1_rot
        h5f.attrs['pixel_size'] = CONFIG.simulation.setup.pm.pixel_size
        h5f.attrs['rnd_seed'] = rnd_seed
        h5f['fiber_bundles'] = file

        fiber_bundles = models.rotate(fiber_bundles, f0_inc, f1_rot)

        nstep = int(np.round(np.sqrt(args.repeat)))
        step = 60 / nstep  # 60er cube
        for n in tqdm.tqdm(list(range(args.repeat))):
            rnd_dim_origin = np.array(
                (-30 + (n % nstep) * step, -30 + (n // nstep) * step))
            for voxel_size in VOXEL_SIZES:

                species_list = [
                    #('Human', CONFIG.species.human.mu),
                    ('Vervet', CONFIG.species.vervet.mu),
                    #('Roden', CONFIG.species.roden.mu)
                ]
                for species, mu in species_list:
                    model_list = ['r']  # , 'p']
                    for model in model_list:
                        LAYERS = CONFIG.models.layers
                        layers = [(LAYERS.b.radius, LAYERS.b.dn, LAYERS.b.mu,
                                   LAYERS.b.model)]
                        if model == 'r':
                            layers.append((LAYERS.r.radius, LAYERS.r.dn,
                                           LAYERS.r.mu, LAYERS.r.model))
                        elif model == 'p':
                            layers.append((LAYERS.p.radius, LAYERS.p.dn,
                                           LAYERS.p.mu, LAYERS.p.model))
                        else:
                            raise ValueError('FOOO')

                        setup_list = [
                            ('PM', CONFIG.simulation.setup.pm),
                            #('LAP', CONFIG.simulation.setup.lap) SETUP.pixel_size!!!
                        ]
                        for setup, SETUP in setup_list:

                            # Setup Simpli
                            simpli = fastpli.simulation.Simpli()
                            simpli.omp_num_threads = args.num_threads
                            simpli.pixel_size = SETUP.pixel_size
                            simpli.filter_rotations = np.linspace(
                                0, np.pi, CONFIG.simulation.num_filter_rot,
                                False)
                            simpli.interpolate = "Slerp"
                            simpli.untilt_sensor_view = True
                            simpli.wavelength = CONFIG.simulation.wavelength  # in nm
                            simpli.optical_sigma = CONFIG.simulation.optical_sigma  # in pixel size
                            simpli.light_intensity = SETUP.light_intensity  # a.u.
                            simpli.fiber_bundles = fiber_bundles
                            simpli.tilts = np.deg2rad(np.array([(0, 0)]))

                            simpli.voxel_size = voxel_size

                            voi = (3 * SETUP.pixel_size, 3 * SETUP.pixel_size,
                                   CONFIG.simulation.volume[-1])
                            simpli.set_voi(-0.5 * np.array(voi),
                                           0.5 * np.array(voi))

                            # print(simpli.dim_origin)
                            simpli.dim_origin[:2] = rnd_dim_origin
                            # print(simpli.dim_origin)

                            if simpli.memory_usage() * args.num_proc > 200000:
                                print(
                                    str(round(simpli.memory_usage(), 2)) + 'MB')
                                return

                            dset = h5f.create_group(
                                f'simpli/{voxel_size}/{setup}/{species}/{model}/{n}'
                            )
                            dset.attrs['dim_origin'] = rnd_dim_origin

                            simpli.fiber_bundles.layers = [layers
                                                          ] * len(fiber_bundles)

                            label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                            )

                            unique_elements, counts_elements = np.unique(
                                label_field, return_counts=True)
                            dset.attrs['label_field_stats'] = np.asarray(
                                (unique_elements, counts_elements))

                            if np.all(unique_elements == 0):
                                print(file, f0_inc, f1_rot, counts_elements,
                                      unique_elements, rnd_dim_origin,
                                      simpli.dim, simpli.dim_origin)

                            # Simulate PLI Measurement
                            # simpli.save_parameter_h5(h5f=dset)
                            for t, tilt in enumerate(simpli._tilts):
                                theta, phi = tilt[0], tilt[1]
                                images = simpli.run_simulation(
                                    label_field, vector_field,
                                    tissue_properties, theta, phi)

                                # absorption
                                images *= np.exp(
                                    -mu * CONFIG.simulation.volume[-1] * 1e-3 *
                                    simpli.voxel_size)

                                dset['simulation/data/' + str(t)] = images
                                dset['simulation/data/' +
                                     str(t)].attrs['theta'] = theta
                                dset['simulation/data/' +
                                     str(t)].attrs['phi'] = phi

                                for m in range(args.repeat_noise):
                                    if m == 0:
                                        simpli.noise_model = lambda x: np.round(
                                            x).astype(np.float32)
                                    else:
                                        simpli.noise_model = lambda x: np.round(
                                            np.random.normal(
                                                x, np.sqrt(SETUP.gain * x))
                                        ).astype(np.float32)
                                    # apply optic to simulation
                                    _, images_ = simpli.apply_optic(images)
                                    dset[f'simulation/optic/{t}/{m}'] = images_
                                    # calculate modalities
                                    epa = simpli.apply_epa(images_)
                                    dset[
                                        f'analysis/epa/{t}/transmittance/{m}'] = epa[
                                            0]
                                    dset[
                                        f'analysis/epa/{t}/direction/{m}'] = epa[
                                            1]
                                    dset[
                                        f'analysis/epa/{t}/retardation/{m}'] = epa[
                                            2]

                                dset[f'simulation/optic/{t}'].attrs[
                                    'theta'] = theta
                                dset[f'simulation/optic/{t}'].attrs['phi'] = phi
                                dset[f'analysis/epa/{t}'].attrs['theta'] = theta
                                dset[f'analysis/epa/{t}'].attrs['phi'] = phi

                            del label_field
                            del vector_field
                            del simpli


def check_file(p):
    file_pref = get_file_pref(p)
    return not os.path.isfile(file_pref + '.h5')


if __name__ == "__main__":
    file_list = glob.glob(os.path.join(args.input, "*.solved.h5"))

    # four case study -> ||,+,*,*|
    filtered_files = []
    for file in file_list:
        psi = helper.file.value(file, "psi")
        omega = helper.file.value(file, "omega")
        r = helper.file.value(file, "r")
        v0 = helper.file.value(file, "v0")
        if psi == 1:
            filtered_files.append(file)
        elif psi == 0.5 and omega == 90:
            filtered_files.append(file)

    parameters = []
    for file, f0_inc in list(itertools.product(filtered_files, [0, 90])):
        parameters.append((file, f0_inc, 0))

    print("total:", len(parameters))

    # DEBUG
    # run(parameters[0])
    with mp.Pool(processes=args.num_proc) as pool:
        [
            d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]

# sort -t- -k3.1,3.4
