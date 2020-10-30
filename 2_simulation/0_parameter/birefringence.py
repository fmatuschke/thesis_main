import numpy as np
import multiprocessing as mp
import subprocess
import itertools
import argparse
import logging
import pandas as pd
import datetime
import warnings
import time
import h5py
import glob
import sys
import os

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.tools
import fastpli.io

import tqdm

VOXEL_SIZE = 0.1
PIXEL_SIZE = 1.25
THICKNESS = 60

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

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
output_name = os.path.join(args.output)
os.makedirs(args.output, exist_ok=True)


def run(parameter):
    file = parameter[0]
    dn = parameter[1]
    model = parameter[2]

    with h5py.File(file, 'r') as h5f_:
        fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
        psi = h5f_['/'].attrs["psi"]
        omega = h5f_['/'].attrs["omega"]
        # radius = h5f_['/'].attrs["radius"] # FIXME
        radius = float(file.split("_r_")[1].split("_")[0])
        v0 = float(file.split("_v0_")[1].split("_")[0])

    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = args.num_threads
    simpli.pixel_size = PIXEL_SIZE
    # simpli.sensor_gain = 1.5  # pm
    simpli.optical_sigma = 0.71  # in voxel size
    simpli.filter_rotations = np.linspace(0, np.pi, 9, False)
    simpli.interpolate = "Slerp"
    simpli.untilt_sensor_view = True
    simpli.wavelength = 525  # in nm
    simpli.light_intensity = 50000  # a.u.
    simpli.fiber_bundles = fiber_bundles
    simpli.tilts = np.deg2rad(np.array([(0, 0)]))

    simpli.voxel_size = VOXEL_SIZE
    simpli.set_voi(-0.5 * np.array([60, 60, THICKNESS]),
                   0.5 * np.array([60, 60, THICKNESS]))

    # print(simpli.dim_origin)
    # simpli.dim_origin[:2] = rnd_dim_origin
    # print(simpli.dim_origin)

    simpli.fiber_bundles_properties = [[(0.75, 0, 10, 'b'), (1.0, dn, 10, model)
                                       ]] * len(fiber_bundles)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="objects overlap")
        label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
        )

    # Simulate PLI Measurement
    for t, tilt in enumerate(simpli._tilts):
        theta, phi = tilt[0], tilt[1]
        images = simpli.run_simulation(label_field, vector_field,
                                       tissue_properties, theta, phi)

        # simpli.sensor_gain = 0
        simpli.sensor_gain = 1.5

        images_ = simpli.apply_optic(images)
        t, d, r = simpli.apply_epa(images_)

        df = pd.DataFrame([[
            simpli.voxel_size,
            radius,
            v0,
            model,
            omega,
            psi,
            simpli.pixel_size,
            dn,
            t,
            d,
            r,
        ]],
                          columns=[
                              "voxel_size",
                              "radius",
                              "v0",
                              "model",
                              "omega",
                              "psi",
                              "pixel_size",
                              "dn",
                              "transmittance",
                              "direction",
                              "retardation",
                          ])
    return df


if __name__ == "__main__":

    files = glob.glob(
        "../../1_model/1_cubes/output/cube_2pop_1/cube_2pop_psi_1.00_omega_0.00_r_*.solved.h5"
    )

    parameters = []
    for file in files:
        for fn in [1, 2, 3, 4]:
            for dn, model in [(-0.001 * fn, 'p'), (0.002 * fn, 'r')]:
                parameters.append((file, dn, model))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]

    df = pd.concat(df, ignore_index=True)

    df.to_pickle(os.path.join(args.output, "bf.pkl"))
