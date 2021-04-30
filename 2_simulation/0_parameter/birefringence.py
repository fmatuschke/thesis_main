#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import argparse
import pandas as pd
import warnings
import h5py
import glob
import os

import fastpli.simulation
import fastpli.analysis
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.tools
import fastpli.io

import tqdm

VOXEL_SIZE = 0.125
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
os.makedirs(args.output, exist_ok=False)
subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)


def run(parameter):
    file = parameter[0]
    dn = parameter[1]
    model = parameter[2]
    name = parameter[3]
    gain = parameter[4]
    intensity = parameter[5]
    res = parameter[6]
    # tilt_angle = parameter[7]
    sigma = parameter[8]
    species = parameter[9]
    mu = parameter[10]

    with h5py.File(file, 'r') as h5f_:
        fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
        psi = h5f_['/'].attrs["psi"]
        omega = h5f_['/'].attrs["omega"]
        # radius = h5f_['/'].attrs["radius"] # FIXME
        radius = float(file.split("_r_")[1].split("_")[0])
        v0 = float(file.split("_v0_")[1].split("_")[0])

    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = args.num_threads
    simpli.pixel_size = res
    simpli.optical_sigma = sigma  # in voxel size
    simpli.filter_rotations = np.linspace(0, np.pi, 9, False)
    simpli.interpolate = "Slerp"
    simpli.untilt_sensor_view = True
    simpli.wavelength = 525  # in nm
    simpli.light_intensity = intensity  # a.u.
    simpli.fiber_bundles = fiber_bundles
    simpli.tilts = np.deg2rad(np.array([(0, 0)]))

    simpli.voxel_size = VOXEL_SIZE
    simpli.set_voi(-0.5 * np.array([60, 60, THICKNESS]),
                   0.5 * np.array([60, 60, THICKNESS]))

    # print(simpli.dim_origin)
    # simpli.dim_origin[:2] = rnd_dim_origin
    # print(simpli.dim_origin)

    simpli.fiber_bundles.layers = [[(0.75, 0, 0, 'b'),
                                    (1.0, dn, 0, model)]] * len(fiber_bundles)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="objects overlap")
        label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
        )

    # Simulate PLI Measurement
    for tilt in simpli._tilts:
        theta, phi = tilt[0], tilt[1]
        images = simpli.run_simulation(label_field, vector_field,
                                       tissue_properties, theta, phi)

        # absorption
        images *= np.exp(-mu * THICKNESS * 1e-3 * simpli.voxel_size)

        simpli.noise_model = lambda x: np.round(
            np.random.normal(x, np.sqrt(gain * x))).astype(np.uint16)

        _, images = simpli.apply_optic(images)
        t, d, r = simpli.apply_epa(images)

        df = pd.DataFrame([[
            species,
            simpli.voxel_size,
            radius,
            v0,
            model,
            name,
            omega,
            psi,
            simpli.pixel_size,
            dn,
            mu,
            t,
            d,
            r,
        ]],
                          columns=[
                              "species",
                              "voxel_size",
                              "radius",
                              "v0",
                              "model",
                              "setup",
                              "omega",
                              "psi",
                              "pixel_size",
                              "dn",
                              "mu",
                              "transmittance",
                              "direction",
                              "retardation",
                          ])
    return df


if __name__ == "__main__":

    files = glob.glob(
        "/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/cube_2pop_psi_1.00_omega_0.00_r_*.solved.h5"
    )

    parameters = []
    for file in files:
        for fn in np.arange(1, 5.001, 0.125):
            for dn, model in [(-0.001 * fn, 'p'), (0.002 * fn, 'r')]:
                for name, gain, intensity, res, tilt_angle, sigma in [
                    ('LAP', 3, 35000, 20, 5.5, 0.75),
                    ('PM', 0.1175, 8000, 1.25, 3.9, 0.75)
                ]:
                    for species, mu in [('Roden', 8), ('Vervet', 30),
                                        ('Human', 65)]:
                        parameters.append(
                            (file, dn, model, name, gain, intensity, res,
                             tilt_angle, sigma, species, mu))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]

    df = pd.concat(df, ignore_index=True)

    df.to_pickle(os.path.join(args.output, "bf.pkl"))
