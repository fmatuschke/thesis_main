#! /usr/bin/env python3

import argparse
import ast
import glob
import itertools
import multiprocessing as mp
import os
import warnings

# import pretty_errors
import fastpli.tools
import h5py
import helper.file
import numba
import numpy as np
import pandas as pd
import tqdm

import parameter

CONFIG = parameter.get_tupleware()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input path.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()


def _calc_intensity(phi, alpha, t_rel, theta, phii):
    num_rotations = 9
    number_tilts = 5
    rotation_angles = np.linspace(0, np.pi, num_rotations + 1)[:-1]

    # tilt orientation
    u = np.array([
        np.cos(phii) * np.sin(theta),
        np.sin(phii) * np.sin(theta),
        np.cos(theta)
    ])

    # fiber orientation
    v = np.array([
        np.cos(phi) * np.cos(alpha),
        np.sin(phi) * np.cos(alpha),
        np.sin(alpha)
    ])

    # tilted fiber orientation
    rot = fastpli.tools.rotation.a_on_b([0, 0, 1], v)
    v = np.dot(rot, u)
    phi_v = np.arctan2(v[1], v[0])
    alpha_v = np.arcsin(v[2])

    I = np.sin(np.pi / 2 * t_rel * np.cos(alpha_v)**2) * np.sin(
        2 * (rotation_angles - phi_v))
    return I


def run(file):
    df = []

    with h5py.File(file, 'r') as h5f:
        for microscope, species, model in list(
                itertools.product(["PM", "LAP"], ["Roden", "Vervet", "Human"],
                                  ["r", "p"])):

            if f"/{microscope}/{species}/{model}/" not in h5f:
                continue
            h5f_sub = h5f[f"/{microscope}/{species}/{model}/"]

            # R
            rofl_direction = h5f_sub['analysis/rofl/direction'][...]
            rofl_inclination = h5f_sub['analysis/rofl/inclination'][...]
            rofl_trel = h5f_sub['analysis/rofl/t_rel'][...]

            fit_data = np.empty(
                (5, rofl_direction.shape[0], rofl_direction.shape[1], 9))

            try:
                tilt_angle = h5f_sub['simulation'].attrs['tilt_angle']
            except:
                warnings.warn("using config value")
                if microscope == "PM":
                    SETUP = CONFIG.simulation.setup.pm
                elif microscope == "LAP":
                    SETUP = CONFIG.simulation.setup.lap
                tilt_angle = SETUP.tilt_angle

            # TODO:
            warnings.warn('LOOK AT THIS')
            print(tilt_angle)  # For safty

            optic_data = []
            phis = [0, 0, 90, 180, 270]
            for t, phi in enumerate(phis):
                for i in range(fit_data.shape[1]):
                    for j in range(fit_data.shape[2]):
                        fit_data[t, i, j, :] = _calc_intensity(
                            rofl_direction[i, j], rofl_inclination[i, j],
                            rofl_trel[i, j], np.deg2rad(tilt_angle),
                            np.deg2rad(phi))

                optic_data.append(h5f_sub[f'simulation/optic/{t}'][...])

            optic_data = np.array(optic_data)
            optic_data = np.divide(optic_data,
                                   np.mean(optic_data, axis=-1)[:, :, :,
                                                                None]) - 1

            R = np.mean(np.abs(fit_data.ravel() - optic_data.ravel()))
            R2 = np.mean(np.power(fit_data.ravel() - optic_data.ravel(), 2))

            # print("R:", R, "R2:", R2)
            # print(fit_data.ravel())
            # print(((optic_data.ravel() / np.mean(optic_data.ravel()) - 1)))

            df.append(
                pd.DataFrame([[
                    microscope, species, model,
                    float(h5f_sub.attrs['parameter/radius']),
                    float(h5f_sub.attrs['parameter/omega']),
                    float(h5f_sub.attrs['parameter/psi']),
                    float(h5f_sub.attrs['parameter/f0_inc']),
                    float(h5f_sub.attrs['parameter/f1_rot']),
                    h5f_sub['analysis/rofl/direction'][...].ravel(),
                    h5f_sub['analysis/rofl/inclination'][...].ravel(),
                    h5f_sub['analysis/rofl/t_rel'][...].ravel(),
                    h5f_sub['analysis/epa/0/transmittance'][...].ravel(),
                    h5f_sub['analysis/epa/0/direction'][...].ravel(),
                    h5f_sub['analysis/epa/0/retardation'][...].ravel(), R, R2
                ]],
                             columns=[
                                 "microscope", "species", "model", "radius",
                                 "omega", "psi", "f0_inc", "f1_rot", "rofl_dir",
                                 "rofl_inc", "rofl_trel", "epa_trans",
                                 "epa_dir", "epa_ret", "R", "R2"
                             ]))
    return df


if __name__ == "__main__":
    os.makedirs(os.path.join(args.input, "analysis"), exist_ok=True)
    files = glob.glob(os.path.join(args.input, "*.h5"))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, files),
                                 total=len(files),
                                 smoothing=0.1)
        ]

    df = [item for sublist in df for item in sublist]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(
        os.path.join(os.path.join(args.input, "analysis"),
                     f"cube_2pop_simulation.pkl"))
