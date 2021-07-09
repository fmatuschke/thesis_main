#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import os

import pandas as pd
from tqdm import tqdm

import fastpli.io
import fastpli.analysis
import fastpli.simulation

import helper.circular

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()

hist_polar_bin = lambda n: np.linspace(
    -np.pi / 2, np.pi / 2, n + 1, endpoint=True)


def polar_hist(phi, theta):
    incl = np.pi / 2 - theta

    # remap to phi->[-np.pi/2, np.pi/2], incl->[-np.pi/2, np.pi/2]
    phi[phi > np.pi * 3 / 2] -= 2 * np.pi
    incl[phi > np.pi / 2] = -incl[phi > np.pi / 2]
    phi[phi > np.pi / 2] -= np.pi

    if np.any(incl < -np.pi / 2) or np.any(incl > np.pi / 2):
        raise ValueError("FOOO incl")
    if np.any(phi < -np.pi / 2) or np.any(phi > np.pi / 2):
        raise ValueError("FOOO phi")

    # direction
    h, x = np.histogram(phi, hist_polar_bin(180), density=True)
    phi_x = x[:-1] + (x[1] - x[0]) / 2
    phi_h = h / np.amax(h)

    # inclination
    h, x = np.histogram(incl, hist_polar_bin(180), density=True)
    incl_x = x[:-1] + (x[1] - x[0]) / 2
    incl_h = h / np.amax(h)

    return phi_x, phi_h, incl_x, incl_h


def run(file):
    omega = float(file.split("_omega_")[1].split("_")[0])
    psi = float(file.split("_psi_")[1].split("_")[0])
    r = float(file.split("_r_")[1].split("_")[0])
    v0 = float(file.split("_v0_")[1].split("_")[0])
    fl = float(file.split("_fl_")[1].split("_")[0])
    fr = float(file.split("_fr_")[1].split("_")[0])
    n = float(file.split("_n_")[1].split("_")[0])
    state = file.split(".")[-2].split(".")[-1]

    try:
        fbs = fastpli.io.fiber_bundles.load(file)
    except:
        raise ValueError("failed to open file")

    if state != "init":
        # Setup Simpli
        simpli = fastpli.simulation.Simpli()
        warnings.filterwarnings("ignore", message="objects overlap")
        # simpli.omp_num_threads = 0
        simpli.voxel_size = 0.1
        simpli.set_voi([-30] * 3, [30] * 3)
        simpli.fiber_bundles = fbs
        simpli.fiber_bundles.layers = [[(1.0, 0, 0, 'p')]] * len(fbs)

        if run.flag.value == 0:
            with run.lock:
                if run.flag.value == 0:
                    print(
                        f"Single Memory: {simpli.memory_usage(item='tissue'):.0f} MB"
                    )
                    print(
                        f"Total Memory: {simpli.memory_usage(item='tissue') * args.num_proc:.0f} MB"
                    )
                    run.flag.value = 1

        tissue, _, _ = simpli.generate_tissue(only_tissue=True)
        unique_elements, count_elements = np.unique(tissue, return_counts=True)
    else:
        unique_elements, count_elements = 0, 0

    with h5py.File(file, "r") as h5f:

        obj_mean_length = h5f['/'].attrs['obj_mean_length']
        obj_min_radius = h5f['/'].attrs['obj_min_radius']
        if state != "init":
            overlap = h5f['/'].attrs['overlap']
            num_col_obj = h5f['/'].attrs['num_col_obj']
            num_obj = h5f['/'].attrs['num_obj']
            num_steps = h5f['/'].attrs['num_steps']
            time = h5f['/'].attrs['time']
        else:
            overlap = np.nan
            num_col_obj = np.nan
            num_obj = np.nan
            num_steps = np.nan
            time = np.nan

    fbs_ = fbs.cut([[-30] * 3, [30] * 3])

    if len([f for fb in fbs for f in fb]) == 0:
        raise ValueError(f"FOO: no fibers inside cube: {file}")

    phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs_)
    phi_x, phi_h, incl_x, incl_h = polar_hist(phi, theta)

    return pd.DataFrame([[
        omega,
        psi,
        r,
        v0,
        fl,
        fr,
        n,
        state,
        overlap,
        obj_mean_length,
        obj_min_radius,
        num_col_obj,
        num_obj,
        num_steps,
        time,
        unique_elements,
        count_elements,
        phi_x,
        phi_h,
        incl_x,
        incl_h,
    ]],
                        columns=[
                            "omega",
                            "psi",
                            "r",
                            "v0",
                            "fl",
                            "fr",
                            "n",
                            "state",
                            "overlap",
                            "obj_mean_length",
                            "obj_min_radius",
                            "num_col_obj",
                            "num_obj",
                            "num_steps",
                            "time",
                            "unique_elements",
                            "count_elements",
                            "phi_x",
                            "phi_h",
                            "incl_x",
                            "incl_h",
                        ])


files = glob.glob(os.path.join(args.input, "*.h5"))

run.flag = mp.Value('i', 0)
run.lock = mp.Lock()
with mp.Pool(processes=args.num_proc) as pool:
    df = [
        d for d in tqdm(
            pool.imap_unordered(run, files), total=len(files), smoothing=0.0)
    ]
    df = pd.concat(df, ignore_index=True)

df.to_pickle(os.path.join(args.input, "parameter_statistic.pkl"))
