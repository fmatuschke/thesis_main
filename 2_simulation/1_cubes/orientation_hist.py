#! /usr/bin/env python3

import argparse
import multiprocessing as mp
import os

import fastpli.analysis
import fastpli.tools
import matplotlib.pyplot as plt
import models
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

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

hist_polar_bin = lambda n: np.linspace(
    -np.pi / 2, np.pi / 2, n + 1, endpoint=True)


def polar_hist(phi, theta):
    phi = np.array(phi)
    theta = np.array(theta)

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


def plot_polar(phi, theta, data, file_name):
    _, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    ax.pcolormesh(phi, theta, data)
    plt.save(file_name)


def run(df):
    df = df[1]
    phi = df['rofl_dir']
    theta = np.pi / 2 - df['rofl_inc']

    _, ax = plt.subplots(subplot_kw=dict(projection="polar"))

    _ = fastpli.analysis.orientation.histogram(phi.ravel(),
                                               theta.ravel(),
                                               n_phi=36 * 2,
                                               n_theta=18,
                                               weight_area=True,
                                               ax=ax)

    plt.tight_layout()
    plt.savefig(
        f'output/test/{df.microscope}_{df.species}_{df.model}_r_{df.radius:.2f}_o_{df.omega:.2f}_p_{df.psi:.2f}_f0_{df.f0_inc:.2f}_f1_{df.f1_rot:.2f}_.pdf'
    )

    plt.close()


def run_init(df):
    df = df[1]

    file = f'../../1_model/1_cubes/output/cube_2pop_135_rc1/cube_2pop_psi_{df.psi:.2f}_omega_{df.omega:.2f}_r_{df.radius:.2f}_v0_135_.solved.h5'

    phi, theta = models.ori_from_file(file, 0, 0, CONFIG.simulation.voi)

    _, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    _ = fastpli.analysis.orientation.histogram(phi.ravel(),
                                               theta.ravel(),
                                               n_phi=36 * 2,
                                               n_theta=18,
                                               weight_area=True,
                                               ax=ax)

    plt.tight_layout()
    plt.savefig(
        f'output/test/{df.microscope}_{df.species}_{df.model}_r_{df.radius:.2f}_o_{df.omega:.2f}_p_{df.psi:.2f}_f0_{df.f0_inc:.2f}_f1_{df.f1_rot:.2f}_.init.pdf'
    )

    plt.close()


def main():
    args = parser.parse_args()

    df = pd.read_pickle(
        os.path.join(args.input, "analysis", f"cube_2pop_simulation.pkl"))

    # df = df[(df.omega == 0) | (df.omega == 30) | (df.omega == 60) |
    #         (df.omega == 90)]
    # df = df[(df.psi == 0.3) | (df.psi == 0.6) | (df.psi == 0.9)]
    df = df[(df.f0_inc == 0)]
    df = df[(df.species == 'Vervet')]
    df = df[(df.model == 'r')]
    df = df[(df.f1_rot == 0)]

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, df.iterrows()),
                                 total=len(df),
                                 smoothing=0.1)
        ]

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run_init, df.iterrows()),
                                 total=len(df),
                                 smoothing=0.1)
        ]


if __name__ == "__main__":
    main()
