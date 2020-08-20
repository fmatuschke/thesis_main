import numpy as np
import multiprocessing as mp
import itertools
import h5py
import os
import sys
import glob

import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import pandas as pd
import tikzplotlib

from tqdm import tqdm

import helper.circular
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

model_path = "output/cube_2pop_juron_0_"
out_path = "output/cube_2pop_juron_0_hist"
os.makedirs(out_path, exist_ok=True)

hist_bin = lambda n: np.linspace(0, np.pi, n + 1, endpoint=True)

# files = glob.glob(os.path.join(model_path, "*solved.pkl"))


def run(file):
    for phase, _ in [("init", "g"), ("solved", "r")]:
        df = pd.read_pickle(file + f".{phase}.pkl")

        # sub = (df.f0_inc == 0) & (df.f1_rot == 0)

        # if len(df[sub]) != 1:
        #     sys.exit(1)

        # phi = df[sub].explode("phi").phi.to_numpy(float)
        # theta = df[sub].explode("theta").theta.to_numpy(float)

    if state == "init":
        f0_list = [0]
        f1_list = [0]
    else:
        f0_list = fibers.inclinations()
        f1_list = fibers.omega_rotations(omega)

    df = []
    for f0_inc in f0_list:
        for f1_rot in f1_list:
            # for v in [120, 60]:  # ascending order!
            for v in [60]:  # ascending order!
                phi, theta = fibers.ori_from_fbs(fbs, omega, f0_inc, f1_rot, v)

                #TODO:!!!!!!!!!!!!!!!!!!!!!!!

                df.append(
                    pd.DataFrame([[
                        f0_inc, f1_rot, v0, v, r, meta, phi, theta,
                        os.path.abspath(file)
                    ]],
                                 columns=[
                                     "f0_inc", "f1_rot", "v0", "v", "r", "meta",
                                     "phi", "theta", "file"
                                 ]))

        theta[phi > np.pi] = np.pi - theta[phi > np.pi]
        phi = helper.circular.remap(phi, np.pi, 0)

        # direction
        h, x = np.histogram(phi, hist_bin(180), density=True)
        x = x[:-1] + (x[1] - x[0]) / 2
        x = np.append(np.concatenate((x, x - np.pi), axis=0), x[0])
        h = np.append(np.concatenate((h, h), axis=0), h[0])
        h = h / np.max(h)
        h = h[np.logical_and(x >= -np.pi / 2, x < np.pi / 2)]
        x = x[np.logical_and(x >= -np.pi / 2, x < np.pi / 2)]
        data[f'{phase}_phi_x'] = np.rad2deg(x)
        data[f'{phase}_phi_h'] = h

        # inclination
        h, x = np.histogram(theta, hist_bin(180), density=True)
        x = np.pi / 2 - (x[:-1] + (x[1] - x[0]) / 2)  # np/2 for incl,
        # x = np.append(np.concatenate((x, x - np.pi), axis=0), x[0])
        # h = np.append(np.concatenate((h, h), axis=0), h[0])
        h = h / np.max(h)
        h = h[np.logical_and(x >= -np.pi / 2, x < np.pi / 2)]
        x = x[np.logical_and(x >= -np.pi / 2, x < np.pi / 2)]

        # append 0 line
        df.loc[len(df)] = 0

        data = pd.concat([
            pd.DataFrame({
                f'{phase}_incl_x': np.rad2deg(x),
                f'{phase}_incl_h': h
            }), data
        ],
                         axis=1)

    data.to_csv(os.path.join(out_path,
                             os.path.basename(file) + ".csv"),
                index=False,
                float_format='%.3f')


if __name__ == "__main__":

    if os.path.isdir(args.input):
        args.input = os.path.join(args.input, "cube_2pop.pkl")

    df = pd.read_pickle(args.input)

    files = df.files.tolist()
    with mp.Pool(processes=24) as pool:
        [_ for _ in tqdm(pool.imap_unordered(run, files), total=len(files))]
