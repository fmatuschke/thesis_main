#! /usr/bin/env python3

import pretty_errors

import numpy as np
import multiprocessing as mp
import argparse
import os
import sys
import glob
import warnings

import tqdm
import pandas as pd

import fibers
import fastpli.analysis

import matplotlib.pyplot as plt

df_model = pd.read_pickle(
    os.path.join(os.getcwd(),
                 f"1_model/1_cubes/output/cube_2pop_120/cube_2pop.pkl"))

df_model = df_model[df_model.state != "init"]

f0_inc = 0
f1_rot = 0

sub = (df_model.psi == 0.3) & (df_model.omega == 30) & (df_model.radius == 1.0)

print(os.getcwd())
print(df_model[sub].fiber.iloc[0])
print(os.path.join(os.getcwd(), "1_model/1_cubes/",
                   df_model[sub].fiber.iloc[0]))

phi, theta = fibers.ori_from_file(
    os.path.join(os.getcwd(), "1_model/1_cubes/", df_model[sub].fiber.iloc[0]),
    f0_inc, f1_rot)

_, ax = plt.subplots(subplot_kw=dict(projection="polar"))
h, x, y, pc = fastpli.analysis.orientation.histogram(phi,
                                                     theta,
                                                     ax=ax,
                                                     n_phi=30,
                                                     n_theta=15,
                                                     weight_area=True)

print(np.rad2deg(x))
print(np.rad2deg(y))
print(x.shape)
print(y.shape)
print(h.shape)

cbar = plt.colorbar(pc, ax=ax)
cbar.ax.set_title('#')
ax.set_rmax(90)
ax.set_rticks(range(0, 90, 10))
ax.set_rlabel_position(22.5)
ax.set_yticklabels([])
ax.grid(True)

# # save 2d hist
with open("test.dat", "w") as f:

    x_axis = x[:-1] + (x[1] - x[0]) / 2  # x over 360 for latex visualization
    y_axis = y[:-1] + (y[1] - y[0]) / 2
    # H = np.vstack([h, h[0, :]]).T
    # H = H / np.sum(H.ravel())
    H = h.T / np.sum(h.ravel())

    print(np.rad2deg(x_axis))
    print(np.rad2deg(y_axis))
    print(x_axis.shape)
    print(y_axis.shape)
    print(h.shape)
    print(H.shape)

    X, Y = np.meshgrid(np.rad2deg(x_axis), np.rad2deg(y_axis))

    print(X.shape)
    print(Y.shape)

    # norm
    # H /= np.amax(H)

    for h_array, x_array, y_array in zip(H, X, Y):
        for h, x, y in zip(h_array, x_array, y_array):
            f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
        f.write('\n')

plt.show()
