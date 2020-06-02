import numpy as np
import h5py
import os
import sys
import glob

import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib
from tqdm import tqdm

import fastpli.tools
import helper.circular
import helper.spherical_interpolation

sim_path = "output/simulation/*.h5"
ana_file = "output/analysis/"
out_file = "output/images/spheres/"

microscope = "PM"
model = "r"
df = pd.read_pickle(
    os.path.join(ana_file,
                 f"cube_2pop_simulation_{microscope}_model_{model}_.pkl"))

df_acc = pd.read_pickle(
    os.path.join(
        ana_file,
        f"cube_2pop_simulation_{microscope}_model_{model}_schilling.pkl"))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

f0_inc = 60
psi = 0.3

sub = (df_acc.psi == psi) & (df_acc.f0_inc == f0_inc)
f1_rot = df_acc[sub].f1_rot.to_numpy(float)
omega = df_acc[sub].omega.to_numpy(float)
data = df_acc[sub].acc.to_numpy(float)

print((data == 0).any())

# get points on sphere
phi = []
theta = []
for f1, om in zip(f1_rot, omega):
    v = np.array([np.cos(np.deg2rad(om)), np.sin(np.deg2rad(om)), 0])
    rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
    rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1))
    rot = np.dot(rot_inc, rot_phi)
    v = np.dot(rot, v)
    theta.extend([np.arccos(v[2])])
    phi.extend([np.arctan2(v[1], v[0])])

phi_ = phi.copy()
theta_ = theta.copy()
data_ = data.copy()

# apply symmetries
phi = np.array(phi)
theta = np.array(theta)

phi = np.concatenate((phi, -phi), axis=0)
theta = np.concatenate((theta, theta), axis=0)
data = np.concatenate((data, data), axis=0)

phi = np.concatenate((phi, phi), axis=0)
theta = np.concatenate((theta, np.pi + theta), axis=0)
data = np.concatenate((data, data), axis=0)

# rm multiple
phi, theta = helper.spherical_interpolation.remap_sph_angles(phi, theta)
tmp = np.concatenate(
    (np.atleast_2d(phi), np.atleast_2d(theta), np.atleast_2d(data)), axis=0)
tmp = np.unique(tmp, axis=1)
phi, theta, data = tmp[0, :], tmp[1, :], tmp[2, :]

# interplate mesh on sphere
x, y, z, data_i = helper.spherical_interpolation.on_mesh(
    phi, theta, data, 40, 40)

ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(data_i))

x = np.multiply(np.cos(phi_), np.sin(theta_)) * 1.05
y = np.multiply(np.sin(phi_), np.sin(theta_)) * 1.05
z = np.cos(theta_) * 1.05

sc = ax.scatter(x,
                y,
                z,
                marker='o',
                s=50,
                c=data_,
                alpha=1,
                vmin=0,
                vmax=1,
                cmap="viridis")
plt.colorbar(sc)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.view_init(30, 30)

tikzplotlib.clean_figure()
tikzplotlib.save(
    filepath=os.path.join(
        out_file,
        f"sphere_{microscope}_model_{model}_psi_{psi:1f}_f0inc_{f0_inc:1f}.tikz"
    ),
    # extra_tikzpicture_parameters=tikzpicpara,
    encoding="utf-8",
)
