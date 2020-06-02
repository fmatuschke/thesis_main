import numpy as np
import h5py
import os
import sys
import glob
import subprocess
import itertools

import pandas as pd
import tikzplotlib
from tqdm import tqdm

import fastpli.tools
import helper.circular
import helper.tikz
import helper.spherical_interpolation

sim_path = "output/simulation/*.h5"
ana_file = "output/analysis/"
out_file = "output/images/spheres/"

os.makedirs(out_file, exist_ok=True)

for microscope, model in tqdm(list(itertools.product(["PM", "LAP"],
                                                     ["r", "p"]))):

    df = pd.read_pickle(
        os.path.join(ana_file,
                     f"cube_2pop_simulation_{microscope}_model_{model}_.pkl"))

    df_acc = pd.read_pickle(
        os.path.join(
            ana_file,
            f"cube_2pop_simulation_{microscope}_model_{model}_schilling.pkl"))

    for psi in tqdm(df.psi.unique()):
        for f0_inc in tqdm(df.f0_inc.unique()):

            file_name = f"sphere_{microscope}_model_{model}_psi_{psi:.1f}_f0_inc_{f0_inc:.1f}"
            # if os.path.isfile(f"{os.path.join(out_file,file_name)}.pdf"):
            #     continue

            sub = (df_acc.psi == psi) & (df_acc.f0_inc == f0_inc)
            f1_rot = df_acc[sub].f1_rot.to_numpy(float)
            omega = df_acc[sub].omega.to_numpy(float)
            data = df_acc[sub].acc.to_numpy(float)

            # get points on sphere
            phi = []
            theta = []
            for f1, om in zip(f1_rot, omega):
                v = np.array(
                    [np.cos(np.deg2rad(om)),
                     np.sin(np.deg2rad(om)), 0])
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
            phi, theta = helper.spherical_interpolation.remap_sph_angles(
                phi, theta)
            tmp = np.concatenate(
                (np.atleast_2d(phi), np.atleast_2d(theta), np.atleast_2d(data)),
                axis=0)
            tmp = np.unique(tmp, axis=1)
            phi, theta, data = tmp[0, :], tmp[1, :], tmp[2, :]

            # interplate mesh on sphere
            x, y, z, data_i = helper.spherical_interpolation.on_mesh(
                phi, theta, data, 40, 40)

            x2 = np.multiply(np.cos(phi_), np.sin(theta_)) * 1.05
            y2 = np.multiply(np.sin(phi_), np.sin(theta_)) * 1.05
            z2 = np.cos(theta_) * 1.05

            helper.tikz.sphere(x,
                               y,
                               z,
                               data_i,
                               f"{os.path.join(out_file,file_name)}.tikz",
                               x2,
                               y2,
                               z2,
                               data,
                               standalone=True)

            subprocess.run(
                f"cd {out_file} && pdflatex -interaction=nonstopmode {file_name}.tikz && rm {file_name}.aux {file_name}.log",
                shell=True,
                stdout=subprocess.DEVNULL,
                check=True)
