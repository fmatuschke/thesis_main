import numpy as np
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


def _nx2_dat(x, y, file, info=None):

    with open(file, 'w') as f:

        if info:
            if not isinstance(info, list):
                info = [info]

            for i in info:
                f.write(f"%{i}\n")

        f.write(f"x,y\n")

        for i, j, in zip(x, y):
            f.write(f"{i:.3f},{j:.3f}\n")


def to_tikz(data,
            file,
            path_to_data=None,
            standalone=False,
            only_dat=False,
            info=None):

    file_path = os.path.dirname(file)
    file_base = os.path.basename(file)
    file_name, _ = os.path.splitext(file_base)
    file_pre = os.path.join(file_path, file_name)

    for i, d in enumerate(data):
        _nx2_dat(d[0, :], d[1, :], f"{file_pre}_{i}.dat", info)

    if only_dat:
        return

    if path_to_data:
        file_name = path_to_data + "/" + file_name

    with open(file, 'w') as f:
        if standalone:
            f.write("" \
                "\\documentclass[]{standalone}\n"\
                "\\usepackage{pgfplots}\n" \
                "\\usepgfplotslibrary{polar}\n" \
                "\\pgfplotsset{compat=1.17}\n" \
                "\\usepackage{siunitx}\n" \
                "\\begin{document}\n" \
                "\\tikzset{>=latex}\n" \
                "%\n" )
        f.write("" \
            "\\begin{tikzpicture}\n"
        )
        f.write("" \
            "\\begin{polaraxis}[\n" \
            "    domain=-90:90,\n" \
            "    ymin=0, ymax=1,\n" \
            "    xmin=-90, xmax=90,\n" \
            "    xtick={-90,-45,...,90},\n" \
            "    ytick=\empty,\n" \
            "]\n" \
            "\\addplot [thick, green!50!black]\n" \
            f"    table [x=x, y=y, col sep=comma] {{{file_name}_0.dat}};\n" \
            "\\addplot [thick, red, dashed]\n" \
            f"    table [x=x, y=y, col sep=comma] {{{file_name}_2.dat}};\n" \
            "\\end{polaraxis}\n"
        )
        f.write("%\n")
        f.write("" \
            "\\begin{scope}[shift={(-4,0)}]\n" \
            "\\begin{polaraxis}[\n" \
            "    domain=90:270,\n" \
            "    ymin=0, ymax=1,\n" \
            "    xmin=90, xmax=270,\n" \
            "    xtick={90,135,...,270},\n" \
            "    xticklabels={90, 45, 0, -45, -90},\n" \
            "    ytick=\empty,\n" \
            "]\n" \
            "\\addplot [thick, green!50!black]\n" \
            f"    table [x=x, y=y, col sep=comma] {{{file_name}_1.dat}};\n" \
            "\\addplot [thick, red, dashed]\n" \
            f"    table [x=x, y=y, col sep=comma] {{{file_name}_3.dat}};\n" \
            "\\end{polaraxis}\n"
            "\\end{scope}\n" \
        )
        f.write("" \
            "\\end{tikzpicture}\n" \
            )

        if standalone:
            f.write("%\n")
            f.write("\\end{document}\n")


if __name__ == "__main__":
    out_path = "output/images/voxel_size_simulation"
    os.makedirs(out_path, exist_ok=True)

    hist_bin = lambda n: np.linspace(0, np.pi, n + 1, endpoint=True)
    df = pd.read_pickle(os.path.join("output/vs__2/voxel_size_simulation.pkl"))
    for model in df.model.unique():
        fig, axs = plt.subplots(len(df.f0_inc.unique()),
                                len(df.voxel_size.unique()),
                                subplot_kw={'projection': 'polar'},
                                figsize=(20, 20))
        axs = np.atleast_2d(axs)
        for i, f0_inc in enumerate(sorted(df.f0_inc.unique())):
            for j, vs in enumerate(sorted(df.voxel_size.unique())):
                sub = (df.omega == 0.0) & (df.psi == 1.0) & (
                    df.f1_rot == 0.0) & (df.voxel_size == vs) & (
                        df.f0_inc == f0_inc) & (df.model == model)
                data = []
                df_sub = df[sub]
                if len(df_sub) > 1:
                    print("fooo", omega, psi)
                if len(df_sub) == 0:
                    continue

                phi = df_sub.explode("f_phi").f_phi.to_numpy(float)
                theta = df_sub.explode("f_theta").f_theta.to_numpy(float)

                theta[phi > np.pi] = np.pi - theta[phi > np.pi]
                phi = helper.circular.remap(phi, np.pi, 0)

                h, x = np.histogram(phi, hist_bin(4 * 18), density=True)
                x = x[:-1] + (x[1] - x[0]) / 2
                x = np.append(np.concatenate((x, x + np.pi), axis=0), x[0])
                h = np.append(np.concatenate((h, h), axis=0), h[0])
                h = h / np.max(h)
                data.append(np.vstack((np.rad2deg(x), h)))

                h, x = np.histogram(theta, hist_bin(4 * 18), density=True)
                x = np.pi - (np.pi / 2 -
                             (x[:-1] +
                              (x[1] - x[0]) / 2))  # np/2 for incl, - for plot
                h = h / np.max(h)
                data.append(np.vstack((np.rad2deg(x), h)))

                phi = df_sub.explode("epa_dir").epa_dir.to_numpy(float)
                phi = helper.circular.remap(phi, np.pi, 0)

                h, x = np.histogram(phi, hist_bin(4 * 18), density=True)
                x = x[:-1] + (x[1] - x[0]) / 2
                x = np.append(np.concatenate((x, x + np.pi), axis=0), x[0])
                h = np.append(np.concatenate((h, h), axis=0), h[0])
                h = h / np.max(h)
                data.append(np.vstack((np.rad2deg(x), h)))

                # theta data
                data.append(np.vstack(([0, f0_inc], [0, 1])))

                to_tikz(
                    data,
                    os.path.join(
                        out_path,
                        f"voxel_size_simulation_f0_inc_{f0_inc:.2f}_vs_{vs:.4f}.tikz"
                    ),
                    path_to_data="\\currfiledir",
                    standalone=False)
