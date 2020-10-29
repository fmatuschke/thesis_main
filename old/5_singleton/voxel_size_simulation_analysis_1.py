import numpy as np
import itertools
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of files.")
args = parser.parse_args()


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
            "\\addplot [thick, blue, dash dot]\n" \
            f"    table [x=x, y=y, col sep=comma] {{{file_name}_4.dat}};\n" \
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
            "\\addplot [thick, blue, dash dot]\n" \
            f"    table [x=x, y=y, col sep=comma] {{{file_name}_5.dat}};\n" \
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

    hist_phi = lambda n: np.linspace(
        -np.pi / 2, np.pi / 2, n + 1, endpoint=True)
    hist_phi_inc = lambda n: np.linspace(
        np.pi / 2, 3 * np.pi / 2, n + 1, endpoint=True)
    df = pd.read_pickle(args.input)
    for f0_inc in tqdm(sorted(df.f0_inc.unique())):
        for vs in sorted(df.voxel_size.unique()):
            sub = (df.omega == 0.0) & (df.psi == 1.0) & (df.f1_rot == 0.0) & (
                df.voxel_size == vs) & (df.f0_inc == f0_inc) & (df.m > 0)
            data = []
            df_sub = df[sub]

            phi = df_sub.explode("f_phi").f_phi.to_numpy(float)
            theta = df_sub.explode("f_theta").f_theta.to_numpy(float)

            phi, phi_inc = helper.circular.orientation_sph_plot(phi, theta)
            h, x = np.histogram(phi, hist_phi(4 * 18), density=True)
            x = x[:-1] + (x[1] - x[0]) / 2
            h = h / np.amax(h)
            data.append(np.vstack((np.rad2deg(x), h)))
            h, x = np.histogram(phi_inc, hist_phi_inc(4 * 18), density=True)
            x = x[:-1] + (x[1] - x[0]) / 2
            h = h / np.amax(h)
            data.append(np.vstack((np.rad2deg(x), h)))

            phi = df_sub[df_sub.model == 'r'].explode(
                "epa_dir").epa_dir.to_numpy(float)
            phi[phi > np.pi] -= 2 * np.pi
            phi[phi > np.pi / 2] -= np.pi
            h, x = np.histogram(phi, hist_phi(10 * 18), density=True)
            x = x[:-1] + (x[1] - x[0]) / 2
            h = h / np.amax(h)
            data.append(np.vstack((np.rad2deg(x), h)))
            data.append(np.vstack(([180 - f0_inc, 180 - f0_inc], [0, 1])))

            phi = df_sub[df_sub.model == 'p'].explode(
                "epa_dir").epa_dir.to_numpy(float)
            phi[phi > np.pi] -= 2 * np.pi
            phi[phi > np.pi / 2] -= np.pi
            h, x = np.histogram(phi, hist_phi(4 * 18), density=True)
            x = x[:-1] + (x[1] - x[0]) / 2
            h = h / np.amax(h)
            data.append(np.vstack((np.rad2deg(x), h)))
            data.append(np.vstack(([180 - f0_inc, 180 - f0_inc], [0, 1])))

            to_tikz(
                data,
                os.path.join(
                    out_path,
                    f"voxel_size_simulation_f0_inc_{f0_inc:.2f}_vs_{vs:.4f}.tikz"
                ),
                path_to_data="\\currfiledir",
                standalone=False)
