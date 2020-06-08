import numpy as np
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

import helper.tikz
import helper.circular
import fastpli.io
import fastpli.analysis
import fastpli.tools
import fastpli.objects

out_file = "output/analysis/"
for device in ["pm", "lap"]:
    df = pd.read_pickle(os.path.join(out_file, f"{device}.pkl"))

    f0_inc = 0.0
    f1_rot = 0.0
    epa_dir = df[(df.f0 == f0_inc) & (df.f1 == f1_rot)].explode("epa_dir")
    epa_ret = df[(df.f0 == f0_inc) & (df.f1 == f1_rot)].explode("epa_ret")

    tikzpicpara = ["trim axis left", "trim axis right", "baseline"]

    flag = False
    for i, omega in tqdm(enumerate(epa_dir.omega.unique())):
        plt.clf()
        sns.boxplot(x="psi",
                    y="epa_dir",
                    data=epa_dir[epa_dir.omega == omega],
                    orient='v',
                    color=sns.color_palette("dark", 10)[i])
        tikzplotlib.clean_figure()
        tikzplotlib.save(
            filepath=os.path.join(out_file,
                                  f"{device}_omega_{omega}_psi_epa_dir.tikz"),
            extra_tikzpicture_parameters=tikzpicpara,
            encoding="utf-8",
        )

        if not flag:
            for psi in epa_dir.psi.unique():
                data = epa_dir[(epa_dir.omega == omega) &
                               (epa_dir.psi == psi)].epa_dir.to_numpy(
                                   dtype=float)

                data = helper.circular.remap(data, np.pi, 0)

                h, x = np.histogram(data, 18, density=True)

                helper.tikz.direction_hist(
                    np.rad2deg(x[:-1] + x[1] - x[0]),
                    h,
                    os.path.join(
                        out_file,
                        f"{device}_omega_{omega}_psi_{psi}_epa_dir.tikz"),
                    standalone=True)

                # ground truth
                fbs = fastpli.io.fiber_bundles.load(
                    f"../data/models/cube_2pop_psi_{psi:.2f}_omega_{omega:.2f}_.solved.h5"
                )
                rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
                rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
                rot = np.dot(rot_inc, rot_phi)
                fbs = fastpli.objects.fiber_bundles.Rotate(fbs, rot)
                phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs)

                ax = plt.subplot(111, projection='polar')

        flag = True
        sys.exit()

        plt.clf()
        sns.boxplot(x="psi",
                    y="epa_ret",
                    data=epa_ret[epa_ret.omega == omega],
                    orient='v',
                    color=sns.color_palette("dark", 10)[i])
        tikzplotlib.clean_figure()
        tikzplotlib.save(
            filepath=os.path.join(out_file,
                                  f"{device}_omega_{omega}_psi_epa_ret.tikz"),
            extra_tikzpicture_parameters=tikzpicpara,
            encoding="utf-8",
        )
