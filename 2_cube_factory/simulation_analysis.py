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

out_file = "output/analysis/"
df_pm = pd.read_pickle(os.path.join(out_file, "pm.pkl"))
df_lap = pd.read_pickle(os.path.join(out_file, "lap.pkl"))

epa_dir = df_pm[(df_pm.f0 == 0.0) & pm.f1 == 0.0)].explode("epa_dir")
epa_ret = df_pm[(df_pm.f0 == 0.0) & (df_pm.f1 == 0.0)].explode("epa_ret")

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
        filepath=os.path.join(out_file, f"pm_omega_{omega}_psi_epa_dir.tikz"),
        extra_tikzpicture_parameters=tikzpicpara,
        encoding="utf-8",
    )

    if not flag:
        for psi in epa_dir.psi.unique():
            data = epa_dir[(epa_dir.omega == omega) &
                           (epa_dir.psi == psi)].epa_dir.to_numpy()

            data = helper.circular.remap(data, np.pi, 0)

            h, x = np.histogram(data, 18, density=True)

            x *= 180 / np.pi

            # h = np.append(h, h[0])
            # x = x + (x[1] - x[0]) / 2
            helper.tikz.direction_hist(
                x[:-1],
                h,
                os.path.join(out_file,
                             f"pm_omega_{omega}_psi_{psi}_epa_dir.tikz"),
                standalone=True)
    flag = True

    plt.clf()
    sns.boxplot(x="psi",
                y="epa_ret",
                data=epa_ret[epa_ret.omega == omega],
                orient='v',
                color=sns.color_palette("dark", 10)[i])
    tikzplotlib.clean_figure()
    tikzplotlib.save(
        filepath=os.path.join(out_file, f"pm_omega_{omega}_psi_epa_ret.tikz"),
        extra_tikzpicture_parameters=tikzpicpara,
        encoding="utf-8",
    )
