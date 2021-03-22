import numpy as np
import os
import argparse
import multiprocessing as mp
import pandas as pd
import tqdm

import fastpli.tools
import helper.spherical_interpolation
import polar_hist_to_tikz

import matplotlib.pyplot as plt

import odf

df = pd.read_pickle(
    os.path.join("output", "sim_120_ime_r_0.5", "analysis",
                 f"cube_2pop_simulation.pkl"))
df_ = pd.read_pickle(
    os.path.join("output", "sim_120_ime_r_0.5_", "analysis",
                 f"cube_2pop_simulation.pkl"))

df = pd.concat([df, df_])

df["trel_mean"] = df["rofl_trel"].apply(lambda x: np.mean(x))
df["ret_mean"] = df["epa_ret"].apply(lambda x: np.mean(x))
df["trans_mean"] = df["epa_trans"].apply(lambda x: np.mean(x))

df = df[df.microscope == "PM"]
df = df[df.species == "Vervet"]
df = df[df.model == "r"]
df = df[df.radius == 0.5]

df_ = df[df.f1_rot == 0]
df_ = df_[df_.f0_inc == 0]

fig, axs = plt.subplots(len(df_.psi.unique()),
                        len(df_.omega.unique()),
                        figsize=(25, 25),
                        subplot_kw=dict(projection='3d'))

print(sorted(df_.psi.unique()))
print(sorted(df_.omega.unique()))

for i, psi in enumerate(sorted(df_.psi.unique())):
    for j, omega in enumerate(sorted(df_.omega.unique())):
        df__ = df_[(df_.omega == omega) & (df_.psi == psi)]

        # print(len(df__))
        # print(df__)
        if len(df__) == 0:
            continue
        if len(df__) != 1:
            print(psi, omega, "not len==1:", len(df__))
        odf_table = odf.table(df__.rofl_dir.iloc[0], df__.rofl_inc.iloc[0])
        odf.plot(odf_table, axs[i, j])
        axs[i, j].set_title(f"psi: {psi:.2f}, omega: {omega:.2f}")

radius = df_.radius.unique()[0]
model = df_.model.unique()[0]
species = df_.species.unique()[0]
microscope = df_.microscope.unique()[0]

inc = df_.f0_inc.unique()[0]
rot = df_.f1_rot.unique()[0]

plt.tight_layout(pad=0, w_pad=-42, h_pad=0)
plt.savefig(
    f"test_r_{radius}_species_{species}_microscope_{microscope}_model_{model}_inc_{inc:.2f}_rot_{rot:.2f}.pdf",
    bbox_inches='tight')

#     break
# break
# pass
