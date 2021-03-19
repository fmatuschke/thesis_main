#%%
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

        print(len(df__))
        # print(df__)
        if len(df__) == 1:
            odf_table = odf.table(df__.rofl_dir.iloc[0], df__.rofl_inc.iloc[0])
            odf.plot(odf_table, axs[i, j])
            # print(odf_table)

plt.savefig("test.pdf")

#     break
# break
# pass

# %%
