#%%
import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import tqdm
import seaborn as sns

import fastpli.analysis
import models
import helper.circular

#%%
sim_path = "output/sim_120_ime_r_0.5"
df = pd.read_pickle(
    os.path.join(sim_path, "analysis", f"cube_2pop_simulation.pkl"))

df = df[df.microscope == "PM"]
df = df[df.species == "Vervet"]
df = df[df.model == "r"]
df = df[df.radius == 0.5]
df = df[df.psi == 1.0]

df["trel_mean"] = df["rofl_trel"].apply(lambda x: np.mean(x))
df["ret_mean"] = df["epa_ret"].apply(lambda x: np.mean(x))
df["trans_mean"] = df["epa_trans"].apply(lambda x: np.mean(x))


#%%
def get_file_from_parameter(psi="0.30",
                            omega="30.00",
                            radius="0.50",
                            incl="30.00",
                            rot="0.00"):

    model_str = os.path.join(
        "../..", "1_model/1_cubes/output/cube_2pop_120",
        f"cube_2pop_psi_{psi}_omega_{omega}_r_{radius}_v0_120_.solved.h5")

    sim_str = os.path.join(
        sim_path,
        f"cube_2pop_psi_{psi}_omega_{omega}_r_{radius}_v0_120_.solved_vs_0.1250_inc_{incl}_rot_{rot}.h5"
    )

    return model_str, sim_str


def get_file_from_series(df):

    return get_file_from_parameter(psi=f"{df.psi:.2f}",
                                   omega=f"{df.omega:.2f}",
                                   radius=f"{df.radius:.2f}",
                                   incl=f"{df.f0_inc:.2f}",
                                   rot=f"{df.f1_rot:.2f}")


#%%

fig, axs = plt.subplots(nrows=len(df),
                        ncols=2,
                        subplot_kw=dict(projection="polar"),
                        figsize=(5, 10))

for (_, row), ax in zip(df.sort_values("f0_inc").iterrows(), axs):
    phi, theta = fastpli.analysis.orientation.remap_orientation(
        row.rofl_dir, np.pi / 2 - row.rofl_inc)
    ax[0].hist2d(
        phi,
        np.rad2deg(theta),
        bins=[np.linspace(0, 2 * np.pi, 36 + 1),
              np.linspace(0, 90, 9 + 1)],
        norm=mpl.colors.LogNorm())
    ax[0].set_title(f"incl:{row.f0_inc}")
    #
    #
    phi, theta = models.ori_from_file(
        get_file_from_series(row)[0], row.f0_inc, row.f1_rot, 60)
    phi, theta = fastpli.analysis.orientation.remap_orientation(phi, theta)
    ax[1].hist2d(
        phi,
        np.rad2deg(theta),
        bins=[np.linspace(0, 2 * np.pi, 36 + 1),
              np.linspace(0, 90, 9 + 1)],
        norm=mpl.colors.LogNorm())
    ax[1].set_title(f"incl:{row.f0_inc}")

# plt.tight_layout()
plt.tight_layout(pad=0, w_pad=0, h_pad=0)

plt.savefig(f"output/{os.path.basename(__file__)[:-3]}_hist.pdf")
# %%
# fig, axs = plt.subplots(1, 1)

sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

df_ = df.apply(pd.Series.explode).reset_index()

phi, theta = df_["rofl_dir"].to_numpy(
    float), np.pi / 2 - df_["rofl_inc"].to_numpy(float)
phi, theta = fastpli.analysis.orientation.remap_orientation(phi, theta)
df_["rofl_dir"], df_["rofl_inc"] = np.rad2deg(phi), np.rad2deg(np.pi / 2 -
                                                               theta)

df_["epa_dir"] = np.rad2deg(df_["epa_dir"].to_numpy(float))

for name in [
        "rofl_inc", "rofl_dir", "rofl_trel", "epa_trans", "epa_dir", "epa_ret"
]:
    if "dir" in name:
        df_[name] = helper.circular.remap(df_[name], 90, -90)

    # Draw a nested boxplot to show bills by day and time
    fig, axs = plt.subplots(1, 1)
    sns.boxplot(
        x="f0_inc",
        y=name,
        # hue="smoker",
        # palette=["m", "g"],
        data=df_)
    sns.despine(offset=10, trim=True)
    # plt.tight_layout()
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(f"output/{os.path.basename(__file__)[:-3]}_{name}.pdf")

# %%
