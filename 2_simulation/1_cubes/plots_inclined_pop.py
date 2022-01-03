#%%
import collections
import itertools
import multiprocessing as mp
import os

import fastpli.analysis
import helper.circular
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from scipy.stats import circmean

import models
import parameter

#%%

CONFIG = parameter.get_tupleware()

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

MODEL = 'cube_2pop_135_rc1'
DATASET = 'cube_2pop_135_rc1_inclined'

df = pd.read_pickle(
    os.path.join(FILE_PATH, 'output', DATASET,
                 'analysis/cube_2pop_simulation.pkl'))

df = df[df.microscope == "PM"]
df = df[df.species == "Vervet"]
df = df[df.model == "r"]
df = df[df.radius == 0.5]
# df = df[df.psi == 0.3]
# df = df[df.psi == 0.5]
# psi = df.psi.unique()[-1]

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
        THESIS, f"1_model/1_cubes/output/{MODEL}",
        f"cube_2pop_psi_{psi}_omega_{omega}_r_{radius}_v0_135_.solved.h5")

    sim_str = os.path.join(
        FILE_PATH, DATASET,
        f"cube_2pop_psi_{psi}_omega_{omega}_r_{radius}_v0_135_.solved_vs_0.1000_inc_{incl}_rot_{rot}.h5"
    )

    return model_str, sim_str


def get_file_from_series(df):

    return get_file_from_parameter(psi=f"{df.psi:.2f}",
                                   omega=f"{df.omega:.2f}",
                                   radius=f"{df.radius:.2f}",
                                   incl=f"{df.f0_inc:.2f}",
                                   rot=f"{df.f1_rot:.2f}")


#%%
def calc_omega(p, t):
    v0 = np.array([np.cos(p) * np.sin(t), np.sin(p) * np.sin(t), np.cos(t)])
    v1 = v0[:, 0].copy()

    for v in v0.T[1:, :]:
        s = np.dot(v1, v)

        if s > 0:
            v1 += v
        else:
            v1 -= v
    v1 /= np.linalg.norm(v1)

    # print(v1)

    data = np.empty(v0.shape[1])

    for i in range(v0.shape[1]):
        d = np.abs(np.dot(v0[:, i], v1))  # because orientation
        data[i] = np.arccos(d)

    return data
    # return v1, np.mean(data), np.std(data), np.quantile(data, [0.25, 0.5, 0.75])


asd = []
for i, row in df.iterrows():
    # row = df.iloc[i]
    phi, theta = fastpli.analysis.orientation.remap_orientation(
        row.rofl_dir, np.pi / 2 - row.rofl_inc)

    asd.append(np.rad2deg(calc_omega(phi, theta)))
df['domega'] = asd

#%%

#%%
os.makedirs(os.path.join(FILE_PATH, 'output', DATASET, "hist"), exist_ok=True)
if True:
    for _, row in tqdm.tqdm(df.sort_values("omega").iterrows(), total=len(df)):
        phi, theta = fastpli.analysis.orientation.remap_orientation(
            row.rofl_dir, np.pi / 2 - row.rofl_inc)

        # fig, ax = plt.subplots(nrows=1,
        #                        ncols=2,
        #                        subplot_kw=dict(projection="polar"),
        #                        figsize=(3.5, 2))
        # fig.suptitle(f"omega:{row.omega}")
        #
        # simulation values
        h, x, y, _ = fastpli.analysis.orientation.histogram(
            phi,
            theta,
            # ax=ax[0],
            n_phi=36 * 2,
            n_theta=18,
            weight_area=True,
            fun=lambda x: np.log(x + 1),
            cmap='cividis')
        #
        # 2d hist
        with open(
                os.path.join(
                    FILE_PATH, 'output', DATASET, "hist",
                    f"sim_hists_p_{row.psi:.1f}_o_{row.omega:.1f}_r_{row.radius:.1f}_f0_{row.f0_inc:.1f}_f1_{row.f1_rot:.1f}.dat"
                ), "w") as f:

            H = h
            x_axis = x
            y_axis = y

            # for pgfplots matrix plot*
            x_axis = x_axis[:-1] + (x_axis[1] - x_axis[0]) / 2
            y_axis = y_axis[:-1] + (y_axis[1] - y_axis[0]) / 2
            H = H.T / np.sum(H.ravel())

            X, Y = np.meshgrid(np.rad2deg(x_axis), np.rad2deg(y_axis))
            for h_array, x_array, y_array in zip(H, X, Y):
                for h, x, y in zip(h_array, x_array, y_array):
                    if y <= 90:
                        f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
                f.write('\n')

        # GT
        phi, theta = models.ori_from_file(
            get_file_from_series(row)[0], row.f0_inc, row.f1_rot,
            CONFIG.simulation.voi)

        # to tex
        h, x, y, _ = fastpli.analysis.orientation.histogram(phi,
                                                            theta,
                                                            n_phi=36 * 2,
                                                            n_theta=18,
                                                            weight_area=True)

        # 2d hist
        with open(
                os.path.join(
                    FILE_PATH, 'output', DATASET, "hist",
                    f"gt_hists_p_{row.psi:.1f}_o_{row.omega:.1f}_r_{row.radius:.1f}_f0_{row.f0_inc:.1f}_f1_{row.f1_rot:.1f}.dat"
                ), "w") as f:

            H = h
            x_axis = x
            y_axis = y

            # for pgfplots matrix plot*
            x_axis = x_axis[:-1] + (x_axis[1] - x_axis[0]) / 2
            y_axis = y_axis[:-1] + (y_axis[1] - y_axis[0]) / 2
            H = H.T / np.sum(H.ravel())

            X, Y = np.meshgrid(np.rad2deg(x_axis), np.rad2deg(y_axis))
            for h_array, x_array, y_array in zip(H, X, Y):
                for h, x, y in zip(h_array, x_array, y_array):
                    if y <= 90:
                        f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
                f.write('\n')

# %%
# fig, axs = plt.subplots(1, 1)

sns.set_theme(style="ticks", palette="pastel")

df_ = df.apply(pd.Series.explode).reset_index()

phi, theta = df_["rofl_dir"].to_numpy(
    float), np.pi / 2 - df_["rofl_inc"].to_numpy(float)
# phi, theta = fastpli.analysis.orientation.remap_orientation(phi, theta)
theta[phi > 3 / 4 * np.pi] = np.pi - theta[phi > 3 / 4 * np.pi]
phi[phi > 3 / 4 * np.pi] -= np.pi
df_["rofl_dir"], df_["rofl_inc"] = np.rad2deg(phi), np.rad2deg(np.pi / 2 -
                                                               theta)
df_["epa_dir"] = np.rad2deg(df_["epa_dir"].to_numpy(float))

for omega in df_.omega.unique():
    theta = df_.loc[df_.omega == omega, "rofl_inc"]
    t_mean = circmean(theta, 180, -180)
    theta[theta < t_mean - 90] = theta[theta < t_mean - 90] + 180
    df_.loc[df_.omega == omega, "rofl_inc"] = theta

for omega in df_.omega.unique():
    for psi in df_.psi.unique():
        for name in ["epa_dir", "rofl_dir"]:
            df_.loc[(df_['omega'] == omega) & (df_['psi'] == psi),
                    name] = helper.circular.remap(
                        df_[(df_['omega'] == omega) &
                            (df_['psi'] == psi)][name], 90, -90)

#%%

for psi in df_.psi.unique():
    df__ = df_[df_.psi == psi]

    dff = pd.DataFrame()

    for o in df__.omega.unique():
        for n in [
                "rofl_inc", "rofl_dir", "rofl_trel", "epa_trans", "epa_ret",
                "domega"
        ]:
            # print(psi, n, o)
            # print(df__[df__.omega == o][n])
            dff[f'{n}_{o}'] = df__[df__.omega == o][n].to_numpy()

    dff.to_csv(os.path.join(
        FILE_PATH, 'output', DATASET, 'analysis',
        f"{DATASET}_{os.path.basename(__file__)[:-3]}_psi_{psi}.csv"),
               index=False)

    df_theo = pd.DataFrame()
    N = 10
    x = np.linspace(0, 90, N, True)
    y = np.repeat(np.linspace(0, 90, N)[:, None], 10, 1)
    y[:, 0:int(psi * 10)] = 0
    y = circmean(y, 90, -90, axis=1)
    df_theo['x'] = np.linspace(0, 9, N, True)
    df_theo['y'] = y

    df_theo.to_csv(os.path.join(
        FILE_PATH, 'output', DATASET, 'analysis',
        f"{DATASET}_{os.path.basename(__file__)[:-3]}_psi_{psi}_theo_incl.csv"),
                   index=False)

#%%
if False:
    for name in tqdm.tqdm(
        ["rofl_inc", "rofl_dir", "rofl_trel", "epa_trans", "epa_ret",
         "domega"]):
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

        if "epa_ret" == name:
            x = np.linspace(0, np.pi, 42)
            y_max = np.mean(df_[df_.f0_inc == 0].epa_ret)
            y_min = np.mean(df_[df_.f0_inc == 90].epa_ret)
            y = (np.cos(x) + 1) / 2
            # plt.plot(x / np.pi * 3, y, linewidth=4.2)
            plt.plot(x / np.pi * (len(df) - 1), y * y_max, linewidth=4.2)
            # plt.plot(x / np.pi * (len(df) - 1), y * 0.85, linewidth=4.2)
            # plt.plot(x / np.pi * (len(df) - 1),
            #          y * (y_max - y_min) + y_min,
            #          linewidth=4.2)

        if "rofl_inc" == name:
            x = [0, (len(df) - 1)]
            y = [0, 90]
            plt.plot(x, y, linewidth=4.2)
            # axs.set_ylim(-15, 105)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.savefig(
            os.path.join(
                FILE_PATH,
                f"output/{DATASET}_{os.path.basename(__file__)[:-3]}_{name}.pdf"
            ))
# %%
