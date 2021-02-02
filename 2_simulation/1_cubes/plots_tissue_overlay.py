import h5py
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import os
import tqdm
import fastpli.analysis

import pretty_errors

import models

sim_path = "2_simulation/1_cubes/output/sim_120_ime/"

df_model = pd.read_pickle(
    os.path.join(os.getcwd(), "../..",
                 "1_model/1_cubes/output/cube_2pop_120/cube_2pop.pkl"))
# print("df_model:", df_model.columns)

# df_model_hist = pd.read_pickle(
#     os.path.join(os.getcwd(), "../..",
#                  "1_model/1_cubes/output/cube_2pop_120/hist/cube_2pop.pkl"))
# print("df_model_hist:", df_model_hist.columns)

df_sim = pd.read_pickle(
    os.path.join(os.getcwd(), "../..", sim_path,
                 "analysis/cube_2pop_simulation.pkl"))
# print("df_sim:", df_sim.columns)

# df_sim_sch = pd.read_pickle(
#     os.path.join(os.getcwd(), "../..", sim_path,
#                  "analysis/cube_2pop_simulation_schilling.pkl"))
# print("df_sim_sch:", df_sim_sch.columns)


def get_file_from_parameter(psi="0.30",
                            omega="30.00",
                            r="0.50",
                            incl="30.00",
                            rot="0.00"):

    model_str = os.path.join(
        os.getcwd(), "../..", "1_model/1_cubes/output/cube_2pop_120",
        f"cube_2pop_psi_{psi}_omega_{omega}_r_{r}_v0_120_.solved.h5")

    sim_str = os.path.join(
        os.getcwd(), "../..", sim_path,
        f"cube_2pop_psi_{psi}_omega_{omega}_r_{r}_v0_120_.solved_vs_0.1250_inc_{incl}_rot_{rot}.h5"
    )

    return model_str, sim_str


def get_rotations(omega):
    return models.omega_rotations(omega)


def to_str(value):
    if isinstance(value, str):
        return value
    else:
        return f"{value:.2f}"


def get_sim(microscope="PM",
            species="Vervet",
            model="r",
            psi="0.30",
            omega="30.00",
            radius="0.50",
            incl="30.00",
            rot="0.00"):

    _, sim_str = get_file_from_parameter(to_str(psi), to_str(omega),
                                         to_str(radius), to_str(incl),
                                         to_str(rot))

    return h5py.File(sim_str)[f"{microscope}/{species}/{model}"]


def get_tissue(psi="0.30",
               omega="30.00",
               radius="0.50",
               incl="30.00",
               rot="0.00"):

    _, sim_str = get_file_from_parameter(to_str(psi), to_str(omega),
                                         to_str(radius), to_str(incl),
                                         to_str(rot))

    path, file = os.path.split(sim_str)
    return h5py.File(os.path.join(path, "tissue",
                                  f"{file[:-3]}.tissue.h5"))["tissue"][...]


def plot(p):

    p = p[1]

    radius = p.radius
    incl = p.f0_inc
    rot = p.f1_rot
    psi = p.psi
    omega = p.omega

    h5_sim = get_sim(microscope, species, model, psi, omega, radius, incl, rot)
    tissue = get_tissue(psi, omega, radius, incl, rot)

    data = h5_sim["analysis/epa/0/retardation"][...]

    tissue_b = tissue.flatten().astype(np.float32)
    # tissue_b[tissue_b == 0] = np.nan
    tissue_b -= 1
    tissue_b //= 2
    tissue_b += 1
    tissue_b.shape = tissue.shape

    extent = (0, data.shape[0], 0, data.shape[1])
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    fig.suptitle(
        f'{microscope}_{species}_{model}_psi_{to_str(psi)}_' +
        f'omega_{to_str(omega)}_r_{to_str(radius)}_f0_{to_str(incl)}_f1_{to_str(rot)}',
        fontsize=24)

    for i, name in enumerate(["transmittance", "direction", "retardation"]):
        data = h5_sim[f"analysis/epa/0/{name}"][...]

        if name == "direction":
            data[data > np.pi / 2] -= np.pi
            data = np.rad2deg(data)

        axs[0, i].imshow(
            np.flip(np.mean(tissue_b, -1).T, 0),
            cmap=plt.cm.gray,
            interpolation='nearest',
            extent=extent,
        )
        pcm = axs[0, i].imshow(
            np.flip(data.T, 0),
            cmap=plt.cm.viridis,
            alpha=.5,
            interpolation='nearest',
            extent=extent,
        )
        fig.colorbar(pcm, ax=axs[0, i], shrink=0.75)

    for i, name in enumerate(["inclination", "t_rel", "fom"]):
        if name == "fom":
            data = fastpli.analysis.images.fom_hsv_black(
                h5_sim[f"analysis/rofl/direction"][...],
                h5_sim[f"analysis/rofl/inclination"][...])
            data = np.swapaxes(data, 0, 1)
            axs[1, i].imshow(np.flip(data, 0))
        else:
            data = h5_sim[f"analysis/rofl/{name}"][...]

            if name == "inclination":

                direction = h5_sim[f"analysis/rofl/direction"][...]
                bmap = direction > np.pi / 2
                # bmap = np.logical_and(data < 0, direction > np.pi / 2)
                data[bmap] *= -1

                data = np.rad2deg(data)
                pcm = axs[1, i].imshow(
                    np.flip(data.T, 0),
                    cmap=plt.cm.viridis,
                    # alpha=.5,
                    interpolation='nearest',
                    extent=extent,
                )
            else:  # trel
                pcm = axs[1, i].imshow(
                    np.flip(data.T, 0),
                    cmap=plt.cm.viridis,
                    vmin=0,
                    vmax=1,
                    # alpha=.5,
                    interpolation='nearest',
                    extent=extent,
                )

            fig.colorbar(pcm, ax=axs[1, i], shrink=0.75)
            axs[1, i].imshow(
                np.flip(np.mean(tissue_b, -1).T, 0),
                cmap=plt.cm.gray,
                alpha=.5,
                interpolation='nearest',
                extent=extent,
            )

    plt.savefig(
        os.path.join(
            os.getcwd(), "../..", sim_path, "tissue",
            f"tissue_overlay_{microscope}_{species}_{model}_{to_str(psi)}_" +
            f"{to_str(omega)}_{to_str(radius)}_{to_str(incl)}_{to_str(rot)}_.pdf"
        ))

    plt.close(fig)


############### PLOT ##############

microscope = "PM"
species = "Vervet"
model = "r"

df_ = df_sim[[
    "radius",
    "psi",
    "omega",
    "f0_inc",
    "f1_rot",
]].drop_duplicates()

df_ = df_[df_.f1_rot == 0]

with mp.Pool(processes=48) as pool:
    [
        _ for _ in tqdm.tqdm(pool.imap_unordered(plot, df_.iterrows()),
                             total=len(df_),
                             smoothing=0)
    ]