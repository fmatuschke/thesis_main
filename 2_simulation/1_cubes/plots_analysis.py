import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import tqdm
import itertools
import multiprocessing as mp

import fastpli.analysis
import fastpli.io
import models
import odf

sim_path = "2_simulation/1_cubes/output/sim_120_ime/"

df_sim = pd.read_pickle(
    os.path.join(os.getcwd(), "../..", sim_path,
                 "analysis/cube_2pop_simulation.pkl"))


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


def to_str(value):
    if isinstance(value, str):
        return value
    else:
        return f"{value:.2f}"


def get_model(psi="0.30",
              omega="30.00",
              radius="0.50",
              incl="30.00",
              rot="0.00"):

    model_str, _ = get_file_from_parameter(to_str(psi), to_str(omega),
                                           to_str(radius), to_str(incl),
                                           to_str(rot))

    return model_str


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


def run(p):
    radius, incl, rot, psi, omega = p

    microscope = "PM"
    species = "Vervet"
    model = "r"

    # # # radius = 0.50
    # # incl = 0.00
    # # rot = 0
    # # # rot = 90
    # # psi = 0.5
    # # omega = 90.00

    # if omega == 0:
    #     rot = 0

    model_str, sim_str = get_file_from_parameter(f"{psi:.2f}", f"{omega:.2f}",
                                                 f"{radius:.2f}", f"{incl:.2f}",
                                                 f"{rot:.2f}")

    if not os.path.isfile(model_str) or not os.path.isfile(sim_str):
        return

    h5_sim = get_sim(microscope, species, model, psi, omega, radius, incl, rot)
    data = h5_sim["analysis/epa/0/retardation"][...]

    TISSUE = False

    # if TISSUE:
    tissue = get_tissue(psi, omega, radius, incl, rot)
    # tissue_b = tissue.flatten().astype(np.float32)
    # # tissue_b[tissue_b == 0] = np.nan
    # tissue_b -= 1
    # tissue_b //= 2
    # tissue_b += 1
    # tissue_b.shape = tissue.shape

    tissue_b = np.array(np.logical_and(tissue > 0, tissue < 3), np.uint8)

    # fig.colorbar(pcm, ax=axs[1], shrink=0.275)

    extent = (0, data.shape[0], 0, data.shape[1])
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    fig.suptitle(
        f'{microscope}_{species}_{model}_psi_{to_str(psi)}_' +
        f'omega_{to_str(omega)}_r_{to_str(radius)}_f0_{to_str(incl)}_f1_{to_str(rot)}',
        fontsize=24)

    for i, name in enumerate(["transmittance", "direction", "retardation"]):
        data = h5_sim[f"analysis/epa/0/{name}"][...]

        if name == "direction":
            data[data > np.pi / 2] -= np.pi
            data = np.rad2deg(data)

        pcm = axs[0, i].imshow(
            np.flip(data.T, 0),
            cmap=plt.cm.viridis,
            # alpha=.5,
            interpolation='nearest',
            extent=extent,
        )
        if TISSUE:
            axs[0, i].imshow(
                np.flip(np.mean(tissue_b, -1).T, 0),
                cmap=plt.cm.gray,
                alpha=.5,
                interpolation='nearest',
                extent=extent,
            )
        axs[0, i].set_title(name)
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

                # data[data > np.pi / 2] -= np.pi
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
            if TISSUE:
                axs[1, i].imshow(
                    np.flip(np.mean(tissue_b, -1).T, 0),
                    cmap=plt.cm.gray,
                    alpha=.5,
                    interpolation='nearest',
                    extent=extent,
                )
        axs[1, i].set_title(name)

    axs[-1, -1].imshow(
        np.flip(np.mean(tissue_b, -1).T, 0),
        cmap=plt.cm.gray,
        alpha=.5,
        interpolation='nearest',
        extent=extent,
    )
    axs[-1, -1].set_title("np.mean(tissue_b)")

    # TODO: plot fibers in imshow
    #   - rotate fbs
    #   - cut fbs
    #   - color segments in hsv space

    # fbs = fastpli.io.fiber_bundles.load(model_str)
    # shape = tissue_b.shape
    # fastpli.objects.
    # for fb in fbs:
    #     for f in fb:
    #         axs[-1, -1].plot(f[:, 0] / shape[1], f[:, 1] / shape[0])

    # plot odfs
    phi, theta = models.ori_from_file(get_model(psi, omega, radius, incl, rot),
                                      incl, rot, [65,65,60])
    odf_table = odf.table(phi, np.pi / 2 - theta)

    process = mp.current_process()

    odf.plot(odf_table)
    plt.savefig(f"/tmp/tmp_odf_gt.{process.pid}.png")
    axs[2, 0].imshow(mpimg.imread(f"/tmp/tmp_odf_gt.{process.pid}.png"))
    axs[2, 0].set_title("odf_gt")

    odf_table = odf.table(h5_sim[f"analysis/rofl/direction"][...].ravel(),
                          h5_sim[f"analysis/rofl/inclination"][...].ravel())

    odf.plot(odf_table)
    plt.savefig(f"/tmp/tmp_odf_sim.{process.pid}.png")
    axs[2, 1].imshow(mpimg.imread(f"/tmp/tmp_odf_sim.{process.pid}.png"))
    axs[2, 1].set_title("odf_sim")

    fig.savefig(
        "output/" +
        f"nine_m_{microscope}_s_{species}_m_{model}_psi_{psi:.2f}_" +
        f"omega_{omega:.2f}_r_{radius:.2f}_f0_{incl:.2f}_f1_{rot:.2f}_.pdf")


with mp.Pool(32) as pool:

    parameters = [
        (radius, incl, rot, psi, omega)
        for radius, incl, rot, psi, omega in itertools.product(
            df_sim.radius.unique(), df_sim.f0_inc.unique(),
            df_sim.f1_rot.unique(), df_sim.psi.unique(), df_sim.omega.unique())
    ]

    # parameters = parameters[:5]

    [
        _ for _ in tqdm.tqdm(pool.imap_unordered(run, parameters),
                             total=len(parameters))
    ]
