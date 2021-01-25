import numpy as np
import os
import argparse
import multiprocessing as mp
import pandas as pd
import tqdm

import fastpli.tools
import helper.spherical_interpolation

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input path.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()

sim_path = args.input
os.makedirs(os.path.join(sim_path, "hist"), exist_ok=True)
df = pd.read_pickle(
    os.path.join(sim_path, "analysis", f"cube_2pop_simulation.pkl"))
df_acc = pd.read_pickle(
    os.path.join(sim_path, "analysis", f"cube_2pop_simulation_schilling.pkl"))


def run(p):
    radius = p[1].radius
    microscope = p[1].microscope
    species = p[1].species
    model = p[1].model
    psi = p[1].psi
    f0_inc = p[1].f0_inc

    file_name = (f"sphere_r_{radius}_{microscope}_species_{species}"
                 f"_model_{model}_psi_{psi:.2f}_f0_inc_{f0_inc:.2f}")

    sub = (df_acc.radius == radius) & (df_acc.microscope == microscope) & (
        df_acc.species == species) & (df_acc.model == model) & (
            df_acc.psi == psi) & (df_acc.f0_inc == f0_inc)

    sub_ref = (df_acc.radius == radius) & (df_acc.microscope == microscope) & (
        df_acc.species == species) & (df_acc.model == model) & (
            df_acc.psi == 0.0) & (df_acc.f0_inc == 0.0)

    if len(df_acc[sub_ref]) != 1:
        print(df_acc[sub_ref])
        raise ValueError(f"FOOO:1: {len(df_acc[sub_ref])}")
    if psi != 0.0:
        if len(df_acc[sub]) != len(df_acc[sub].groupby(['f1_rot', 'omega'
                                                       ]).size()):
            raise ValueError(f"FOOO:22: {len(df_acc[sub])}")

    f1_rot = df_acc[sub].f1_rot.to_numpy(float)
    omega = df_acc[sub].omega.to_numpy(float)

    for name, norm in [
        ("acc", False),
            # ("R", False),
            # ("R2", False),
            # ("angle", False),
            # ("acc", True),
            # ("R", True),
            # ("R2", True),
    ]:

        data = df_acc[sub][name].to_numpy(float)

        # norm R and R2 to ref
        if norm:
            data /= np.mean(df_acc[sub_ref][name].to_numpy(float))

        # get points on sphere
        phi = []
        theta = []
        for f1, om in zip(f1_rot, omega):
            v = np.array([np.cos(np.deg2rad(om)), np.sin(np.deg2rad(om)), 0])
            rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
            rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1))
            rot = np.dot(rot_inc, rot_phi)
            v = np.dot(rot, v)
            theta.extend([np.arccos(v[2])])
            phi.extend([np.arctan2(v[1], v[0])])

        phi = np.array(phi)
        theta = np.array(theta)

        phi_ = phi.copy()
        theta_ = theta.copy()
        data_ = data.copy()

        # # apply symmetry
        phi_ = phi.copy()
        theta_ = theta.copy()
        data_ = data.copy()
        # measurement symmetry
        phi_ = np.concatenate((phi_, -phi_), axis=0)
        theta_ = np.concatenate((theta_, theta_), axis=0)
        data_ = np.concatenate((data_, data_), axis=0)
        # orientation symmetry
        phi_ = np.concatenate((phi_, phi_), axis=0)
        theta_ = np.concatenate((theta_, np.pi + theta_), axis=0)
        data_ = np.concatenate((data_, data_), axis=0)

        # rm multiple
        phi_, theta_ = helper.spherical_interpolation.remap_spherical(
            phi_, theta_)

        x_ = np.multiply(np.cos(phi_), np.sin(theta_))
        y_ = np.multiply(np.sin(phi_), np.sin(theta_))
        z_ = np.cos(theta_)

        data__ = []
        for i in range(phi_.size):
            flag = True
            for j in range(i + 1, phi_.size):
                dist = np.linalg.norm(
                    np.array((x_[i] - x_[j], y_[i] - y_[j], z_[i] - z_[j])))

                if dist < 1e-6:
                    flag = False
                    break
            if flag:
                data__.append((phi_[i], theta_[i], data_[i]))

        data__ = np.array(data__)
        phi_ = data__[:, 0]
        theta_ = data__[:, 1]
        data_ = data__[:, 2]

        # interplate mesh on sphere
        x_i, y_i, z_i, data_i = helper.spherical_interpolation.on_mesh(
            phi_, theta_, data_, 37, 19)

        # 2d hist
        with open(
                os.path.join(
                    args.input, "hist",
                    f"sim_r_{radius}_setup_{microscope}_s_{species}_m_{model}_psi_{psi}_f0_{f0_inc}_{name}_{norm}_hist.dat"
                ), "w") as f:

            for h_array, x_array, y_array, z_array in zip(
                    data_i, x_i, y_i, z_i):
                for h, x, y, z in zip(h_array, x_array, y_array, z_array):
                    p = np.rad2deg(np.arctan2(y, x))
                    t = np.rad2deg(np.arccos(z))

                    if t <= 90:
                        f.write(f'{p:.2f} {t:.2f} {h:.6f}\n')
                f.write('\n')

        with open(
                os.path.join(
                    args.input, "hist",
                    f"sim_r_{radius}_setup_{microscope}_s_{species}_m_{model}_psi_{psi}_f0_{f0_inc}_{name}_{norm}_init.dat"
                ), "w") as f:

            f.write(f'{0} {90-f0_inc}\n')
            f.write('\n')

        with open(
                os.path.join(
                    args.input, "hist",
                    f"sim_r_{radius}_setup_{microscope}_s_{species}_m_{model}_psi_{psi}_f0_{f0_inc}_{name}_{norm}_data.dat"
                ), "w") as f:

            # f.write(f'{0} {f0_inc} {np.deg2rad(data[0])}\n')

            for d, p, t in zip(data_, phi_, theta_):
                if t <= np.pi / 2:
                    f.write(
                        f'{np.rad2deg(p):.2f} {np.rad2deg(t):.2f} {d:.6f}\n')
            f.write('\n')


if __name__ == "__main__":

    df_p = df[[
        "radius",
        "microscope",
        "species",
        "model",
        "psi",
        "f0_inc",
    ]].drop_duplicates()

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(run, df_p.iterrows()),
                                 total=len(df_p),
                                 smoothing=0.1)
        ]
