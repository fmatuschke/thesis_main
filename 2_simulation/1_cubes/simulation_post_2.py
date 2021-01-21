#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import argparse
import os

import pandas as pd
import tqdm

import helper.spherical_harmonics
import helper.schilling
import fibers

import fastpli.analysis

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

# share data
manager = mp.Manager()
gt_dict = manager.dict()
sim_dict = manager.dict()


def save2dHist(file, H, x, y, norm=False):
    with open(file, "w") as f:

        x = x + (x[1] - x[0]) / 2
        y = y[1:] + (y[1] - y[0]) / 2
        H = np.vstack([H, H[0, :]])
        H = H / np.sum(H.ravel())

        X, Y = np.meshgrid(np.rad2deg(x), np.rad2deg(x))

        # norm
        if norm:
            H /= np.amax(H)

        for h_array, x_array, y_array in zip(H.T, X, Y):
            for h, x, y in zip(h_array, x_array, y_array):
                f.write(f'{x:.2f} {y:.2f} {h:.6f}\n')
            f.write('\n')


def calcShGT(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot, radius = parameter

    # ground truth
    sub = (df_org.psi == psi) & (df_org.omega == omega) & (df_org.radius
                                                           == radius)

    if len(df_org[sub]) != 1:
        df_ = df_org[sub]
        for col in df_.columns:
            try:
                if len(df_[col].unique()) == 1:
                    df_.drop(col, inplace=True, axis=1)
            except:
                pass
        print(df_.columns)
        print(len(df_))
        print("FOOO:3")
        exit(1)

    phi, theta = fibers.ori_from_file(
        f"/data/PLI-Group/felix/data/thesis/1_model/1_cubes/{df_org[sub].fiber.iloc[0]}",
        f0_inc, f1_rot)
    sh1 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    gt_dict[
        f'r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}'] = sh1


def calcShSim(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot, microscope, species, model, radius = parameter
    sub = (df_sim.psi == psi) & (df_sim.omega == omega) & (
        df_sim.f0_inc == f0_inc) & (df_sim.f1_rot == f1_rot) & (
            df_sim.microscope == microscope) & (df_sim.species == species) & (
                df_sim.model == model) & (df_sim.radius == radius)

    if len(df_sim[sub]) != 1:
        print("FOOO:2")
        return pd.DataFrame()

    phi = df_sim[sub].explode("rofl_dir").rofl_dir.to_numpy(dtype=float)
    theta = np.pi / 2 - df_sim[sub].explode("rofl_inc").rofl_inc.to_numpy(
        dtype=float)
    sh0 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    sim_dict[
        f'r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_'
        + f'microscope_{microscope}_species_{species}_model_{model}_'] = sh0


def calcACC(parameter):
    psi, omega, f0_inc, f1_rot, microscope, species, model, radius0 = parameter

    df = []
    for radius1 in sorted(df_sim.radius.unique()):

        sh0 = gt_dict[
            f'r_{radius0:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}']
        sh1 = gt_dict[
            f'r_{radius1:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}']
        accGtGt = helper.schilling.angular_correlation_coefficient(sh0, sh1)

        sh0 = gt_dict[
            f'r_{radius0:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}']
        sh1 = sim_dict[
            f'r_{radius1:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_'
            + f'microscope_{microscope}_species_{species}_model_{model}_']
        accGtSim = helper.schilling.angular_correlation_coefficient(sh0, sh1)

        sh0 = sim_dict[
            f'r_{radius0:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_'
            + f'microscope_{microscope}_species_{species}_model_{model}_']
        sh1 = gt_dict[
            f'r_{radius1:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}']
        accSimGt = helper.schilling.angular_correlation_coefficient(sh0, sh1)

        sh0 = sim_dict[
            f'r_{radius0:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_'
            + f'microscope_{microscope}_species_{species}_model_{model}_']
        sh1 = sim_dict[
            f'r_{radius1:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_'
            + f'microscope_{microscope}_species_{species}_model_{model}_']
        accSimSim = helper.schilling.angular_correlation_coefficient(sh0, sh1)

        df.append(
            pd.DataFrame(
                {
                    'microscope': microscope,
                    'species': species,
                    'model': model,
                    'f0_inc': f0_inc,
                    'f1_rot': f1_rot,
                    'omega': omega,
                    'psi': psi,
                    'radius0': radius0,
                    'radius1': radius1,
                    'accGtGt': accGtGt,
                    'accGtSim': accGtSim,
                    'accSimGt': accSimGt,
                    'accSimSim': accSimSim,
                },
                index=[0]))

    return df


if __name__ == "__main__":
    df_sim = pd.read_pickle(
        os.path.join(args.input, "analysis", f"cube_2pop_simulation.pkl"))

    df_org = pd.read_pickle(
        f"/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/cube_2pop.pkl"
    )  # TODO: same number as simulation

    df_org = df_org[df_org.state != "init"]
    if len(df_org) == 0:
        print("FOOO:1")
        exit(1)

    # # TEST
    # df_sim = df_sim[df_sim.omega == 30]
    # df_org = df_org[df_org.omega == 30]
    # df_sim = df_sim[df_sim.radius == 2.0]
    # df_org = df_org[df_org.radius == 2.0]

    # GROUND TRUTH sh coeff
    parameters = []
    for _, p in df_sim[[
            "radius",
            "psi",
            "omega",
            "f0_inc",
            "f1_rot",
    ]].drop_duplicates().iterrows():
        parameters.append((p.psi, p.omega, p.f0_inc, p.f1_rot, p.radius))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(calcShGT, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]

    # Sim sh coeff
    parameters = []
    for _, p in df_sim[[
            "radius",
            "microscope",
            "species",
            "model",
            "psi",
            "omega",
            "f0_inc",
            "f1_rot",
    ]].drop_duplicates().iterrows():
        parameters.append((p.psi, p.omega, p.f0_inc, p.f1_rot, p.microscope,
                           p.species, p.model, p.radius))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(calcShSim, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]

    parameters = []
    for _, p in df_sim[[
            "radius",
            "microscope",
            "species",
            "model",
            "psi",
            "omega",
            "f0_inc",
            "f1_rot",
    ]].drop_duplicates().iterrows():
        parameters.append((p.psi, p.omega, p.f0_inc, p.f1_rot, p.microscope,
                           p.species, p.model, p.radius))

    df = []
    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(calcACC, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]
    df = [item for sublist in df for item in sublist]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(
        os.path.join(args.input, "analysis", "cube_2pop_simulation_accs.pkl"))
