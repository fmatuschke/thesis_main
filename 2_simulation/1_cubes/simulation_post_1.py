import argparse
import multiprocessing as mp
import os

import fastpli.analysis
import helper.schilling
import helper.spherical_harmonics
import numpy as np
import pandas as pd
import tqdm

import models
import parameter

CONFIG = parameter.get_tupleware()

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


def calcGroundTruth(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot, radius, rep_n = parameter

    # ground truth
    sub = (df_org.psi == psi) & (df_org.omega == omega) & (df_org.radius
                                                           == radius)
    if "rep_n" in df_org:
        sub = sub & (df_org.rep_n == rep_n)

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

    phi, theta = models.ori_from_file(
        f"/data/PLI-Group/felix/data/thesis/1_model/1_cubes/{df_org[sub].fiber.iloc[0]}",
        f0_inc, f1_rot, [CONFIG.simulation.voi[0], CONFIG.simulation.voi[1]])
    sh1 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    # hist
    h, x, y, _ = fastpli.analysis.orientation.histogram(phi,
                                                        theta,
                                                        n_phi=36,
                                                        n_theta=18,
                                                        weight_area=True)
    # save2dHist(
    #     os.path.join(
    #         args.input, "analysis",
    #         f"orientation_hist_r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_org.dat"
    #     ), h, x, y)

    gt_dict[
        f'r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_rep_{rep_n}'] = sh1


def run(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot, microscope, species, model, radius, rep_n = parameter
    sub = (df.psi == psi) & (df.omega == omega) & (df.f0_inc == f0_inc) & (
        df.f1_rot
        == f1_rot) & (df.microscope == microscope) & (df.species == species) & (
            df.model == model) & (df.radius == radius) & (df.rep_n == rep_n)

    if len(df[sub]) != 1:
        print("FOOO:2")
        return pd.DataFrame()

    phi = df[sub].explode("rofl_dir").rofl_dir.to_numpy(dtype=float)
    theta = np.pi / 2 - df[sub].explode("rofl_inc").rofl_inc.to_numpy(
        dtype=float)
    sh0 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    # hist
    h, x, y, _ = fastpli.analysis.orientation.histogram(phi,
                                                        theta,
                                                        n_phi=36,
                                                        n_theta=18,
                                                        weight_area=True)
    # save2dHist(
    #     os.path.join(
    #         args.input, "analysis",
    #         f"orientation_hist_r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_rofl.dat"
    #     ), h, x, y)

    # ground truth
    sh1 = gt_dict[
        f'r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}_rep_{rep_n}']

    # ACC
    acc = helper.schilling.angular_correlation_coefficient(sh0, sh1)

    return pd.DataFrame(
        {
            'microscope': microscope,
            'species': species,
            'model': model,
            'radius': radius,
            'f0_inc': f0_inc,
            'f1_rot': f1_rot,
            'rep_n': rep_n,
            'omega': omega,
            'psi': psi,
            'acc': acc,
            'R': df[sub].R.to_numpy(float),
            'R2': df[sub].R2.to_numpy(float)
        },
        index=[0])


if __name__ == "__main__":
    # with mp.Pool(processes=args.num_proc) as pool:
    df = pd.read_pickle(
        os.path.join(args.input, "analysis", f"cube_2pop_simulation.pkl"))

    org_path = os.path.basename(args.input)
    org_path = org_path.replace('_single', '')
    org_path = org_path.replace('_flat', '')
    org_path = org_path.replace('_r_0.5', '')
    print("USING:", org_path)

    df_org = pd.read_pickle(
        f"/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/{org_path}/cube_2pop.pkl"
    )  # TODO: same number as simulation

    # df_org = df_org[df_org.r == df.r.unique()[0]]
    df_org = df_org[df_org.state != "init"]
    if len(df_org) == 0:
        print("FOOO:1")
        exit(1)

    # TEST
    # df = df[df.omega == 30]
    # df_org = df_org[df_org.omega == 30]
    # df = df[df.r == 2.0]
    # df_org = df_org[df_org.r == 2.0]

    # GROUND TRUTH sh coeff
    parameters_gt = []

    for _, p in df[[
            "radius",
            "psi",
            "omega",
            "f0_inc",
            "f1_rot",
            "rep_n",
    ]].drop_duplicates().iterrows():
        parameters_gt.append(
            (p.psi, p.omega, p.f0_inc, p.f1_rot, p.radius, int(p.rep_n)))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(calcGroundTruth,
                                                     parameters_gt),
                                 total=len(parameters_gt),
                                 smoothing=0)
        ]

    # schilling
    parameters = []
    for _, p in df[[
            "radius",
            "microscope",
            "species",
            "model",
            "psi",
            "omega",
            "f0_inc",
            "f1_rot",
            "rep_n",
    ]].drop_duplicates().iterrows():
        parameters.append((p.psi, p.omega, p.f0_inc, p.f1_rot, p.microscope,
                           p.species, p.model, p.radius, int(p.rep_n)))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(
        os.path.join(args.input, "analysis",
                     "cube_2pop_simulation_schilling.pkl"))
