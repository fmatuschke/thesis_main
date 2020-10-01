import numpy as np
import multiprocessing as mp
import itertools
import argparse
import os

import pandas as pd
import tqdm

import fastpli.io
import fastpli.analysis

import helper.spherical_harmonics
import helper.schilling
import fibers

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


def calcGroundTruth(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot = parameter

    # ground truth
    sub = (df_org.psi == psi) & (df_org.omega == omega)
    phi, theta = fibers.ori_from_file(df_org[sub].fiber.iloc[0], f0_inc, f1_rot)
    sh1 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    gt_dict[
        f'f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}'] = sh1

    # return {
    #     f'f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi.2f}':
    #         sh1
    # }


def run(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot, microscope, model = parameter
    sub = (df.psi == psi) & (df.omega == omega) & (df.f0_inc == f0_inc) & (
        df.f1_rot == f1_rot) & (df.microscope == microscope) & (df.model
                                                                == model)

    if len(df[sub]) != 1:
        print("FOOO")
        exit(1)

    phi = df[sub].explode("rofl_dir").rofl_dir.to_numpy(dtype=float)
    theta = np.pi / 2 - df[sub].explode("rofl_inc").rofl_inc.to_numpy(
        dtype=float)
    sh0 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    # # ground truth
    # sub = (df_org.psi == psi) & (df_org.omega == omega)
    # phi, theta = fibers.ori_from_file(df_org[sub].fiber.iloc[0], f0_inc, f1_rot)
    # sh1 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    sh1 = gt_dict[
        f'f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}']

    # ACC
    acc = helper.schilling.angular_correlation_coefficient(sh0, sh1)

    return pd.DataFrame(
        {
            'microscope': microscope,
            'model': model,
            'f0_inc': f0_inc,
            'f1_rot': f1_rot,
            'omega': omega,
            'psi': psi,
            'acc': acc
        },
        index=[0])


if __name__ == "__main__":
    # with mp.Pool(processes=args.num_proc) as pool:
    df = pd.read_pickle(
        os.path.join(args.input, "analysis", f"cube_2pop_simulation.pkl"))

    df_org = pd.read_pickle(
        f"output/cube_2pop_1/cube_2pop.pkl")  # TODO: same number as simulation

    df_org = df_org[df_org.r == df.r.unique()[0]]
    df_org = df_org[df_org.state != "init"]
    if len(df_org) == 0:
        print("FOOO")
        exit(1)

    # # TEST
    # df = df[df.omega == 10]
    # df_org = df_org[df_org.omega == 10]

    # GROUND TRUTH sh coeff
    parameters_gt = []
    for psi in df.psi.unique():
        for omega in df[df.psi == psi].omega.unique():
            df_sub = df[(df.psi == psi) & (df.omega == omega)]
            for f0_inc in df_sub.f0_inc.unique():
                for f1_rot in df_sub[df_sub.f0_inc == f0_inc].f1_rot.unique():
                    parameters_gt.append((psi, omega, f0_inc, f1_rot))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(calcGroundTruth,
                                                     parameters_gt),
                                 total=len(parameters_gt),
                                 smoothing=0)
        ]

    # schilling
    parameters = []
    for microscope in df.microscope.unique():
        for model in df.model.unique():
            for psi in df.psi.unique():
                for omega in df[df.psi == psi].omega.unique():
                    df_sub = df[(df.psi == psi) & (df.omega == omega)]
                    for f0_inc in df_sub.f0_inc.unique():
                        for f1_rot in df_sub[df_sub.f0_inc ==
                                             f0_inc].f1_rot.unique():
                            parameters.append(
                                (psi, omega, f0_inc, f1_rot, microscope, model))

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
