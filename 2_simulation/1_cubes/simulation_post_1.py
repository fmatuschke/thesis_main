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
    psi, omega, f0_inc, f1_rot, radius = parameter

    # ground truth
    sub = (df_org.psi == psi) & (df_org.omega == omega) & (df_org.r == radius)

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


def run(parameter):
    # rofl
    psi, omega, f0_inc, f1_rot, microscope, species, model, radius = parameter
    sub = (df.psi == psi) & (df.omega == omega) & (df.f0_inc == f0_inc) & (
        df.f1_rot == f1_rot) & (df.microscope == microscope) & (
            df.species == species) & (df.model == model) & (df.r == radius)

    if len(df[sub]) != 1:
        print("FOOO:2")
        return pd.DataFrame()

    phi = df[sub].explode("rofl_dir").rofl_dir.to_numpy(dtype=float)
    theta = np.pi / 2 - df[sub].explode("rofl_inc").rofl_inc.to_numpy(
        dtype=float)
    sh0 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    # ground truth
    sh1 = gt_dict[
        f'r_{radius:.2f}_f0_inc_{f0_inc:.2f}_f1_rot_{f1_rot:.2f}_omega_{omega:.2f}_psi_{psi:.2f}']

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
        f"/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_1/cube_2pop.pkl"
    )  # TODO: same number as simulation

    # df_org = df_org[df_org.r == df.r.unique()[0]]
    df_org = df_org[df_org.state != "init"]
    if len(df_org) == 0:
        print("FOOO:1")
        sys.exit(1)

    # TEST
    # df = df[df.omega == 30]
    # df_org = df_org[df_org.omega == 30]
    # df = df[df.r == 2.0]
    # df_org = df_org[df_org.r == 2.0]

    # GROUND TRUTH sh coeff
    parameters_gt = []
    for radius in df.r.unique():
        for psi in df.psi.unique():
            for omega in df[df.psi == psi].omega.unique():
                df_sub = df[(df.psi == psi) & (df.omega == omega)]
                for f0_inc in df_sub.f0_inc.unique():
                    for f1_rot in df_sub[df_sub.f0_inc ==
                                         f0_inc].f1_rot.unique():
                        parameters_gt.append(
                            (psi, omega, f0_inc, f1_rot, radius))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(calcGroundTruth,
                                                     parameters_gt),
                                 total=len(parameters_gt),
                                 smoothing=0)
        ]

    # schilling
    parameters = []
    for radius in df.r.unique():
        for microscope in df.microscope.unique():
            for species in df.species.unique():
                for model in df.model.unique():
                    for psi in df.psi.unique():
                        for omega in df[df.psi == psi].omega.unique():
                            df_sub = df[(df.psi == psi) & (df.omega == omega)]
                            for f0_inc in df_sub.f0_inc.unique():
                                for f1_rot in df_sub[df_sub.f0_inc ==
                                                     f0_inc].f1_rot.unique():
                                    parameters.append(
                                        (psi, omega, f0_inc, f1_rot, microscope,
                                         species, model, radius))

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
