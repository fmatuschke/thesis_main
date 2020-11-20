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
    radius, psi, f1_theta, f1_phi = parameter

    # ground truth
    sub = (df_org.radius == radius) & (df_org.psi == psi) & (
        df_org.f1_theta == f1_theta) & (df_org.f1_phi == f1_phi) & (df_org.state
                                                                    == "solved")

    if len(df_org[sub]) != 1:
        df_ = df_org[sub]
        for col in df_.columns:
            try:
                if len(df_[col].unique()) == 1:
                    df_.drop(col, inplace=True, axis=1)
            except:
                pass
        print(df_.columns)
        print(df_)
        print(f"FOOO:3: {len(df_)}")
        exit(1)

    phi, theta = fibers.ori_from_file(
        f"/data/PLI-Group/felix/data/thesis/1_model/2_htm/{df_org[sub].fiber.iloc[0]}",
        0, 0, 60)
    sh1 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    gt_dict[
        f'r_{radius:.2f}_f1_theta_{f1_theta:.2f}_f1_phi_{f1_phi:.2f}_psi_{psi:.2f}'] = sh1


def run(parameter):
    # rofl
    radius, psi, f1_theta, f1_phi, model, microscope, species = parameter
    sub = (df.radius == radius) & (df.psi == psi) & (
        df.f1_theta == f1_theta) & (df.f1_phi == f1_phi) & (
            df.model == model) & (df.microscope == microscope) & (df.species
                                                                  == species)

    if len(df[sub]) != 1:
        print("FOOO:2")
        return pd.DataFrame()

    phi = df[sub].explode("rofl_dir").rofl_dir.to_numpy(dtype=float)
    theta = np.pi / 2 - df[sub].explode("rofl_inc").rofl_inc.to_numpy(
        dtype=float)
    sh0 = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    # ground truth
    sh1 = gt_dict[
        f'r_{radius:.2f}_f1_theta_{f1_theta:.2f}_f1_phi_{f1_phi:.2f}_psi_{psi:.2f}']

    # ACC
    acc = helper.schilling.angular_correlation_coefficient(sh0, sh1)

    return pd.DataFrame(
        {
            'microscope': microscope,
            'species': species,
            'model': model,
            'radius': radius,
            'f1_theta': f1_theta,
            'f1_phi': f1_phi,
            'psi': psi,
            'acc': acc
        },
        index=[0])


if __name__ == "__main__":
    # with mp.Pool(processes=args.num_proc) as pool:
    df = pd.read_pickle(
        os.path.join(args.input, "analysis", f"htm_simulation.pkl"))

    df_org = pd.read_pickle(
        f"/data/PLI-Group/felix/data/thesis/1_model/2_htm/output/htm_/data.pkl"
    )  # TODO: same number as simulation

    # # df_org = df_org[df_org.r == df.r.unique()[0]]
    # df_org = df_org[df_org.state != "init"]
    # if len(df_org) == 0:
    #     print("FOOO:1")
    #     sys.exit(1)

    # TEST
    # df = df[df.omega == 30]
    # df_org = df_org[df_org.omega == 30]
    # df = df[df.r == 2.0]
    # df_org = df_org[df_org.r == 2.0]

    # GROUND TRUTH sh coeff

    # print(df.columns)

    sub = df[['radius', 'psi', 'f1_theta', 'f1_phi']]
    sub = sub.drop_duplicates()
    # for f1_theta, f1_phi, psi in sub.itertuples(index=False):

    parameters_gt = []
    for radius, psi, f1_theta, f1_phi in sub.itertuples(index=False):
        parameters_gt.append((radius, psi, f1_theta, f1_phi))

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(calcGroundTruth,
                                                     parameters_gt),
                                 total=len(parameters_gt),
                                 smoothing=0)
        ]

    # schilling
    parameters = []

    sub = df[[
        'radius', 'psi', 'f1_theta', 'f1_phi', 'model', 'microscope', 'species'
    ]]
    sub = sub.drop_duplicates()
    for radius, psi, f1_theta, f1_phi, model, microscope, species in sub.itertuples(
            index=False):
        parameters.append(
            (radius, psi, f1_theta, f1_phi, model, microscope, species))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]
    df = pd.concat(df, ignore_index=True)
    df.to_pickle(
        os.path.join(args.input, "analysis", "simulation_schilling.pkl"))
