import fastpli.model.solver
import fastpli.objects
import fastpli.io

import numpy as np
import subprocess
import argparse
import sys
import os
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    nargs='+',
                    required=True,
                    help="input files.")

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="output path of images.")

parser.add_argument("-v",
                    "--volume",
                    type=float,
                    required=True,
                    help="cutting volume size.")

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

solver = fastpli.model.solver.Solver()


def run(file):
    # solver.toggle_axis()

    # df = pd.read_pickle("output/analysis/cube_2pop_simulation_LAP_model_p_.pkl")

    # for file in tqdm(args.input):
    fbs = fastpli.io.fiber_bundles.load(file)

    # omega = float((file.split("omega_")[-1]).split("_")[0])
    # psi = float((file.split("psi_")[-1]).split("_")[0])

    # df_sub = df[(df.omega == omega) & (df.psi == psi)]
    # for f0_inc in df_sub.f0_inc.unique():
    #     for f1_rot in df_sub[df_sub.f0_inc == f0_inc].f1_rot.unique():
    #         rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
    #         rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
    #         rot = np.dot(rot_inc, rot_phi)
    #         fbs_ = fastpli.objects.fiber_bundles.Rotate(fbs, rot)
    #         fbs_ = fastpli.objects.fiber_bundles.Cut(fbs_,
    #                                                  [[-30] * 3, [30] * 3])

    fbs = fastpli.objects.fiber_bundles.Cut(
        fbs, [[-args.volume / 2] * 3, [args.volume / 2] * 3])

    solver.fiber_bundles = fbs
    solver.draw_scene()

    file = os.path.join(os.path.abspath(args.output),
                        os.path.splitext(os.path.basename(file))[0])
    solver.save_ppm(file + ".ppm")
    subprocess.run(f"convert {file}.ppm {file}.png && rm {file}.ppm",
                   shell=True,
                   check=True)


with mp.Pool(processes=4) as pool:
    [
        f for f in tqdm(pool.imap_unordered(run, args.input),
                        total=len(args.input))
    ]
