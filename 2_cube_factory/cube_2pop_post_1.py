import fastpli.model.solver
import fastpli.objects
import fastpli.io

import numpy as np
import subprocess
import argparse
import sys
import os
import pandas as pd

from tqdm import tqdm

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    nargs='+',
                    required=True,
                    help="input path path of solver.")

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="output path path of solver.")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

solver = fastpli.model.solver.Solver()
# solver.toggle_axis()

# df = pd.read_pickle("output/analysis/cube_2pop_simulation_LAP_model_p_.pkl")

for file in tqdm(args.input):
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

    solver.fiber_bundles = fbs
    solver.draw_scene()

    file = os.path.join(args.output,
                        os.path.splitext(os.path.basename(file))[0])
    solver.save_ppm(os.path.splitext(file)[0] + ".ppm")
    subprocess.run(
        ["convert", file + ".ppm", file + ".png", "&&", "rm", file + ".ppm"])
