import fastpli.model.solver
import fastpli.objects
import fastpli.io

import numpy as np
import subprocess
import argparse
import sys
import os

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
for file in tqdm(args.input):
    fbs = fastpli.io.fiber_bundles.load(file)
    file = os.path.join(args.output,
                        os.path.splitext(os.path.basename(file))[0])
    solver.fiber_bundles = fastpli.objects.fiber_bundles.Cut(
        fbs, [[-30] * 3, [30] * 3])
    solver.draw_scene()
    solver.save_ppm(file + ".ppm")
    subprocess.run(
        f'convert {file + ".ppm"} {file + ".png"} && rm {file + ".ppm"}',
        shell=True)
