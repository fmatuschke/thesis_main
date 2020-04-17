import fastpli.model.solver
import fastpli.io

import numpy as np
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
args = parser.parse_args()

solver = fastpli.model.solver.Solver()
for file in tqdm(args.input):
    solver.fiber_bundles = fastpli.io.fiber_bundles.load(file)
    solver.draw_scene()
    solver.save_ppm(os.path.splitext(file)[0] + ".ppm")
