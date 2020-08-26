#! /usr/bin/env python3

import fastpli.model.solver
import fastpli.objects
import fastpli.io

import numpy as np
import subprocess
import argparse
import glob
import sys
import os
import multiprocessing as mp

import tqdm

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input files.")

parser.add_argument("-v",
                    "--volume",
                    type=float,
                    required=True,
                    help="cutting volume size.")

parser.add_argument("-p", "--num_proc", type=int, required=True, help="")

args = parser.parse_args()

os.makedirs(os.path.join(args.input, "images"), exist_ok=True)

solver = fastpli.model.solver.Solver()


def run(file):
    fbs = fastpli.io.fiber_bundles.load(file)

    fbs = fastpli.objects.fiber_bundles.Cut(
        fbs, [[-args.volume / 2] * 3, [args.volume / 2] * 3])

    solver.fiber_bundles = fbs
    # solver.reset_view()
    # solver.set_view_angles(30, 30, 0)
    # solver.set_view_distance(120)
    # solver.set_view_center(0, 0, 0)
    solver.draw_scene()

    file = os.path.join(os.path.abspath(args.input), "images",
                        os.path.splitext(os.path.basename(file))[0])
    solver.save_ppm(file + ".ppm")
    subprocess.run(f"convert {file}.ppm {file}.png && rm {file}.ppm",
                   shell=True,
                   check=True)


if __name__ == "__main__":
    files = glob.glob(os.path.join(args.input, "*.h5"))
    with mp.Pool(processes=args.num_proc) as pool:
        [
            f for f in tqdm.tqdm(pool.imap_unordered(run, files),
                                 total=len(files))
        ]
