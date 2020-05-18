import numpy as np
import argparse
import os

import fastpli.analysis
import fastpli.io

from helper import hist2d_2_tikz

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    nargs='*',
                    required=True,
                    help="fiber bundle input file(s)")

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)
file_list = args.input

FILE_NAME = os.path.abspath(__file__)

file_list = list(
    filter(lambda file: True if os.path.isfile(file) else False, file_list))

for file in tqdm(file_list):
    if not os.path.isfile(file):
        continue

    fbs = fastpli.io.fiber_bundles.load(file)
    phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs)
    h, x, y, _ = fastpli.analysis.orientation.histogram(phi,
                                                        theta,
                                                        n_angle=50,
                                                        n_radius=25,
                                                        normed=True)

    file_path = os.path.dirname(file)
    file_base = os.path.basename(file)
    file_name, _ = os.path.splitext(file_base)
    file_pre = os.path.join(file_path, file_name)

    hist2d_2_tikz(h,
                  np.rad2deg(x),
                  np.rad2deg(y),
                  f"{args.output}/{file_name}.tikz",
                  path_to_data="\currfiledir",
                  info=[FILE_NAME])
