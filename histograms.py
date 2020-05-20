import numpy as np
import argparse
import os
import subprocess

import fastpli.analysis
import fastpli.io

from helper import hist2d_2_tikz

from tqdm import tqdm
from mpi4py import MPI

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

for file in tqdm(
        file_list[MPI.COMM_WORLD.Get_rank()::MPI.COMM_WORLD.Get_size()]):

    fbs = fastpli.io.fiber_bundles.load(file)
    phi, theta = fastpli.analysis.orientation.fiber_bundles(fbs)
    h, x, y, _ = fastpli.analysis.orientation.histogram(
        phi,
        theta,
        n_phi=60,
        n_theta=30,
        weight_area=True,
        # fun=lambda x: np.log2(x),
    )

    file_path = os.path.dirname(file)
    file_base = os.path.basename(file)
    file_name, _ = os.path.splitext(file_base)
    file_pre = os.path.join(file_path, file_name)

    hist2d_2_tikz(
        h,
        np.rad2deg(x),
        np.rad2deg(y),
        f"{args.output}/{file_name}.tikz",
        #   path_to_data="\currfiledir",
        standalone=True,
        info=[FILE_NAME])

    # subprocess.run(
    #     f"cd {args.output} && pdflatex {file_name}.tikz && rm {file_name}.aux {file_name}.log",
    #     shell=True,
    #     stdout=subprocess.DEVNULL,
    #     check=True)

    hist2d_2_tikz(
        h,
        np.rad2deg(x),
        np.rad2deg(y),
        f"{args.output}/{file_name}.tikz",
        path_to_data="\currfiledir",
        # standalone=True,
        info=[FILE_NAME])
