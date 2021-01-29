#! /usr/bin/env python3

import numpy as np
import multiprocessing as mp
import argparse
import warnings
import h5py
import glob
import gc
import os

import pandas as pd
from tqdm import tqdm

import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

import helper.file
import helper.circular

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

if __name__ == "__main__":

    file = os.path.join(args.input, "cube_2pop.pkl")

    df = pd.read_pickle(file)
