import numpy as np
import h5py
import argparse
import logging
import subprocess
import glob
import sys
import os

import warnings

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

import simulation_filled_helper

import tqdm

# from simulation_repeat import run_simulation_pipeline_n
import helper.mpi
import helper.file
import models

from mpi4py import MPI
comm = MPI.COMM_WORLD
import multiprocessing as mp

# reproducability
np.random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input path.")
parser.add_argument("-f",
                    "--filled",
                    type=str,
                    required=True,
                    help="filled path.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")

args = parser.parse_args()

df_in = pd.read_pickle(
    os.path.join(args.input, "analysis", f"cube_2pop_simulation.pkl"))
df_in["trel_mean"] = df_in["rofl_trel"].apply(lambda x: np.mean(x))
df_in["ret_mean"] = df_in["epa_ret"].apply(lambda x: np.mean(x))
df_in["trans_mean"] = df_in["epa_trans"].apply(lambda x: np.mean(x))

df_fill = pd.read_pickle(
    os.path.join(args.filled, "analysis", f"cube_2pop_simulation.pkl"))
df_fill["trel_mean"] = df_fill["rofl_trel"].apply(lambda x: np.mean(x))
df_fill["ret_mean"] = df_fill["epa_ret"].apply(lambda x: np.mean(x))
df_fill["trans_mean"] = df_fill["epa_trans"].apply(lambda x: np.mean(x))

# def run(p):
#     radius = p[1].radius
#     microscope = p[1].microscope
#     species = p[1].species
#     model = p[1].model

#     # ACC
#     sub = (df_acc.radius == radius) & (df_acc.microscope == microscope) & (
#         df_acc.species == species) & (df_acc.model == model)

# if __name__ == "__main__":

#     df_p = df[[
#         "radius",
#         "microscope",
#         "species",
#         "model",
#     ]].drop_duplicates()

#     df_p = df_p[df_p.radius == 0.5]
#     df_p = df_p[df_p.species == "Vervet"]
#     df_p = df_p[df_p.model == "r"]
#     df_p = df_p[df_p.microscope == "PM"]

#     with mp.Pool(processes=args.num_proc) as pool:
#         [
#             _ for _ in tqdm.tqdm(pool.imap_unordered(run, df_p.iterrows()),
#                                  total=len(df_p),
#                                  smoothing=0.1)
#         ]
