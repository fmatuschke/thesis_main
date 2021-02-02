import numpy as np
import os
import argparse
import multiprocessing as mp
import pandas as pd
import tqdm

import fastpli.tools
import helper.spherical_interpolation
import polar_hist_to_tikz

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

sim_path = args.input
os.makedirs(os.path.join(sim_path, "hist"), exist_ok=True)
df = pd.read_pickle(
    os.path.join(sim_path, "analysis", f"cube_2pop_simulation.pkl"))
# df_acc = pd.read_pickle(
#     os.path.join(sim_path, "analysis", f"cube_2pop_simulation_schilling.pkl"))

# print(df.columns)
df["trel_mean"] = df["rofl_trel"].apply(lambda x: np.mean(x))
df["ret_mean"] = df["epa_ret"].apply(lambda x: np.mean(x))
df["trans_mean"] = df["epa_trans"].apply(lambda x: np.mean(x))


def run(p):
    radius = p[1].radius
    microscope = p[1].microscope
    species = p[1].species
    model = p[1].model

    sub = (df.radius == radius) & (df.microscope == microscope) & (
        df.species == species) & (df.model == model)

    for name in ["trel_mean", "ret_mean", "trans_mean"]:

        crange = [0, 1]
        if name == "trans_mean":
            crange = None

        crange = polar_hist_to_tikz.generate(
            df[sub],
            name,
            f"simulation_analysis_hist_{radius}_setup_{microscope}_s_{species}_m_{model}_{name}",
            crange=crange,
            psi_list=[0.30, 0.50, 0.60, 0.90],
            f0_list=[0, 30, 60, 90])

        print(name, crange)


if __name__ == "__main__":

    # df.trel_mean = df["rofl_trel"].apply(lambda x: np.mean(x))

    df_p = df[[
        "radius",
        "microscope",
        "species",
        "model",
    ]].drop_duplicates()

    df_p = df_p[df_p.radius == 0.5]
    df_p = df_p[df_p.species == "Vervet"]
    df_p = df_p[df_p.model == "r"]
    df_p = df_p[df_p.microscope == "PM"]

    with mp.Pool(processes=args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(run, df_p.iterrows()),
                                 total=len(df_p),
                                 smoothing=0.1)
        ]