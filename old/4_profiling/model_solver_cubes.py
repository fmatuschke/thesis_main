#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import logging
import os, sys
import datetime
import time
import tqdm
# tqdm.tqdm = lambda x: x

import fastpli.model.solver
import fastpli.model.sandbox

# reproducability
# np.random.seed(42)

# arguments
parser = argparse.ArgumentParser()

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

parser.add_argument("-n",
                    "--n_initial",
                    type=int,
                    required=True,
                    help="Number of max_steps.")

parser.add_argument("-m",
                    "--m_measure",
                    type=int,
                    required=True,
                    help="Number of max_steps.")

parser.add_argument("-r",
                    "--fiber_radius",
                    required=True,
                    type=float,
                    help="mean value of fiber radius")

parser.add_argument("-v",
                    "--volume",
                    required=True,
                    type=int,
                    help="volume args.volume/2")

parser.add_argument("--psi", required=True, type=float, help="")

parser.add_argument("--omega", required=True, type=float, help="")

args = parser.parse_args()
args.output = f"{args.output}_psi_{args.psi}_omega_{args.omega}_n_{args.n_initial}_m_{args.m_measure}_v_{args.volume}_r_{args.fiber_radius}_"
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# logging
log_file = args.output + f'.{datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")}.log'
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
logging.basicConfig(filename=log_file, format=formatter)
logging.info("args: " + " ".join(sys.argv[1:]))


def run(num_threads):

    ### setup solver ###
    logging.info(f"solver setup")
    solver = fastpli.model.solver.Solver()
    solver.drag = 0
    solver.obj_min_radius = 2
    solver.obj_mean_length = 2
    solver.omp_num_threads = num_threads

    ### create fiber_bundle(s) ###
    logging.info(f"create fiber_bundle(s)")
    seeds_0 = np.random.uniform(-args.volume, args.volume,
                                (int(args.psi * (2 * args.volume)**2 /
                                     (np.pi * args.fiber_radius**2)), 2))
    seeds_1 = np.random.uniform(-args.volume, args.volume,
                                (int((1 - args.psi) * (2 * args.volume)**2 /
                                     (np.pi * args.fiber_radius**2)), 2))

    rnd_radii_0 = args.fiber_radius * np.random.lognormal(
        0, 0.1, seeds_0.shape[0])
    rnd_radii_1 = args.fiber_radius * np.random.lognormal(
        0, 0.1, seeds_1.shape[0])

    fiber_bundles = [
        fastpli.model.sandbox.build.cuboid(p=-0.5 * np.array([args.volume] * 3),
                                           q=0.5 * np.array([args.volume] * 3),
                                           phi=np.deg2rad(0),
                                           theta=np.deg2rad(90),
                                           seeds=seeds_0,
                                           radii=rnd_radii_0),
        fastpli.model.sandbox.build.cuboid(p=-0.5 * np.array([args.volume] * 3),
                                           q=0.5 * np.array([args.volume] * 3),
                                           phi=np.deg2rad(args.omega),
                                           theta=np.deg2rad(90),
                                           seeds=seeds_1 + args.fiber_radius,
                                           radii=rnd_radii_1)
    ]

    # add rnd displacement
    logging.info(f"rnd displacement")
    for fb in fiber_bundles:
        for f in fb:
            f[:, :-1] += np.random.normal(0, 0.05 * args.fiber_radius,
                                          (f.shape[0], 3))
            f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

    solver.fiber_bundles = fiber_bundles
    solver.apply_boundary_conditions(100)

    ### run solver ###
    logging.info(f"solving init")
    time0 = time.time()
    for i in range(args.n_initial):
        solver.step()
    time1 = time.time()
    time_n = time1 - time0
    logging.info(f"finished solving initial: {time_n}")

    solver.fiber_bundles = fastpli.objects.fiber_bundles.Cut(
        solver.fiber_bundles, [[-args.volume / 2] * 3, [args.volume / 2] * 3])

    logging.info(f"start solving")
    time0 = time.time()
    for i in range(args.m_measure):
        solver.step()
    time1 = time.time()
    time_m = time1 - time0
    logging.info(f"finished solving: {time_m}")

    logging.info(f"solver.overlap = {solver.overlap}")
    logging.info(f"solver.num_col_obj = {solver.num_col_obj}")
    logging.info(f"solver.num_obj = {solver.num_obj}")
    logging.info(f"solver.num_steps = {solver.num_steps}")
    logging.info(f"solver.obj_mean_length = {solver.obj_mean_length}")
    logging.info(f"solver.obj_min_radius = {solver.obj_min_radius}")

    return time_n, time_m


if __name__ == "__main__":
    df = pd.DataFrame()
    for i in tqdm.tqdm(range(1, 10 + 1)):
        for p in tqdm.tqdm(range(1, 8 + 1), leave=False):
            tn, tm = run(p)
            df = df.append({
                "i": i,
                "p": p,
                "tn": tn,
                "tm": tm
            },
                           ignore_index=True)

            df.to_csv(
                f"{args.output}.csv",
                index=False,
            )
    df.pivot(index="i", columns="p")['tn'].to_csv(f"{args.output}_tn.csv")
    df.pivot(index="i", columns="p")['tm'].to_csv(f"{args.output}_tm.csv")
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(f"{args.output}.csv")
df = pd.melt(df,
             id_vars=["i", "p"],
             value_vars=["tn", "tm"],
             var_name='method',
             value_name='time')
plt.figure()
sns.boxplot(x="p", y="time", hue="method", data=df)
plt.show()
'''
