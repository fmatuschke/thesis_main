#!/usr/bin/env python3
'''
python3 -m venv env-imedv18
env-activate
pip3 install pip --upgrade
cd ../fastpli
make clean
make BUILD=info fastpli
pip3 install .
cd --
'''

import yep

import fastpli.model.solver
import fastpli.model.sandbox
import fastpli.io

import numpy as np
import argparse
import os, sys

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


def run():
    ### setup solver ###
    solver = fastpli.model.solver.Solver()
    solver.drag = 0
    solver.obj_min_radius = 2
    solver.obj_mean_length = 2
    solver.omp_num_threads = 1

    if not os.path.isfile(f"{args.output}.prep.h5"):
        ### create fiber_bundle(s) ###
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
            fastpli.model.sandbox.build.cuboid(
                p=-0.5 * np.array([args.volume] * 3),
                q=0.5 * np.array([args.volume] * 3),
                phi=np.deg2rad(0),
                theta=np.deg2rad(90),
                seeds=seeds_0,
                radii=rnd_radii_0),
            fastpli.model.sandbox.build.cuboid(
                p=-0.5 * np.array([args.volume] * 3),
                q=0.5 * np.array([args.volume] * 3),
                phi=np.deg2rad(args.omega),
                theta=np.deg2rad(90),
                seeds=seeds_1 + args.fiber_radius,
                radii=rnd_radii_1)
        ]

        # add rnd displacement
        for fb in fiber_bundles:
            for f in fb:
                f[:, :-1] += np.random.normal(0, 0.05 * args.fiber_radius,
                                              (f.shape[0], 3))
                f[:, -1] *= np.random.lognormal(0, 0.05, f.shape[0])

        solver.fiber_bundles = fiber_bundles
        solver.apply_boundary_conditions(100)

        ### run solver ###
        for _ in range(args.n_initial):
            solver.step()

        solver.fiber_bundles = fastpli.objects.fiber_bundles.Cut(
            solver.fiber_bundles,
            [[-args.volume / 2] * 3, [args.volume / 2] * 3])

        fastpli.io.fiber_bundles.save(f"{args.output}.prep.h5",
                                      solver.fiber_bundles)

    solver.fiber_bundles = fastpli.io.fiber_bundles.load(
        f"{args.output}.prep.h5",)

    yep.start(f"{args.output}.prof")
    for _ in range(args.m_measure):
        solver.step()
    yep.stop


if __name__ == "__main__":
    run()

    # os.system(
    #     f"google-pprof --callgrind {sys.executable} {args.output}.prof > {args.output}.callgrind"
    # )
