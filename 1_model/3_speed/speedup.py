import argparse
import os
import time
import typing
import multiprocessing as mp

import fastpli.io
import fastpli.model.solver
import numpy as np
import pandas as pd
import tqdm

rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
np.random.seed(rnd_seed)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    nargs='+',
                    required=True,
                    help="input string.")

parser.add_argument("-r",
                    "--repeat",
                    type=int,
                    required=True,
                    help="Number of repeat.")

parser.add_argument("-s",
                    "--steps",
                    type=int,
                    required=True,
                    help="Number of steps.")

parser.add_argument("-a",
                    "--after",
                    nargs='+',
                    type=int,
                    required=True,
                    help="Number of after steps.")

parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path")


class Parameter(typing.NamedTuple):
    """  """
    file: str
    steps: int
    repeat: int
    nthreads: int
    after: int


def run(p):
    fiber_bundle_org = fastpli.io.fiber_bundles.load(p.file)
    solver = fastpli.model.solver.Solver()
    solver.fiber_bundles = fiber_bundle_org
    solver.omp_num_threads = 4
    for _ in tqdm.trange(p.after, desc='a', leave=False):
        solver.step()
    fiber_bundle = solver.fiber_bundles

    df = pd.DataFrame(columns=['n', 'm', 'dt', 'p'])

    for n in tqdm.trange(p.repeat, desc='n', leave=False):
        solver = fastpli.model.solver.Solver()
        solver.omp_num_threads = p.nthreads
        solver.fiber_bundles = fiber_bundle

        for i in tqdm.trange(p.steps, desc='i', leave=False):
            dt = time.time()
            solver.step()
            dt = time.time() - dt
            # t[n, i] = dt
            df = df.append(
                {
                    'file': p.file,
                    'n': n,
                    'm': i,
                    'a': p.after,
                    'dt': dt,
                    'p': p.nthreads
                },
                ignore_index=True)

    return df


def main():
    args = parser.parse_args()
    parameters = [
        Parameter(file=file,
                  steps=args.steps,
                  repeat=args.repeat,
                  nthreads=t,
                  after=a) for file in args.input
        for t in [1, 2, 3, 4, 5, 6, 7, 8] for a in args.after
    ]

    df = []
    for p in tqdm.tqdm(parameters, smoothing=0):
        df.append(run(p))

    df = pd.concat(df, ignore_index=True)
    df.to_pickle(args.output)


if __name__ == '__main__':
    main()
