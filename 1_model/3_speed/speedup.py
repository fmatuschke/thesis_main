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
    repeat: int
    nthreads: list
    after: int


def run(p):
    fiber_bundle_org = fastpli.io.fiber_bundles.load(p.file)
    solver = fastpli.model.solver.Solver()
    solver.fiber_bundles = fiber_bundle_org
    solver.omp_num_threads = 8
    for _ in tqdm.trange(p.after, desc='a', leave=False):
        solver.step()
    fiber_bundle = solver.fiber_bundles

    df = pd.DataFrame(columns=['file', 'n', 'a', 'dt', 'p'])

    for t in tqdm.tqdm(p.nthreads, desc='p', leave=False):
        for n in tqdm.trange(p.repeat, desc='n', leave=False):
            solver = fastpli.model.solver.Solver()
            solver.omp_num_threads = t
            solver.fiber_bundles = fiber_bundle

            dt = time.time()
            solver.step()
            dt = time.time() - dt
            df = df.append(
                {
                    'file': p.file,
                    'n': n,
                    'a': p.after,
                    'dt': dt,
                    'p': t
                },
                ignore_index=True)

    return df


def main():
    args = parser.parse_args()
    parameters = [
        Parameter(file=file,
                  repeat=args.repeat,
                  nthreads=[1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48],
                  after=a) for file in args.input for a in args.after
    ]

    df = []
    for p in tqdm.tqdm(parameters, smoothing=0):
        df.append(run(p))

    df = pd.concat(df, ignore_index=True)
    df.to_pickle(args.output)


if __name__ == '__main__':
    main()
