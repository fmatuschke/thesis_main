import subprocess
import multiprocessing as mp
import os

import tqdm

NAME = 'rc1.1'
HOST = os.uname()[1]
PATH = os.path.dirname(os.path.realpath(__file__))
THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')


def run(p):
    radius = p[0]
    volume = p[1]
    output = p[2]
    n_max = p[3]
    n_thread = p[4]
    index = p[5]
    subprocess.call([
        os.path.join(THESIS, f"env-{HOST}/bin/python3"), "cube_2pop.py", "-o",
        os.path.join(output, f"cube_2pop_{volume}_{NAME}"), "-r", f"{radius}",
        "-v", f"{volume}", "-n", f"{n_max}", "-p", f"{n_thread}", "--index",
        f"{index}"
    ])


N = 92
with mp.Pool(mp.cpu_count() // 2) as pool:
    for r in [10, 5, 2, 1, 0.5]:
        parameters = [(r, 135, "output", 100000, 1, s) for s in range(N)]
        for _ in tqdm.tqdm(pool.imap_unordered(run, parameters),
                           total=len(parameters),
                           desc=f"radius: {r}"):
            pass
