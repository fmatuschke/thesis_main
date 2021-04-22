import subprocess
import multiprocessing as mp
import os

import tqdm


def run(p):
    radius = p[0]
    volume = p[1]
    output = p[2]
    n_max = p[3]
    n_thread = p[4]
    start = p[5]
    subprocess.call([
        "/data/PLI-Group/felix/data/thesis/env-ime656/bin/python3",
        "cube_2pop.py", "-o",
        os.path.join(output, f"cube_2pop_{volume}"), "-r", f"{radius}", "-v",
        f"{volume}", "-n", f"{n_max}", "-p", f"{n_thread}", "--start",
        f"{start}"
    ])


N = 92
with mp.Pool(48) as pool:
    for r in [10, 5, 2, 1, 0.5]:
        parameters = [(r, 135, "output", 10000, 1, s) for s in range(N)]
        print(f"radius: {r}")
        for _ in pool.imap_unordered(run, parameters):
            pass
