import numpy as np
import h5py
import tqdm


def read(file):
    with h5py.File(file, 'r') as h5f:
        fiber_bundles = []
        h5f = h5f['fiber_bundles']
        fb_list = list(map(int, list(h5f.keys())))
        fb_list.sort()
        for fb in tqdm.tqdm(fb_list):
            fiber_bundles.append([])
            f_list = list(map(int, list(h5f[str(fb)].keys())))
            f_list.sort()
            for f in tqdm.tqdm(f_list):
                fiber_bundles[-1].append(
                    np.insert(h5f[str(fb)][str(f)]['points'][:].astype(float),
                              3,
                              1,
                              axis=-1))

        return fiber_bundles
