import numpy as np
import multiprocessing as mp
import h5py
import argparse
import glob
import os

import fastpli.simulation
import tqdm

# reproducability
# np.random.seed(42)
rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
np.random.seed(rnd_seed)

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

PIXEL_PM = 1.25
PIXEL_LAP = 20


def run(file):

    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = 1
    simpli.filter_rotations = np.linspace(0, np.pi, 9, False)

    with h5py.File(file, "r") as h5f:  #r+ # FIXME WTF???????
        h5f['/'].attrs['rnd_seed'] = rnd_seed

        for species, mu in [('Roden', 8), ('Vervet', 30), ('Human', 65)]:
            for m, (dn, model) in enumerate([(-0.008 / 2, 'p'), (0.008, 'r')]):
                for name, gain, intensity, res, tilt_angle, sigma in [
                    ('LAP', 3, 35000, PIXEL_LAP, 5.5, 0.75),
                    ('PM', 0.1175, 8000, PIXEL_PM, 3.9, 0.75)
                ]:
                    simpli.tilts = np.deg2rad(
                        np.array([(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                                  (tilt_angle, 180), (tilt_angle, 270)]))

                    h5f_ = h5f[f'{name}/{species}/{model}/']

                    tilting_stack = [None] * 5
                    for t in range(5):
                        tilting_stack[t] = h5f_['simulation/optic/' + str(t)]
                    tilting_stack = np.array(tilting_stack)
                    while tilting_stack.ndim < 4:
                        tilting_stack = np.expand_dims(tilting_stack, axis=-2)

                    rofl_direction, rofl_incl, rofl_t_rel, (
                        rofl_direction_conf, rofl_incl_conf, rofl_t_rel_conf,
                        rofl_func,
                        rofl_n_iter) = simpli.apply_rofl(tilting_stack)

                    # if not np.allclose(rofl_direction,
                    #                    h5f_['analysis/rofl/direction'][...],
                    #                    1e-5):
                    #     raise ValueError("rofl direction")
                    # if not np.allclose(rofl_incl,
                    #                    h5f_['analysis/rofl/inclination'][...],
                    #                    1e-4, 1e-4):
                    #     raise ValueError("rofl inclination")
                    # if not np.allclose(rofl_t_rel,
                    #                    h5f_['analysis/rofl/t_rel'][...], 1e-5):
                    #     raise ValueError("rofl t_rel")

                    h5f_['analysis/rofl/direction_conf'] = rofl_direction_conf,
                    h5f_['analysis/rofl/inclination_conf'] = rofl_incl_conf,
                    h5f_['analysis/rofl/t_rel_conf'] = rofl_t_rel_conf,
                    h5f_['analysis/rofl/func'] = rofl_func,
                    h5f_['analysis/rofl/n_iter'] = rofl_n_iter


if __name__ == "__main__":
    files = glob.glob(os.path.join(args.input, "*.h5"))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, files),
                                 total=len(files),
                                 smoothing=0.1)
        ]
