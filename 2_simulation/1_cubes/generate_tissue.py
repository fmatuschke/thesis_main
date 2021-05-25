import argparse
import glob
import multiprocessing as mp
import os
import typing
import warnings

import fastpli.analysis
import fastpli.io
import fastpli.simulation
import fastpli.tools
import h5py
import helper.file
import numpy as np
import tqdm

import parameter

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

CONFIG = parameter.get_tupleware()


class Parameter(typing.NamedTuple):
    """  """
    file: str


def run(p):

    file_path, file_name = os.path.split(p.file)
    file_name = os.path.splitext(file_name)[0]
    file_name = os.path.join(file_path, "tissue", file_name)

    with h5py.File(p.file, 'r') as h5f:
        dset = h5f['PM/Vervet/p']
        fiber_bundles = fastpli.io.fiber_bundles.load(
            dset.attrs['parameter/fiber_path'])
        psi = dset.attrs["parameter/psi"]
        omega = dset.attrs["parameter/omega"]
        radius = dset.attrs["parameter/radius"]
        f0_inc = dset.attrs["parameter/f0_inc"]
        f1_rot = dset.attrs["parameter/f1_rot"]
    voxel_size = helper.file.value(p.file, "vs")

    rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
    rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
    rot = np.dot(rot_inc, rot_phi)

    with h5py.File(file_name + '.tissue.h5', 'w') as h5f:
        with open(os.path.abspath(__file__), 'r') as script:
            h5f.attrs['script'] = script.read()
            h5f.attrs['input_file'] = p.file

        # Setup Simpli
        simpli = fastpli.simulation.Simpli()
        warnings.filterwarnings("ignore", message="objects overlap")
        simpli.omp_num_threads = 1
        simpli.voxel_size = voxel_size

        simpli.set_voi(CONFIG.simulation.voi[0], CONFIG.simulation.voi[1])
        simpli.tilts = np.deg2rad(np.array([(0, 0)]))

        simpli.fiber_bundles = fiber_bundles.rotate(rot)

        LAYERS = CONFIG.models.layers
        layers = [(LAYERS.b.radius, LAYERS.b.dn, 0, LAYERS.b.model),
                  (LAYERS.p.radius, LAYERS.p.dn, 0, LAYERS.p.model)
                 ]  # p is faster
        simpli.fiber_bundles.layers = [layers] * len(fiber_bundles)

        tissue, _, _ = simpli.generate_tissue(only_tissue=True)
        dset = h5f.create_dataset('tissue',
                                  tissue.shape,
                                  dtype=np.uint8,
                                  compression='gzip')

        dset[:] = tissue

        # tissue = tissue.astype(np.float32)
        # tmp = tissue.ravel()
        # tmp[tmp > 0] = ((tmp[tmp > 0] - 1) // 2) + 1
        # tmp[tmp == 0] = np.nan

        # with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         "ignore", message='RuntimeWarning: Mean of empty slice')
        #     h5f[f"/tissue_bin_median"] = np.nanmedian(tissue, axis=-1)
        #     h5f[f"/tissue_bin_mean"] = np.nanmean(tissue, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input string.")
    parser.add_argument("-p",
                        "--num_proc",
                        type=int,
                        required=True,
                        help="Number of processes.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.input, "tissue"), exist_ok=True)

    parameters = []
    for file in glob.glob(os.path.join(args.input, '*.h5')):
        parameters.append(Parameter(file=file))

    # Memory Check
    simpli = fastpli.simulation.Simpli()
    simpli.voxel_size = CONFIG.simulation.voxel_size  # in mu meter
    simpli.pixel_size = CONFIG.simulation.setup.pm.pixel_size  # in mu meter
    simpli.set_voi(CONFIG.simulation.voi[0], CONFIG.simulation.voi[1])
    print(f"Single Memory: {simpli.memory_usage(item='tissue'):.0f} MB")
    print(f"Total Memory: {simpli.memory_usage(item='tissue')* args.num_proc:.0f} MB")

    with mp.Pool(args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]


if __name__ == "__main__":
    main()
