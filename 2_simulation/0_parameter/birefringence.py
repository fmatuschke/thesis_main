import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import warnings

import fastpli.analysis
import fastpli.io
import fastpli.model.sandbox
import fastpli.model.solver
import fastpli.simulation
import fastpli.tools
import h5py
import numpy as np
import pandas as pd
import tqdm
import typing

import parameter

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
CONFIG = parameter.get_tupleware()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path of solver.")

parser.add_argument("-p",
                    "--num_proc",
                    type=int,
                    required=True,
                    help="Number of parallel processes.")

parser.add_argument("-t",
                    "--num_threads",
                    type=int,
                    required=True,
                    help="Number of threads per process.")

args = parser.parse_args()
output_name = os.path.join(args.output)
os.makedirs(args.output, exist_ok=False)
subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)


class Parameter(typing.NamedTuple):
    """  """
    file: str
    dn: float
    model: float
    setup: str
    gain: float
    intensity: float
    pixel_size: float
    species: str
    sigma: float
    mu: float


def run(p):

    with h5py.File(p.file, 'r') as h5f_:
        fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f_)
        psi = h5f_['/'].attrs["psi"]
        omega = h5f_['/'].attrs["omega"]
        # radius = h5f_['/'].attrs["radius"] # FIXME
        radius = float(p.file.split("_r_")[1].split("_")[0])
        v0 = float(p.file.split("_v0_")[1].split("_")[0])

    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = args.num_threads
    simpli.pixel_size = p.pixel_size
    simpli.optical_sigma = p.sigma  # in voxel size
    simpli.filter_rotations = np.linspace(0, np.pi,
                                          CONFIG.simulation.num_filter_rot,
                                          False)
    simpli.interpolate = "Slerp"
    simpli.untilt_sensor_view = True
    simpli.wavelength = CONFIG.simulation.wavelength  # in nm
    simpli.light_intensity = p.intensity  # a.u.
    simpli.fiber_bundles = fiber_bundles
    simpli.tilts = np.deg2rad(np.array([(0, 0)]))

    simpli.voxel_size = CONFIG.simulation.voxel_size
    simpli.set_voi(CONFIG.simulation.voi[0], CONFIG.simulation.voi[1])

    # print(simpli.dim_origin)
    # simpli.dim_origin[:2] = rnd_dim_origin
    # print(simpli.dim_origin)

    if p.species == 'Roden':
        SPECIES = CONFIG.species.roden
    elif p.species == 'Vervet':
        SPECIES = CONFIG.species.vervet
    elif p.species == 'Human':
        SPECIES = CONFIG.species.human
    else:
        raise ValueError('wrong species')

    simpli.fiber_bundles.layers = [[(0.75, 0, SPECIES.mu, 'b'),
                                    (1.0, p.dn, SPECIES.mu, p.model)]
                                  ] * len(fiber_bundles)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message="objects overlap")
    label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline()

    # Simulate PLI Measurement
    for tilt in simpli._tilts:
        theta, phi = tilt[0], tilt[1]
        images = simpli.run_simulation(label_field, vector_field,
                                       tissue_properties, theta, phi)

        # absorption
        # images *= np.exp(-mu * THICKNESS * 1e-3)

        simpli.noise_model = lambda x: np.round(
            np.random.normal(x, np.sqrt(p.gain * x))).astype(np.float32)

        _, images = simpli.apply_optic(images)
        t, d, r = simpli.apply_epa(images)

        df = pd.DataFrame([[
            p.species,
            simpli.voxel_size,
            radius,
            v0,
            p.model,
            p.setup,
            omega,
            psi,
            simpli.pixel_size,
            p.dn,
            p.mu,
            t,
            d,
            r,
        ]],
                          columns=[
                              "species",
                              "voxel_size",
                              "radius",
                              "v0",
                              "model",
                              "setup",
                              "omega",
                              "psi",
                              "pixel_size",
                              "dn",
                              "mu",
                              "transmittance",
                              "direction",
                              "retardation",
                          ])
    return df


def main():

    files = glob.glob(
        "/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/cube_2pop_psi_1.00_omega_0.00_r_*.solved.h5"
    )

    parameters = []
    for file in files:
        for fn in np.arange(1, 5.001, 1):
            for dn, model in [(-0.001 * fn, 'p'), (0.002 * fn, 'r')]:
                for setup, SETUP in [('PM', CONFIG.simulation.setup.pm),
                                     ('LAP', CONFIG.simulation.setup.lap)]:
                    for species, mu in [('Human', CONFIG.species.human.mu),
                                        ('Vervet', CONFIG.species.vervet.mu),
                                        ('Roden', CONFIG.species.roden.mu)]:
                        parameters.append(
                            Parameter(file=file,
                                      dn=dn,
                                      model=model,
                                      setup=setup,
                                      gain=SETUP.gain,
                                      intensity=SETUP.light_intensity,
                                      pixel_size=SETUP.pixel_size,
                                      sigma=SETUP.sigma,
                                      species=species,
                                      mu=mu))

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]

    df = pd.concat(df, ignore_index=True)

    df.to_pickle(os.path.join(args.output, "bf.pkl"))


if __name__ == "__main__":
    main()
