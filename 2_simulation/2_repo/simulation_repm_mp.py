import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import typing
import warnings

import fastpli.analysis
import fastpli.io
import fastpli.simulation
import fastpli.tools
import h5py
import helper.file
# import helper.mpi
import numpy as np
import tqdm
import yaml

import models
import parameter

THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]

CONFIG = parameter.get_tupleware()

parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Output path.")

parser.add_argument("-i",
                    "--input",
                    nargs='+',
                    required=True,
                    help="input string.")

parser.add_argument("-p",
                    "--num_proc",
                    type=int,
                    required=True,
                    help="Number of processes.")

parser.add_argument("-m",
                    "--rep_m",
                    type=int,
                    required=True,
                    help="Number of processes.")

# parser.add_argument("--start", type=int, required=True, help="mpi start.")
parser.add_argument('--vervet', default=False, action='store_true')
parser.add_argument('--flat', default=False, action='store_true')
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--radial', default=False, action='store_true')
parser.add_argument('--pm', default=False, action='store_true')

# # DEBUG
# CONFIG = parameter.get_namespace()
# CONFIG.simulation.voxel_size = 1.3


class Parameter(typing.NamedTuple):
    """  """
    file: str
    output: str
    voxel_size: float
    f0_inc: float
    f1_rot: float
    rep_m: int
    vervet_only: bool
    radial_only: bool
    pm_only: bool


def run(p):
    # logger
    # logger = logging.getLogger(f"rank[{MPI.COMM_WORLD.Get_rank()}]")
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler(
    #     os.path.join(
    #         p.output,
    #         f'simulation_{p.voxel_size}_{MPI.COMM_WORLD.Get_size()}_{MPI.COMM_WORLD.Get_rank()}.log',
    #     ), 'a')
    # formatter = logging.Formatter(
    #     '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # logger.info(
    #     f"git: {subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()}")

    # reproducability
    # np.random.seed(42)
    rnd_seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(rnd_seed)

    _, file_name = os.path.split(p.file)
    file_name = os.path.splitext(file_name)[0]
    file_name += f'_vs_{p.voxel_size:.4f}'
    file_name += f'_inc_{p.f0_inc:.2f}'
    file_name += f'_rot_{p.f1_rot:.2f}'
    file_name += f'_msim_{p.rep_m}'
    file_name = os.path.join(p.output, file_name)
    # logger.info(f"input file: {p.file}")
    # logger.info(f"output file: {file_name}")

    if os.path.isfile(file_name + '.h5'):
        # logger.info(f"file exists: {file_name}.h5")
        raise ValueError("file exists")

    with h5py.File(p.file, 'r') as h5f:
        fiber_bundles = fastpli.io.fiber_bundles.load_h5(h5f)
        psi = h5f['/'].attrs["psi"]
        omega = h5f['/'].attrs["omega"]
        radius = h5f['/'].attrs["radius"]
        v0 = h5f['/'].attrs["v0"]

        if 'rep_n' in h5f['/'].attrs:
            rep_n = h5f['/'].attrs['rep_n']
        else:
            rep_n = 0

    # logger.info(f"omega: {omega}")
    # logger.info(f"psi: {psi}")
    # logger.info(f"inclination : {p.f0_inc}")
    # logger.info(f"rotation : {p.f1_rot}")

    rot_inc = fastpli.tools.rotation.y(-np.deg2rad(p.f0_inc))
    rot_phi = fastpli.tools.rotation.x(np.deg2rad(p.f1_rot))
    rot = np.dot(rot_inc, rot_phi)

    with h5py.File(file_name + '.h5', 'w-') as h5f:
        with open(os.path.abspath(__file__), 'r') as script:
            h5f.attrs['script'] = script.read()
            h5f.attrs['input_file'] = p.file
            h5f.attrs['rnd_seed'] = rnd_seed
            h5f.attrs['parameter/rep_n'] = rep_n

        model_list = ['r', 'p']
        if p.radial_only:
            model_list = ['r']

        for model in model_list:
            setup_list = [('PM', CONFIG.simulation.setup.pm),
                          ('LAP', CONFIG.simulation.setup.lap)]
            if p.pm_only:
                setup_list = [('PM', CONFIG.simulation.setup.pm)]

            for setup_name, SETUP in setup_list:

                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                warnings.filterwarnings("ignore", message="objects overlap")
                simpli.omp_num_threads = 1
                simpli.voxel_size = CONFIG.simulation.voxel_size
                simpli.pixel_size = SETUP.pixel_size
                simpli.filter_rotations = np.linspace(
                    0, np.pi, CONFIG.simulation.num_filter_rot, False)
                simpli.interpolate = "Slerp"
                simpli.wavelength = CONFIG.simulation.wavelength  # in nm
                simpli.optical_sigma = CONFIG.simulation.optical_sigma  # in pixel size
                simpli.verbose = 0

                simpli.set_voi(CONFIG.simulation.voi[0],
                               CONFIG.simulation.voi[1])
                tilt_angle = SETUP.tilt_angle
                simpli.tilts = np.deg2rad(
                    np.array([(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                              (tilt_angle, 180), (tilt_angle, 270)]))
                simpli.add_crop_tilt_halo()

                simpli.fiber_bundles = fiber_bundles.rotate(rot)

                LAYERS = CONFIG.models.layers
                layers = [(LAYERS.b.radius, LAYERS.b.dn, 0, LAYERS.b.model)]
                if model == 'r':
                    layers.append(
                        (LAYERS.r.radius, LAYERS.r.dn, 0, LAYERS.r.model))
                elif model == 'p':
                    layers.append(
                        (LAYERS.p.radius, LAYERS.p.dn, 0, LAYERS.p.model))
                else:
                    raise ValueError('FOOO')
                simpli.fiber_bundles.layers = [layers] * len(fiber_bundles)
                # logger.info(
                #     f"tissue_pipeline: model: {[layers] * len(fiber_bundles)}")

                tissue, optical_axis, tissue_properties = simpli.run_tissue_pipeline(
                )

                # Simulate PLI Measurement
                # logger.info(f"simulation_pipeline: model:{model}")

                simpli.light_intensity = SETUP.light_intensity  # a.u.

                gain = SETUP.gain
                simpli.noise_model = lambda x: np.round(
                    np.random.normal(x, np.sqrt(gain * x))).astype(np.float32)

                dset = h5f.create_group(f'{setup_name}/{model}')
                simpli.save_parameter_h5(h5f=dset)
                if 'tissue_stats' not in dset:
                    unique_elements, counts_elements = np.unique(
                        tissue, return_counts=True)
                    dset.attrs['tissue_stats'] = np.asarray(
                        (unique_elements, counts_elements))

                species_list = [('Human', CONFIG.species.human.mu),
                                ('Vervet', CONFIG.species.vervet.mu),
                                ('Roden', CONFIG.species.roden.mu)]

                if p.vervet_only:
                    species_list = [('Vervet', CONFIG.species.vervet.mu)]

                for species, mu in species_list:
                    tissue_properties[1:,
                                      1] = mu  # fiber absorp, background not

                    dset = h5f.create_group(f'{setup_name}/{species}/{model}')
                    dset.create_group('simulation')
                    dset['simulation'].attrs['tilt_angle'] = tilt_angle

                    tilting_stack = [None] * 5
                    for t, (theta, phi) in enumerate(simpli.tilts):
                        images = simpli.run_simulation(tissue, optical_axis,
                                                       tissue_properties, theta,
                                                       phi)

                        images = simpli.rm_crop_tilt_halo(images)

                        # apply optic to simulation
                        resample, images = simpli.apply_optic(images)
                        dset[f'simulation/optic/{t}'] = images
                        dset[f'simulation/resample/{t}'] = resample
                        dset['simulation/optic'].attrs['theta'] = theta
                        dset['simulation/optic'].attrs['phi'] = phi

                        # calculate modalities
                        epa = simpli.apply_epa(images)
                        dset[f'analysis/epa/{t}/transmittance'] = epa[0]
                        dset[f'analysis/epa/{t}/direction'] = epa[1]
                        dset[f'analysis/epa/{t}/retardation'] = epa[2]

                        tilting_stack[t] = images

                    mask = None  # keep analysing all pixels

                    # print('Run ROFL analysis:')
                    rofl_direction, rofl_incl, rofl_t_rel, param = simpli.apply_rofl(
                        tilting_stack, mask=mask)

                    dset['analysis/rofl/direction'] = rofl_direction
                    dset['analysis/rofl/inclination'] = rofl_incl
                    dset['analysis/rofl/t_rel'] = rofl_t_rel

                    dset['analysis/rofl/direction_conf'] = param[0]
                    dset['analysis/rofl/inclination_conf'] = param[1]
                    dset['analysis/rofl/t_rel_conf'] = param[2]
                    dset['analysis/rofl/func'] = param[3]
                    dset['analysis/rofl/n_iter'] = param[4]

                    dset.attrs['parameter/CONFIG'] = yaml.dump(CONFIG)

                    for name, value in p._asdict().items():
                        dset.attrs[f'parameter/p/{name}'] = value

                    dset.attrs['parameter/simpli'] = str(simpli.get_dict())
                    dset.attrs['parameter/rep_n'] = rep_n
                    dset.attrs['parameter/rep_m'] = p.rep_m
                    dset.attrs['parameter/v0'] = v0
                    dset.attrs['parameter/radius'] = radius
                    dset.attrs['parameter/psi'] = psi
                    dset.attrs['parameter/omega'] = omega
                    dset.attrs['parameter/fiber_path'] = p.file
                    dset.attrs['parameter/volume'] = CONFIG.simulation.volume
                    dset.attrs['parameter/f0_inc'] = p.f0_inc
                    dset.attrs['parameter/f1_rot'] = p.f1_rot
                    dset.attrs[
                        'parameter/crop_tilt_voxel'] = simpli.crop_tilt_voxel()

                h5f.flush()
                del tissue
                del optical_axis
                del simpli


def main():

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    subprocess.run([f'touch {args.output}/$(git rev-parse HEAD)'], shell=True)
    subprocess.run([f'touch {args.output}/$(hostname)'], shell=True)

    for f in args.input:
        if not os.path.isfile(f):
            raise ValueError('not a file')

    # Memory Check
    simpli = fastpli.simulation.Simpli()
    simpli.voxel_size = CONFIG.simulation.voxel_size  # in mu meter
    simpli.pixel_size = CONFIG.simulation.setup.pm.pixel_size  # in mu meter
    simpli.set_voi(CONFIG.simulation.voi[0], CONFIG.simulation.voi[1])
    simpli.tilts = np.deg2rad(
        np.array([(CONFIG.simulation.setup.pm.tilt_angle, 0)]))
    simpli.add_crop_tilt_halo()

    print(f"Single Memory: {simpli.memory_usage():.0f} MB")
    print(f"Total Memory: {simpli.memory_usage()* args.num_proc:.0f} MB")
    del simpli

    parameters = []
    file_list = args.input

    if args.single:
        fiber_inc = [(f, i) for f in file_list for i in models.inclinations(2)]
        for file, f0_inc in fiber_inc:
            for m in range(args.rep_m):
                parameters.append(
                    Parameter(file=file,
                              output=args.output,
                              voxel_size=CONFIG.simulation.voxel_size,
                              f0_inc=f0_inc,
                              f1_rot=0,
                              rep_m=m,
                              vervet_only=args.vervet,
                              radial_only=args.radial,
                              pm_only=args.pm))
    elif args.flat:
        for file in file_list:
            for m in range(args.rep_m):
                parameters.append(
                    Parameter(file=file,
                              output=args.output,
                              voxel_size=CONFIG.simulation.voxel_size,
                              f0_inc=0,
                              f1_rot=0,
                              rep_m=m,
                              vervet_only=args.vervet,
                              radial_only=args.radial,
                              pm_only=args.pm))
    else:
        raise ValueError('Wrong input arguments')

    # DEBUG
    # run(parameters[0])
    # parameters = parameters[:10]
    with mp.Pool(args.num_proc) as pool:
        [
            _ for _ in tqdm.tqdm(pool.imap_unordered(run, parameters),
                                 total=len(parameters),
                                 smoothing=0)
        ]


if __name__ == "__main__":
    main()
