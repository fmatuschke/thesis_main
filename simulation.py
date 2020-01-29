import numpy as np
import copy
import h5py
import argparse
import logging
import glob
import sys
import os
import warnings

import fastpli.simulation
import fastpli.objects
import fastpli.tools

from tqdm import tqdm

from MPIFileHandler import MPIFileHandler
from mpi4py import MPI
comm = MPI.COMM_WORLD
import multiprocessing as mp
NUM_THREADS = 4
mp_pool = mp.Pool(NUM_THREADS)

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)


def run_simulation_pipeline_n(simpli,
                              label_field,
                              vector_field,
                              tissue_properties,
                              n_repeat,
                              h5f=None,
                              crop_tilt=False,
                              mp_pool=None):
    '''
    copy from _simpli.py
    '''

    if simpli._tilts is None:
        raise ValueError("tilts is not set")
    if simpli._optical_sigma is None:
        raise ValueError("optical_sigma is not set")

    flag_rofl = True
    if np.any(simpli._tilts[:, 1] != np.deg2rad([0, 0, 90, 180, 270])
             ) or simpli._tilts[0, 0] != 0 or np.any(
                 simpli._tilts[1:, 0] != simpli._tilts[1, 0]):
        warnings.warn("Tilts not suitable for ROFL. Skipping analysis")
        flag_rofl = False

    tilting_stack = [None] * len(simpli._tilts)

    simpli._print("Simulate tilts:")
    for t, tilt in enumerate(simpli._tilts):
        theta, phi = tilt[0], tilt[1]
        simpli._print("{}: theta: {} deg, phi: {} deg".format(
            t, round(np.rad2deg(theta), 2), round(np.rad2deg(phi), 2)))

        images_n = []
        new_images_n = []
        for n in range(n_repeat):
            images = simpli.run_simulation(label_field, vector_field,
                                           tissue_properties, theta, phi)

            if crop_tilt:
                delta_voxel = simpli.crop_tilt_voxel()
                print(delta_voxel)
                images = images[delta_voxel:-1 - delta_voxel,
                                delta_voxel:-1 - delta_voxel, :]

            images_n.append(images)

            # apply optic to simulation
            new_images = simpli.apply_optic(images, mp_pool=mp_pool)
            new_images_n.append(new_images)

        images = np.vstack(images_n)
        new_images = np.vstack(new_images_n)

        if h5f:
            h5f['simulation/data/' + str(t)] = images
            h5f['simulation/data/' + str(t)].attrs['theta'] = theta
            h5f['simulation/data/' + str(t)].attrs['phi'] = phi
        if h5f:
            h5f['simulation/optic/' + str(t)] = new_images
            h5f['simulation/optic/' + str(t)].attrs['theta'] = theta
            h5f['simulation/optic/' + str(t)].attrs['phi'] = phi

        # calculate modalities
        epa = simpli.apply_epa(new_images)

        if h5f:
            h5f['analysis/epa/' + str(t) + '/transmittance'] = epa[0]
            h5f['analysis/epa/' + str(t) + '/direction'] = np.rad2deg(epa[1])
            h5f['analysis/epa/' + str(t) + '/retardation'] = epa[2]

            h5f['analysis/epa/' + str(t) +
                '/transmittance'].attrs['theta'] = theta
            h5f['analysis/epa/' + str(t) + '/transmittance'].attrs['phi'] = phi
            h5f['analysis/epa/' + str(t) + '/direction'].attrs['theta'] = theta
            h5f['analysis/epa/' + str(t) + '/direction'].attrs['phi'] = phi
            h5f['analysis/epa/' + str(t) +
                '/retardation'].attrs['theta'] = theta
            h5f['analysis/epa/' + str(t) + '/retardation'].attrs['phi'] = phi

        tilting_stack[t] = new_images

    # pseudo mask
    mask = np.sum(label_field, 2) > 0
    mask = simpli.apply_optic_resample(1.0 * mask, mp_pool=mp_pool) > 0.1
    h5f['simulation/optic/mask'] = np.uint8(mask)

    tilting_stack = np.array(tilting_stack)
    while tilting_stack.ndim < 4:
        tilting_stack = np.expand_dims(tilting_stack, axis=-2)

    if flag_rofl:
        simpli._print("Analyse tilts")
        rofl_direction, rofl_incl, rofl_t_rel, (
            rofl_direction_conf, rofl_incl_conf, rofl_t_rel_conf, rofl_func,
            rofl_n_iter) = simpli.apply_rofl(tilting_stack,
                                             mask=None,
                                             mp_pool=mp_pool)
    else:
        rofl_direction = None
        rofl_incl = None
        rofl_t_rel = None

        rofl_direction_conf = None
        rofl_incl_conf = None
        rofl_t_rel_conf = None
        rofl_func = None
        rofl_n_iter = None

    if h5f and flag_rofl:
        h5f['analysis/rofl/direction'] = rofl_direction
        h5f['analysis/rofl/inclination'] = rofl_incl
        h5f['analysis/rofl/t_rel'] = rofl_t_rel

        h5f['analysis/rofl/direction_conf'] = rofl_direction_conf,
        h5f['analysis/rofl/inclination_conf'] = rofl_incl_conf,
        h5f['analysis/rofl/t_rel_conf'] = rofl_t_rel_conf,
        h5f['analysis/rofl/func'] = rofl_func,
        h5f['analysis/rofl/n_iter'] = rofl_n_iter

    if flag_rofl:
        fom = fom_hsv_black(rofl_direction, rofl_incl)
    else:
        fom = None

    return tilting_stack, (rofl_direction, rofl_incl, rofl_t_rel), fom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="Output path of solver.")

    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="input regex string.")

    args = parser.parse_args()
    os.makedirs(os.path.join(args.output, 'simulations'), exist_ok=True)

    # logger
    logger = logging.getLogger("rank[%i]" % comm.rank)
    logger.setLevel(logging.DEBUG)

    log_file = fastpli.tools.helper.version_file_name(
        os.path.join(args.output, 'simulation')) + ".log"
    mh = MPIFileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:\t%(message)s')
    mh.setFormatter(formatter)
    logger.addHandler(mh)

    # PARAMETER
    VOXEL_SIZE = 0.1
    LENGTH = 65
    THICKNESS = 60

    file_list = sorted(
        glob.glob(os.path.join(args.output, 'models', args.input)))
    if not file_list:
        logger.error("no files detected")

    # print Memory
    simpli = fastpli.simulation.Simpli()
    simpli.voxel_size = VOXEL_SIZE  # in mu meter
    simpli.set_voi(-0.5 * np.array([65, 65, 60]),
                   0.5 * np.array([65, 65, 60]))  # in mu meter

    logger.info(f"Single Memory: {round(simpli.memory_usage())} MB")
    logger.info(
        f"Total Memory: {round(simpli.memory_usage()* comm.Get_size())} MB")
    del simpli

    # simulation loop
    for file in tqdm(file_list[comm.Get_rank()::comm.Get_size()]):
        logger.info(f"input file: {file}")

        # Loading fiber models and prepair rotations
        fiber_bundles = fastpli.io.fiber.load(file, 'fiber_bundles')

        with h5py.File(file, 'r') as h5f:
            psi = h5f['fiber_bundles'].attrs["psi"]
            dphi = h5f['fiber_bundles'].attrs["dphi"]

        n_rot = int(
            np.round(
                np.sqrt((1 - np.cos(2 * np.deg2rad(dphi))) /
                        (1 - np.cos(np.deg2rad(10))))))

        if n_rot > 0:
            n_rot = max(n_rot, 2)

        fiber_phi_rotations = np.linspace(0, 90, n_rot, True)
        logger.info(f"dphi: {dphi}, n_rot: {n_rot}")

        if n_rot == 0:
            fiber_phi_rotations = [0]

        for f_phi in fiber_phi_rotations:
            _, file_name = os.path.split(file)
            file_name = os.path.splitext(file_name)[0]
            file_name += '_phi_{:.2f}'.format(f_phi)
            file_name = fastpli.tools.helper.version_file_name(
                os.path.join(args.output, 'simulations', file_name))

            logger.info(f"rotation : {f_phi}")
            logger.info(f"output file: {file_name}")

            rot = fastpli.tools.rotation.x(np.deg2rad(f_phi))

            with h5py.File(file_name + '.h5', 'w-') as h5f:
                with open(os.path.abspath(__file__), 'r') as script:
                    h5f['parameter/script'] = script.read()
                h5f['parameter/input_file'] = file

                for m, (dn, model) in enumerate([(-0.001, 'p'), (0.002, 'r')]):
                    for name, gain, intensity, res, tilt_angle, sigma in [
                        ('LAP', 3, 26000, 20, 5.5, 0),
                        ('PM', 1.5, 50000, 1.25, 3.9, 0.71)
                    ]:
                        dset = h5f.create_group(name + '/' + model)

                        # Setup Simpli
                        simpli = fastpli.simulation.Simpli()
                        simpli.omp_num_threads = NUM_THREADS
                        simpli.voxel_size = VOXEL_SIZE
                        simpli.resolution = res
                        simpli.filter_rotations = np.deg2rad(
                            [0, 30, 60, 90, 120, 150])
                        simpli.interpolate = True
                        simpli.wavelength = 525  # in nm
                        simpli.optical_sigma = 0.71  # in pixel size
                        simpli.verbose = 0

                        simpli.set_voi(
                            -0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                            0.5 * np.array([LENGTH, LENGTH, THICKNESS]))
                        simpli.tilts = np.deg2rad(
                            np.array([(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                                      (tilt_angle, 180), (tilt_angle, 270)]))
                        simpli.add_crop_tilt_halo()

                        simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                            copy.deepcopy(fiber_bundles), rot)
                        simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                            (1.0, dn, 1, model)]
                                                          ] * len(fiber_bundles)

                        logger.info(f"tissue_pipeline: model:{model}")
                        label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                            h5f=dset, save=["label_field"])

                        # Simulate PLI Measurement
                        logger.info(f"simulation_pipeline: model:{model}")
                        # TODO: LAP run multiple pixels for statistics
                        # FIXME: LAP sigma ist bei einem pixel sinnfrei

                        simpli.light_intensity = intensity  # a.u.
                        simpli.sensor_gain = gain

                        simpli.save_parameter(h5f=dset)

                        if name == 'LAP':
                            run_simulation_pipeline_n(simpli,
                                                      label_field,
                                                      vector_field,
                                                      tissue_properties,
                                                      int(20 / 1.25),
                                                      h5f=dset,
                                                      crop_tilt=True,
                                                      mp_pool=mp_pool)
                        else:
                            simpli.run_simulation_pipeline(label_field,
                                                           vector_field,
                                                           tissue_properties,
                                                           h5f=dset,
                                                           crop_tilt=True,
                                                           mp_pool=mp_pool)

                        dset['parameter/model/psi'] = psi
                        dset['parameter/model/dphi'] = dphi
                        dset['parameter/model/phi'] = f_phi

                        del label_field
                        del vector_field
                        del simpli


if __name__ == "__main__":
    try:
        main()
    except:
        comm.Abort()
