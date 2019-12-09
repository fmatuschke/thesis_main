import numpy as np
import h5py
import os
import glob
import sys

import fastpli.simulation
import fastpli.analysis
import fastpli.io
import fastpli_helper as helper

import multiprocessing as mp

from mpi4py import MPI
from tqdm import tqdm
from time import sleep
import imageio

NUM_PROC = 8


def data2image(data):
    return np.swapaxes(np.flip(data, 1), 0, 1)


mp_pool = mp.Pool(NUM_PROC)

VOI = [(-200, -200, 0), (200, 200, 60)]

# reproducability
np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

# PARAMETERS
PIXEL_SIZE_LAP = 65
PIXEL_SIZE_PM = 1.25
FACTOR = 10
VOXEL_SIZE = PIXEL_SIZE_PM / FACTOR
THICKNESS = 60

comm = MPI.COMM_WORLD
os.makedirs(os.path.join(FILE_PATH, 'output', 'simulations'), exist_ok=True)

input_file_name = sys.argv[1]

h5_file_name = os.path.join(FILE_PATH, 'output/simulations', input_file_name)
h5_file_name = helper.version_file_name(h5_file_name) + '.h5'
output_base_name = os.path.basename(h5_file_name)

with h5py.File(os.path.join(FILE_PATH, 'output/simulations/', h5_file_name),
               'w-') as h5f:

    # save script
    h5f['version'] = fastpli.__version__
    with open(os.path.abspath(__file__), 'r') as f:
        h5f['script'] = f.read()
        h5f['pip_freeze'] = helper.pip_freeze()

    # Setup Simpli for Tissue Generation
    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = NUM_PROC
    simpli.voxel_size = VOXEL_SIZE  # in mu meter
    simpli.set_voi(VOI[0], VOI[1])  # in mu meter

    tqdm.write('Memory: ' + str(simpli.memory_usage()))
    # sleep(10)

    tqdm.write('loading models')
    simpli.fiber_bundles = fastpli.io.fiber.load(
        os.path.join(FILE_PATH, 'output/models', input_file_name),
        '/fiber_bundles/')

    tqdm.write('starting simlation')
    for m, (dn, model) in enumerate([(-0.001, 'p'), (0.002, 'r')]):
        tqdm.write("tissue: " + str(m) + ', ' + str(dn) + ', ' + str(model))

        simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                            (1.0, dn, 1, model)]] * len(
                                                simpli.fiber_bundles)

        # Generate Tissue
        label_field, vec_field, tissue_properties = simpli.generate_tissue()

        if m == 0:
            dset = h5f.create_dataset('tissue_222',
                                      np.array(label_field.shape) // 2,
                                      dtype=np.uint8,
                                      compression='gzip',
                                      compression_opts=1)
            dset[:] = label_field[::2, ::2, ::2]

        # h5f['vectorfield'] = vec_field
        h5f[model + '/tissue_properties'] = tissue_properties

        # Simulate PLI Measurement
        for name, gain, intensity, res, tilt_angle in [
                # ('LAP', 3, 26000, PIXEL_SIZE_LAP, 5.5),
            ('PM', 1.5, 50000, PIXEL_SIZE_PM, 3.9)  # aachen tilt pm
        ]:
            tqdm.write('simulation: ' + ', '.join(
                tuple(
                    [str(x)
                     for x in (name, gain, intensity, res, tilt_angle)])))

            simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
            simpli.light_intensity = intensity  # a.u.
            simpli.untilt_sensor_view = True
            simpli.wavelength = 525  # in nm
            simpli.resolution = res  # in mu meter

            if name == 'LAP':
                simpli.flip_z_beam = False
            elif name == 'PM':
                simpli.flip_z_beam = True
            else:
                raise ValueError("name error")

            TILTS = [(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                     (tilt_angle, 180), (tilt_angle, 270)]

            tilting_stack = [None] * len(TILTS)

            for t, (theta, phi) in tqdm(enumerate(TILTS)):
                tqdm.write('tilt: ' + str(theta) + ', ' + str(phi))

                images = simpli.run_simulation(label_field, vec_field,
                                               tissue_properties,
                                               np.deg2rad(theta),
                                               np.deg2rad(phi))
                h5f[name + '/' + model + '/data/' + str(t)] = images

                # apply optic to simulation
                tqdm.write('optic')
                new_images = simpli.apply_optic(images,
                                                gain=gain,
                                                mp_pool=mp_pool)
                h5f[name + '/' + model + '/optic/' + str(t)] = new_images

                # calculate modalities
                tqdm.write('epa')
                epa = simpli.apply_epa(new_images)
                h5f[name + '/' + model + '/epa/' + str(t) +
                    '/transmittance'] = epa[0]
                h5f[name + '/' + model + '/epa/' + str(t) +
                    '/direction'] = np.rad2deg(epa[1])
                h5f[name + '/' + model + '/epa/' + str(t) +
                    '/retardation'] = epa[2]

                tilting_stack[t] = new_images

            # save mask for analysis
            mask = np.sum(label_field, 2) > 0
            mask = simpli.apply_resize(1.0 * mask, mp_pool=mp_pool) > 0.1
            h5f[name + '/' + model + '/optic/mask'] = np.uint8(mask)
            mask = None  # keep analysing all pixels

            tqdm.write('rofl')
            rofl_direction, rofl_incl, rofl_t_rel, _ = simpli.apply_rofl(
                tilting_stack,
                tilt_angle=np.deg2rad(tilt_angle),
                gain=gain,
                mask=mask,
                mp_pool=mp_pool)

            h5f[name + '/' + model +
                '/rofl/direction'] = np.rad2deg(rofl_direction)
            h5f[name + '/' + model +
                '/rofl/inclination'] = np.rad2deg(rofl_incl)
            h5f[name + '/' + model + '/rofl/trel'] = rofl_t_rel
            h5f[name + '/' + model].attrs['simpli'] = str(simpli.as_dict())

            imageio.imwrite(
                'output/simulations/' + output_base_name + '_' + name + '_' +
                model + '.png',
                data2image(
                    fastpli.analysis.images.fom_hsv_black(
                        rofl_direction, rofl_incl)))
