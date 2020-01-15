import numpy as np
import copy
import h5py
import os
import sys
import glob

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

from mpi4py import MPI
from tqdm import tqdm
import imageio


def data2image(data):
    return np.swapaxes(np.flip(data, 1), 0, 1)


comm = MPI.COMM_WORLD
NUM_THREADS = 4

# reproducability
np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

# PARAMETERS
PIXEL_SIZE_LAP = 65
PIXEL_SIZE_PM = 1.25
NUM_LAP_PIXEL = 1
THICKNESS = 60

if len(sys.argv) > 1:
    FILE_PATH = sys.argv[1]

# FIBER_ROTATIONS_PHI = np.linspace(0, 90, 10, True)

os.makedirs(os.path.join(FILE_PATH, 'output', 'simulations'), exist_ok=True)
file_list = sorted(
    glob.glob(os.path.join(FILE_PATH, 'output', 'models', '*.solved*.h5')))

for file in tqdm(file_list[comm.Get_rank()::comm.Get_size()]):

    tqdm.write(file)

    tqdm.write('loading models')
    fiber_bundles = fastpli.io.fiber.load(file, 'fiber_bundles')

    dphi = float(file.split("dphi_")[-1].split("_psi")[0])
    n_rot = int(
        np.round(
            np.sqrt((1 - np.cos(2 * np.deg2rad(dphi))) /
                    (1 - np.cos(np.deg2rad(10))))))

    if n_rot > 0:
        n_rot = max(n_rot, 2)

    FIBER_ROTATIONS_PHI = np.linspace(0, 90, n_rot, True)
    tqdm.write("{}: {}".format(str(dphi), str(n_rot)))

    if n_rot == 0:
        FIBER_ROTATIONS_PHI = [0]

    for f_phi in FIBER_ROTATIONS_PHI:
        tqdm.write("rotation: " + str(f_phi))

        _, file_name = os.path.split(file)
        file_name = os.path.splitext(file_name)[0]
        file_name += '_phi_{:.2f}'.format(f_phi)

        file_name = fastpli.tools.helper.version_file_name(
            os.path.join(FILE_PATH, 'output/simulations', file_name))

        tqdm.write(file_name)
        with h5py.File(file_name + '.h5', 'w-') as h5f:

            # save script
            h5f['version'] = fastpli.__version__
            with open(os.path.abspath(__file__), 'r') as f:
                h5f['script'] = f.read()
                h5f['pip_freeze'] = fastpli.tools.helper.pip_freeze()

            # Setup Simpli for Tissue Generation
            simpli = fastpli.simulation.Simpli()
            simpli.omp_num_threads = NUM_THREADS
            simpli.voxel_size = 0.25  # in mu meter
            simpli.set_voi([
                -0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP,
                -0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP, -0.5 * THICKNESS
            ], [
                0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP,
                0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP, 0.5 * THICKNESS
            ])  # in mu meter

            tqdm.write('Memory: ' + str(simpli.memory_usage()))

            tqdm.write('rotating models')
            rot = fastpli.tools.rotation.x(np.deg2rad(f_phi))
            simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                copy.deepcopy(fiber_bundles), rot)

            tqdm.write('starting simlation')
            for m, (dn, model) in enumerate([(-0.001, 'p'), (0.001, 'r')]):
                tqdm.write("tissue: " + str(m) + ', ' + str(dn) + ', ' +
                           str(model))
                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 1, model)]
                                                  ] * len(fiber_bundles)

                # Generate Tissue
                label_field, vec_field, tissue_properties = simpli.generate_tissue(
                )

                if m == 0:
                    dset = h5f.create_dataset('tissue',
                                              label_field.shape,
                                              dtype=np.uint16,
                                              compression='gzip',
                                              compression_opts=1)
                    dset[:] = label_field

                # h5f['vectorfield'] = vec_field
                h5f[model + '/tissue_properties'] = tissue_properties

                # Simulate PLI Measurement
                for name, gain, intensity, res, tilt_angle in [
                    ('LAP', 3, 26000, PIXEL_SIZE_LAP, 5.5),
                    ('PM', 1.5, 50000, PIXEL_SIZE_PM, 3.9)
                ]:
                    tqdm.write('simulation: ' + ', '.join(
                        tuple([
                            str(x)
                            for x in (name, gain, intensity, res, tilt_angle)
                        ])))

                    simpli.filter_rotations = np.deg2rad(
                        [0, 30, 60, 90, 120, 150])
                    simpli.light_intensity = intensity  # a.u.
                    simpli._untilt_sensor_view = True
                    simpli.wavelength = 525  # in nm
                    simpli.resolution = res  # in mu meter

                    TILTS = [(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                             (tilt_angle, 180), (tilt_angle, 270)]

                    tilting_stack = [None] * len(TILTS)

                    if name is 'PM':
                        label_field = np.flip(label_field, 2)
                        vec_field = np.flip(vec_field, 2)

                    for t, (theta, phi) in enumerate(TILTS):
                        tqdm.write('tilt: ' + str(theta) + ', ' + str(phi))

                        images = simpli.run_simulation(label_field, vec_field,
                                                       tissue_properties,
                                                       np.deg2rad(theta),
                                                       np.deg2rad(phi))
                        h5f[name + '/' + model + '/data/' + str(t)] = images

                        # apply optic to simulation
                        # tqdm.write('optic')
                        new_images = simpli.apply_optic(images, gain=gain)
                        h5f[name + '/' + model + '/optic/' +
                            str(t)] = new_images

                        # calculate modalities
                        # tqdm.write('epa')
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
                    mask = simpli.apply_resize(1.0 * mask) > 0.1
                    h5f[name + '/' + model + '/optic/mask'] = np.uint8(mask)
                    mask = None  # keep analysing all pixels

                    tqdm.write('rofl')

                    tilting_stack = np.array(tilting_stack)
                    while tilting_stack.ndim < 4:
                        tilting_stack = np.expand_dims(tilting_stack, axis=-2)

                    rofl_direction, rofl_incl, rofl_t_rel, _ = simpli.apply_rofl(
                        tilting_stack,
                        tilt_angle=np.deg2rad(tilt_angle),
                        gain=gain,
                        mask=mask)

                    h5f[name + '/' + model +
                        '/rofl/direction'] = np.rad2deg(rofl_direction)
                    h5f[name + '/' + model +
                        '/rofl/inclination'] = np.rad2deg(rofl_incl)
                    h5f[name + '/' + model + '/rofl/trel'] = rofl_t_rel
                    h5f[name + '/' + model].attrs['simpli'] = str(
                        simpli.as_dict())

                    if name != "LAP":
                        imageio.imwrite(
                            file_name + '_' + name + '_' + model + '.png',
                            data2image(
                                fastpli.analysis.images.fom_hsv_black(
                                    rofl_direction, rofl_incl)))
