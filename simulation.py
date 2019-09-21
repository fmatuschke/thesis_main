import numpy as np
import h5py
import os
import glob

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.io

from mpi4py import MPI
from tqdm import tqdm
import imageio


def data2image(data):
    return np.swapaxes(np.flip(data, 1), 0, 1)


# reproducability
np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

# PARAMETERS
PIXEL_SIZE_LAP = 65
PIXEL_SIZE_PM = 1.25
NUM_LAP_PIXEL = 5
THICKNESS = 60

FIBER_ROTATIONS_PHI = [0, 45]
FIBER_ROTATIONS_THETA = [0, 20, 40, 60, 70, 80, 90]
FIBER_ROTATIONS = []
for phi in FIBER_ROTATIONS_PHI:
    for theta in FIBER_ROTATIONS_THETA:
        FIBER_ROTATIONS.append((phi, theta))

comm = MPI.COMM_WORLD
os.makedirs(os.path.join(FILE_PATH, 'output', 'simulations'), exist_ok=True)
file_list = glob.glob(os.path.join(FILE_PATH, 'output', 'models', '*.h5'))

for file in tqdm(file_list[comm.Get_rank()::comm.Get_size()]):
    file_name = os.path.basename(file)
    tqdm.write(file_name)

    for (f_phi, f_theta) in FIBER_ROTATIONS:

        file_name = os.path.splitext(file_name)[0] + '_theta_' + str(
            f_theta) + '_phi_' + str(f_phi) + '.h5'

        f_phi = np.deg2rad(f_phi)
        f_theta = np.deg2rad(f_theta)
        with h5py.File(
                os.path.join(FILE_PATH, 'output/simulations/', file_name),
                'w-') as h5f:

            # save script
            h5f['version'] = fastpli.__version__
            with open(os.path.abspath(__file__), 'r') as f:
                h5f['script'] = f.read()

            # Setup Simpli for Tissue Generation
            simpli = fastpli.simulation.Simpli()
            simpli.omp_num_threads = 4
            simpli.voxel_size = 0.25  # in mu meter
            simpli.set_voi([
                -0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP,
                0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP,
                -0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP,
                0.5 * NUM_LAP_PIXEL * PIXEL_SIZE_LAP, -0.5 * THICKNESS,
                0.5 * THICKNESS
            ])  # in mu meter

            tqdm.write('Memory: ' + str(simpli.memory_usage()))

            tqdm.write('loading models')
            rot = fastpli.tools.rotation.a_on_b([0, 0, 1], [
                np.sin(f_theta) * np.cos(f_phi),
                np.sin(f_theta) * np.sin(f_phi),
                np.cos(f_theta)
            ])
            simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                fastpli.io.fiber.load(file, 'fiber_bundles'), rot)

            tqdm.write('starting simlation')
            for m, (dn, model) in enumerate([(-0.001, 'p'), (0.001, 'r')]):
                tqdm.write("tissue: " + str(m) + ', ' + str(dn) + ', ' +
                           str(model))
                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 1, model)]]

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
                    simpli.untilt_sensor = True
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
                        tqdm.write('optic')
                        new_images = simpli.apply_optic(images, gain=gain)
                        h5f[name + '/' + model + '/optic/' +
                            str(t)] = new_images

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
                    mask = simpli.apply_resize(1.0 * mask) > 0.1
                    h5f[name + '/' + model + '/optic/mask'] = np.uint8(mask)
                    mask = None  # keep analysing all pixels

                    tqdm.write('rofl')
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

                    imageio.imwrite(
                        'output/simulations/' + file_name + '_' + name + '_' +
                        model + '.png',
                        data2image(
                            fastpli.analysis.images.fom_hsv_black(
                                rofl_direction, rofl_incl)))
