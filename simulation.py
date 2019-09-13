import numpy as np
import h5py
import os

import fastpli.simulation
import fastpli.analysis
import fastpli.io

import imageio


def data2image(data):
    return np.swapaxes(np.flip(data, 1), 0, 1)


FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

PHI_LIST = np.linspace(0, 45, 5)
THETA_LIST = np.linspace(0, 90, 5)
RND_LIST = np.array([0, 0.1, 0.2, 0.5])

np.random.seed(42)

PIXEL_LAP_REL = 48
PIXEL_PM_REL = 1
SCALE = PIXEL_LAP_REL / 64  # scale to coordinates/fiber_radius to bring 1 LAP pixel to 64 µm
THICKNESS = np.round(60 * SCALE)
NUM_LAP_PIXEL = 5
FIBER_RADIUS = 2 * SCALE

# mpi
index = 0
comm = MPI.COMM_WORLD

for _phi in tqdm(PHI_LIST):
    for _theta in tqdm(THETA_LIST):

        index += 1
        if index % comm.Get_size() != comm.Get_rank():
            continue

        for _rnd in tqdm(RND_LIST * FIBER_RADIUS):

            file_name = 'phi_' + str(round(_phi, 0)) + '_theta_' + str(
                round(_theta, 0)) + '_rnd_' + str(round(_rnd, 1))

            with h5py.File(
                    os.path.join(
                        FILE_PATH,
                        'output/simulations/simulation_' + file_name + '.h5'),
                    'w') as h5f:

                ### save script ###
                with open(os.path.abspath(__file__), 'r') as f:
                    h5f['script'] = f.read()

                ### Setup Simpli for Tissue Generation
                simpli = fastpli.simulation.Simpli()
                simpli.omp_num_threads = 4
                simpli.voxel_size = 0.2  # in mu meter
                simpli.set_voi([
                    0, NUM_LAP_PIXEL * PIXEL_LAP_REL, 0,
                    NUM_LAP_PIXEL * PIXEL_LAP_REL, 0, THICKNESS
                ])  # in mu meter

                simpli.fiber_bundles = fastpli.io.fiber.load(
                    os.path.join(FILE_PATH,
                                 'output/models/model_' + file_name + '.h5'),
                    'fiber_bundles')

                simpli.fiber_bundles_properties = [[(1.0, -0.001, 1, 'p')]]

                ### Generate Tissue
                label_field, vec_field, tissue_properties = simpli.generate_tissue(
                )

                dset = h5f.create_dataset('tissue',
                                          label_field.shape,
                                          dtype=np.uint16,
                                          compression='gzip',
                                          compression_opts=1)
                dset[:] = label_field

                # h5f['vectorfield'] = vec_field
                h5f['tissue_properties'] = tissue_properties

                ### Simulate PLI Measurement ###

                for name, gain, intensity, res, tilt_angle in [
                    ('LAP', 3, 26000, 64, 5.5), ('PM', 1.5, 50000, 1.33, 3.9)
                ]:

                    simpli.filter_rotations = np.deg2rad(
                        [0, 30, 60, 90, 120, 150])
                    simpli.light_intensity = intensity  # a.u.
                    simpli.untilt_sensor = True
                    simpli.wavelength = 525  # in nm

                    TILTS = [(0, 0), (tilt_angle, 0), (tilt_angle, 90),
                             (tilt_angle, 180), (tilt_angle, 270)]

                    tilting_stack_lap = [None] * len(TILTS)
                    tilting_stack_pm = [None] * len(TILTS)
                    for t, (theta, phi) in enumerate(TILTS):
                        images = simpli.run_simulation(label_field, vec_field,
                                                       tissue_properties,
                                                       np.deg2rad(theta),
                                                       np.deg2rad(phi))
                        h5f[name + '/data/' + str(t)] = images

                        simpli.resolution = res  # in mu meter

                        # apply optic to simulation
                        new_images = simpli.apply_optic(images, gain=gain)
                        h5f[name + '/optic/' + str(t)] = new_images

                        # calculate modalities
                        epa = simpli.apply_epa(new_images)
                        h5f[name + '/epa/' + str(t) + '/transmittance'] = epa[0]
                        h5f[name + '/epa/' + str(t) +
                            '/direction'] = np.rad2deg(epa[1])
                        h5f[name + '/epa/' + str(t) + '/retardation'] = epa[2]

                        tilting_stack[t] = new_images

                    # save mask for analysis
                    simpli.resolution = PIXEL_LAP_REL
                    mask = np.sum(label_field, 2) > 0
                    mask = simpli.apply_resize(1.0 * mask) > 0.1
                    h5f[name + '/optic/mask'] = np.uint8(mask)
                    mask = None  # keep analysing all pixels

                    rofl_direction, rofl_incl, rofl_t_rel, _ = simpli.apply_rofl(
                        tilting_stack,
                        tilt_angle=np.deg2rad(tilt_angle),
                        gain=gain,
                        mask=mask)

                    h5f[name + '/rofl/direction'] = np.rad2deg(rofl_direction)
                    h5f[name + '/rofl/inclination'] = np.rad2deg(rofl_incl)
                    h5f[name + '/rofl/trel'] = rofl_t_rel
                    h5f[name].attrs['simpli'] = str(simpli.as_dict())

                    imageio.imwrite(
                        'output/simulations/' + file_name + '_' + name + '.png',
                        data2image(
                            fastpli.analysis.images.fom_hsv_black(
                                rofl_direction, rofl_incl)))
