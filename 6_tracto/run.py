# /data/PLI-Group/Marius/Master/Nicole/tractogramsMSA/tractogram_msa50.h5

import fastpli.simulation
import fastpli.analysis
import fastpli.tools
import fastpli.io

import multiprocessing as mp
import numpy as np
import h5py
import tqdm
import os

import imageio
import nibabel as nib

from load_fbs_plivis import read

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
os.makedirs(os.path.join(FILE_PATH, 'output'), exist_ok=True)
FILE_OUT = os.path.join(FILE_PATH, 'output', f'fastpli.example.{FILE_BASE}')

fbs = fastpli.io.fiber_bundles.load("output/model.dat")


def run(z, s):
    if os.path.isfile(f'{FILE_OUT}_{s}.h5'):
        print(f'{FILE_OUT}_{s}.h5 --- EXISTS')
        return
    with h5py.File(f'{FILE_OUT}_{s}.h5', 'w-') as h5f:
        simpli = fastpli.simulation.Simpli()
        simpli.omp_num_threads = 24
        mp_pool = mp.Pool(simpli.omp_num_threads)
        simpli.voxel_size = 10  # in micro meter
        simpli.set_voi(
            np.array([220, 175, z]) * 60,
            np.array([1124, 1448, z + 1]) * 60)  # in micro meter
        simpli.fiber_bundles = fbs
        # simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rescale(fbs, 60)
        simpli.fiber_bundles_properties = [[(1.0, -0.004, 1, 'p')]]

        # print('VOI:', simpli.get_voi())
        # print('Memory:', str(round(simpli.memory_usage('MB'), 2)) + ' MB')

        # Generate Tissue
        # print('Run Generation:')
        tissue, optical_axis, tissue_properties = simpli.generate_tissue()

        # h5f['tissue/tissue'] = tissue.astype(np.uint16)
        # h5f['tissue/optical_axis'] = optical_axis
        # h5f['tissue/tissue_properties'] = tissue_properties

        # Simulate PLI Measurement
        simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
        simpli.light_intensity = 26000  # a.u.
        simpli.interpolate = "Slerp"
        simpli.wavelength = 525  # in nm
        simpli.pixel_size = 60  # in micro meter
        simpli.sensor_gain = 3
        simpli.optical_sigma = 0.71  # in voxel size
        simpli.tilts = np.deg2rad([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180),
                                   (5.5, 270)])

        tilting_stack = [None] * 5
        # print('Run Simulation:')
        for t, (theta, phi) in enumerate(tqdm.tqdm(simpli.tilts)):
            # print(round(np.rad2deg(theta), 1), round(np.rad2deg(phi), 1))
            images = simpli.run_simulation(tissue, optical_axis,
                                           tissue_properties, theta, phi)

            h5f['simulation/data/' + str(t)] = images

            # apply optic to simulation
            images = simpli.apply_optic(images, mp_pool=mp_pool)
            h5f['simulation/optic/' + str(t)] = images

            # calculate modalities
            epa = simpli.apply_epa(images)
            h5f['analysis/epa/' + str(t) + '/transmittance'] = epa[0]
            h5f['analysis/epa/' + str(t) + '/direction'] = np.rad2deg(epa[1])
            h5f['analysis/epa/' + str(t) + '/retardation'] = epa[2]

            tilting_stack[t] = images

        # save mask for analysis
        mask = np.sum(tissue, 2) > 0
        mask = simpli.apply_optic_resample(1.0 * mask, mp_pool=mp_pool) > 0.1
        h5f['simulation/optic/mask'] = np.uint8(mask)
        mask = None  # keep analysing all pixels

        # print('Run ROFL analysis:')
        # tqdm.tqdm.write("ROFL")
        rofl_direction, rofl_incl, rofl_t_rel, _ = simpli.apply_rofl(
            tilting_stack, mask=mask, mp_pool=mp_pool)

        h5f['analysis/rofl/direction'] = np.rad2deg(rofl_direction)
        h5f['analysis/rofl/inclination'] = np.rad2deg(rofl_incl)
        h5f['analysis/rofl/trel'] = rofl_t_rel

        def data2image(data):
            return np.swapaxes(np.flip(data, 1), 0, 1)

        # print(f'creating Fiber Orientation Map: {FILE_OUT}_{z}.png')

        imageio.imwrite(
            f'{FILE_OUT}_{s}.png',
            data2image(
                fastpli.analysis.images.fom_hsv_black(rofl_direction,
                                                      rofl_incl)))

        UnitX, UnitY, UnitZ = fastpli.analysis.images.unit_vectors(
            rofl_direction, rofl_incl)

        # save nifti
        img = nib.Nifti1Image(UnitX.astype(np.float32), np.eye(4))
        img.to_filename(f'{FILE_OUT}_{z}.s{s:04}.UnitX.nii')
        img = nib.Nifti1Image(UnitY.astype(np.float32), np.eye(4))
        img.to_filename(f'{FILE_OUT}_{z}.s{s:04}.UnitY.nii')
        img = nib.Nifti1Image(UnitZ.astype(np.float32), np.eye(4))
        img.to_filename(f'{FILE_OUT}_{z}.s{s:04}.UnitZ.nii')

    # print('Done')
    # print('You can look at the data e.g with Fiji and the hdf5 plugin')


parameters = list(range(-1, 49))
# with mp.Pool(processes=10) as pool:
#     [
#         d for d in tqdm.tqdm(pool.imap_unordered(run, parameters),
#                              total=len(parameters))
#     ]

for s, z in enumerate(tqdm.tqdm(parameters)):
    run(z, s)
