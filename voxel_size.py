import numpy as np
import h5py
import glob
import sys
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

NUM_THREADS = 16

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
FILE_NAME = os.path.join('output/', FILE_NAME)

if False:

    import fastpli.simulation
    import fastpli.analysis
    import fastpli.objects
    import fastpli.tools
    import fastpli.io

    # PARAMETER
    LENGTH = 0.1
    THICKNESS = 60
    FIBER_INCLINATION = np.linspace(0, 90, 10, True)
    # VOXEL_SIZES = [0.0625 * 2**i for i in range(7)]  # 1/2**4
    # VOXEL_SIZES = [
    #     0.0625 * i for i in
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 60, 120, 240, 480, 960]
    # ]  # 1/2**4

    VOXEL_SIZES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    fiber_bundles = fastpli.io.fiber_bundles.load(
        'input/cube_2pop_psi_0.5_omega_0.0.solved.h5', '/')

    os.makedirs('output', exist_ok=True)

    with h5py.File(FILE_NAME + '.h5', 'w-') as h5f:
        with open(os.path.abspath(__file__), 'r') as script:
            h5f.attrs['script'] = script.read()

        for voxel_size in tqdm(VOXEL_SIZES):
            for dn, model in [(-0.001, 'p'), (0.002, 'r')]:
                dset = h5f.create_group(str(voxel_size) + '/' + model)

                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                simpli.omp_num_threads = NUM_THREADS
                simpli.voxel_size = voxel_size
                simpli.resolution = voxel_size
                simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
                simpli.interpolate = True
                simpli.untilt_sensor_view = True
                simpli.wavelength = 525  # in nm

                simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                               0.5 * np.array([LENGTH, LENGTH, THICKNESS]))

                # simpli.tilts = np.deg2rad(
                #     np.array([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180),
                #               (5.5, 270)]))
                simpli.tilts = np.deg2rad(np.array([(0, 0)]))
                simpli.add_crop_tilt_halo()

                # print(simpli.dim)

                print(str(round(simpli.memory_usage(), 2)) + 'MB')
                time.sleep(4.2)

                simpli.fiber_bundles = fiber_bundles
                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 0, model)]
                                                  ] * len(fiber_bundles)
                label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                    h5f=dset, save=["label_field"])

                # Simulate PLI Measurement
                simpli.light_intensity = 1  # a.u.
                simpli.save_parameter_h5(h5f=dset)
                for t, tilt in enumerate(simpli._tilts):
                    theta, phi = tilt[0], tilt[1]
                    images = simpli.run_simulation(label_field, vector_field,
                                                   tissue_properties, theta,
                                                   phi)

                    images = simpli.rm_crop_tilt_halo(images)

                    dset['simulation/data/' + str(t)] = images
                    dset['simulation/data/' + str(t)].attrs['theta'] = theta
                    dset['simulation/data/' + str(t)].attrs['phi'] = phi

                del label_field
                del vector_field
                del simpli

if True:
    import fastpli.simulation.optic
    with h5py.File(FILE_NAME + '.h5', 'r') as h5f:
        voxel_size_ref = str(min([float(i) for i in h5f]))
        print(voxel_size_ref)
        data_ref = {}
        data_ref['r'] = h5f[voxel_size_ref + '/r/simulation/data/0'][:]
        data_ref['p'] = h5f[voxel_size_ref + '/p/simulation/data/0'][:]

        df = pd.DataFrame()
        for voxel_size in h5f:
            for model in h5f[voxel_size]:
                dset = h5f[voxel_size + '/' + model]

                data = dset['simulation/data/0'][:]

                data_ref_optic = np.empty_like(data)
                scale = float(voxel_size_ref) / float(voxel_size)

                if scale != 1:
                    for i in range(data_ref[model].shape[-1]):
                        data_ref_optic[:, :,
                                       i] = fastpli.simulation.optic.resample(
                                           data_ref[model][:, :, i],
                                           float(voxel_size_ref) /
                                           float(voxel_size))
                else:
                    data_ref_optic = data_ref[model].copy()

                data_ref_optic = np.array(data_ref_optic)

                df_ = pd.DataFrame({
                    'voxel_size': [float(voxel_size)] * data.size,
                    'model': [model] * data.size,
                    'data':
                        data.flatten(),
                    'data_ref':
                        data_ref_optic.flatten(),
                    'diff':
                        np.abs(data - data_ref_optic).flatten(),
                    'dev':
                        np.abs(data - data_ref_optic).flatten() /
                        data_ref_optic.flatten(),
                })

                df = df.append(df_)

        df.to_pickle(FILE_NAME + '.pkl')

df = pd.read_pickle(FILE_NAME + '.pkl')
# df = df.where(df['voxel_size'] < 0.1)

# print(df['voxel_size'])
# sys.exit()
# print(df['voxel_size'])

df1 = df[['model', 'dev', 'voxel_size']]
# df1 = df1.loc[df1['model'] == 'r']

fig, axes = plt.subplots(nrows=2, ncols=1)
# df1.boxplot()
df1[df1['model'] == 'p'].boxplot(ax=axes[0], by=['voxel_size'], column=['dev'])
df1[df1['model'] == 'r'].boxplot(ax=axes[1], by=['voxel_size'], column=['dev'])
plt.show()

# # import tikzplotlib
# # tikzplotlib.save("test.tex")
