import numpy as np
import h5py
import glob
import sys
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from mpi4py import MPI
comm = MPI.COMM_WORLD

NUM_THREADS = 4

# reproducability
np.random.seed(42)

# path
FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
FILE_NAME = os.path.join('output/', FILE_NAME)

LENGTH = 5
VOXEL_SIZES = [
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16,
    0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
]

for omega in list(range(0, 100, 10))[comm.Get_rank()::comm.Get_size()]:

    INPUT_FILE = f'input/cube_2pop_psi_0.5_omega_{omega}.0.solved.h5'

    OUTPUT_NAME = FILE_NAME + '_' + os.path.basename(
        INPUT_FILE) + '_vref_' + str(VOXEL_SIZES[0]) + '_size_' + str(LENGTH)

    if not os.path.isfile(OUTPUT_NAME + '.h5'):
        print("simulate data")

        import fastpli.simulation
        import fastpli.analysis
        import fastpli.objects
        import fastpli.tools
        import fastpli.io

        # PARAMETER
        THICKNESS = 60
        FIBER_INCLINATION = np.linspace(0, 90, 10, True)

        fiber_bundles = fastpli.io.fiber_bundles.load(INPUT_FILE, '/')

        os.makedirs('output', exist_ok=True)

        with h5py.File(OUTPUT_NAME + '.h5', 'w-') as h5f:
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
                    simpli.filter_rotations = np.deg2rad(
                        [0, 30, 60, 90, 120, 150])
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
                    if simpli.memory_usage() > 1e3:
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
                        images = simpli.run_simulation(label_field,
                                                       vector_field,
                                                       tissue_properties, theta,
                                                       phi)

                        images = simpli.rm_crop_tilt_halo(images)

                        dset['simulation/data/' + str(t)] = images
                        dset['simulation/data/' + str(t)].attrs['theta'] = theta
                        dset['simulation/data/' + str(t)].attrs['phi'] = phi

                    del label_field
                    del vector_field
                    del simpli

if not os.path.isfile(FILE_NAME + '.pkl'):

    print("analyse data")
    import fastpli.simulation.optic
    import fastpli.analysis

    df = pd.DataFrame()

    for omega in list(range(0, 100, 10)):
        INPUT_FILE = f'input/cube_2pop_psi_0.5_omega_{omega}.0.solved.h5'

        OUTPUT_NAME = FILE_NAME + '_' + os.path.basename(
            INPUT_FILE) + '_vref_' + str(
                VOXEL_SIZES[0]) + '_size_' + str(LENGTH)

        with h5py.File(OUTPUT_NAME + '.h5', 'r') as h5f:
            voxel_size_ref = str(min([float(i) for i in h5f]))
            print(voxel_size_ref)
            data_ref = {}
            data_ref['r'] = h5f[voxel_size_ref + '/r/simulation/data/0'][:]
            data_ref['p'] = h5f[voxel_size_ref + '/p/simulation/data/0'][:]

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
                                               data_ref['r'][:, :, i],
                                               float(voxel_size_ref) /
                                               float(voxel_size))
                    else:
                        data_ref_optic = data_ref[model].copy()

                    data_ref_optic = np.array(data_ref_optic)

                    trans, dirc, ret = fastpli.analysis.epa.epa(data)
                    ref_trans, ref_dirc, ref_ret = fastpli.analysis.epa.epa(
                        data_ref_optic)

                    # print(np.sum(np.abs(data - data_ref_optic)),
                    #       np.sum(np.abs(trans - ref_trans)),
                    #       np.sum(np.abs(dirc - ref_dirc)),
                    #       np.sum(np.abs(ret - ref_ret)))

                    # dirc[dirc > np.pi / 2] -= np.pi

                    df = df.append(
                        {
                            'voxel_size': float(voxel_size),
                            'model': model,
                            'omega': float(omega),
                            'data': data.flatten().tolist(),
                            'data_ref': data_ref_optic.flatten().tolist(),
                            'transmittance': trans.flatten().tolist(),
                            'direction': dirc.flatten().tolist(),
                            'retardation': ret.flatten().tolist(),
                            'opt_transmittance': ref_trans.flatten().tolist(),
                            'opt_direction': ref_dirc.flatten().tolist(),
                            'opt_retardation': ref_ret.flatten().tolist()
                        },
                        ignore_index=True)

    df.to_pickle(FILE_NAME + '.pkl')

import seaborn as sns
import statistic
import scipy.stats
import scipy.optimize
import astropy.stats

df_analyse = pd.DataFrame()

df = pd.read_pickle(FILE_NAME + '.pkl')

fig, ax = plt.subplots()
# for m in ['p', 'r']:
for m, m_group in df.groupby('model'):
    for vs, vs_group in m_group.groupby('voxel_size'):
        for o, o_group in vs_group.groupby('omega'):

            # print(str(vs) + '_' + m + '_' + str(o))
            # print(o_group)

            df_analyse = df_analyse.append(
                {
                    "voxel_size":
                        float(vs),
                    "model":
                        m,
                    "omega":
                        float(o),
                    "mean_diff_dir":
                        np.mean(
                            np.abs(
                                np.array(o_group['opt_direction'].iloc[0]) -
                                np.array(o_group['direction'].iloc[0]))),
                    "mean_diff_ret":
                        np.mean(
                            np.abs(
                                np.array(o_group['opt_retardation'].iloc[0]) -
                                np.array(o_group['retardation'].iloc[0]))),
                    "mean_diff_rel_ret":
                        np.mean(
                            np.abs(
                                np.array(o_group['opt_retardation'].iloc[0]) -
                                np.array(o_group['retardation'].iloc[0])) /
                            np.array(o_group['opt_retardation'].iloc[0])),
                    "mean_diff_trans":
                        np.mean(
                            np.abs(
                                np.array(o_group['opt_transmittance'].iloc[0]) -
                                np.array(o_group['transmittance'].iloc[0])))
                },
                ignore_index=True)

            # x = np.array(o_group['opt_retardation'].iloc[0])
            # y = np.array(o_group['retardation'].iloc[0])

            # # ax.scatter(x, y)
            # ax.hist2d(x, y, bins=(50, 50), cmap=plt.cm.viridis)
            # ax.plot([np.min(np.append(x, y)),
            #          np.max(np.append(x, y))],
            #         [np.min(np.append(x, y)),
            #          np.max(np.append(x, y))])
            # ax.set_title(str(vs) + '_' + m + '_' + str(o))
            # ax.axis('equal')
            # plt.ion()
            # plt.show()
            # plt.pause(0.25)

            # ax.clear()

for mod in ['trans', 'dir', 'ret', 'rel_ret']:
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(mod)
    for i, (k, g) in enumerate(df_analyse.groupby(['model'])):
        ax[i].set_title(k)

        # sns.boxplot(x='voxel_size', y='mean_diff_rel_ret', hue='omega', data=g)

        for key, grp in g.groupby(['omega']):
            # print(g)
            grp.plot(ax=ax[i],
                     kind='line',
                     x='voxel_size',
                     y='mean_diff_' + mod,
                     marker='d',
                     label=str(key) + '_' + k)

plt.show()
