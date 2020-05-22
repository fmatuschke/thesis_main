#%% Imports
# %matplotlib inline

import numpy as np
import os
from numba import njit

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)

import fastpli.simulation
import fastpli.analysis
import fastpli.tools
import fastpli.io
import sys
import h5py

import matplotlib.pyplot as plt

import tikzplotlib
from skimage.external import tifffile as tif

from vector_field_generation import *

np.random.seed(42)

print(fastpli.__version__)

#%% Vector Field Generation

# Setup Simpli for Tissue Generation
# simpli = fastpli.simulation.Simpli()
# simpli.omp_num_threads = 2
# simpli.voxel_size = 0.25  # in µm meter
fiber_bundles = fastpli.io.fiber_bundles.load(
    os.path.join(FILE_PATH, '..', 'data', 'models',
                 'cube_2pop_psi_0.5_omega_0.0_.solved.h5'))

# define layers (e.g. axon, myelin) inside fibers of each fiber_bundle fiber_bundle

voxel_size = 0.1  # in µm meter
with h5py.File('data.h5', 'w') as h5f:
    for s, scale in enumerate([3]):
        fig, axs = plt.subplots(2, 3)
        for m, (dn, model) in enumerate([(-0.002, 'p'), (0.004, 'r')]):

            # low resolution
            simpli = fastpli.simulation.Simpli()
            simpli.omp_num_threads = 2
            simpli.fiber_bundles = fiber_bundles
            simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                (1.0, dn, 0, model)]] * len(
                                                    simpli.fiber_bundles)

            simpli.voxel_size = voxel_size  # in µm meter
            simpli.set_voi([-5] * 3, [5] * 3)  # in µm meter
            if simpli.memory_usage('MB') > 2**12:
                print("MEMORY!")
                sys.exit(1)
            tissue, optical_axis, tissue_properties = simpli.generate_tissue()

            # high resolution
            simpli = fastpli.simulation.Simpli()
            simpli.omp_num_threads = 2
            simpli.fiber_bundles = fiber_bundles
            simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                (1.0, dn, 0, model)]] * len(
                                                    simpli.fiber_bundles)

            simpli.voxel_size = voxel_size / scale  # in µm meter
            simpli.set_voi([-5] * 3, [5] * 3)  # in µm meter
            print(
                f"scale: {scale}, model: {model} -> {simpli.memory_usage('MB'):.0f} MB"
            )
            if simpli.memory_usage('MB') > 2**12:
                print("MEMORY!")
                sys.exit(1)
            tissue_high, optical_axis_high, tissue_properties = simpli.generate_tissue(
            )
            print("IntpVecField")
            vf_intp = IntpVecField(tissue, optical_axis, scale, True)
            print("diff")
            vf_diff = np.linalg.norm(VectorOrientationSubstractionField(
                optical_axis_high, vf_intp),
                                     axis=-1)

            h5f[f'{scale}/{model}/tissue'] = tissue.astype(np.uint8)
            h5f[f'{scale}/{model}/optical_axis'] = optical_axis
            h5f[f'{scale}/{model}/vf_intp'] = vf_intp
            h5f[f'{scale}/{model}/tissue_high'] = tissue_high.astype(np.uint8)
            h5f[f'{scale}/{model}/optical_axis_high'] = optical_axis_high
            h5f[f'{scale}/{model}/vf_diff'] = vf_diff

            vmax = np.amax(vf_diff)
            axs[m, 0].imshow(vf_diff[vf_diff.shape[0] // 2, :, :],
                             vmin=0,
                             vmax=vmax)
            axs[m, 1].imshow(vf_diff[:, vf_diff.shape[1] // 2, :],
                             vmin=0,
                             vmax=vmax)
            pcm = axs[m, 2].imshow(vf_diff[:, :, vf_diff.shape[2] // 2],
                                   vmin=0,
                                   vmax=vmax)

            if m == 0:
                fig.colorbar(pcm, ax=axs[:, 2])

            # print(tissue_high.shape)
            # np.savez(f"test_vfdiff_{scale}.npz")
            # tif.imsave(f"vfdiff_{scale}.tiff", vf_diff, bigtiff=True)
            # tif.imsave(f"label_{scale}.tiff", tissue_high, bigtiff=True)

        # plt.show()
        # tikzplotlib.save(f"test_{scale}.tex")

    plt.show()

# %%
