import os
import sys
import numpy as np
import h5py
import yaml
print(sys.version)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
FILE_NAME = os.path.join('output/', FILE_NAME)
os.makedirs(os.path.join(FILE_PATH, 'output'), exist_ok=True)

NUM_PROC = 16
with open(os.path.join(FILE_PATH, '.numba_config.yaml'), 'w') as file:
    yaml.dump({'num_threads': NUM_PROC}, file)
print(f'NUM_PROC: {NUM_PROC}')

import fastpli.simulation
import fastpli.analysis
import fastpli.tools
import fastpli.io
import vector_field_generation as vfg

import matplotlib.pyplot as plt
import tikzplotlib

np.random.seed(42)
print(fastpli.__version__)

for voxel_size in [1, 0.5, 0.2, 0.1]:
    scale = voxel_size // 0.05

    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = NUM_PROC
    print(FILE_PATH)
    simpli.fiber_bundles = fastpli.io.fiber_bundles.load(
        os.path.join(FILE_PATH, '..', 'data', 'models', '1_rnd_seed',
                     'cube_2pop_psi_1.00_omega_0.00_r_1.00_v0_210_.solved.h5'))

    simpli.voxel_size = voxel_size  # in µm meter
    simpli.set_voi([-30] * 3, [30] * 3)  # in µm meter

    # fig, axs = plt.subplots(2, 3)
    for m, (dn, model) in enumerate([(0.004, 'r'), (-0.002, 'p')]):
        simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                            (1.0, dn, 0, model)]] * len(
                                                simpli.fiber_bundles)

        # low resolution
        simpli.voxel_size = voxel_size  # in µm meter
        print(
            f"low res: {scale}, model: {model} -> {simpli.memory_usage('MB'):.0f} MB"
        )
        tissue, optical_axis, tissue_properties = simpli.generate_tissue()

        with h5py.File(f"{FILE_NAME}_{voxel_size:.02f}_{scale}_{model}.h5",
                       'w') as h5f:
            with open(os.path.abspath(__file__), 'r') as f:
                simpli.save_parameter_h5(h5f, f.read())

        # high resolution
        simpli.voxel_size = voxel_size / scale  # in µm meter
        print(
            f"scale: {scale}, model: {model} -> {simpli.memory_usage('MB'):.0f} MB"
        )
        if simpli.memory_usage('MB') > 128 / 3 * 1e3:
            print("MEMORY!")
            sys.exit(1)
        tissue_high, optical_axis_high, tissue_properties = simpli.generate_tissue(
        )
        print("IntpVecField")

        tissue_int = np.empty_like(tissue_high)
        vf_intp = np.empty_like(optical_axis_high)

        simpli._Simpli__sim.__field_interpolation(tissue.shape,
                                                  tissue_int.shape, tissue,
                                                  optical_axis, tissue_int,
                                                  vf_intp, False)
        print("diff")
        vf_diff = vfg.VectorOrientationDiffNorm(optical_axis_high, vf_intp)

        vmax = np.amax(vf_diff)
        print(f"vmax: {vmax}")

        tmp = np.logical_or(
            np.logical_and(tissue_high > 0, tissue_high % 2 == 0), vf_diff != 0)
        tmp = vf_diff[tmp].ravel()

        print("sort")
        tmp.sort()
        if tmp.size > 1000:
            s = tmp.size // 1000
            tmp = tmp[::s]

        fig, axs = plt.subplots(1, 1)
        x = np.arange(tmp.size) / tmp.size
        plt.plot(x, tmp)
        plt.plot([0, 1], [0, 0])
        tikzplotlib.save(
            f"{FILE_NAME}_line_{voxel_size:.02f}_{scale}_{model}.tex")

        # show optical_axis_high_norm
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.linalg.norm(optical_axis_high[vf_diff.shape[0] //
                                                       2, :, :],
                                     axis=-1),
                      vmin=0,
                      vmax=1)
        axs[1].imshow(np.linalg.norm(optical_axis_high[:, vf_diff.shape[1] //
                                                       2, :],
                                     axis=-1),
                      vmin=0,
                      vmax=1)
        axs[2].imshow(np.linalg.norm(optical_axis_high[:, :,
                                                       vf_diff.shape[2] // 2],
                                     axis=-1),
                      vmin=0,
                      vmax=1)

        tikzplotlib.save(
            f"{FILE_NAME}_optical_axis_high_norm_{voxel_size:.02f}_{scale}_{model}.tex",
            tex_relative_path_to_data="\currfiledir")

        # show vf+intp
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.linalg.norm(vf_intp[vf_diff.shape[0] // 2, :, :],
                                     axis=-1),
                      vmin=0,
                      vmax=1)
        axs[1].imshow(np.linalg.norm(vf_intp[:, vf_diff.shape[1] // 2, :],
                                     axis=-1),
                      vmin=0,
                      vmax=1)
        axs[2].imshow(np.linalg.norm(vf_intp[:, :, vf_diff.shape[2] // 2],
                                     axis=-1),
                      vmin=0,
                      vmax=1)

        tikzplotlib.save(
            f"{FILE_NAME}_vf_intp_norm_{voxel_size:.02f}_{scale}_{model}.tex",
            tex_relative_path_to_data="\currfiledir")

        # show vf_diff with tissue_high overlay
        fig, axs = plt.subplots(1, 3, frameon=False)
        axs[0].imshow(vf_diff[vf_diff.shape[0] // 2, :, :], vmin=0, vmax=vmax)
        axs[1].imshow(vf_diff[:, vf_diff.shape[1] // 2, :], vmin=0, vmax=vmax)
        axs[2].imshow(vf_diff[:, :, vf_diff.shape[2] // 2], vmin=0, vmax=vmax)
        axs[0].imshow(tissue_high[vf_diff.shape[0] // 2, :, :],
                      vmin=0,
                      vmax=4,
                      cmap='gray',
                      alpha=0.25)
        axs[1].imshow(tissue_high[:, vf_diff.shape[1] // 2, :],
                      vmin=0,
                      vmax=4,
                      cmap='gray',
                      alpha=0.25)
        axs[2].imshow(tissue_high[:, :, vf_diff.shape[2] // 2],
                      vmin=0,
                      vmax=4,
                      cmap='gray',
                      alpha=0.25)
        for ax in axs:
            ax.axis('off')

        tikzplotlib.save(
            f"{FILE_NAME}_vdiff_{voxel_size:.02f}_{scale}_{model}.tex",
            tex_relative_path_to_data="\currfiledir")
