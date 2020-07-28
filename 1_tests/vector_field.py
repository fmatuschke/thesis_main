import os
import sys
import numpy as np

import fastpli.simulation
import fastpli.io

import matplotlib.pyplot as plt
import tikzplotlib

import vector_field_generation as vfg

np.random.seed(42)

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_NAME = os.path.splitext(FILE_BASE)[0]
FILE_NAME = os.path.join('output/', FILE_NAME)
os.makedirs(os.path.join(FILE_PATH, 'output'), exist_ok=True)
NUM_PROC = 16


def data2image(data, name, vmin=-1, vmax=1, cmap='viridis'):
    # data = np.swapaxes(np.flip(data, 1), 0, 1)
    # print(data.shape)
    if data.ndim == 3:
        plt.imsave(f'{name}_x.png',
                   data[data.shape[0] // 2, :, :],
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)
        plt.imsave(f'{name}_y.png',
                   data[:, data.shape[1] // 2, :],
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)
        plt.imsave(f'{name}_z.png',
                   data[:, :, data.shape[2] // 2],
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)
    if data.ndim == 4:
        plt.imsave(f'{name}_x.png',
                   data[data.shape[0] // 2, :, :, 0],
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)
        plt.imsave(f'{name}_y.png',
                   data[:, data.shape[1] // 2, :, 1],
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)
        plt.imsave(f'{name}_z.png',
                   data[:, :, data.shape[2] // 2, 2],
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)


def mask_contour(data, name):

    # plt.imshow(data[data.shape[0] // 2, :, :] >= 1, interpolation='none')
    # plt.contour(data[data.shape[0] // 2, :, :] >= 1, 1)
    # plt.show()

    for i, p in enumerate(
            plt.contour(data[data.shape[0] // 2, :, :] >= 1,
                        1).collections[0].get_paths()):
        np.savetxt(f"{name}_{i}_x.dat", p.vertices, delimiter=',')
    for i, p in enumerate(
            plt.contour(data[:, data.shape[1] // 2, :] >= 1,
                        1).collections[0].get_paths()):
        np.savetxt(f"{name}_{i}_y.dat", p.vertices, delimiter=',')
    for i, p in enumerate(
            plt.contour(data[:, :, data.shape[2] // 2] >= 1,
                        1).collections[0].get_paths()):
        np.savetxt(f"{name}_{i}_z.dat", p.vertices, delimiter=',')


# def imshow_crosssection(data, name):
#     fig, axs = plt.subplots(1, 3, figsize=(10, 10))
#     axs[0].imshow(data[data.shape[0] // 2, :, :, 0],
#                   vmin=-1,
#                   vmax=1,
#                   interpolation='none')
#     axs[1].imshow(data[:, data.shape[1] // 2, :, 1],
#                   vmin=-1,
#                   vmax=1,
#                   interpolation='none')
#     axs[2].imshow(data[:, :, data.shape[2] // 2, 2],
#                   vmin=-1,
#                   vmax=1,
#                   interpolation='none')

#     for ax in axs:
#         ax.axis('off')

#     tikzplotlib.save(name, tex_relative_path_to_data="\currfiledir")

x_data = []
y_data = []
z_data = []
for voxel_size in [1, 0.5, 0.2]:
    scale = 5
    print("voxel_size:", voxel_size)

    for m, mode in enumerate(["NN", "Lerp", "Slerp"]):

        simpli = fastpli.simulation.Simpli()
        simpli.omp_num_threads = NUM_PROC
        simpli.fiber_bundles = fastpli.io.fiber_bundles.load(
            os.path.join(
                FILE_PATH, '..', 'data', 'models', '1_rnd_seed',
                'cube_2pop_psi_1.00_omega_0.00_r_1.00_v0_210_.solved.h5'))

        simpli.voxel_size = voxel_size  # in µm meter
        simpli.set_voi([-8] * 3, [8] * 3)  # in µm meter
        voxel_size_0 = simpli.voxel_size

        simpli.fiber_bundles_properties = [[(0.75, 0.000, 0, 'b'),
                                            (1.0, 0.002, 0, 'r')]] * len(
                                                simpli.fiber_bundles)
        model = simpli.fiber_bundles_properties[0][-1][-1]

        if m == 0:
            # low resolution
            simpli.voxel_size = voxel_size_0  # in µm meter
            # print(
            #     f"low res: {scale}, mode: {mode} -> {simpli.memory_usage('MB'):.0f} MB"
            # )

            tissue, optical_axis, tissue_properties = simpli.generate_tissue()

            # high resolution
            simpli.voxel_size = voxel_size_0 / scale  # in µm meter
            print(
                f"scale: {scale}, mode: {mode} -> {simpli.memory_usage('MB'):.0f} MB"
            )
            if simpli.memory_usage('MB') > 128 * 1e3:
                print("MEMORY!")
                sys.exit(1)
            tissue_high, optical_axis_high, tissue_properties = simpli.generate_tissue(
            )

            data2image(
                tissue_high,
                f"{FILE_NAME}_tissue_high_{mode}_{voxel_size:.02f}_{scale}_{model}",
                0, 1, 'gray')
            mask_contour(
                tissue_high,
                f"{FILE_NAME}_tissue_high_{voxel_size:.02f}_{scale}_{model}")

        # print(f"IntpVecField: {mode}")
        tissue_int = np.empty_like(tissue_high)
        vf_intp = np.empty_like(optical_axis_high)

        simpli._Simpli__sim.__field_interpolation(tissue.shape,
                                                  tissue_int.shape, tissue,
                                                  optical_axis, tissue_int,
                                                  vf_intp, mode)

        data2image(
            vf_intp,
            f"{FILE_NAME}_vf_intp_{mode}_{voxel_size:.02f}_{scale}_{model}")

        # print("diff")

        vf_diff = np.empty(optical_axis_high.shape[:-1], dtype=np.float32)
        simpli._Simpli__sim.__diff_angle(optical_axis_high, vf_intp, vf_diff)

        # print("sorting")

        tmp = vf_diff.ravel()

        # sort plot
        tmp.sort()
        if tmp.size > 1000:
            s = tmp.size // 1000
            tmp = tmp[::s]

        tmp = np.rad2deg(tmp)

        x = np.arange(tmp.size) / tmp.size

        x = x[tmp > 0]
        tmp = tmp[tmp > 0]
        x = x[tmp < 90]
        tmp = tmp[tmp < 90]
        x_data.append(x.copy())
        y_data.append(tmp.copy())
        z_data.append(f"{voxel_size}_{mode}")

        # print("sort done")
        # ax.plot(x, tmp, label=mode)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for x, y, mode in zip(x_data, y_data, z_data):
    ax.plot(x, y, label=mode)
ax.legend()
tikzplotlib.save(
    f"{FILE_NAME}_mode_plot_{voxel_size:.02f}_{scale}_{model}.tikz",
    tex_relative_path_to_data="\currfiledir")
