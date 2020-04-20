import numpy as np
import os
from numba import njit

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)


@njit(cache=True)
def GetLabel(x, y, z, label_field):
    if x < 0 or x >= label_field.shape[0] or y < 0 or y >= label_field.shape[
            1] or z < 0 or z >= label_field.shape[2]:
        return 0

    return label_field[int(x), int(y), int(z)]


@njit(cache=True)
def VectorOrientationAddition(v, u):
    if np.dot(v, u) >= 0:
        return v + u
    else:
        return v - u


@njit(cache=True)
def InterpolateVec(x, y, z, vector_field):
    # Trilinear interpolate
    x0 = int(min(max(np.floor(x), 0), vector_field.shape[0] - 1))
    y0 = int(min(max(np.floor(y), 0), vector_field.shape[1] - 1))
    z0 = int(min(max(np.floor(z), 0), vector_field.shape[2] - 1))

    x1 = int(max(min(np.ceil(x), vector_field.shape[0] - 1), 0))
    y1 = int(max(min(np.ceil(y), vector_field.shape[1] - 1), 0))
    z1 = int(max(min(np.ceil(z), vector_field.shape[2] - 1), 0))

    if x0 == x1 and y0 == y1 and z0 == z1:
        return vector_field[x0, y0, z0]

    xd = 0 if x0 == x1 else (x - x0) / (x1 - x0)
    yd = 0 if y0 == y1 else (y - y0) / (y1 - y0)
    zd = 0 if z0 == z1 else (z - z0) / (z1 - z0)

    c000 = vector_field[x0, y0, z0]
    c100 = vector_field[x1, y0, z0]
    c010 = vector_field[x0, y1, z0]
    c110 = vector_field[x1, y1, z0]
    c001 = vector_field[x0, y0, z1]
    c101 = vector_field[x1, y0, z1]
    c011 = vector_field[x0, y1, z1]
    c111 = vector_field[x1, y1, z1]

    c00 = VectorOrientationAddition(c000 * (1 - xd), c100 * xd)
    c01 = VectorOrientationAddition(c001 * (1 - xd), c101 * xd)
    c10 = VectorOrientationAddition(c010 * (1 - xd), c110 * xd)
    c11 = VectorOrientationAddition(c011 * (1 - xd), c111 * xd)

    c0 = VectorOrientationAddition(c00 * (1 - yd), c10 * yd)
    c1 = VectorOrientationAddition(c01 * (1 - yd), c11 * yd)

    return VectorOrientationAddition(c0 * (1 - zd),
                                     c1 * zd).astype(vector_field.dtype)


@njit(cache=True)
def GetVec(x, y, z, label_field, vector_field, interpolate):
    if interpolate:
        # only interpolate if all neighbors are the same tissue

        label = GetLabel(x, y, z, label_field)

        labels = (
            GetLabel(np.floor(x), np.floor(y), np.floor(z),
                     label_field) == label,
            GetLabel(np.floor(x), np.floor(y), np.ceil(z),
                     label_field) == label,
            GetLabel(np.floor(x), np.ceil(y), np.floor(z),
                     label_field) == label,
            GetLabel(np.floor(x), np.ceil(y), np.ceil(z), label_field) == label,
            GetLabel(np.ceil(x), np.floor(y), np.floor(z),
                     label_field) == label,
            GetLabel(np.ceil(x), np.floor(y), np.ceil(z), label_field) == label,
            GetLabel(np.ceil(x), np.ceil(y), np.floor(z), label_field) == label,
            GetLabel(np.ceil(x), np.ceil(y), np.ceil(z), label_field) == label,
        )

        for i in range(7):
            if labels[0] != labels[i + 1]:
                # Nearest Neighbor
                return vector_field[int(x), int(y), int(z)]

        return InterpolateVec(x, y, z, vector_field)

    # Nearest Neighbor
    return vector_field[int(x), int(y), int(z)]


@njit(cache=True)
def IntpVecField(label_field, vector_field, scale, interpolate=True):

    sx = int(round(vector_field.shape[0] * scale))
    sy = int(round(vector_field.shape[1] * scale))
    sz = int(round(vector_field.shape[2] * scale))

    vector_field_intp = np.empty((sx, sy, sz, 3), vector_field.dtype)
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                vector_field_intp[x, y, z, :] = GetVec(
                    (x * vector_field.shape[0]) / sx,
                    (y * vector_field.shape[1]) / sy,
                    (z * vector_field.shape[2]) / sz, label_field, vector_field,
                    interpolate)

    return vector_field_intp


if __name__ == "__main__":
    import fastpli.simulation
    import fastpli.analysis
    import fastpli.tools
    import fastpli.io
    import sys

    import matplotlib.pyplot as plt
    import scipy.ndimage

    np.random.seed(42)

    file_name = 'fastpli.example.' + FILE_BASE + '.h5'
    print(f"creating file: {file_name}")

    # with h5py.File(file_name, 'w') as h5f:
    # save script
    # h5f['version'] = fastpli.__version__
    # with open(os.path.abspath(__file__), 'r') as f:
    #     h5f['parameter/script'] = f.read()
    #     h5f['parameter/pip_freeze'] = fastpli.tools.helper.pip_freeze()

    # Setup Simpli for Tissue Generation
    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = 2
    # simpli.voxel_size = 0.25  # in µm meter
    simpli.fiber_bundles = fastpli.io.fiber_bundles.load(
        os.path.join(FILE_PATH, '..', 'data', 'models',
                     'cube_2pop_psi_0.5_omega_0.0_.solved.h5'))

    # define layers (e.g. axon, myelin) inside fibers of each fiber_bundle fiber_bundle
    for dn, model in [(-0.002, 'r')]:  #, (0.004, 'r')
        simpli.fiber_bundles_properties = [[  #(0.75, 0, 0, 'b'),
            (1.0, dn, 0, model)
        ]] * len(simpli.fiber_bundles)

        # Generate Tissue
        print("Run Generation:")
        simpli.voxel_size = 0.25  # in µm meter
        simpli.set_voi([-10] * 3, [10] * 3)  # in µm meter

        print('VOI:', simpli.get_voi())
        print('Memory:', f'{simpli.memory_usage("MB"):.0f} MB')
        if simpli.memory_usage('MB') > 2**12:
            print("MEMORY!")
            sys.exit(1)
        tissue, optical_axis, tissue_properties = simpli.generate_tissue()

        for scale in [2]:
            print(f"scale: {scale}")
            simpli.voxel_size = simpli.voxel_size / scale  # in µm meter
            simpli.set_voi([-10] * 3, [10] * 3)  # in µm meter

            print('VOI:', simpli.get_voi())
            print('Memory:', f'{simpli.memory_usage("MB"):.0f} MB')
            if simpli.memory_usage('MB') > 2**12:
                print("MEMORY!")
                sys.exit(1)
            _, optical_axis_high, tissue_properties = simpli.generate_tissue()

            # vf_zoom = scipy.ndimage.zoom(optical_axis, scale, order=0)
            vf_intp = IntpVecField(tissue, optical_axis, scale, True)

            vf_diff = optical_axis_high - vf_intp

            # plt.imshow(optical_axis_high[:, :, 5, 0], vmin=-1, vmax=1)
            # plt.show()
            # plt.imshow(vf_intp[:, :, 0, 0], vmin=-1, vmax=1)
            # plt.show()
            plt.imshow(vf_diff[:, :, 5, 0], vmin=-1, vmax=1)
            plt.show()
