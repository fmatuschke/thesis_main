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
def VectorOrientationSubstraction(v, u):
    if np.dot(v, u) >= 0:
        return v - u
    else:
        return v + u


@njit(cache=True)
def VectorOrientationSubstractionField(v, u):
    shape = v.shape
    v = np.reshape(v, (-1, 3))
    u = np.reshape(u, (-1, 3))
    result = np.empty_like(v)
    for i in range(v.shape[0]):
        result[i, :] = VectorOrientationSubstraction(v[i, :], u[i, :])
    result = np.reshape(result, shape)
    return result


@njit(cache=True)
def InterpolateVec(x, y, z, vector_field, label_field):
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

    label = GetLabel(x0, y0, z0, label_field)  # nearest neighbor

    c000 = vector_field[x0, y0, z0] * (GetLabel(x0, y0, z0,
                                                label_field) == label)
    c100 = vector_field[x1, y0, z0] * (GetLabel(x1, y0, z0,
                                                label_field) == label)
    c010 = vector_field[x0, y1, z0] * (GetLabel(x0, y1, z0,
                                                label_field) == label)
    c110 = vector_field[x1, y1, z0] * (GetLabel(x1, y1, z0,
                                                label_field) == label)
    c001 = vector_field[x0, y0, z1] * (GetLabel(x0, y0, z1,
                                                label_field) == label)
    c101 = vector_field[x1, y0, z1] * (GetLabel(x1, y0, z1,
                                                label_field) == label)
    c011 = vector_field[x0, y1, z1] * (GetLabel(x0, y1, z1,
                                                label_field) == label)
    c111 = vector_field[x1, y1, z1] * (GetLabel(x1, y1, z1,
                                                label_field) == label)

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
        return InterpolateVec(x, y, z, vector_field, label_field)

    # Nearest Neighbor
    return vector_field[int(np.floor(x)), int(np.floor(y)), int(np.floor(z))]


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
                    (0.5 + x) / scale, (0.5 * y) / scale, (0.5 + z) / scale,
                    label_field, vector_field, interpolate)

    return vector_field_intp
