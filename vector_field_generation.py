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

    l000 = GetLabel(x0, y0, z0, label_field) == label
    l100 = GetLabel(x1, y0, z0, label_field) == label
    l010 = GetLabel(x0, y1, z0, label_field) == label
    l110 = GetLabel(x1, y1, z0, label_field) == label
    l001 = GetLabel(x0, y0, z1, label_field) == label
    l101 = GetLabel(x1, y0, z1, label_field) == label
    l011 = GetLabel(x0, y1, z1, label_field) == label
    l111 = GetLabel(x1, y1, z1, label_field) == label

    c000 = vector_field[x0, y0, z0]
    c100 = vector_field[x1, y0, z0]
    c010 = vector_field[x0, y1, z0]
    c110 = vector_field[x1, y1, z0]
    c001 = vector_field[x0, y0, z1]
    c101 = vector_field[x1, y0, z1]
    c011 = vector_field[x0, y1, z1]
    c111 = vector_field[x1, y1, z1]

    c00 = VectorOrientationAddition(c000 * (1 - xd * l100), c100 * xd * l000)
    c01 = VectorOrientationAddition(c001 * (1 - xd * l101), c101 * xd * l001)
    c10 = VectorOrientationAddition(c010 * (1 - xd * l110), c110 * xd * l010)
    c11 = VectorOrientationAddition(c011 * (1 - xd * l111), c111 * xd * l011)

    c0 = VectorOrientationAddition(c00 * (1 - yd * np.any(c10)),
                                   c10 * yd * np.any(c00))
    c1 = VectorOrientationAddition(c01 * (1 - yd * np.any(c11)),
                                   c11 * yd * np.any(c01))

    return VectorOrientationAddition(c0 * (1 - zd * np.any(c1)),
                                     c1 * zd * np.any(c0)).astype(
                                         vector_field.dtype)


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
                xs = (0.5 + x) / scale
                ys = (0.5 + y) / scale
                zs = (0.5 + z) / scale

                # print(x, y, z, "<->", xs, ys, zs)
                vector_field_intp[x, y,
                                  z, :] = GetVec(xs, ys, zs, label_field,
                                                 vector_field, interpolate)

    return vector_field_intp
