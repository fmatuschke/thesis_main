import numpy as np
import os
from numba import njit

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)


# @njit(cache=True)
def ind3(x, y, z, shape):
    return x * np.prod(shape[1:]) + y * np.prod(shape[2:]) + z * np.prod(
        shape[2:])


# @njit(cache=True)
def ind4(x, y, z, shape):
    return ind3(x, y, z, shape) * 3


# @njit(cache=True)
def GetLabel(x, y, z, label_field, shape):
    if x < 0 or x >= shape[0] or y < 0 or y >= shape[1] or z < 0 or z >= shape[
            2]:
        return 0

    return label_field[ind3(int(x), int(y), int(z), shape)]


# @njit(cache=True)
def GetVec(x, y, z, vector_field, shape):
    if x < 0 or x >= shape[0] or y < 0 or y >= shape[1] or z < 0 or z >= shape[
            2]:
        return 0

    return vector_field[ind4(int(x), int(y), int(z), shape)]


# @njit(cache=True)
def VectorOrientationAddition(v, u):
    if np.dot(v, u) >= 0:
        return v + u
    else:
        return v - u


# @njit(cache=True)
def VectorOrientationSubstraction(v, u):
    if np.dot(v, u) >= 0:
        return v - u
    else:
        return v + u


# @njit(cache=True)
def VectorOrientationSubstractionField(v, u):
    shape = v.shape
    v = np.reshape(v, (-1, 3))
    u = np.reshape(u, (-1, 3))
    result = np.empty_like(v)
    for i in range(v.shape[0]):
        result[i, :] = VectorOrientationSubstraction(v[i, :], u[i, :])
    result = np.reshape(result, shape)
    return result


# @njit(cache=True)
def InterpolateVec(x, y, z, vector_field, label_field, shape):
    # Trilinear interpolate
    x0 = int(min(max(np.floor(x), 0), shape[0] - 1))
    y0 = int(min(max(np.floor(y), 0), shape[1] - 1))
    z0 = int(min(max(np.floor(z), 0), shape[2] - 1))

    x1 = int(max(min(np.ceil(x), shape[0] - 1), 0))
    y1 = int(max(min(np.ceil(y), shape[1] - 1), 0))
    z1 = int(max(min(np.ceil(z), shape[2] - 1), 0))

    if x0 == x1 and y0 == y1 and z0 == z1:
        return GetVec(x0, y0, z0, vector_field, shape)

    xd = 0 if x0 == x1 else (x - x0) / (x1 - x0)
    yd = 0 if y0 == y1 else (y - y0) / (y1 - y0)
    zd = 0 if z0 == z1 else (z - z0) / (z1 - z0)

    label = GetLabel(x0, y0, z0, label_field, shape)  # nearest neighbor

    l000 = GetLabel(x0, y0, z0, label_field, shape) == label
    l100 = GetLabel(x1, y0, z0, label_field, shape) == label
    l010 = GetLabel(x0, y1, z0, label_field, shape) == label
    l110 = GetLabel(x1, y1, z0, label_field, shape) == label
    l001 = GetLabel(x0, y0, z1, label_field, shape) == label
    l101 = GetLabel(x1, y0, z1, label_field, shape) == label
    l011 = GetLabel(x0, y1, z1, label_field, shape) == label
    l111 = GetLabel(x1, y1, z1, label_field, shape) == label

    c000 = GetVec(x0, y0, z0, vector_field, shape)
    c100 = GetVec(x1, y0, z0, vector_field, shape)
    c010 = GetVec(x0, y1, z0, vector_field, shape)
    c110 = GetVec(x1, y1, z0, vector_field, shape)
    c001 = GetVec(x0, y0, z1, vector_field, shape)
    c101 = GetVec(x1, y0, z1, vector_field, shape)
    c011 = GetVec(x0, y1, z1, vector_field, shape)
    c111 = GetVec(x1, y1, z1, vector_field, shape)

    c00 = VectorOrientationAddition(c000 * (1 - xd * l100), c100 * xd * l000)
    c01 = VectorOrientationAddition(c001 * (1 - xd * l101), c101 * xd * l001)
    c10 = VectorOrientationAddition(c010 * (1 - xd * l110), c110 * xd * l010)
    c11 = VectorOrientationAddition(c011 * (1 - xd * l111), c111 * xd * l011)

    c0 = VectorOrientationAddition(c00 * (1 - yd * np.any(c10)),
                                   c10 * yd * np.any(c00))
    c1 = VectorOrientationAddition(c01 * (1 - yd * np.any(c11)),
                                   c11 * yd * np.any(c01))

    c = VectorOrientationAddition(c0 * (1 - zd * np.any(c1)),
                                  c1 * zd * np.any(c0)).astype(
                                      vector_field.dtype)

    if np.any(c):
        c /= np.linalg.norm(c)

    return c


# @njit(cache=True)
def GetVector(x, y, z, label_field, vector_field, shape, interpolate):
    if interpolate:
        return InterpolateVec(x, y, z, vector_field, label_field, shape)

    # Nearest Neighbor
    return GetVec(int(np.floor(x)), int(np.floor(y)), int(np.floor(z)),
                  vector_field, shape)


# @njit(cache=True)
def IntpVecField_(x, scale, label_field, vector_field, vector_field_intp, shape,
                  interpolate):
    sy = int(round(shape[1] * scale))
    sz = int(round(shape[2] * scale))

    # vector_field_intp = np.empty((sy, sz, 3), vector_field.dtype)
    for y in range(sy):
        for z in range(sz):
            xs = (0.5 + x) / scale
            ys = (0.5 + y) / scale
            zs = (0.5 + z) / scale

            # print(x, y, z, "<->", xs, ys, zs)
            vector_field_intp[ind4(x, y, z, shape):ind4(x, y, z, shape) +
                              3] = GetVector(xs, ys, zs, label_field,
                                             vector_field, shape, interpolate)
    # return vector_field_intp


# @njit(cache=True)
def IntpVecField(label_field,
                 vector_field,
                 scale,
                 interpolate=True,
                 mp_pool=None):

    sx = int(round(vector_field.shape[0] * scale))
    sy = int(round(vector_field.shape[1] * scale))
    sz = int(round(vector_field.shape[2] * scale))

    vector_field_intp = np.empty((sx, sy, sz, 3), vector_field.dtype)

    if mp_pool is None:
        for x in range(sx):
            IntpVecField_(x, scale,
                          label_field.flatten(), vector_field.flatten(),
                          vector_field_intp.flatten(), label_field.shape,
                          interpolate)
            # for y in range(sy):
            #     for z in range(sz):
            #         xs = (0.5 + x) / scale
            #         ys = (0.5 + y) / scale
            #         zs = (0.5 + z) / scale

            #         vector_field_intp[x, y,
            #                           z, :] = GetVec(xs, ys, zs,
            #                                          label_field.flatten(),
            #                                          vector_field.flatten(),
            #                                          label_field.shape,
            #                                          interpolate)
    else:
        from multiprocessing import shared_memory

        label_field_shr_ = shared_memory.SharedMemory(create=True,
                                                      size=label_field.nbytes)
        vector_field_shr_ = shared_memory.SharedMemory(create=True,
                                                       size=vector_field.nbytes)
        vector_field_intp_shr_ = shared_memory.SharedMemory(
            create=True, size=vector_field_intp.nbytes)

        # TODO: check if shr memory is copied!
        chunk = [(x, scale, label_field_shr_, vector_field_shr_,
                  vector_field_intp_shr_, label_field.shape, interpolate)
                 for x in range(sx)]

        mp_pool.starmap(IntpVecField_, chunk)
        # for i in range(len(results)):
        #     vector_field_intp[i, :] = results[i]

    return vector_field_intp
