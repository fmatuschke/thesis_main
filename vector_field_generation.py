import numpy as np
import os
from numba import njit

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)


@njit(cache=True)
def ind3(x, y, z, shape):
    return x * shape[1] * shape[2] + y * shape[2] + z


@njit(cache=True)
def ind4(x, y, z, shape):
    return ind3(x, y, z, shape) * 3


@njit(cache=True)
def GetLabel(x, y, z, label_field, shape):
    return label_field[ind3(x, y, z, shape)]


@njit(cache=True)
def GetVec(x, y, z, vector_field, shape):
    i = ind4(x, y, z, shape)
    return vector_field[i:i + 3]


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
def InterpolateVec(x, y, z, vector_field, label_field, shape):
    # Trilinear interpolate
    x0 = min(max(int(np.floor(x)), 0), shape[0] - 1)
    y0 = min(max(int(np.floor(y)), 0), shape[1] - 1)
    z0 = min(max(int(np.floor(z)), 0), shape[2] - 1)

    x1 = max(min(int(np.ceil(x)), shape[0] - 1), 0)
    y1 = max(min(int(np.ceil(y)), shape[1] - 1), 0)
    z1 = max(min(int(np.ceil(z)), shape[2] - 1), 0)

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


@njit(cache=True)
def GetVector(x, y, z, label_field, vector_field, shape, interpolate):
    if interpolate:
        return InterpolateVec(x, y, z, vector_field, label_field, shape)

    # Nearest Neighbor
    return GetVec(int(np.floor(x)), int(np.floor(y)), int(np.floor(z)),
                  vector_field, shape)


@njit(cache=True)
def IntpVecField_(x, scale, label_field, vector_field, vector_field_intp, shape,
                  interpolate):
    sx = int(round(shape[0] * scale))
    sy = int(round(shape[1] * scale))
    sz = int(round(shape[2] * scale))

    scale_int = np.array([sx, sy, sz])
    for y in range(sy):
        for z in range(sz):
            xs = (0.5 + x) / scale
            ys = (0.5 + y) / scale
            zs = (0.5 + z) / scale

            i = ind4(x, y, z, scale_int)
            vector_field_intp.ravel()[i:i + 3] = GetVector(
                xs, ys, zs, label_field, vector_field, shape, interpolate)


# @njit(cache=True)
def IntpVecField(label_field,
                 vector_field,
                 scale,
                 interpolate=True,
                 mp_pool=None):

    sx = int(round(vector_field.shape[0] * scale))
    sy = int(round(vector_field.shape[1] * scale))
    sz = int(round(vector_field.shape[2] * scale))

    vector_field_intp = np.zeros((sx, sy, sz, 3), vector_field.dtype)

    if mp_pool is None:
        for x in range(sx):
            IntpVecField_(x, scale, label_field.ravel(), vector_field.ravel(),
                          vector_field_intp.ravel(), label_field.shape,
                          interpolate)
    else:
        import ctypes
        from multiprocessing.sharedctypes import RawArray

        label_field_base_ptr_ = RawArray(ctypes.c_int32, label_field.size)
        vector_field_base_ptr_ = RawArray(ctypes.c_float, vector_field.size)
        vector_field_intp_base_ptr_ = RawArray(ctypes.c_float,
                                               vector_field_intp.size)
        label_field_ptr_ = np.frombuffer(label_field_base_ptr_)
        vector_field_ptr_ = np.frombuffer(vector_field_base_ptr_)
        vector_field_intp_ptr_ = np.frombuffer(vector_field_intp_base_ptr_)
        label_field_ptr_ = label_field
        vector_field_ptr_ = vector_field
        vector_field_intp_ptr_ = vector_field_intp

        # TODO: check if shr memory is copied!
        chunk = [(x, scale, label_field_ptr_, vector_field_ptr_,
                  vector_field_intp_ptr_, label_field.shape, interpolate)
                 for x in range(sx)]

        mp_pool.starmap(IntpVecField_, chunk)
        # for i in range(len(results)):
        #     vector_field_intp[i, :] = results[i]

    return vector_field_intp
