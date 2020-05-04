import numpy as np
import os
from numba import njit, vectorize, float32, float64, prange


@njit(cache=True)
def VectorOrientationAddition(v, u):
    if np.dot(v, u) >= 0:
        return v + u
    else:
        return v - u


# @vectorize([float32[:](float32[:], float32[:]),
#             float64[:](float64[:], float64[:])])
@njit(cache=True)
def VectorOrientationSubstraction(v, u):
    if np.dot(v, u) >= 0:
        return v - u
    else:
        return v + u


@njit(cache=True, parallel=True)
def VectorOrientationSubstractionField(v, u):
    shape = v.shape
    v = np.reshape(v, (-1, 3))
    u = np.reshape(u, (-1, 3))
    result = np.empty_like(v)
    for i in prange(v.shape[0]):
        result[i, :] = VectorOrientationSubstraction(v[i, :], u[i, :])
    result = np.reshape(result, shape)
    return result


# @njit(cache=True)
# def InterpolateVec(x, y, z, vector_field, label_field):
#     shape = label_field.shape

#     # Trilinear interpolate
#     x0 = min(max(int(np.floor(x)), 0), shape[0] - 1)
#     y0 = min(max(int(np.floor(y)), 0), shape[1] - 1)
#     z0 = min(max(int(np.floor(z)), 0), shape[2] - 1)

#     x1 = min(max(int(np.ceil(x)), 0), shape[0] - 1)
#     y1 = min(max(int(np.ceil(y)), 0), shape[1] - 1)
#     z1 = min(max(int(np.ceil(z)), 0), shape[2] - 1)

#     if x0 == x1 and y0 == y1 and z0 == z1:
#         return vector_field[x0, y0, z0, :]

#     xd = 0 if x0 == x1 else (x - x0) / (x1 - x0)
#     yd = 0 if y0 == y1 else (y - y0) / (y1 - y0)
#     zd = 0 if z0 == z1 else (z - z0) / (z1 - z0)

#     label = label_field[x0, y0, z0]  #  nearest neighbor

#     l000 = label_field[x0, y0, z0] == label
#     l100 = label_field[x1, y0, z0] == label
#     l010 = label_field[x0, y1, z0] == label
#     l110 = label_field[x1, y1, z0] == label
#     l001 = label_field[x0, y0, z1] == label
#     l101 = label_field[x1, y0, z1] == label
#     l011 = label_field[x0, y1, z1] == label
#     l111 = label_field[x1, y1, z1] == label

#     c000 = vector_field[x0, y0, z0, :]
#     c100 = vector_field[x1, y0, z0, :]
#     c010 = vector_field[x0, y1, z0, :]
#     c110 = vector_field[x1, y1, z0, :]
#     c001 = vector_field[x0, y0, z1, :]
#     c101 = vector_field[x1, y0, z1, :]
#     c011 = vector_field[x0, y1, z1, :]
#     c111 = vector_field[x1, y1, z1, :]

#     c00 = VectorOrientationAddition(c000 * (1 - xd * l100), c100 * xd * l000)
#     c01 = VectorOrientationAddition(c001 * (1 - xd * l101), c101 * xd * l001)
#     c10 = VectorOrientationAddition(c010 * (1 - xd * l110), c110 * xd * l010)
#     c11 = VectorOrientationAddition(c011 * (1 - xd * l111), c111 * xd * l011)

#     c0 = VectorOrientationAddition(c00 * (1 - yd * np.any(c10)),
#                                    c10 * yd * np.any(c00))
#     c1 = VectorOrientationAddition(c01 * (1 - yd * np.any(c11)),
#                                    c11 * yd * np.any(c01))

#     c = VectorOrientationAddition(c0 * (1 - zd * np.any(c1)),
#                                   c1 * zd * np.any(c0)).astype(
#                                       vector_field.dtype)

#     if np.any(c):
#         c /= np.linalg.norm(c)

#     return c


# @njit(cache=True)
# def InterpolateVec_new(x, y, z, vector_field, label_field):
#     shape = label_field.shape

#     # Trilinear interpolate
#     x0 = min(max(int(np.floor(x - 0.5)), 0), shape[0] - 1)
#     y0 = min(max(int(np.floor(y - 0.5)), 0), shape[1] - 1)
#     z0 = min(max(int(np.floor(z - 0.5)), 0), shape[2] - 1)

#     x1 = min(max(int(np.ceil(x - 0.5)), 0), shape[0] - 1)
#     y1 = min(max(int(np.ceil(y - 0.5)), 0), shape[1] - 1)
#     z1 = min(max(int(np.ceil(z - 0.5)), 0), shape[2] - 1)

#     if x0 == x1 and y0 == y1 and z0 == z1:
#         return vector_field[x0, y0, z0, :]

#     xd = 0 if x0 == x1 else (x - x0) / (x1 - x0)
#     yd = 0 if y0 == y1 else (y - y0) / (y1 - y0)
#     zd = 0 if z0 == z1 else (z - z0) / (z1 - z0)

#     xn = int(np.floor(x))
#     yn = int(np.floor(y))
#     zn = int(np.floor(z))
#     label = label_field[xn, yn, zn]  #  nearest neighbor

#     l000 = label_field[x0, y0, z0] == label
#     l100 = label_field[x1, y0, z0] == label
#     l010 = label_field[x0, y1, z0] == label
#     l110 = label_field[x1, y1, z0] == label
#     l001 = label_field[x0, y0, z1] == label
#     l101 = label_field[x1, y0, z1] == label
#     l011 = label_field[x0, y1, z1] == label
#     l111 = label_field[x1, y1, z1] == label

#     c000 = vector_field[x0, y0, z0, :]
#     c100 = vector_field[x1, y0, z0, :]
#     c010 = vector_field[x0, y1, z0, :]
#     c110 = vector_field[x1, y1, z0, :]
#     c001 = vector_field[x0, y0, z1, :]
#     c101 = vector_field[x1, y0, z1, :]
#     c011 = vector_field[x0, y1, z1, :]
#     c111 = vector_field[x1, y1, z1, :]

#     c00 = VectorOrientationAddition(c000 * (1 - xd * l100), c100 * xd * l000)
#     c01 = VectorOrientationAddition(c001 * (1 - xd * l101), c101 * xd * l001)
#     c10 = VectorOrientationAddition(c010 * (1 - xd * l110), c110 * xd * l010)
#     c11 = VectorOrientationAddition(c011 * (1 - xd * l111), c111 * xd * l011)

#     c0 = VectorOrientationAddition(c00 * (1 - yd * np.any(c10)),
#                                    c10 * yd * np.any(c00))
#     c1 = VectorOrientationAddition(c01 * (1 - yd * np.any(c11)),
#                                    c11 * yd * np.any(c01))

#     c = VectorOrientationAddition(c0 * (1 - zd * np.any(c1)),
#                                   c1 * zd * np.any(c0)).astype(
#                                       vector_field.dtype)

#     if np.any(c):
#         c /= np.linalg.norm(c)

#     return c


# @njit(cache=True)
# def GetVector(x, y, z, label_field, vector_field, interpolate):
#     if interpolate:
#         # return InterpolateVec(x, y, z, vector_field, label_field)
#         return InterpolateVec_new(x, y, z, vector_field, label_field)

#     # Nearest Neighbor
#     return vector_field[int(np.floor(x)), int(np.floor(y)), int(np.floor(z)), :]


# @njit(cache=True)
# def IntpVecField_(x, scale, label_field, vector_field, interpolate):

#     shape = label_field.shape

#     sy = int(round(shape[1] * scale))
#     sz = int(round(shape[2] * scale))

#     vector_field_intp = np.empty((sy, sz, 3), vector_field.dtype)

#     for y in range(sy):
#         for z in range(sz):
#             xs = (0.5 + x) / scale
#             ys = (0.5 + y) / scale
#             zs = (0.5 + z) / scale

#             vector_field_intp[y, z, :] = GetVector(xs, ys, zs, label_field,
#                                                    vector_field, interpolate)

#     return vector_field_intp


# # @njit(cache=True)
# def IntpVecField(label_field,
#                  vector_field,
#                  scale,
#                  interpolate=True,
#                  mp_pool=None):

#     sx = int(round(vector_field.shape[0] * scale))
#     sy = int(round(vector_field.shape[1] * scale))
#     sz = int(round(vector_field.shape[2] * scale))

#     vector_field_intp = np.zeros((sx, sy, sz, 3), vector_field.dtype)

#     if mp_pool is None:
#         for x in range(sx):
#             vector_field_intp[x, :] = IntpVecField_(x, scale, label_field,
#                                                     vector_field, interpolate)
#     else:
#         chunk = [(x, scale, label_field, vector_field, interpolate)
#                  for x in range(sx)]

#         # WARNING copies every argument n_pool times
#         results = mp_pool.starmap(IntpVecField_, chunk)

#         for i in range(len(results)):
#             vector_field_intp[i, :] = results[i]

#     return vector_field_intp
