import numpy as np


def GetLabel(x, y, z, label_field):

    if x < 0 or x >= label_field.shape[0] or y < 0 or y >= label_field.shape[
            1] or z < 0 or z >= label_field.shape[2]:
        return 0

    return label_field[int(x), int(y), int(z)]


def VectorOrientationAddition(v, u):

    if np.dot(v, u) >= 0:
        return v + u
    else:
        return v - u


def InterpolateVec(x, y, z, vector_field):
    # Trilinear interpolate
    x0 = max(np.floor(x), 0)
    y0 = max(np.floor(y), 0)
    z0 = max(np.floor(z), 0)

    x1 = min(np.ceil(x), vector_field.shape[0] - 1)
    y1 = min(np.ceil(y), vector_field.shape[1] - 1)
    z1 = min(np.ceil(z), vector_field.shape[2] - 1)

    if x0 == x1 and y0 == y1 and z0 == z1:
        return vector_field[x0, y0, z0]

    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)

    if x0 == x1:
        xd = 0
    if y0 == y1:
        yd = 0
    if z0 == z1:
        zd = 0

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

    return VectorOrientationAddition(c0 * (1 - zd), c1 * zd)


def GetVec(x, y, z, label_field, vector_field, interpolate):
    if interpolate:
        # only interpolate if all neighbors are the same tissue

        label = GetLabel(x, y, z, label_field)

        labels = [
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
        ]

        if labels.count() == len(labels):
            return InterpolateVec(x, y, z, vector_field)

    # Nearest Neighbor
    return vector_field[int(x), int(y), int(z)]


if __name__ == "__main__":
    import fastpli.simulation
    import fastpli.analysis
    import fastpli.tools
    import fastpli.io
