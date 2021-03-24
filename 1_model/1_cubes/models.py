import numpy as np
import h5py

import fastpli.analysis
import fastpli.tools
import fastpli.io


def inclinations(n=10):
    return np.linspace(0, 90, n, True)


def omega_rotations(omega, dphi=10):

    rot = []

    n_rot = int(
        np.round(
            np.sqrt((1 - np.cos(2 * np.deg2rad(omega))) /
                    (1 - np.cos(np.deg2rad(dphi))))))
    if n_rot == 0:
        rot.append(0)
    else:
        n_rot += (n_rot + 1) % 2
        n_rot = max(n_rot, 3)
        for f_rot in np.linspace(-90, 90, n_rot, True):
            f_rot = np.round(f_rot, 2)
            rot.append(f_rot)

    return rot


def rotate(fbs, f0_inc, f1_rot):
    if f0_inc != 0 or f1_rot != 0:
        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)
        fbs = fbs.rotate(rot)
    return fbs


def ori_from_fbs(fbs, f0_inc=0, f1_rot=0, cut=None):
    fbs = rotate(fbs, f0_inc, f1_rot)
    if cut:
        if isinstance(cut, (int, float)):
            cut = [[-cut / 2] * 3, [cut / 2] * 3]
        fbs = fbs.cut(cut)
    return fastpli.analysis.orientation.fiber_bundles(fbs)


def ori_from_file(file, f0_inc, f1_rot, cut=None):
    fbs = fastpli.io.fiber_bundles.load(file)
    return ori_from_fbs(fbs, f0_inc, f1_rot, cut)


def vec_from_fbs(fbs, f0_inc=0, f1_rot=0, cut=None):
    fbs = rotate(fbs, f0_inc, f1_rot)
    if cut:
        if isinstance(cut, (int, float)):
            cut = [[-cut / 2] * 3, [cut / 2] * 3]
        fbs = fbs.cut(cut)

    vecs = []
    for fb in fbs:
        vec = []
        for f in fb:
            if f.shape[0] <= 1:
                continue

            vec.extend(f[1:, :-1] - f[0:-1, :-1])
            # print(f[1:, :-1], f[0:-1, :-1], f[1:, :-1] - f[0:-1, :-1])

        vec = np.atleast_2d(np.array(vec))

        # if vec.size > 0:
        #     vec[vec[:, 0] < 0, :] = -vec[vec[:, 0] < 0, :]
        vecs.append(vec)

    if len(vecs) == 1:
        vecs.append(np.empty(shape=(0, 0)))

    return vecs


def vec_from_file(file, f0_inc, f1_rot, cut=None):
    fbs = fastpli.io.fiber_bundles.load(file)
    return vec_from_fbs(fbs, f0_inc, f1_rot, cut)
